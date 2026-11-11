# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The attn_then_moe pattern: a reference forward and what each part must be.

Framework-owned. A model owner declares that a decoder layer conforms to this
pattern with a binding function returning ``AttnMoEParts``
(``declare_block``). The conformance check then holds the layer to two
things:

- **Kinds.** Each part is what its field says (``AttnMoEParts.problems``, run
  when the parts are built): two RMSNorm-family norms; an attention child
  holding an attention layer, whose output projection is a
  ``RowParallelLinear`` to the hidden size; a MoE child owning the MoE runner
  and the ``GateLinear`` router.
- **Wiring.** With each role made opaque, the layer's forward computes the
  same graph as ``attn_then_moe`` below, and the MoE child's forward the same
  graph as ``moe_explicit_router`` or ``moe_runner_router``.
"""

import dataclasses
from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.blocks.attn_moe.plan import MoEPath, RouterMode
from vllm_torchtpu.blocks.parts import BlockParts, Problem, as_kind
from vllm_torchtpu.blocks.stream import Attend, Stream

Norm = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Op = Callable[[torch.Tensor], torch.Tensor]


def attn_then_moe(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    *,
    in_norm: Norm,
    attn: Op,
    post_norm: Norm,
    moe: Op,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pattern: norm, attention, norm, MoE, on the residual stream."""
    x, residual = in_norm(hidden, residual)
    x = attn(x)
    x, residual = post_norm(x, residual)
    return moe(x), residual


def moe_explicit_router(
    x: torch.Tensor,
    *,
    router: Op,
    experts: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """The MoE child computes the logits and hands them to the runner."""
    return experts(x, router(x))


def moe_runner_router(x: torch.Tensor, *, experts: Op) -> torch.Tensor:
    """The runner holds the router and computes the logits itself."""
    return experts(x)


@dataclasses.dataclass(frozen=True)
class BlockKey:
    """What the block plan is chosen by (the type names the pattern)."""

    tp: int
    hidden: int
    dtype: torch.dtype
    attention_kind: type
    router_mode: RouterMode
    has_shared_experts: bool
    sp_moe: bool
    fused_output_reduced: bool
    moe_path: MoEPath


def is_residual_norm(module: object) -> bool:
    """vLLM's RMSNorm family, whose contract takes ``(x, residual)``."""
    return isinstance(module, (RMSNorm, GemmaRMSNorm))


def holds_attention(module: nn.Module) -> bool:
    """Whether ``module``'s subtree holds an attention layer."""
    return any(isinstance(m, AttentionLayerBase) for m in module.modules())


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardParts:
    """What the block forward calls: the layer's own objects, and how.

    Execution needs only each part's call contract, so tests build this from
    stand-in modules. A model's declaration builds ``AttnMoEParts``, which
    extends it with the parts conformance and the plan read, and checks every
    kind.

    Attributes:
        input_norm, post_norm: The two norms.
        router: The router, returning logits (``(out, bias)`` unless built
            with ``return_bias=False``).
        experts: The MoE runner, taking ``(hidden_states, router_logits)``.
        stream: How the layer's forward is called and read.
        attend: How the layer calls attention on the normed tokens.
    """

    input_norm: nn.Module
    post_norm: nn.Module
    router: nn.Module
    experts: nn.Module
    stream: Stream
    attend: Attend


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttnMoEParts(ForwardParts, BlockParts):
    """A declared layer's parts for attn_then_moe.

    The forward's parts plus the attention child (opaque to conformance; its
    kind keys the plan), attention's output projection (declared, so the
    block never searches for it; the overlap stops it reducing) and the MoE
    child that owns the router and the experts.
    """

    attention: nn.Module
    out_proj: RowParallelLinear
    moe: nn.Module

    pattern = staticmethod(attn_then_moe)

    def problems(self) -> list[Problem]:
        problems = [
            Problem(name, norm, "is not an RMSNorm called with a residual")
            for name, norm in (
                ("input_norm", self.input_norm),
                ("post_norm", self.post_norm),
            )
            if not is_residual_norm(norm)
        ]
        if not holds_attention(self.attention):
            problems.append(
                Problem("attention", self.attention, "holds no attention layer")
            )
        if not isinstance(self.out_proj, RowParallelLinear):
            problems.append(
                Problem("out_proj", self.out_proj, "is not a RowParallelLinear")
            )
        elif not any(m is self.out_proj for m in self.attention.modules()):
            problems.append(
                Problem("out_proj", self.out_proj, "is not a module of attention")
            )
        children = list(self.moe.children())
        kinds = (
            ("router", self.router, GateLinear, "a GateLinear router"),
            ("experts", self.experts, MoERunnerInterface, "a MoE runner"),
        )
        for name, part, kind, words in kinds:
            if not isinstance(part, kind):
                problems.append(Problem(name, part, f"is not {words}"))
            elif not any(c is part for c in children):
                problems.append(Problem(name, part, "is not a child of moe"))
        router = self.router
        if (
            not problems
            and isinstance(router, GateLinear)
            and self.out_proj.output_size != router.input_size
        ):
            problems.append(
                Problem(
                    "out_proj",
                    self.out_proj,
                    f"projects to {self.out_proj.output_size}, not the hidden "
                    f"size {router.input_size}",
                )
            )
        return problems


class PreNormLayer(Protocol):
    """A pre-norm decoder layer with vLLM's usual norm names."""

    @property
    def input_layernorm(self) -> nn.Module: ...
    @property
    def post_attention_layernorm(self) -> nn.Module: ...


def pre_norm_moe_parts(
    layer: PreNormLayer,
    attention: nn.Module,
    out_proj: RowParallelLinear,
    moe: nn.Module,
    *,
    router: nn.Module,
    experts: nn.Module,
    stream: Stream,
    attend: Attend,
) -> AttnMoEParts:
    """The parts of a layer whose norms have vLLM's usual names."""
    return AttnMoEParts(
        input_norm=layer.input_layernorm,
        attention=attention,
        out_proj=out_proj,
        post_norm=layer.post_attention_layernorm,
        moe=moe,
        router=router,
        experts=experts,
        stream=stream,
        attend=attend,
    )


def block_key(parts: AttnMoEParts, tp: int, dtype: torch.dtype) -> BlockKey:
    """The plan's key, read from the declared parts."""
    runner = as_kind(parts.experts, MoERunner)
    router = as_kind(parts.router, GateLinear)
    cfg = runner.moe_config
    sp = bool(cfg.is_sequence_parallel)
    if cfg.moe_parallel_config.use_all2all_kernels:
        path = MoEPath.ALL2ALL
    elif sp:
        path = MoEPath.SEQUENCE_PARALLEL
    else:
        path = MoEPath.LATE_ALL_REDUCE
    attention = next(
        m for m in parts.attention.modules() if isinstance(m, AttentionLayerBase)
    )
    return BlockKey(
        tp=tp,
        hidden=router.input_size,
        dtype=dtype,
        attention_kind=type(attention),
        router_mode=(
            RouterMode.FUSED_IN_EXPERTS
            if runner.gate is router
            else RouterMode.EXPLICIT
        ),
        has_shared_experts=runner.shared_experts is not None,
        sp_moe=sp,
        # The runner answers for this layer: the fused EP kernel is armed per
        # layer, so a process-wide flag would let one armed layer answer for
        # the rest.
        fused_output_reduced=runner._fused_output_is_reduced,  # noqa: SLF001 - no public accessor
        moe_path=path,
    )
