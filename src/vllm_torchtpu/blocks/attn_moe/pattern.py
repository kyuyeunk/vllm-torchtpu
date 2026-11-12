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
"""The attn_then_moe pattern: a reference forward and what each role must be.

Framework-owned. A model owner declares that a decoder layer conforms to this
pattern with a binding function returning ``AttnMoEBinding`` (``declare_block``).
The conformance check then holds the layer to two things:

- **Kinds.** Each part is what its field says (``AttnMoEBinding.problems``, run
  when the binding are built): two RMSNorm-family norms called with a residual;
  an attention child holding an attention layer and ending in a
  ``RowParallelLinear`` to the hidden size; a MoE child owning the MoE runner
  and the ``GateLinear`` router.
- **Wiring.** With each role made opaque, the layer's forward computes the
  same graph as ``attn_then_moe`` below, and the MoE child's forward the same
  graph as ``moe_explicit_router`` or ``moe_runner_router``.
"""

import dataclasses
from collections.abc import Callable
from typing import ClassVar, Protocol, override

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.blocks.attn_moe.plan import BlockKey, MoEPath, RouterMode
from vllm_torchtpu.blocks.attn_moe.runner import MoEBinding
from vllm_torchtpu.blocks.binding import BlockBinding, as_kind
from vllm_torchtpu.blocks.norm_attn.pattern import is_residual_norm
from vllm_torchtpu.blocks.stream import Attend, Stream

Norm = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Op = Callable[[torch.Tensor], torch.Tensor]


def attn_then_moe(  # noqa: PLR0913 - the pattern takes one callable per role
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


def block_key(binding: "AttnMoEBinding", tp: int, dtype: torch.dtype) -> BlockKey:
    """The plan's key, read from the declared binding."""
    runner = as_kind(binding.experts, MoERunner)
    router = as_kind(binding.router, GateLinear)
    cfg = runner.moe_config
    sp = bool(cfg.is_sequence_parallel)
    if cfg.moe_parallel_config.use_all2all_kernels:
        path = MoEPath.ALL2ALL
    elif sp:
        path = MoEPath.SEQUENCE_PARALLEL
    else:
        path = MoEPath.LATE_ALL_REDUCE
    return BlockKey(
        tp=tp,
        hidden=router.input_size,
        dtype=dtype,
        attention_kind=type(binding.attention),
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
        fused_output_reduced=runner._fused_output_is_reduced,  # noqa: SLF001 - no public accessor  # pyright: ignore[reportPrivateUsage]
        moe_path=path,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttnMoEBinding(MoEBinding, BlockBinding):
    """A declared layer's binding for attn_then_moe.

    The runner's binding (``MoEBinding``: post norm, router, experts) plus the
    layer's other roles: the input norm, the attention child (run whole, or
    through the nested ``AttentionRunner``), attention's output projection
    (declared, so the block never searches for it; only a layer without a
    nested sub-block needs it, for the interim unreduced write) and the MoE
    child that owns the router and the experts.
    """

    input_norm: nn.Module
    attention: nn.Module
    out_proj: RowParallelLinear
    moe: nn.Module
    stream: Stream
    attend: Attend

    pattern: ClassVar[Callable[..., object]] = attn_then_moe

    @override
    def layer_children(self) -> dict[str, object]:
        return {
            "input_norm": self.input_norm,
            "attention": self.attention,
            "post_norm": self.post_norm,
            "moe": self.moe,
        }

    @override
    def problems(self) -> list[tuple[str, object, str]]:
        problems: list[tuple[str, object, str]] = [
            (name, norm, f"is a {type(norm).__name__}, not a vLLM RMSNorm")
            for name, norm in (
                ("input_norm", self.input_norm),
                ("post_norm", self.post_norm),
            )
            if not is_residual_norm(norm)
        ]
        out_proj = self.out_proj
        if not isinstance(out_proj, RowParallelLinear):  # pyright: ignore[reportUnnecessaryIsInstance] - a binding outside the type checker
            why = f"is a {type(out_proj).__name__}, not a RowParallelLinear"
            problems.append(("out_proj", out_proj, why))
        if not any(m is out_proj for m in self.attention.modules()):
            problems.append(("out_proj", out_proj, "is not a module of attention"))
        kinds: tuple[tuple[str, object, type, str], ...] = (
            ("router", self.router, GateLinear, "a GateLinear"),
            ("experts", self.experts, MoERunner, "a MoE runner"),
        )
        children = list(self.moe.children())
        for name, part, kind, words in kinds:
            if not isinstance(part, kind):
                problems.append(
                    (name, part, f"is a {type(part).__name__}, not {words}")
                )
            if not any(c is part for c in children):
                problems.append((name, part, "is not a child of moe"))
        router = self.router
        if (
            not problems
            and isinstance(router, GateLinear)
            and out_proj.output_size != router.input_size
        ):
            why = f"projects to {out_proj.output_size}, not hidden {router.input_size}"
            problems.append(("out_proj", out_proj, why))
        return problems


class RoutedMoE(Protocol):
    """A MoE child with vLLM's usual part names."""

    @property
    def gate(self) -> GateLinear:
        """The part's ``gate``."""
        ...

    @property
    def experts(self) -> nn.Module:
        """The part's ``experts``."""
        ...


class PreNormMoELayer(Protocol):
    """A pre-norm decoder layer with vLLM's usual child names (attention aside)."""

    @property
    def input_layernorm(self) -> nn.Module:
        """The part's ``input_layernorm``."""
        ...

    @property
    def post_attention_layernorm(self) -> nn.Module:
        """The part's ``post_attention_layernorm``."""
        ...


def pre_norm_moe_binding(  # noqa: PLR0913 - one argument per declared part
    layer: PreNormMoELayer,
    attention: nn.Module,
    out_proj: RowParallelLinear,
    moe: RoutedMoE,
    *,
    stream: Stream,
    attend: Attend,
) -> AttnMoEBinding:
    """The binding of a layer, its attention and its MoE, under vLLM's usual names."""
    return AttnMoEBinding(
        input_norm=layer.input_layernorm,
        attention=attention,
        out_proj=out_proj,
        post_norm=layer.post_attention_layernorm,
        moe=moe,  # pyright: ignore[reportArgumentType] - a Protocol over an nn.Module
        router=moe.gate,
        experts=moe.experts,
        stream=stream,
        attend=attend,
    )
