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
pattern with a binding function returning ``AttnMoEParts`` (``declare_block``).
The conformance check then holds the layer to two things:

- **Kinds.** Each part is what its field says (``AttnMoEParts.problems``, run
  when the parts are built): two RMSNorm-family norms called with a residual;
  an attention child holding an attention layer and ending in a
  ``RowParallelLinear`` to the hidden size; a MoE child owning the MoE runner
  and the ``GateLinear`` router.
- **Wiring.** With each role made opaque, the layer's forward computes the
  same graph as ``attn_then_moe`` below, and the MoE child's forward the same
  graph as ``moe_explicit_router`` or ``moe_runner_router``.
"""

import dataclasses
from collections.abc import Callable
from typing import Literal, Protocol

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.blocks.attn_moe.runner import MoEParts
from vllm_torchtpu.blocks.parts import BlockParts

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
    """What the block plan is chosen by."""

    kind: Literal["attn_then_moe"]
    tp: int
    hidden: int
    dtype: str
    attention_kind: str
    router_mode: Literal["explicit", "fused_in_experts"]
    has_shared_experts: bool
    sp_moe: bool
    fused_output_reduced: bool
    moe_path: Literal["late_all_reduce", "all2all", "sequence_parallel"]


@dataclasses.dataclass(frozen=True)
class AttnMoERoles:
    """The declared binding, resolved on one layer, plus what the check found.

    Attributes:
        input_norm, post_norm, attention, moe: Child names bound to the roles.
        out_proj, experts, router: Dotted paths found by the kind check.
        attention_args: Attention's keyword arguments and their sources,
            "normed" or a forward parameter, recorded by the conformance check.
        hidden_param, residual_param, positions_param: The forward's stream
            parameters ("" when there is no positions parameter).
    """

    input_norm: str
    post_norm: str
    attention: str
    out_proj: str
    moe: str
    experts: str
    router: str
    attention_args: tuple[tuple[str, str], ...]
    hidden_param: str
    residual_param: str
    positions_param: str


def out_proj_of(attn: nn.Module, hidden: int) -> str | None:
    """The last RowParallelLinear to ``hidden`` in attention's subtree."""
    found = None
    for name, sub in attn.named_modules():
        if isinstance(sub, RowParallelLinear) and sub.output_size == hidden:
            found = name
    return found


def block_key(parts: "AttnMoEParts", tp: int, dtype: str) -> BlockKey:
    """The plan's key, read from the declared parts."""
    runner, router = parts.experts, parts.router
    cfg = runner.moe_config
    sp = bool(getattr(cfg, "is_sequence_parallel", False))
    kind = next(
        type(m).__name__
        for m in parts.attention.modules()
        if isinstance(m, AttentionLayerBase)
    )
    return BlockKey(
        kind="attn_then_moe",
        tp=tp,
        hidden=int(router.input_size),
        dtype=dtype,
        attention_kind=kind,
        router_mode="fused_in_experts" if runner.gate is router else "explicit",
        has_shared_experts=getattr(runner, "shared_experts", None) is not None,
        sp_moe=sp,
        # The runner answers for this layer: the fused EP kernel is armed per
        # layer, so a process-wide flag would let one armed layer answer for
        # the rest.
        fused_output_reduced=runner._fused_output_is_reduced,  # noqa: SLF001 - no public accessor  # pyright: ignore[reportPrivateUsage]
        moe_path=(
            "all2all"
            if getattr(cfg, "use_all2all_kernels", False)
            else "sequence_parallel"
            if sp
            else "late_all_reduce"
        ),
    )


def _is_residual_norm(module: nn.Module) -> bool:
    return isinstance(module, (RMSNorm, GemmaRMSNorm)) and (
        "residual" in module.forward_native.__code__.co_varnames
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttnMoEParts(MoEParts, BlockParts):
    """A declared layer's parts for attn_then_moe.

    The runner's parts (``MoEParts``: post norm, router, experts) plus the
    layer's other roles: the input norm, the attention child (the front half,
    run whole or through the nested ``AttentionRunner``) and the MoE child
    that owns the router and the experts.
    """

    input_norm: nn.Module
    attention: nn.Module
    moe: nn.Module

    pattern = staticmethod(attn_then_moe)

    def layer_children(self) -> dict[str, nn.Module]:
        return {
            "input_norm": self.input_norm,
            "attention": self.attention,
            "post_norm": self.post_norm,
            "moe": self.moe,
        }

    def problems(self) -> list[tuple[str, str]]:
        problems = [
            (name, f"is a {type(norm).__name__}, not an RMSNorm called with a residual")
            for name, norm in (
                ("input_norm", self.input_norm),
                ("post_norm", self.post_norm),
            )
            if not _is_residual_norm(norm)
        ]
        if not any(isinstance(m, AttentionLayerBase) for m in self.attention.modules()):
            problems.append(("attention", "holds no attention layer"))
        kinds = (
            ("router", GateLinear, "a GateLinear"),
            ("experts", MoERunnerInterface, "a MoE runner"),
        )
        children = list(self.moe.children())
        for name, kind, words in kinds:
            part = getattr(self, name)
            if not isinstance(part, kind):
                problems.append((name, f"is a {type(part).__name__}, not {words}"))
            if not any(c is part for c in children):
                problems.append((name, "is not a child of moe"))
        router = self.router
        if (
            not problems
            and isinstance(router, GateLinear)
            and out_proj_of(self.attention, int(router.input_size)) is None
        ):
            problems.append(
                ("attention", "has no RowParallelLinear to the hidden size")
            )
        return problems


class RoutedMoE(Protocol):
    """A MoE child with vLLM's usual part names."""

    @property
    def gate(self) -> nn.Module: ...
    @property
    def experts(self) -> nn.Module: ...


class PreNormMoELayer(Protocol):
    """A pre-norm decoder layer with vLLM's usual child names (attention aside)."""

    @property
    def input_layernorm(self) -> nn.Module: ...
    @property
    def post_attention_layernorm(self) -> nn.Module: ...


def pre_norm_moe_parts(
    layer: PreNormMoELayer, attention: nn.Module, moe: RoutedMoE
) -> AttnMoEParts:
    """The parts of a layer, its attention and its MoE, under vLLM's usual names."""
    return AttnMoEParts(
        input_norm=layer.input_layernorm,
        attention=attention,
        post_norm=layer.post_attention_layernorm,
        moe=moe,  # type: ignore[arg-type]  # a Protocol over an nn.Module
        router=moe.gate,
        experts=moe.experts,
    )
