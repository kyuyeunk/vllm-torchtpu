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
pattern and binds its children to the four roles (``declare_block``). The
conformance check then holds the layer to two things:

- **Kinds.** Each bound child is what its role says (``check_roles``): two
  RMSNorm-family norms called with a residual; an attention child holding an
  attention layer and ending in a ``RowParallelLinear`` to the hidden size; a
  MoE child owning a MoE runner and a ``GateLinear`` router.
- **Wiring.** With each role made opaque, the layer's forward computes the
  same graph as ``attn_then_moe`` below, and the MoE child's forward the same
  graph as ``moe_explicit_router`` or ``moe_runner_router``.
"""

import dataclasses
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import RowParallelLinear

Norm = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Op = Callable[[torch.Tensor], torch.Tensor]
ROLES = ("in_norm", "attn", "post_norm", "moe")


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


def _is_norm(module: nn.Module) -> bool:
    return isinstance(module, (RMSNorm, GemmaRMSNorm)) and (
        "residual" in module.forward_native.__code__.co_varnames
    )


def check_roles(layer: nn.Module, binding: dict[str, str]) -> list[str]:
    """Why the bound children are not the kinds their roles need."""
    problems = []
    for role in ("in_norm", "post_norm"):
        if not _is_norm(layer.get_submodule(binding[role])):
            problems.append(
                f"{role}={binding[role]} is not an RMSNorm called with a residual"
            )
    attn = layer.get_submodule(binding["attn"])
    if not any(isinstance(m, AttentionLayerBase) for m in attn.modules()):
        problems.append(f"attn={binding['attn']} holds no attention layer")
    moe = layer.get_submodule(binding["moe"])
    runners = [c for c in moe.children() if isinstance(c, MoERunnerInterface)]
    routers = [c for c in moe.children() if isinstance(c, GateLinear)]
    if len(runners) != 1 or len(routers) != 1:
        problems.append(
            f"moe={binding['moe']} owns {len(runners)} MoE runners and "
            f"{len(routers)} GateLinear routers, not one each"
        )
    elif out_proj_of(attn, int(routers[0].input_size)) is None:
        problems.append(
            f"attn={binding['attn']} has no RowParallelLinear to the hidden size"
        )
    return problems


def out_proj_of(attn: nn.Module, hidden: int) -> str | None:
    """The last RowParallelLinear to ``hidden`` in attention's subtree."""
    found = None
    for name, sub in attn.named_modules():
        if isinstance(sub, RowParallelLinear) and sub.output_size == hidden:
            found = name
    return found


def moe_parts(moe: nn.Module) -> tuple[str, str]:
    """The MoE child's runner and router names (after ``check_roles``)."""
    runner = next(
        n for n, c in moe.named_children() if isinstance(c, MoERunnerInterface)
    )
    router = next(n for n, c in moe.named_children() if isinstance(c, GateLinear))
    return runner, router


def block_key(
    layer: nn.Module, binding: dict[str, str], tp: int, dtype: str
) -> BlockKey:
    """The plan's key, read from the bound children."""
    moe = layer.get_submodule(binding["moe"])
    runner_name, router_name = moe_parts(moe)
    runner, router = moe.get_submodule(runner_name), moe.get_submodule(router_name)
    cfg = runner.moe_config
    sp = bool(getattr(cfg, "is_sequence_parallel", False))
    attn = layer.get_submodule(binding["attn"])
    kind = next(
        type(m).__name__ for m in attn.modules() if isinstance(m, AttentionLayerBase)
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
