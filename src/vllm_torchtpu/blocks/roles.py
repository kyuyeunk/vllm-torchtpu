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
"""What a model owner declares: the modules that fill a pattern's roles.

A pattern names its roles in its reference forward (``in_norm``, ``attn``,
``post_norm``, ...). A model owner's declaration returns the pattern's
``Roles`` dataclass, whose top-level fields carry exactly those names. A role
that needs more than one module states the rest under its own name:
``Attention`` (standard attention) and ``LinearAttention`` (Gated DeltaNet)
carry attention's parts, and ``MoE`` the router and experts, so the
declaration reads like the pattern, one keyword per role. Each role class
knows how the layer calls its module.
"""

import dataclasses
from typing import Protocol, Self, cast, runtime_checkable

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.blocks.binding import BlockBinding
from vllm_torchtpu.blocks.calls import Experts, Router
from vllm_torchtpu.blocks.stream import (
    Attend,
    AttentionWithoutPositions,
    AttentionWithPositions,
    Stream,
    attend_with_positions,
    attend_without_positions,
)


class BlockRoles:
    """Base of every pattern's ``Roles`` dataclass.

    Subclasses are frozen keyword-only dataclasses whose fields are the
    pattern's role names, plus ``layer_call``. ``binding`` turns them into
    what the pattern's runner takes.
    """

    layer_call: Stream  # pyright: ignore[reportUninitializedInstanceVariable] - each Roles dataclass sets it
    """How the layer's forward is called and read."""

    def binding(self) -> BlockBinding:
        """The runner's binding, checked for kinds as it is built."""
        raise NotImplementedError


@runtime_checkable
class ModuleRole(Protocol):
    """A role of several modules (``Attention``, ``MoE``): the one called."""

    @property
    def module(self) -> nn.Module:
        """The module the layer calls for this role."""
        ...


@runtime_checkable
class CalledRole(ModuleRole, Protocol):
    """A role whose module the layer calls with more than the pattern passes."""

    @property
    def call(self) -> Attend:
        """How the layer calls ``module``: the pattern's input and the positions."""
        ...


class PreNormAttention(Protocol):
    """An attention module with vLLM's usual part names, called by keyword."""

    def forward(
        self, *, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The call the layer makes."""
        ...

    @property
    def qkv_proj(self) -> nn.Module:
        """The part's ``qkv_proj``."""
        ...

    @property
    def attn(self) -> nn.Module:
        """The part's ``attn`` (the attention core)."""
        ...

    @property
    def o_proj(self) -> RowParallelLinear:
        """The part's ``o_proj``."""
        ...

    @property
    def rotary_emb(self) -> nn.Module:
        """The part's ``rotary_emb``."""
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class Attention:
    """The ``attn`` role for standard attention: the module and its parts.

    Build it with ``Attention.from_module``, which reads vLLM's usual part
    names by typed attribute access. Every part is stated, so a sub-block
    (``NormThenAttention``) can fuse inside it. The layer calls the module
    with the normed tokens and the positions.

    Attributes:
        module: The attention module the layer calls.
        qkv_proj: Its input projection.
        core: The attention core on ``(q, k, v)``.
        o_proj: Its output projection.
        rotary: RoPE on ``(positions, q, k)``.
        q_norm, k_norm: Per-head norms on q and k, both or neither.
        output_gate: Whether the projection also emits a per-head output gate.
    """

    module: nn.Module
    qkv_proj: nn.Module
    core: nn.Module
    o_proj: RowParallelLinear
    rotary: nn.Module
    q_norm: nn.Module | None = None
    k_norm: nn.Module | None = None
    output_gate: bool = False

    @classmethod
    def from_module(
        cls,
        module: PreNormAttention,
        *,
        qk_norm: tuple[nn.Module, nn.Module] | None = None,
        output_gate: bool = False,
    ) -> Self:
        """The role of an attention module with vLLM's usual part names.

        Args:
            module: The attention module.
            qk_norm: Its per-head q and k norms, where it has them.
            output_gate: Whether its projection also emits an output gate.
        """
        q_norm, k_norm = qk_norm if qk_norm is not None else (None, None)
        return cls(
            module=module,  # pyright: ignore[reportArgumentType] - a Protocol over an nn.Module
            qkv_proj=module.qkv_proj,
            core=module.attn,
            o_proj=module.o_proj,
            rotary=module.rotary_emb,
            q_norm=q_norm,
            k_norm=k_norm,
            output_gate=output_gate,
        )

    @property
    def call(self) -> Attend:
        """How the layer calls the module: normed tokens and positions."""
        # Standard attention's forward takes both (``PreNormAttention``).
        return attend_with_positions(cast("AttentionWithPositions", self.module))


class LinearAttentionModule(Protocol):
    """A linear-attention module (Gated DeltaNet) with vLLM's usual names."""

    def forward(self, *, hidden_states: torch.Tensor) -> torch.Tensor:
        """The call the layer makes; no positions."""
        ...

    @property
    def out_proj(self) -> RowParallelLinear:
        """The part's ``out_proj``."""
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class LinearAttention:
    """The ``attn`` role for linear attention (Gated DeltaNet).

    A block needs only the module and its output projection from it; its
    forward is not the standard attention composition, so no sub-block fuses
    inside it. The layer calls the module with the normed tokens only.

    Attributes:
        module: The linear-attention module the layer calls.
        o_proj: Its output projection.
    """

    module: nn.Module
    o_proj: RowParallelLinear

    @classmethod
    def from_module(cls, module: LinearAttentionModule) -> Self:
        """The role of a linear-attention module with vLLM's usual names."""
        return cls(module=module, o_proj=module.out_proj)  # pyright: ignore[reportArgumentType] - a Protocol over an nn.Module

    @property
    def call(self) -> Attend:
        """How the layer calls the module: normed tokens only."""
        # Linear attention's forward reads no positions (``LinearAttentionModule``).
        return attend_without_positions(cast("AttentionWithoutPositions", self.module))


class RoutedMoE(Protocol):
    """A MoE module with vLLM's usual part names."""

    @property
    def gate(self) -> GateLinear:
        """The part's ``gate`` (the router)."""
        ...

    @property
    def experts(self) -> nn.Module:
        """The part's ``experts`` (the MoE runner)."""
        ...


@dataclasses.dataclass(frozen=True, kw_only=True)
class MoE:
    """The ``moe`` role: the MoE module, its router and its experts.

    Attributes:
        module: The MoE module the layer calls.
        router: The router, returning logits (a ``GateLinear``).
        experts: The MoE runner, taking ``(hidden_states, router_logits)``.
    """

    module: nn.Module
    router: Router
    experts: Experts

    @classmethod
    def from_module(cls, module: RoutedMoE) -> Self:
        """The role of a MoE module with vLLM's usual part names."""
        return cls(
            module=module,  # pyright: ignore[reportArgumentType] - a Protocol over an nn.Module
            router=module.gate,
            experts=module.experts,
        )
