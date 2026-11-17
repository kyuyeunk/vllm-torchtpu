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
``Attention`` carries attention's parts and how the layer calls it, and
``MoE`` the router and experts, so the declaration reads like the pattern,
one keyword per role.
"""

import dataclasses
from typing import Protocol, Self

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.blocks.binding import BlockBinding
from vllm_torchtpu.blocks.calls import Experts, Router
from vllm_torchtpu.blocks.stream import Attend, attend_with_positions


class BlockRoles:
    """Base of every pattern's ``Roles`` dataclass.

    Subclasses are frozen keyword-only dataclasses whose fields are the
    pattern's role names. ``binding`` turns them into what the pattern's
    runner and conformance check take.
    """

    def binding(self) -> BlockBinding:
        """The runner's binding, checked for kinds as it is built."""
        raise NotImplementedError


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
    """The ``attn`` role: the attention module, its parts and how it is called.

    Build it with ``Attention.of`` for an attention module with vLLM's usual
    part names; every read is typed attribute access. A block that only needs
    the module and its output projection (a Gated DeltaNet layer inside
    ``attn_then_moe``) uses ``Attention.opaque``; such a role has no parts, so
    no sub-block can fuse inside it.

    Attributes:
        module: The attention module the layer calls.
        o_proj: Its output projection.
        call: How the layer calls ``module`` on the normed tokens.
        qkv_proj: The input projection, or None for an opaque role.
        core: The attention core on ``(q, k, v)``, or None for an opaque role.
        rotary: RoPE on ``(positions, q, k)``, or None for an opaque role.
        q_norm, k_norm: Per-head norms on q and k, both or neither.
        output_gate: Whether the projection also emits a per-head output gate.
    """

    module: nn.Module
    o_proj: RowParallelLinear
    call: Attend
    qkv_proj: nn.Module | None = None
    core: nn.Module | None = None
    rotary: nn.Module | None = None
    q_norm: nn.Module | None = None
    k_norm: nn.Module | None = None
    output_gate: bool = False

    @classmethod
    def of(
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
            o_proj=module.o_proj,
            call=attend_with_positions(module),
            qkv_proj=module.qkv_proj,
            core=module.attn,
            rotary=module.rotary_emb,
            q_norm=q_norm,
            k_norm=k_norm,
            output_gate=output_gate,
        )

    @classmethod
    def opaque(
        cls, module: nn.Module, *, o_proj: RowParallelLinear, call: Attend
    ) -> Self:
        """A role that states only the module, its output projection and its call."""
        return cls(module=module, o_proj=o_proj, call=call)


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
    def of(cls, module: RoutedMoE) -> Self:
        """The role of a MoE module with vLLM's usual part names."""
        return cls(
            module=module,  # pyright: ignore[reportArgumentType] - a Protocol over an nn.Module
            router=module.gate,
            experts=module.experts,
        )
