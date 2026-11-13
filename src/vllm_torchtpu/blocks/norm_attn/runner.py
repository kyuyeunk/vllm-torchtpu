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
"""AttentionRunner: the framework's one forward for a pre-norm attention sub-block.

Shaped like vLLM's ``MoERunner`` (``layers/fused_moe/runner/moe_runner.py``):

- ``__init__`` receives binding the model already built (the input norm, the
  QKV projection, the optional per-head q/k norms, RoPE, the attention core,
  the output projection) and keeps references to them. It creates no
  parameter, so weight loading and the model's ``state_dict`` are unchanged;
  the model keeps owning its binding.
- ``forward`` calls the binding in order. The runner owns the composition only.
- ``_select_forward`` resolves the plan once, at install, as ``MoERunner``
  picks its ``_forward_entry``.

Every model that declares ``norm_then_attention`` runs this one forward, so a
fusion of the sub-block (``fused_qkv``, PR #3096's kernel) is one branch here,
written once against the binding, never against a model class.

The fused value has a field of its own, ``residual_add``: where the residual
is added to the sublayer's output. ``FOLDED`` hands the residual to the fused
op, which adds, normalizes and returns the new residual; ``OUTSIDE`` adds it
here, in torch, and the op only normalizes. Both compute the same stream, so
both keep the sub-block's contract, but on TPU an add outside the kernel is a
value XLA recomputes in every fusion that reads it (PR #3096's first binding
added outside: Llama-70B serving regressed 15%). The op declares which
values it implements (``FusedCapability``); the plan picks ``FOLDED`` where
it can.

The runner's contract with its parent is a constructor argument,
``output_reduced``: whether its output is summed over tensor-parallel ranks.
A layer on its own needs the sum; a block that folds attention into a later
collective (``attn_then_moe``) builds the runner with ``output_reduced=False``
and does the reduction itself. Each plan value states whether it can return
an unreduced output (``plan.unreduced_output_supports``); the parent asks
before it builds.
"""

import dataclasses
import enum
from collections.abc import Callable

import torch
from torch import nn

from vllm_torchtpu.blocks import calls
from vllm_torchtpu.blocks.binding import as_kind


class InputProj(enum.Enum):
    """The runner's forwards: whether the input norm joins the projection."""

    _value_: str

    UNFUSED = "unfused"
    FUSED_QKV = "fused_qkv"


class ResidualAdd(enum.Enum):
    """Where the fused value adds the residual; unfused ignores it.

    The input norm of the unfused value takes the residual itself.
    """

    _value_: str

    FOLDED = "folded"
    """The fused op takes the residual and returns the new residual."""
    OUTSIDE = "outside"
    """The runner adds before the op, which only normalizes and projects."""


@dataclasses.dataclass(frozen=True)
class FusedCapability:
    """What a fused op serves: its author's declaration, as typed data.

    Attributes:
        residual_add: The ``ResidualAdd`` values the op implements. ``FOLDED``
            means it accepts the residual and returns the new residual it
            computed; ``OUTSIDE`` means it accepts the stream already added,
            with no residual. The first decoder layer has no residual, so it
            runs ``OUTSIDE`` whatever the plan says; an op without
            ``OUTSIDE`` leaves that layer's calls to unfused.
        unreduced: Whether the op honours ``runner.output_reduced=False``.
    """

    residual_add: frozenset[ResidualAdd]
    unreduced: bool


FUSED_CALLS: dict[str, int] = {}
"""Runs of the fused value, per layer class (eager runs only)."""

FusedOp = Callable[
    ["AttentionRunner", torch.Tensor, torch.Tensor | None, torch.Tensor],
    "tuple[torch.Tensor, torch.Tensor] | None",
]
"""``op(runner, hidden, residual, positions) -> (out, residual)`` or None to decline.

Called with a residual only under ``ResidualAdd.FOLDED``; it then returns the
new residual it computed. Called with None (``OUTSIDE``, or the first layer),
``hidden`` is the stream already added and the runner keeps that stream; the
op's second result is not read. The op reads ``runner.output_reduced`` and
returns an output summed over TP ranks only when it is True.
"""


@dataclasses.dataclass(frozen=True)
class AttnConfig:
    """The role properties the composition reads, from the attention it serves.

    Attributes:
        num_heads, num_kv_heads, head_dim: This rank's query and KV heads.
        qk_norm: Per-head RMSNorm on q and k after the split.
        output_gate: The projection also emits a per-head gate that scales
            attention's output by its sigmoid (Qwen3-Next, Qwen3.5).
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    qk_norm: bool
    output_gate: bool

    @property
    def q_size(self) -> int:
        """The part's ``q_size``."""
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        """The part's ``kv_size``."""
        return self.num_kv_heads * self.head_dim


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttentionBinding:
    """What ``AttentionRunner`` composes: the model's own part objects.

    Execution needs only each part's call contract, so tests build this from
    synthetic modules. A model's declaration builds ``NormAttnBinding``, which
    extends it with the layer's other roles and checks every kind.

    Attributes:
        input_norm: The input RMSNorm, called as ``(x, residual)``.
        qkv_proj: Attention's input projection, returning ``(out, bias)``.
        attn: The attention core on ``(q, k, v)``.
        o_proj: The output projection (a ``RowParallelLinear``).
        rotary: RoPE on ``(positions, q, k)``.
        q_norm, k_norm: Per-head norms on q and k, both or neither.
    """

    input_norm: calls.ResidualNorm
    qkv_proj: calls.Linear
    attn: calls.AttentionCore
    o_proj: calls.RowLinear
    rotary: calls.Rope
    q_norm: calls.HeadNorm | None = None
    k_norm: calls.HeadNorm | None = None


class AttentionRunner:
    """The sub-block's composition. Composition only; the model owns the parts.

    A plain object, not an ``nn.Module``: it holds no parameter or buffer, and
    the decoder layer holds it as an ordinary attribute, so the model's
    parameter tree, ``state_dict`` and weight loading are unchanged.
    """

    layer_name: str
    config: AttnConfig
    input_proj: InputProj
    residual_add: ResidualAdd
    fused_op: FusedOp | None
    fused_capability: FusedCapability | None
    output_reduced: bool
    _forward_entry: Callable[
        [torch.Tensor, torch.Tensor | None, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]

    def __init__(  # noqa: PLR0913 - the binding, then the plan, the op and the contract
        self,
        layer_name: str,
        binding: AttentionBinding,
        *,
        config: AttnConfig,
        input_proj: InputProj = InputProj.UNFUSED,
        residual_add: ResidualAdd = ResidualAdd.FOLDED,
        fused_op: FusedOp | None = None,
        fused_capability: FusedCapability | None = None,
        output_reduced: bool = True,
    ) -> None:
        """Keeps references to the binding; creates no parameter."""
        self.input_norm: calls.ResidualNorm = binding.input_norm
        self.qkv_proj: calls.Linear = binding.qkv_proj
        self.attn: calls.AttentionCore = binding.attn
        self.o_proj: calls.RowLinear = binding.o_proj
        self.rotary: calls.Rope = binding.rotary
        self.q_norm: calls.HeadNorm | None = binding.q_norm
        self.k_norm: calls.HeadNorm | None = binding.k_norm
        self.layer_name = layer_name
        self.config = config
        self.input_proj = input_proj
        self.residual_add = residual_add
        self.fused_op = fused_op
        self.fused_capability = fused_capability
        self.output_reduced = output_reduced
        self._forward_entry = self._select_forward()

    def _select_forward(
        self,
    ) -> Callable[
        [torch.Tensor, torch.Tensor | None, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """The plan, resolved once (install resolves rule, pin and capability)."""
        if self.input_proj is InputProj.FUSED_QKV:
            if self.fused_op is None:
                msg = f"{self.layer_name}: fused_qkv needs a fused op"
                raise ValueError(msg)
            if self.fused_capability is None:
                msg = f"{self.layer_name}: fused_qkv needs its op's capability"
                raise ValueError(msg)
            if self.residual_add not in self.fused_capability.residual_add:
                msg = (
                    f"{self.layer_name}: the fused op does not implement "
                    f"residual_add={self.residual_add.value}"
                )
                raise ValueError(msg)
            return self._fused_qkv
        return self._unfused

    def __call__(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the plan value ``_select_forward`` picked."""
        return self._forward_entry(hidden, residual, positions)

    # The plan values.

    def _unfused(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream's order: norm, project, split, q/k norm, RoPE, attend, project."""
        stream: torch.Tensor
        if residual is None:  # The first layer: the stream is its own residual.
            x, stream = self.input_norm(hidden), hidden
        else:
            x, stream = self.input_norm(hidden, residual)
        qkv, _ = self.qkv_proj(x)
        return self.attention_from_qkv(qkv, positions), stream

    def _fused_qkv(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fused op on the same binding, the residual added where the plan says.

        Where the op declines a call, or cannot serve it (the first layer,
        which has no residual to fold, with an op that implements only
        ``FOLDED``), unfused runs it.
        """
        op, capability = self.fused_op, self.fused_capability
        if op is None or capability is None:  # _select_forward checked
            raise ValueError(self.layer_name)
        if residual is not None and self.residual_add is ResidualAdd.FOLDED:
            out = op(self, hidden, residual, positions)
        elif ResidualAdd.OUTSIDE in capability.residual_add:
            stream = hidden if residual is None else self.add_outside(hidden, residual)
            fused = op(self, stream, None, positions)
            out = None if fused is None else (fused[0], stream)
        else:
            out = None
        if out is None:
            return self._unfused(hidden, residual, positions)
        if not torch.compiler.is_compiling():
            FUSED_CALLS[self.layer_name] = FUSED_CALLS.get(self.layer_name, 0) + 1
        return out

    @staticmethod
    def add_outside(hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """``ResidualAdd.OUTSIDE``'s add, rounded to the stream's dtype once.

        The platform's fused add-norm adds in float32 and rounds the sum, so
        this equals the input norm's own new residual.
        """
        return (hidden.float() + residual.float()).to(hidden.dtype)

    # The composition after the projection, shared by both values.

    def attention_from_qkv(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """The part's ``attention_from_qkv``."""
        c = self.config
        if c.output_gate:
            q_gate, k, v = torch.split(
                qkv, [c.q_size * 2, c.kv_size, c.kv_size], dim=-1
            )
            shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*shape, c.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*shape, -1)
            gate = gate.reshape(*shape, -1)
        else:
            q, k, v = torch.split(qkv, [c.q_size, c.kv_size, c.kv_size], dim=-1)
            gate = None
        q_norm, k_norm = self.q_norm, self.k_norm
        if q_norm is not None and k_norm is not None:  # config.qk_norm
            q = q_norm(q.view(-1, c.num_heads, c.head_dim)).view(-1, c.q_size)
            k = k_norm(k.view(-1, c.num_kv_heads, c.head_dim)).view(-1, c.kv_size)
        q, k = self.rotary(positions, q, k)
        out = self.attn(q, k, v)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.project_out(out)

    def project_out(self, x: torch.Tensor) -> torch.Tensor:
        """The output projection, reduced over TP ranks iff ``output_reduced``.

        The unreduced form is ``RowParallelLinear.forward`` without its
        all-reduce, the flag passed per call instead of written into the
        projection: ``o_proj.reduce_results`` keeps the value the model built.
        It applies no bias; ``unreduced_output_supports`` declines a
        projection that has one.
        """
        if self.output_reduced:
            out, _ = self.o_proj(x)
            return out
        o_proj = self.o_proj
        return o_proj.quant_method.apply(as_kind(o_proj, nn.Module), x, None)


def reference_fused_qkv(
    runner: AttentionRunner,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused value in plain torch: normalize and project in one call.

    Stands in for PR #3096's kernel on hosts without it. It calls the same
    binding, so it equals unfused; a kernel fuses these calls. Given the
    residual (``ResidualAdd.FOLDED``) the input norm adds it and returns the
    new residual; given none (``OUTSIDE``, the first layer) it normalizes
    the stream it is given. It adds nothing itself.
    """
    stream: torch.Tensor
    if residual is None:
        normed, stream = runner.input_norm(hidden), hidden
    else:
        normed, stream = runner.input_norm(hidden, residual)
    qkv, _ = runner.qkv_proj(normed)
    return runner.attention_from_qkv(qkv, positions), stream


REFERENCE_CAPABILITY = FusedCapability(
    residual_add=frozenset({ResidualAdd.FOLDED, ResidualAdd.OUTSIDE}), unreduced=True
)
"""``reference_fused_qkv``'s declaration: both residual adds, either reduction."""
