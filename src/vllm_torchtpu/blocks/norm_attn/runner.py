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

- ``__init__`` receives parts the model already built (the input norm, the
  QKV projection, the optional per-head q/k norms, RoPE, the attention core,
  the output projection) and keeps references to them. It creates no
  parameter, so weight loading and the model's ``state_dict`` are unchanged;
  the model keeps owning its parts.
- ``forward`` calls the parts in order. The runner owns the composition only.
- ``_select_forward`` resolves the plan once, at install, as ``MoERunner``
  picks its ``_forward_entry``.

Every model that declares ``norm_then_attention`` runs this one forward, so a
fusion of the sub-block (``fused_qkv``, PR #3096's kernel) is one branch here,
written once against the parts, never against a model class.

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
from typing import TYPE_CHECKING, override

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm_torchtpu.blocks import calls


class InputProj(enum.Enum):
    """The runner's forwards: whether the input norm joins the projection."""

    _value_: str

    UNFUSED = "unfused"
    FUSED_QKV = "fused_qkv"


FUSED_CALLS: dict[str, int] = {}
"""Runs of the fused value, per layer class (eager runs only)."""

FusedOp = Callable[
    ["AttentionRunner", torch.Tensor, torch.Tensor | None, torch.Tensor],
    "tuple[torch.Tensor, torch.Tensor] | None",
]
"""``op(runner, hidden, residual, positions) -> (out, residual)`` or None to decline.

The op reads ``runner.output_reduced`` and returns an output summed over TP
ranks only when it is True.
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
class AttentionParts:
    """What ``AttentionRunner`` composes: the model's own part objects.

    Execution needs only each part's call contract, so tests build this from
    synthetic modules. A model's declaration builds ``NormAttnParts``, which
    extends it with the layer's other roles and checks every kind.

    Attributes:
        input_norm: The input RMSNorm, called as ``(x, residual)``.
        qkv_proj: Attention's input projection, returning ``(out, bias)``.
        attn: The attention core on ``(q, k, v)``.
        o_proj: The output projection (a ``RowParallelLinear``).
        rotary: RoPE on ``(positions, q, k)``.
        q_norm, k_norm: Per-head norms on q and k, both or neither.
    """

    input_norm: nn.Module
    qkv_proj: nn.Module
    attn: nn.Module
    o_proj: nn.Module
    rotary: nn.Module
    q_norm: nn.Module | None = None
    k_norm: nn.Module | None = None


class AttentionRunner(nn.Module):
    """The sub-block's composition; the parts stay the model's.

    Parts are held as plain references (not registered submodules), so the
    model's parameter tree, weight loading and checkpoints are unchanged.
    """

    # The parts, typed by their call contracts (``calls``): assigned as plain
    # references in __init__, so the model's parameter tree is unchanged.
    input_norm: "calls.ResidualNorm"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    qkv_proj: "calls.Linear"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    attn: "calls.AttentionCore"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    o_proj: "calls.RowLinear"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    rotary: "calls.Rope"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    q_norm: "calls.HeadNorm"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    k_norm: "calls.HeadNorm"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    layer_name: str
    config: AttnConfig
    input_proj: InputProj
    fused_op: FusedOp | None
    output_reduced: bool
    _forward_entry: Callable[
        [torch.Tensor, torch.Tensor | None, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]

    def __init__(  # noqa: PLR0913 - the parts, then the plan, the op and the contract
        self,
        layer_name: str,
        parts: AttentionParts,
        *,
        config: AttnConfig,
        input_proj: InputProj = InputProj.UNFUSED,
        fused_op: FusedOp | None = None,
        output_reduced: bool = True,
    ) -> None:
        """Keeps references to the parts; creates no parameter."""
        super().__init__()
        refs = (
            ("input_norm", parts.input_norm),
            ("qkv_proj", parts.qkv_proj),
            ("attn", parts.attn),
            ("o_proj", parts.o_proj),
            ("rotary", parts.rotary),
            ("q_norm", parts.q_norm),
            ("k_norm", parts.k_norm),
        )
        for name, part in refs:
            # A reference, not a child: the model keeps owning its parts.
            object.__setattr__(self, name, part)
        self.layer_name = layer_name
        self.config = config
        self.input_proj = input_proj
        self.fused_op = fused_op
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
            return self._fused_qkv
        return self._unfused

    @override
    def forward(
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
        """The fused op on the same parts; where it declines a call, unfused."""
        if self.fused_op is None:  # _select_forward checked
            raise ValueError(self.layer_name)
        out = self.fused_op(self, hidden, residual, positions)
        if out is None:
            return self._unfused(hidden, residual, positions)
        if not torch.compiler.is_compiling():
            FUSED_CALLS[self.layer_name] = FUSED_CALLS.get(self.layer_name, 0) + 1
        return out

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
        if c.qk_norm:
            q = self.q_norm(q.view(-1, c.num_heads, c.head_dim)).view(-1, c.q_size)
            k = self.k_norm(k.view(-1, c.num_kv_heads, c.head_dim)).view(-1, c.kv_size)
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
        return self.o_proj.quant_method.apply(self.o_proj, x, None)


def reference_fused_qkv(
    runner: AttentionRunner,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused value in plain torch: add, normalize and project in one call.

    Stands in for PR #3096's kernel on hosts without it. It calls the same
    parts, keeping the platform's rounding points (the residual sum is
    rounded to the stream's dtype, as the fused add-norm does), so it equals
    unfused exactly; a kernel fuses these calls.
    """
    if residual is None:
        summed = hidden
    else:
        summed = (hidden.float() + residual.float()).to(hidden.dtype)
    qkv, _ = runner.qkv_proj(runner.input_norm(summed))
    return runner.attention_from_qkv(qkv, positions), summed
