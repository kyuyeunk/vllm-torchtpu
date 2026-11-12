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
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

FUSED_CALLS: dict[str, int] = {}
"""Runs of the fused value, per layer class (eager runs only)."""

FusedOp = Callable[
    ["AttentionRunner", torch.Tensor, torch.Tensor | None, torch.Tensor], Any
]
"""``op(runner, hidden, residual, positions) -> (out, residual)`` or None to decline."""


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
        return self.num_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_kv_heads * self.head_dim


class AttentionRunner(nn.Module):
    """The sub-block's composition; the parts stay the model's.

    Parts are held as plain references (not registered submodules), so the
    model's parameter tree, weight loading and checkpoints are unchanged.
    """

    def __init__(
        self,
        layer_name: str,
        *,
        input_norm: nn.Module,
        qkv_proj: nn.Module,
        attn: nn.Module,
        o_proj: nn.Module,
        rotary: nn.Module,
        q_norm: nn.Module | None = None,
        k_norm: nn.Module | None = None,
        config: AttnConfig,
        input_proj: str = "unfused",
        fused_op: FusedOp | None = None,
    ) -> None:
        super().__init__()
        parts = {
            "input_norm": input_norm,
            "qkv_proj": qkv_proj,
            "attn": attn,
            "o_proj": o_proj,
            "rotary": rotary,
            "q_norm": q_norm,
            "k_norm": k_norm,
        }
        for name, part in parts.items():
            object.__setattr__(self, name, part)  # a reference, not a child
        self.layer_name = layer_name
        self.config = config
        self.input_proj = input_proj
        self.fused_op = fused_op
        self._forward_entry = self._select_forward()

    def _select_forward(self) -> Callable[..., Any]:
        """The plan, resolved once (install resolves rule, pin and capability)."""
        if self.input_proj == "fused_qkv":
            if self.fused_op is None:
                raise ValueError(f"{self.layer_name}: fused_qkv needs a fused op")
            return self._fused_qkv
        return self._unfused

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_entry(hidden, residual, positions)

    # The plan values.

    def _unfused(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream's order: norm, project, split, q/k norm, RoPE, attend, project."""
        if residual is None:  # The first layer: the stream is its own residual.
            residual = hidden
            x = self.input_norm(hidden)
        else:
            x, residual = self.input_norm(hidden, residual)
        qkv, _ = self.qkv_proj(x)
        return self.attention_from_qkv(qkv, positions), residual

    def _fused_qkv(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fused op on the same parts; where it declines a call, unfused."""
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
        c = self.config
        if c.output_gate:
            q_gate, k, v = qkv.split([c.q_size * 2, c.kv_size, c.kv_size], dim=-1)
            shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*shape, c.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*shape, -1)
            gate = gate.reshape(*shape, -1)
        else:
            q, k, v = qkv.split([c.q_size, c.kv_size, c.kv_size], dim=-1)
            gate = None
        if c.qk_norm:
            q = self.q_norm(q.view(-1, c.num_heads, c.head_dim)).view(-1, c.q_size)
            k = self.k_norm(k.view(-1, c.num_kv_heads, c.head_dim)).view(-1, c.kv_size)
        q, k = self.rotary(positions, q, k)
        out = self.attn(q, k, v)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        out, _ = self.o_proj(out)
        return out


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
