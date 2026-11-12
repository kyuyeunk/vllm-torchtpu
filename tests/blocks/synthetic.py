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
"""Synthetic parts for the runner execution tests, and a straight-line reference.

Each part has the contract of the vLLM building block it stands for: an
RMSNorm taking ``(x, residual)``, linears returning ``(out, bias)``, a NeoX
rotary on ``(positions, q, k)``, a causal GQA core on ``(q, k, v)``. Shared by
``test_attention_runner.py`` and ``test_attn_moe_runner.py``; imports no model.
"""

import dataclasses

import torch
import torch.nn.functional as F  # noqa: N812 - torch's alias
from torch import nn
from vllm.model_executor.layers import layernorm

from vllm_torchtpu.blocks.norm_attn import runner as runner_mod
from vllm_torchtpu.blocks.norm_attn.runner import (
    AttentionParts,
    AttentionRunner,
    AttnConfig,
)

HIDDEN = 64
TOKENS = 7
NUM_KV_HEADS = 2
EPS = 1e-6


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return (normed * weight.float()).to(x.dtype)


class InputNorm(nn.Module):
    """vLLM's RMSNorm contract: ``x -> normed``, ``(x, residual) -> (normed, sum)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rms(x, self.weight)
        summed = (x.float() + residual.float()).to(x.dtype)
        return _rms(summed, self.weight), summed


class HeadNorm(layernorm.RMSNorm):
    """A per-head RMSNorm over ``(..., heads, head_dim)``.

    Subclasses vLLM's RMSNorm only for its type, which ``fused_qkv``'s
    capability reads; the forward is this file's, and ``CustomOp.__init__``
    (which needs a vLLM config) is skipped.
    """

    def __init__(self, dim: int) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _rms(x, self.weight)


class WholeNorm(nn.Module):
    """A q/k norm that is not an RMSNorm (stands for a TP-wide LayerNorm)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class UnquantizedLinearMethod:
    """Stands for vLLM's; the capability reads the quant method's class name."""

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The GEMM alone, as vLLM's ``apply``: no reduction over ranks."""
        return F.linear(x, layer.weight, bias)


class AwqLinearMethod:
    """A quant method ``fused_qkv`` does not implement."""


class Linear(nn.Module):
    """vLLM's linear contract: ``x -> (out, output_bias)``, the bias already added."""

    def __init__(self, n_in: int, n_out: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) / n_in**0.5)
        self.bias = nn.Parameter(0.1 * torch.randn(n_out)) if bias else None
        self.quant_method: object = UnquantizedLinearMethod()
        self.return_bias = True  # vLLM's linears return (out, bias) by default.
        # RowParallelLinear's fields the unreduced-output capability reads.
        self.input_is_parallel = True
        self.skip_bias_add = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight, self.bias), None


class NeoxRotary(nn.Module):
    """NeoX RoPE on the first ``rotary_dim`` values of each head, on flat q and k."""

    def __init__(self, head_dim: int, rotary_dim: int, *, neox: bool = True) -> None:
        super().__init__()
        self.head_dim, self.rotary_dim = head_dim, rotary_dim
        self.is_neox_style = neox

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(shape[0], -1, self.head_dim)
        rot, keep = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        half = self.rotary_dim // 2
        freqs = 10000.0 ** (-torch.arange(half, dtype=torch.float32) / half)
        angle = positions.float()[:, None] * freqs[None, :]
        cos = torch.cat([angle.cos(), angle.cos()], -1)[:, None, :]
        sin = torch.cat([angle.sin(), angle.sin()], -1)[:, None, :]
        rotated = torch.cat([-rot[..., half:], rot[..., :half]], -1)
        out = (rot.float() * cos + rotated.float() * sin).to(x.dtype)
        return torch.cat([out, keep], -1).view(shape)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rotate(q, positions), self._rotate(k, positions)


class CausalGqa(nn.Module):
    """A causal GQA core on flat ``(q, k, v)``, standing in for the backend."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        sinks: bool = False,
        kv_sharing: str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.sinks = nn.Parameter(torch.zeros(num_heads)) if sinks else None
        self.kv_sharing_target_layer_name = kv_sharing

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        t, h, d = q.shape[0], self.num_heads, self.head_dim
        q = q.view(t, h, d).transpose(0, 1)
        k = k.view(t, self.num_kv_heads, d).transpose(0, 1)
        v = v.view(t, self.num_kv_heads, d).transpose(0, 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return out.transpose(0, 1).reshape(t, h * d)


@dataclasses.dataclass(frozen=True)
class Roles:
    """One role configuration of the sub-block."""

    qk_norm: bool = False
    output_gate: bool = False
    gqa: int = 4
    head_dim: int = 128
    qkv_bias: bool = False
    rotary: str = "full"  # or "partial": the first half of each head.

    def config(self) -> AttnConfig:
        return AttnConfig(
            num_heads=NUM_KV_HEADS * self.gqa,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=self.head_dim,
            qk_norm=self.qk_norm,
            output_gate=self.output_gate,
        )


class Attention(nn.Module):
    """A synthetic attention role: the parts, and the properties vLLM's carry.

    ``fused_qkv_supports`` reads the properties; the runner receives the parts.
    """

    def __init__(
        self,
        roles: Roles,
        *,
        sinks: bool = False,
        kv_sharing: str | None = None,
        q_norm: nn.Module | None = None,
        neox: bool = True,
    ) -> None:
        super().__init__()
        c = roles.config()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            c.num_heads,
            c.num_kv_heads,
            c.head_dim,
        )
        self.attn_output_gate = roles.output_gate
        q_width = c.q_size * (2 if roles.output_gate else 1)
        self.qkv_proj = Linear(HIDDEN, q_width + 2 * c.kv_size, bias=roles.qkv_bias)
        self.o_proj = Linear(c.q_size, HIDDEN)
        rotary_dim = c.head_dim if roles.rotary == "full" else c.head_dim // 2
        self.rotary_emb = NeoxRotary(c.head_dim, rotary_dim, neox=neox)
        self.attn = CausalGqa(
            c.num_heads, c.num_kv_heads, c.head_dim, sinks=sinks, kv_sharing=kv_sharing
        )
        self.q_norm = self.k_norm = None
        if roles.qk_norm:
            self.q_norm = q_norm or HeadNorm(c.head_dim)
            self.k_norm = HeadNorm(c.head_dim)


def setup_parts(
    roles: Roles, dtype: torch.dtype = torch.float32
) -> tuple[InputNorm, Attention]:
    torch.manual_seed(0)
    return InputNorm(HIDDEN).to(dtype), Attention(roles).to(dtype)


def build_runner(
    norm: nn.Module,
    attention: Attention,
    config: AttnConfig,
    input_proj: str = "unfused",
    fused_op: runner_mod.FusedOp | None = None,
    *,
    output_reduced: bool = True,
) -> AttentionRunner:
    parts = AttentionParts(
        input_norm=norm,
        qkv_proj=attention.qkv_proj,
        attn=attention.attn,
        o_proj=attention.o_proj,
        rotary=attention.rotary_emb,
        q_norm=attention.q_norm,
        k_norm=attention.k_norm,
    )
    return AttentionRunner(
        "layer",
        parts,
        config=config,
        input_proj=input_proj,
        fused_op=fused_op,
        output_reduced=output_reduced,
    )


def make_inputs(
    first: bool, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    torch.manual_seed(1)
    hidden = torch.randn(TOKENS, HIDDEN, dtype=dtype)
    residual = None if first else torch.randn(TOKENS, HIDDEN, dtype=dtype)
    return hidden, residual, torch.arange(TOKENS) * 3


def reference_forward(
    roles: Roles,
    norm: InputNorm,
    attention: Attention,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sub-block in straight-line float32 torch, from the parts' weights."""
    c = roles.config()
    h, kvh, d, t = c.num_heads, c.num_kv_heads, c.head_dim, hidden.shape[0]
    stream = hidden if residual is None else hidden + residual

    def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w

    x = rms(stream, norm.weight)
    qkv = x @ attention.qkv_proj.weight.T
    if attention.qkv_proj.bias is not None:
        qkv = qkv + attention.qkv_proj.bias
    q_end = h * d * (2 if roles.output_gate else 1)
    q, k, v = qkv[:, :q_end], qkv[:, q_end : q_end + kvh * d], qkv[:, q_end + kvh * d :]
    gate = None
    if roles.output_gate:  # Per head: [q (d values), gate (d values)].
        per_head = q.reshape(t, h, 2 * d)
        q, gate = per_head[..., :d], per_head[..., d:].reshape(t, h * d)
    q, k, v = q.reshape(t, h, d), k.reshape(t, kvh, d), v.reshape(t, kvh, d)
    if roles.qk_norm:
        q_norm, k_norm = attention.q_norm, attention.k_norm
        assert isinstance(q_norm, HeadNorm)
        assert isinstance(k_norm, HeadNorm)
        q, k = rms(q, q_norm.weight), rms(k, k_norm.weight)
    r = d if roles.rotary == "full" else d // 2
    half = r // 2
    inv_freq = 1.0 / 10000.0 ** (torch.arange(half, dtype=torch.float32) / half)
    angle = positions.float()[:, None, None] * inv_freq
    cos, sin = angle.cos(), angle.sin()

    def rope(x: torch.Tensor) -> torch.Tensor:
        x1, x2, rest = x[..., :half], x[..., half:r], x[..., r:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(h // kvh, dim=1)
    v = v.repeat_interleave(h // kvh, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / d**0.5
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(t, h * d)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return out @ attention.o_proj.weight.T, stream
