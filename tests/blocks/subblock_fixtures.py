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
"""Planted Llama layers for the sub-block's conformance check.

Each reuses a built Llama layer's children and is declared
``norm_then_attention`` with ``llama_binding``; neither conforms:

- ``ScaledAttention``: scales attention's output before the second norm.
- ``ScaledQueryAttention``: its attention scales q after RoPE
  (``scale_query``).
"""

from typing import override

import torch
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

from vllm_torchtpu.blocks.binding import as_kind
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnBinding, pre_norm_binding
from vllm_torchtpu.blocks.stream import positions_first


class ScaledAttention(LlamaDecoderLayer):
    """Planted: scales attention's output before the second norm."""

    @override
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vLLM's norm returns an untyped pair; the planted line is the multiply.
        hidden_states, residual = self.input_layernorm(hidden_states, residual)  # pyright: ignore[reportAny]
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)  # pyright: ignore[reportAny]
        hidden_states = hidden_states * 2.0
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # pyright: ignore[reportAny]
        return self.mlp(hidden_states), residual  # pyright: ignore[reportReturnType]


class ScaledQueryAttention(LlamaDecoderLayer):
    """Planted: its attention scales q after RoPE (see ``scale_query``)."""


def scale_query(attn: LlamaAttention) -> None:
    """Gives ``attn`` a forward the runner does not compute: q * 0.5 after RoPE."""

    def forward(positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = attn.qkv_proj(hidden_states)  # pyright: ignore[reportAny]
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)  # pyright: ignore[reportAny]
        q, k = attn.rotary_emb(positions, q, k)  # pyright: ignore[reportAny]
        q = q * 0.5  # pyright: ignore[reportAny]
        out, _ = attn.o_proj(attn.attn(q, k, v))  # pyright: ignore[reportAny]
        return out  # pyright: ignore[reportAny]

    attn.forward = forward  # The planted forward.


def llama_binding(layer: LlamaDecoderLayer) -> NormAttnBinding:
    """The planted layers' binding function (they reuse Llama's children)."""
    return pre_norm_binding(
        layer,
        as_kind(layer.self_attn, LlamaAttention),
        qk_norm=None,
        output_gate=False,
        stream=positions_first(layer),
    )
