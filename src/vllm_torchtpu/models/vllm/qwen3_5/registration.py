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
"""Qwen3.5's block declarations. Owned by the Qwen3.5 model owner.

The MoE decoder layer conforms to attn_then_moe. Full-attention layers bind
``self_attn``, linear-attention (Gated DeltaNet) layers ``linear_attn``.
Full-attention layers' attention also conforms to norm_then_attention, the
sub-block nested in the block. What runs under the declarations (the plans,
their rules and the forwards) is the framework's.
"""

from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

from vllm_torchtpu.blocks.attn_moe.pattern import AttnThenMoE
from vllm_torchtpu.blocks.norm_attn.pattern import NormThenAttention
from vllm_torchtpu.blocks.roles import Attention, MoE
from vllm_torchtpu.blocks.stream import (
    attend_without_positions,
    hidden_residual_positions,
)


def _attention(layer: Qwen3_5DecoderLayer) -> Attention:
    """The attn role, for either attention kind.

    Full attention states its parts, so the sub-block can fuse inside it.
    Gated DeltaNet reads no positions and states only its output projection.
    """
    if layer.layer_type == "full_attention":  # str-ok: the model's own layer_type
        attention = layer.self_attn
        return Attention.of(
            attention,
            qk_norm=(attention.q_norm, attention.k_norm),
            output_gate=attention.attn_output_gate,
        )
    gdn = layer.linear_attn
    return Attention.opaque(
        gdn, o_proj=gdn.out_proj, call=attend_without_positions(gdn)
    )


@AttnThenMoE.register(Qwen3_5DecoderLayer)
def qwen3_5_block(layer: Qwen3_5DecoderLayer) -> AttnThenMoE.Roles | None:
    """The attention-then-MoE block, for either attention kind.

    On full-attention layers its span ``in_norm, attn`` is the sub-block below.
    """
    if not isinstance(layer.mlp, Qwen3NextSparseMoeBlock):  # Dense Qwen3.5.
        return None
    return AttnThenMoE.Roles(
        in_norm=layer.input_layernorm,
        attn=_attention(layer),
        post_norm=layer.post_attention_layernorm,
        moe=MoE.of(layer.mlp),
        layer_call=hidden_residual_positions(layer),
    )


@NormThenAttention.register(Qwen3_5DecoderLayer)
def qwen3_5_attention(layer: Qwen3_5DecoderLayer) -> NormThenAttention.Roles | None:
    """The pre-norm attention sub-block, on full-attention layers."""
    if layer.layer_type != "full_attention":  # str-ok: the model's own layer_type
        return None
    return NormThenAttention.Roles(
        in_norm=layer.input_layernorm,
        attn=_attention(layer),
        post_norm=layer.post_attention_layernorm,
        mlp=layer.mlp,
        layer_call=hidden_residual_positions(layer),
    )
