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

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts, pre_norm_moe_parts
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnParts, pre_norm_parts


@declare_block(Qwen3_5DecoderLayer)
def qwen3_5_block(layer: Qwen3_5DecoderLayer) -> AttnMoEParts | None:
    """The attention-then-MoE block, for either attention kind."""
    if not isinstance(layer.mlp, Qwen3NextSparseMoeBlock):  # Dense Qwen3.5.
        return None
    full = layer.layer_type == "full_attention"
    attention = layer.self_attn if full else layer.linear_attn
    return pre_norm_moe_parts(layer, attention, layer.mlp)


@declare_block(Qwen3_5DecoderLayer)
def qwen3_5_attention(layer: Qwen3_5DecoderLayer) -> NormAttnParts | None:
    """The pre-norm attention sub-block, on full-attention layers."""
    if layer.layer_type != "full_attention":
        return None
    return pre_norm_parts(layer, layer.self_attn)
