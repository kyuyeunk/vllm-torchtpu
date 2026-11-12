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
"""GLM-4.5 / 4.6's block declarations. Owned by its model owner."""

from vllm.model_executor.models.glm4_moe import Glm4MoE, Glm4MoeDecoderLayer

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts, pre_norm_moe_parts
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnParts, pre_norm_parts


@declare_block(Glm4MoeDecoderLayer)
def glm4_moe_block(layer: Glm4MoeDecoderLayer) -> AttnMoEParts | None:
    """The attention-then-MoE block (attn_then_moe), on the MoE layers."""
    if not isinstance(layer.mlp, Glm4MoE):  # The first_k_dense_replace layers.
        return None
    return pre_norm_moe_parts(layer, layer.self_attn, layer.mlp)


@declare_block(Glm4MoeDecoderLayer)
def glm4_moe_attention(layer: Glm4MoeDecoderLayer) -> NormAttnParts:
    """The pre-norm attention sub-block (norm_then_attention), nested in the block."""
    return pre_norm_parts(layer, layer.self_attn)
