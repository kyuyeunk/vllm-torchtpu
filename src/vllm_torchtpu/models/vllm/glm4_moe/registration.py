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
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEBinding, pre_norm_moe_binding
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnBinding, pre_norm_binding
from vllm_torchtpu.blocks.stream import attend_with_positions, positions_first


@declare_block(Glm4MoeDecoderLayer)
def glm4_moe_block(layer: Glm4MoeDecoderLayer) -> AttnMoEBinding | None:
    """The attention-then-MoE block (attn_then_moe), on the MoE layers."""
    if not isinstance(layer.mlp, Glm4MoE):  # The first_k_dense_replace layers.
        return None
    attention = layer.self_attn
    return pre_norm_moe_binding(
        layer,
        attention,
        attention.o_proj,
        layer.mlp,
        stream=positions_first(layer),
        attend=attend_with_positions(attention),
    )


@declare_block(Glm4MoeDecoderLayer)
def glm4_moe_attention(layer: Glm4MoeDecoderLayer) -> NormAttnBinding:
    """The pre-norm attention sub-block (norm_then_attention), nested in the block."""
    attention = layer.self_attn
    qk_norm = (attention.q_norm, attention.k_norm) if attention.use_qk_norm else None
    return pre_norm_binding(
        layer,
        attention,
        qk_norm=qk_norm,
        output_gate=False,
        stream=positions_first(layer),
    )
