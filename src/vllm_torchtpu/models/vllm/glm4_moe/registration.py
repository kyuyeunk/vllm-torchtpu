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

from vllm_torchtpu.blocks.attn_moe.pattern import AttnThenMoE
from vllm_torchtpu.blocks.norm_attn.pattern import NormThenAttention
from vllm_torchtpu.blocks.roles import Attention, MoE
from vllm_torchtpu.blocks.stream import positions_first


def _attention(layer: Glm4MoeDecoderLayer) -> Attention:
    """The attn role: GLM-4 MoE's attention, with q/k norms where configured."""
    attention = layer.self_attn
    qk_norm = (attention.q_norm, attention.k_norm) if attention.use_qk_norm else None
    return Attention.of(attention, qk_norm=qk_norm)


@AttnThenMoE.register(Glm4MoeDecoderLayer)
def glm4_moe_block(layer: Glm4MoeDecoderLayer) -> AttnThenMoE.Roles | None:
    """The attention-then-MoE block (attn_then_moe), on the MoE layers.

    Its span ``in_norm, attn`` is the sub-block below, built from these roles.
    """
    if not isinstance(layer.mlp, Glm4MoE):  # The first_k_dense_replace layers.
        return None
    return AttnThenMoE.Roles(
        in_norm=layer.input_layernorm,
        attn=_attention(layer),
        post_norm=layer.post_attention_layernorm,
        moe=MoE.of(layer.mlp),
        layer_call=positions_first(layer),
    )


@NormThenAttention.register(Glm4MoeDecoderLayer)
def glm4_moe_attention(layer: Glm4MoeDecoderLayer) -> NormThenAttention.Roles:
    """The pre-norm attention sub-block (norm_then_attention), on every layer.

    On the MoE layers the block above builds it from its own span.
    """
    return NormThenAttention.Roles(
        in_norm=layer.input_layernorm,
        attn=_attention(layer),
        post_norm=layer.post_attention_layernorm,
        mlp=layer.mlp,
        layer_call=positions_first(layer),
    )
