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
"""DeepSeek-V2/V3's block declarations. Owned by the DeepSeek-V2/V3 model owner.

The MoE decoder layer conforms to attn_then_moe. The first
``first_k_dense_replace`` layers build an MLP instead. Models that reuse this
layer class are covered too, if they conform. What runs under the declaration is
the framework's.
"""

from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV2MoE

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts, pre_norm_moe_parts
from vllm_torchtpu.blocks.stream import attend_unscaled, positions_first_unscaled


@declare_block(DeepseekV2DecoderLayer)
def deepseek_v2_block(layer: DeepseekV2DecoderLayer) -> AttnMoEParts | None:
    """The attention-then-MoE block (attn_then_moe), on the MoE layers.

    The layer also takes ``llama_4_scaling``, which the block does not carry:
    it passes attention None and refuses a call that passes one.
    """
    if not isinstance(layer.mlp, DeepseekV2MoE):  # A dense layer.
        return None
    attention, moe = layer.self_attn, layer.mlp
    return pre_norm_moe_parts(
        layer,
        attention,
        attention.o_proj,
        moe,
        router=moe.gate,
        experts=moe.experts,
        stream=positions_first_unscaled(layer),
        attend=attend_unscaled(attention),
    )
