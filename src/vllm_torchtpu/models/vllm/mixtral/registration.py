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
"""Mixtral's block declarations. Owned by the Mixtral model owner.

The MoE decoder layer conforms to attn_then_moe. What runs under the declaration
is the framework's.
"""

from vllm.model_executor.models.mixtral import MixtralDecoderLayer

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts, pre_norm_moe_parts
from vllm_torchtpu.blocks.stream import attend_with_positions, positions_first


@declare_block(MixtralDecoderLayer)
def mixtral_block(layer: MixtralDecoderLayer) -> AttnMoEParts | None:
    """The attention-then-MoE block (attn_then_moe), on the MoE layers."""
    attention, moe = layer.self_attn, layer.block_sparse_moe
    return pre_norm_moe_parts(
        layer,
        attention,
        attention.o_proj,
        moe,
        router=moe.gate,
        experts=moe.experts,
        stream=positions_first(layer),
        attend=attend_with_positions(attention),
    )
