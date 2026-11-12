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
"""Bailing-MoE's block declarations. Owned by the Bailing-MoE model owner.

The MoE decoder layer conforms to attn_then_moe. The first
``first_k_dense_replace`` layers build an MLP instead. What runs under the
declaration is the framework's.
"""

from vllm.model_executor.models.bailing_moe import BailingMoE, BailingMoeBlock

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts, pre_norm_moe_parts
from vllm_torchtpu.blocks.stream import (
    attend_with_position_ids,
    hidden_position_ids_residual,
)


@declare_block(BailingMoeBlock)
def bailing_moe_block(layer: BailingMoeBlock) -> AttnMoEParts | None:
    """The attention-then-MoE block (attn_then_moe), on the MoE layers."""
    if not isinstance(layer.mlp, BailingMoE):  # A dense layer.
        return None
    attention, moe = layer.attention, layer.mlp
    return pre_norm_moe_parts(
        layer,
        attention,
        attention.dense,
        moe,
        router=moe.gate,
        experts=moe.experts,
        stream=hidden_position_ids_residual(layer),
        attend=attend_with_position_ids(attention),
    )
