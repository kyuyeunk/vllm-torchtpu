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
from vllm_torchtpu.blocks.attn_moe.pattern import attn_then_moe
from vllm_torchtpu.blocks.norm_attn.pattern import PRE_NORM_ROLES, norm_then_attention

declare_block(
    Glm4MoeDecoderLayer,
    pattern=attn_then_moe,
    roles={
        "in_norm": "input_layernorm",
        "attn": "self_attn",
        "post_norm": "post_attention_layernorm",
        "moe": "mlp",
    },
    # The first `first_k_dense_replace` layers build the same class with an MLP.
    when=lambda layer: isinstance(layer.mlp, Glm4MoE),
)

declare_block(Glm4MoeDecoderLayer, pattern=norm_then_attention, roles=PRE_NORM_ROLES)
