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
"""Qwen3-MoE's block declarations. Owned by the Qwen3-MoE model owner."""

from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention

declare_block(
    Qwen3MoeDecoderLayer,
    pattern=norm_then_attention,
    roles={
        "in_norm": "input_layernorm",
        "attn": "self_attn",
        "post_norm": "post_attention_layernorm",
        "mlp": "mlp",
    },
)
