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
Full-attention layers' front half also conforms to norm_then_attention, the
sub-block nested in the block. What runs under the declarations (the plans,
their rules and the forwards) is the framework's.
"""

from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.attn_moe.pattern import attn_then_moe
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention

declare_block(
    Qwen3_5DecoderLayer,
    pattern=attn_then_moe,
    roles={
        "in_norm": "input_layernorm",
        "attn": ("self_attn", "linear_attn"),
        "post_norm": "post_attention_layernorm",
        "moe": "mlp",
    },
    # Dense Qwen3.5 builds the same class with an MLP.
    when=lambda layer: isinstance(layer.mlp, Qwen3NextSparseMoeBlock),
)

declare_block(
    Qwen3_5DecoderLayer,
    pattern=norm_then_attention,
    roles={
        "in_norm": "input_layernorm",
        "attn": "self_attn",
        "post_norm": "post_attention_layernorm",
        "mlp": "mlp",
    },
    when=lambda layer: layer.layer_type == "full_attention",
)
