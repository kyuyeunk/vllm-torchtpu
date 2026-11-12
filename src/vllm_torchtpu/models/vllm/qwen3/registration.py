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
"""Qwen3 (dense)'s block declarations. Owned by its model owner."""

from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnParts, pre_norm_parts


@declare_block(Qwen3DecoderLayer)
def qwen3_attention(layer: Qwen3DecoderLayer) -> NormAttnParts:
    """The pre-norm attention sub-block (norm_then_attention)."""
    return pre_norm_parts(layer, layer.self_attn, output_gate=False)
