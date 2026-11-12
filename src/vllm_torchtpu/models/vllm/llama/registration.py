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
"""Llama's block declarations. Owned by its model owner."""

from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

from vllm_torchtpu.blocks import declare_block
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnParts, pre_norm_parts
from vllm_torchtpu.blocks.parts import as_kind


@declare_block(LlamaDecoderLayer)
def llama_attention(layer: LlamaDecoderLayer) -> NormAttnParts:
    """The pre-norm attention sub-block (norm_then_attention).

    Llama builds ``self_attn`` from a class parameter (``attn_layer_type``),
    so the binding names the class it covers.
    """
    return pre_norm_parts(layer, as_kind(layer.self_attn, LlamaAttention))
