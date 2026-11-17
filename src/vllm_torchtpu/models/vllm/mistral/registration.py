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
"""Mistral's block declarations. Owned by its model owner."""

from vllm.model_executor.models.mistral import MistralAttention, MistralDecoderLayer

from vllm_torchtpu.blocks.binding import as_kind
from vllm_torchtpu.blocks.norm_attn.pattern import NormThenAttention
from vllm_torchtpu.blocks.roles import Attention
from vllm_torchtpu.blocks.stream import positions_first


@NormThenAttention.register(MistralDecoderLayer)
def mistral_attention(layer: MistralDecoderLayer) -> NormThenAttention.Roles:
    """The pre-norm attention sub-block (norm_then_attention).

    Mistral builds ``self_attn`` from Llama's class parameter
    (``attn_layer_type=MistralAttention``), so the declaration names the class.
    """
    return NormThenAttention.Roles(
        in_norm=layer.input_layernorm,
        attn=Attention.of(as_kind(layer.self_attn, MistralAttention)),
        post_norm=layer.post_attention_layernorm,
        mlp=layer.mlp,
        layer_call=positions_first(layer),
    )
