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
"""Planted decoder layers for the conformance check.

Each fixture reuses the children of a built Qwen3.5 MoE layer and writes its
own forward, so only the wiring differs from upstream:

- ``Renamed``: every child renamed, forward otherwise upstream's. Declared
  with the new names (``renamed_parts``), it conforms.
- ``ReorderedNorms``: calls the post-attention norm first. It does not.
- ``DeadBranchAndView``: computes a value it never uses, has a branch that
  never runs, and views the stream to its own shape. It conforms: dead code
  and same-shape views are not part of what the layer computes.
"""

import torch
from torch import nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock

from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts
from vllm_torchtpu.blocks.parts import as_kind
from vllm_torchtpu.blocks.stream import (
    attend_with_positions,
    hidden_residual_positions,
)


class _Children(nn.Module):
    def __init__(self, built: nn.Module) -> None:
        super().__init__()
        layer = as_kind(built, Qwen3_5DecoderLayer)
        self.pre_norm = layer.input_layernorm
        self.mixer = layer.self_attn
        self.mid_norm = layer.post_attention_layernorm
        self.ffn = as_kind(layer.mlp, Qwen3NextSparseMoeBlock)


class Renamed(_Children):  # noqa: D101 - the module docstring describes each fixture
    def forward(  # noqa: D102 - upstream's forward under the new names
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, residual = self.pre_norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        hidden_states, residual = self.mid_norm(hidden_states, residual)
        return self.ffn(hidden_states), residual


class ReorderedNorms(_Children):  # noqa: D101 - ditto
    def forward(  # noqa: D102 - the planted order
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, residual = self.mid_norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        hidden_states, residual = self.pre_norm(hidden_states, residual)
        return self.ffn(hidden_states), residual


class DeadBranchAndView(_Children):  # noqa: D101 - ditto
    debug = False

    def forward(  # noqa: D102 - the planted dead code
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states, residual = self.pre_norm(hidden_states, residual)
        unused = hidden_states * 2  # noqa: F841 - planted dead value
        if self.debug:
            hidden_states = hidden_states + 1
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states, residual = self.mid_norm(hidden_states, residual)
        return self.ffn(hidden_states), residual


def renamed_parts(layer: Renamed | ReorderedNorms | DeadBranchAndView) -> AttnMoEParts:
    """The fixtures' binding function: the same parts under the new names."""
    attention, moe = layer.mixer, layer.ffn
    return AttnMoEParts(
        input_norm=layer.pre_norm,
        attention=attention,
        out_proj=attention.o_proj,
        post_norm=layer.mid_norm,
        moe=moe,
        router=moe.gate,
        experts=moe.experts,
        stream=hidden_residual_positions(layer),
        attend=attend_with_positions(attention),
    )
