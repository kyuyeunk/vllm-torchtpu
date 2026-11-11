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
  with the new names, it conforms.
- ``ReorderedNorms``: calls the post-attention norm first. It does not.
- ``DeadBranchAndView``: computes a value it never uses, has a branch that
  never runs, and views the stream to its own shape. It conforms: dead code
  and same-shape views are not part of what the layer computes.
"""

import torch
from torch import nn


class _Children(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.pre_norm = layer.input_layernorm
        self.mixer = layer.self_attn
        self.mid_norm = layer.post_attention_layernorm
        self.ffn = layer.mlp


class Renamed(_Children):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.pre_norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        hidden_states, residual = self.mid_norm(hidden_states, residual)
        return self.ffn(hidden_states), residual


class ReorderedNorms(_Children):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.mid_norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        hidden_states, residual = self.pre_norm(hidden_states, residual)
        return self.ffn(hidden_states), residual


class DeadBranchAndView(_Children):
    debug = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.pre_norm(hidden_states, residual)
        unused = hidden_states * 2  # noqa: F841 - planted dead value
        if self.debug:
            hidden_states = hidden_states + 1
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states, residual = self.mid_norm(hidden_states, residual)
        return self.ffn(hidden_states), residual


RENAMED_ROLES = {
    "in_norm": "pre_norm",
    "attn": "mixer",
    "post_norm": "mid_norm",
    "moe": "ffn",
}
