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
"""The sub-block's one fused value, installed by rewiring two roles.

``fused_qkv`` moves the input norm into attention's projection: the bound
input norm becomes ``AddOnly`` (it adds the residual and returns the sum),
and the bound QKV projection becomes ``FusedNormProjection`` (it normalizes
and projects in one call, through the projection's own matmul, so its weight
layout and quantization stay its own). The layer's own forward is untouched,
so the same two modules serve every declared model, under any compiler. The
conformance check has shown that the normed tokens reach nothing but
attention, and attention's input nothing but its projection.
"""

from typing import Any

import torch
from torch import nn

from vllm_torchtpu.blocks.norm_attn.pattern import norm_scale

FUSED_CALLS: dict[str, int] = {}
"""Fused projections run, per layer class (eager runs only)."""


class AddOnly(nn.Module):
    """The input norm's role under ``fused_qkv``: add the residual, nothing else.

    Returns the float32 sum to attention (its projection normalizes it, as the
    norm would have) and the sum in the stream's dtype as the new residual.
    """

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:  # The first layer: the stream is its own residual.
            return x
        added = x.float() + residual.float()
        return added, added.to(x.dtype)


class FusedNormProjection(nn.Module):
    """``proj(norm(x))`` in one call: the plan value ``fused_qkv``.

    Stands in for PR #3096's fused_qkv kernel; written against the roles, so
    it never names a model.
    """

    def __init__(self, norm: nn.Module, proj: nn.Module, owner: str) -> None:
        super().__init__()
        self.norm = norm
        self.proj = proj
        self.owner = owner

    def forward(self, x: torch.Tensor) -> Any:  # noqa: ANN401 - whatever proj returns
        x32 = x.float()
        normed = x32 * torch.rsqrt(
            x32.pow(2).mean(dim=-1, keepdim=True) + self.norm.variance_epsilon
        )
        normed = (normed * norm_scale(self.norm)).to(self.norm.weight.dtype)
        if not torch.compiler.is_compiling():
            FUSED_CALLS[self.owner] = FUSED_CALLS.get(self.owner, 0) + 1
        return self.proj(normed)


def install_fused_qkv(layer: nn.Module, input_norm: str, qkv_proj: str) -> None:
    """Rewires the two roles on ``layer`` for ``fused_qkv``."""
    norm = layer.get_submodule(input_norm)
    parent, _, name = qkv_proj.rpartition(".")
    owner = layer.get_submodule(parent) if parent else layer
    proj = owner.get_submodule(name)
    setattr(owner, name, FusedNormProjection(norm, proj, type(layer).__qualname__))
    norm_parent, _, norm_name = input_norm.rpartition(".")
    norm_owner = layer.get_submodule(norm_parent) if norm_parent else layer
    setattr(norm_owner, norm_name, AddOnly())
