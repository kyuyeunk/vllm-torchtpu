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
"""The sub-block's forward and its one fused value, written against roles.

``fused_norm_add_qkv`` is the plan value ``fused_qkv``: one function adds the
residual, normalizes and projects, reading the norm's scale from the bound
norm and projecting through the bound projection's own matmul (so its weight
layout and quantization stay its own). It never names a model. Attention
then runs its own forward on the projected tokens (q/k norms, RoPE, gates and
the output projection are its own), handed them in place of its projection.
"""

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnRoles, norm_scale

FUSED_CALLS: dict[str, int] = {}
"""Fused calls per layer class, so a check can see the fused value ran."""


def fused_norm_add_qkv(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    norm: nn.Module,
    proj: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``qkv_proj(norm(hidden + residual))`` and the new residual, in one call."""
    added = hidden.float() + residual.float()
    variance = added.pow(2).mean(dim=-1, keepdim=True)
    normed = added * torch.rsqrt(variance + norm.variance_epsilon)
    normed = (normed * norm_scale(norm)).to(hidden.dtype)
    qkv = proj(normed)
    # A vLLM linear returns (output, bias) unless built with return_bias=False.
    qkv = qkv[0] if getattr(proj, "return_bias", True) else qkv
    return qkv, added.to(hidden.dtype)


@contextlib.contextmanager
def _handed(proj: nn.Module, qkv: torch.Tensor) -> Iterator[None]:
    # A vLLM linear returns (output, bias) unless built with return_bias=False.
    out = (qkv, None) if getattr(proj, "return_bias", True) else qkv
    proj.forward = lambda x: out
    try:
        yield
    finally:
        del proj.forward


def _arg(
    args: tuple[Any, ...], kwargs: dict[str, Any], order: tuple[str, ...], name: str
) -> Any:  # noqa: ANN401 - whatever the caller passed
    if name in kwargs:
        return kwargs[name]
    index = order.index(name)
    return args[index] if index < len(args) else None


class NormThenAttentionBlock(nn.Module):
    """Mixin placed before a conforming decoder layer's class.

    The installer sets ``_subblock_roles`` and ``_subblock_order``.
    """

    _subblock_roles: NormAttnRoles
    _subblock_order: tuple[str, ...]

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        roles = self._subblock_roles
        order = self._subblock_order
        hidden = _arg(args, kwargs, order, roles.hidden_param)
        residual = _arg(args, kwargs, order, roles.residual_param)
        if residual is None:  # The first layer has no residual to add.
            return super().forward(*args, **kwargs)
        stream = {roles.hidden_param: hidden}
        if roles.positions_param:
            stream[roles.positions_param] = _arg(
                args, kwargs, order, roles.positions_param
            )
        proj = self.get_submodule(roles.qkv_proj)
        qkv, residual = fused_norm_add_qkv(
            hidden, residual, self.get_submodule(roles.input_norm), proj
        )
        stream["normed"] = hidden  # unread: attention gets qkv from the handoff
        with _handed(proj, qkv):
            attn = self.get_submodule(roles.attention)(
                **{k: stream[src] for k, src in roles.attention_args}
            )
        FUSED_CALLS[type(self).__qualname__] = (
            FUSED_CALLS.get(type(self).__qualname__, 0) + 1
        )
        x, residual = self.get_submodule(roles.post_norm)(attn, residual)
        return self.get_submodule(roles.mlp)(x), residual
