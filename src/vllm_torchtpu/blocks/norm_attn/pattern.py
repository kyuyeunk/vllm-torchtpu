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
"""The norm_then_attention sub-block: a reference forward and its role kinds.

Framework-owned. A pre-norm decoder layer runs ``input_norm(hidden, residual)``
and then attention on the normed tokens; every model that does so can declare
that its layer conforms, and every fusion written against this pattern's
roles then serves it with no per-model code.

The sub-block is checked in context: the rest of the layer (the second norm
and the MLP, dense or MoE) is opaque, so the check holds the layer to the
skeleton ``in_norm, attn, post_norm, mlp`` without caring what the MLP is.
It is the front half of ``attn_then_moe``; the two plans nest by scope.
"""

import dataclasses
from collections.abc import Callable

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear

Norm = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Op = Callable[[torch.Tensor], torch.Tensor]


def norm_then_attention(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    *,
    in_norm: Norm,
    attn: Op,
    post_norm: Norm,
    mlp: Op,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pattern: the sub-block (norm, attention), then the opaque rest."""
    x, residual = in_norm(hidden, residual)
    x = attn(x)
    x, residual = post_norm(x, residual)
    return mlp(x), residual


@dataclasses.dataclass(frozen=True)
class NormAttnRoles:
    """The declared binding, resolved on one layer, plus what the check found.

    Attributes:
        input_norm, attention, post_norm, mlp: Child names bound to the roles.
        qkv_proj: Dotted path of attention's input projection.
        attention_args: Attention's keyword arguments and their sources.
        hidden_param, residual_param, positions_param: The forward's stream
            parameters ("" when there is no positions parameter).
    """

    input_norm: str
    attention: str
    qkv_proj: str
    post_norm: str
    mlp: str
    attention_args: tuple[tuple[str, str], ...]
    hidden_param: str
    residual_param: str
    positions_param: str


def is_residual_norm(module: nn.Module) -> bool:
    """An RMSNorm-family norm that takes a residual."""
    return isinstance(module, (RMSNorm, GemmaRMSNorm)) and (
        "residual" in module.forward_native.__code__.co_varnames
    )


def norm_scale(norm: nn.Module) -> torch.Tensor:
    """The multiplier the norm applies after normalizing: w, or 1 + w."""
    weight = norm.weight.float()
    return 1.0 + weight if isinstance(norm, GemmaRMSNorm) else weight


def qkv_proj_of(attn: nn.Module, hidden: int) -> list[str]:
    """Attention's input projections from the hidden size."""
    return [
        name
        for name, sub in attn.named_modules()
        if isinstance(sub, QKVParallelLinear) and sub.input_size == hidden
    ]


def check_roles(layer: nn.Module, binding: dict[str, str]) -> list[str]:
    """Why the bound children are not the kinds their roles need."""
    problems = []
    for role in ("in_norm", "post_norm"):
        if not is_residual_norm(layer.get_submodule(binding[role])):
            problems.append(
                f"{role}={binding[role]} is not an RMSNorm called with a residual"
            )
    attn = layer.get_submodule(binding["attn"])
    if not any(isinstance(m, AttentionLayerBase) for m in attn.modules()):
        problems.append(f"attn={binding['attn']} holds no attention layer")
    if not problems:
        hidden = int(layer.get_submodule(binding["in_norm"]).weight.shape[0])
        if len(found := qkv_proj_of(attn, hidden)) != 1:
            problems.append(
                f"attn={binding['attn']} has {len(found)} QKVParallelLinear "
                f"from the hidden size {hidden}, not one"
            )
    return problems
