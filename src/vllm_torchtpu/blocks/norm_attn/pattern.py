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
from typing import Protocol

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

from vllm_torchtpu.blocks.norm_attn.runner import AttentionParts, AttnConfig
from vllm_torchtpu.blocks.parts import BlockParts

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
            parameters.
        parts: The attention role's parts the runner holds: part -> child
            name inside attention (``qkv_proj``, ``attn``, ``o_proj``,
            ``rotary``, and ``q_norm`` / ``k_norm`` when present).
        config: The role properties the runner's composition reads.
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
    parts: tuple[tuple[str, str], ...]
    config: AttnConfig


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class NormAttnParts(AttentionParts, BlockParts):
    """A declared layer's parts for norm_then_attention.

    The runner's parts (``AttentionParts``) plus the layer's other roles:
    ``attention`` (the module holding the runner's parts, whose forward
    conformance compares with the runner's composition), and the rest of the
    layer (``post_norm``, ``mlp``), which stays opaque.
    """

    attention: nn.Module
    post_norm: nn.Module
    mlp: nn.Module

    pattern = staticmethod(norm_then_attention)

    def layer_children(self) -> dict[str, nn.Module]:
        return {
            "input_norm": self.input_norm,
            "attention": self.attention,
            "post_norm": self.post_norm,
            "mlp": self.mlp,
        }

    def problems(self) -> list[tuple[str, str]]:
        problems = [
            (name, f"is a {type(norm).__name__}, not an RMSNorm called with a residual")
            for name, norm in (
                ("input_norm", self.input_norm),
                ("post_norm", self.post_norm),
            )
            if not is_residual_norm(norm)
        ]
        inside = {id(m) for m in self.attention.modules()}
        for name, part in (
            ("qkv_proj", self.qkv_proj),
            ("attn", self.attn),
            ("o_proj", self.o_proj),
            ("rotary", self.rotary),
            ("q_norm", self.q_norm),
            ("k_norm", self.k_norm),
        ):
            if part is not None and id(part) not in inside:
                problems.append((name, "is not a module of attention"))
        kinds = (
            ("attn", AttentionLayerBase, "an attention layer"),
            ("qkv_proj", QKVParallelLinear, "a QKVParallelLinear"),
            ("o_proj", RowParallelLinear, "a RowParallelLinear"),
        )
        for name, kind, words in kinds:
            part = getattr(self, name)
            if not isinstance(part, kind):
                problems.append((name, f"is a {type(part).__name__}, not {words}"))
        weight = self.input_norm.weight
        if not problems and isinstance(weight, torch.Tensor):
            hidden = int(weight.shape[0])
            if self.qkv_proj.input_size != hidden:
                problems.append(
                    (
                        "qkv_proj",
                        f"reads {self.qkv_proj.input_size}, not hidden {hidden}",
                    )
                )
        if (self.q_norm is None) != (self.k_norm is None):
            problems.append(("q_norm", "and k_norm come together or not at all"))
        return problems


class PreNormAttention(Protocol):
    """An attention module with vLLM's usual part names."""

    @property
    def qkv_proj(self) -> nn.Module: ...
    @property
    def attn(self) -> nn.Module: ...
    @property
    def o_proj(self) -> nn.Module: ...
    @property
    def rotary_emb(self) -> nn.Module: ...


class PreNormLayer(Protocol):
    """A pre-norm decoder layer with vLLM's usual child names (attention aside)."""

    @property
    def input_layernorm(self) -> nn.Module: ...
    @property
    def post_attention_layernorm(self) -> nn.Module: ...
    @property
    def mlp(self) -> nn.Module: ...


def pre_norm_parts(layer: PreNormLayer, attention: PreNormAttention) -> NormAttnParts:
    """The parts of a layer and attention that use vLLM's usual names.

    Per-head q/k norms are optional in vLLM's attentions, so they are read if
    present; every other part is typed attribute access.
    """
    return NormAttnParts(
        input_norm=layer.input_layernorm,
        qkv_proj=attention.qkv_proj,
        attn=attention.attn,
        o_proj=attention.o_proj,
        rotary=attention.rotary_emb,
        q_norm=getattr(attention, "q_norm", None),
        k_norm=getattr(attention, "k_norm", None),
        attention=attention,  # type: ignore[arg-type]  # a Protocol over an nn.Module
        post_norm=layer.post_attention_layernorm,
        mlp=layer.mlp,
    )
