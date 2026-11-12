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
from typing import ClassVar, Protocol, override

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear

from vllm_torchtpu.blocks import calls
from vllm_torchtpu.blocks.binding import BlockBinding, as_kind
from vllm_torchtpu.blocks.norm_attn.runner import AttentionBinding
from vllm_torchtpu.blocks.stream import Attend, Stream, attend_with_positions

Norm = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Op = Callable[[torch.Tensor], torch.Tensor]


def norm_then_attention(  # noqa: PLR0913 - the pattern takes one callable per role
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


def is_residual_norm(module: object) -> bool:
    """VLLM's RMSNorm family, whose contract takes ``(x, residual)``."""
    return isinstance(module, (RMSNorm, GemmaRMSNorm))


def norm_scale(norm: nn.Module) -> torch.Tensor:
    """The multiplier the norm applies after normalizing: w, or 1 + w."""
    weight = as_kind(norm.weight, torch.Tensor).float()
    return 1.0 + weight if isinstance(norm, GemmaRMSNorm) else weight


@dataclasses.dataclass(frozen=True, kw_only=True)
class NormAttnBinding(AttentionBinding, BlockBinding):
    """A declared layer's binding for norm_then_attention.

    The runner's binding (``AttentionBinding``) plus the layer's other roles:
    ``attention`` (the module holding the runner's binding, whose forward
    conformance compares with the runner's composition), the rest of the
    layer (``post_norm``, ``mlp``), which stays opaque, ``output_gate``:
    whether the projection also emits a per-head gate (stated, not guessed;
    conformance holds the forward to it), and how the layer's forward and
    attention are called (``stream``, ``attend``), so nothing reads a
    parameter name.
    """

    attention: nn.Module
    post_norm: calls.ResidualNorm
    mlp: calls.Mlp
    output_gate: bool
    stream: Stream
    attend: Attend

    pattern: ClassVar[Callable[..., object]] = norm_then_attention

    @override
    def layer_children(self) -> dict[str, object]:
        return {
            "input_norm": self.input_norm,
            "attention": self.attention,
            "post_norm": self.post_norm,
            "mlp": self.mlp,
        }

    @override
    def problems(self) -> list[tuple[str, object, str]]:
        problems: list[tuple[str, object, str]] = [
            (name, norm, f"is a {type(norm).__name__}, not a vLLM RMSNorm")
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
                problems.append((name, part, "is not a module of attention"))
        kinds: tuple[tuple[str, object, type, str], ...] = (
            ("attn", self.attn, AttentionLayerBase, "an attention layer"),
            ("qkv_proj", self.qkv_proj, QKVParallelLinear, "a QKVParallelLinear"),
            ("o_proj", self.o_proj, RowParallelLinear, "a RowParallelLinear"),
        )
        for name, part, kind, words in kinds:
            if not isinstance(part, kind):
                problems.append(
                    (name, part, f"is a {type(part).__name__}, not {words}")
                )
        norm = self.input_norm
        weight = norm.weight if isinstance(norm, (RMSNorm, GemmaRMSNorm)) else None
        qkv = self.qkv_proj
        if not problems and isinstance(weight, torch.Tensor):
            hidden = int(weight.shape[0])
            if isinstance(qkv, QKVParallelLinear) and qkv.input_size != hidden:
                why = f"reads {qkv.input_size}, not hidden {hidden}"
                problems.append(("qkv_proj", qkv, why))
        if (self.q_norm is None) != (self.k_norm is None):
            why = "and k_norm come together or not at all"
            problems.append(("q_norm", self.q_norm or self.k_norm, why))
        return problems


class PreNormAttention(Protocol):
    """An attention module with vLLM's usual part names, called by keyword."""

    def forward(
        self, *, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """The call the runner makes."""
        ...

    @property
    def qkv_proj(self) -> nn.Module:
        """The part's ``qkv_proj``."""
        ...

    @property
    def attn(self) -> nn.Module:
        """The part's ``attn``."""
        ...

    @property
    def o_proj(self) -> RowParallelLinear:
        """The part's ``o_proj``."""
        ...

    @property
    def rotary_emb(self) -> nn.Module:
        """The part's ``rotary_emb``."""
        ...


class PreNormLayer(Protocol):
    """A pre-norm decoder layer with vLLM's usual child names (attention aside)."""

    @property
    def input_layernorm(self) -> nn.Module:
        """The part's ``input_layernorm``."""
        ...

    @property
    def post_attention_layernorm(self) -> nn.Module:
        """The part's ``post_attention_layernorm``."""
        ...

    @property
    def mlp(self) -> nn.Module:
        """The part's ``mlp``."""
        ...


def pre_norm_binding(
    layer: PreNormLayer,
    attention: PreNormAttention,
    *,
    qk_norm: tuple[nn.Module, nn.Module] | None,
    output_gate: bool,
    stream: Stream,
) -> NormAttnBinding:
    """The binding of a layer and attention that use vLLM's usual names.

    Per-head q/k norms are optional in vLLM's attentions, so the binding
    function states them (``qk_norm``); every part is typed attribute access.
    """
    return NormAttnBinding(
        input_norm=layer.input_layernorm,
        qkv_proj=attention.qkv_proj,
        attn=attention.attn,
        o_proj=attention.o_proj,
        rotary=attention.rotary_emb,
        q_norm=None if qk_norm is None else qk_norm[0],
        k_norm=None if qk_norm is None else qk_norm[1],
        attention=attention,  # pyright: ignore[reportArgumentType] - a Protocol over an nn.Module
        post_norm=layer.post_attention_layernorm,
        mlp=layer.mlp,
        output_gate=output_gate,
        stream=stream,
        attend=attend_with_positions(attention),
    )
