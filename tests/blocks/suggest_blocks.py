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
"""Proposes attn_then_moe declarations for vLLM's MoE decoder layers.

A suggestion tool: it never applies anything. For one meta-built decoder
layer per surveyed model (tests/blocks/_build.py), it looks for children of the
role kinds (two RMSNorm-family norms, an attention child, a MoE child), tries
the binding, and runs the same conformance check the install runs. A layer
that conforms gets the binding function its model owner could add to
``models/<family>/<model>/registration.py``; one that does not gets the
reason. ``python -m prototype.suggest_blocks`` prints the proposals;
``test_suggest_blocks.py`` asserts each layer's verdict.
"""

import dataclasses
import enum
import inspect
import itertools
import json
from collections.abc import Callable

from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)

from tests.blocks import _build
from tests.blocks._report import Emit
from tests.blocks.subblock_reach import guess_stream
from vllm_torchtpu.blocks import BlockDeclaration
from vllm_torchtpu.blocks.attn_moe.install import conform
from vllm_torchtpu.blocks.attn_moe.pattern import AttnThenMoE
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.norm_attn.pattern import is_residual_norm
from vllm_torchtpu.blocks.roles import Attention, LinearAttention, MoE


def _moe_parts(child: nn.Module) -> tuple[str, str] | None:
    """The router's and the MoE runner's names in ``child``, if it has one each."""
    routers = [n for n, c in child.named_children() if isinstance(c, GateLinear)]
    runners = [
        n for n, c in child.named_children() if isinstance(c, MoERunnerInterface)
    ]
    return (routers[0], runners[0]) if len(routers) == len(runners) == 1 else None


@dataclasses.dataclass(frozen=True)
class Names:
    """The attribute names a proposed binding reads (the layer's and its MoE's)."""

    input_norm: str
    attention: str
    post_norm: str
    moe: str
    router: str
    experts: str


def linear_attention(attention: nn.Module) -> bool:
    """Whether the survey reads ``attention`` as linear (no positions argument)."""
    parameters = inspect.signature(attention.forward).parameters
    # The survey reads the forward's parameter names.
    return "positions" not in parameters  # str-ok: a parameter name


def _roles(names: Names) -> Callable[[nn.Module], AttnThenMoE.Roles]:
    def guessed(layer: nn.Module) -> AttnThenMoE.Roles:
        moe = layer.get_submodule(names.moe)
        attention = layer.get_submodule(names.attention)
        role = LinearAttention if linear_attention(attention) else Attention
        try:
            attn = role.from_module(attention)  # pyright: ignore[reportArgumentType]
        except AttributeError as e:
            msg = (
                f"attention={names.attention} fits neither Attention nor "
                f"LinearAttention under vLLM's usual part names ({e})"
            )
            raise BlockConformanceError(msg) from e
        return AttnThenMoE.Roles(
            in_norm=layer.get_submodule(names.input_norm),
            attn=attn,
            post_norm=layer.get_submodule(names.post_norm),
            moe=MoE(
                module=moe,
                router=moe.get_submodule(names.router),
                experts=moe.get_submodule(names.experts),
            ),
            layer_call=guess_stream(layer),
        )

    return guessed


_ROLE_NORMS = 2  # The pattern's two residual norms.


def propose(layer: nn.Module) -> tuple[BlockDeclaration, Names] | str:
    """A conforming declaration for ``layer`` and its names, or why there is none."""
    children = dict(layer.named_children())
    norms = [n for n, c in children.items() if is_residual_norm(c)]
    attns = [
        n
        for n, c in children.items()
        if any(isinstance(m, AttentionLayerBase) for m in c.modules())
    ]
    moes = [n for n, c in children.items() if _moe_parts(c) is not None]
    if len(norms) != _ROLE_NORMS or len(attns) != 1 or len(moes) != 1:
        return (
            f"no binding: {len(norms)} norms called with a residual, "
            f"{len(attns)} attention children, {len(moes)} MoE children"
        )
    error = ""
    found = _moe_parts(children[moes[0]])
    assert found is not None
    for input_norm, post_norm in itertools.permutations(norms):
        names = Names(
            input_norm=input_norm,
            attention=attns[0],
            post_norm=post_norm,
            moe=moes[0],
            router=found[0],
            experts=found[1],
        )
        declaration = AttnThenMoE.declaration(type(layer), _roles(names))
        try:
            with set_current_vllm_config(_build.current_config()):
                conform(layer, declaration)
        except BlockConformanceError as e:
            error = error or str(e)
            continue
        return declaration, names
    return f"does not conform: {error}"


class Suggestion(enum.Enum):
    """What the tool found for one surveyed layer."""

    _value_: str

    CONFORMS = "conforms"
    DECLINES = "declines"
    NOT_BUILT = "not built"


def suggest(spec: _build.Spec, emit: Emit) -> Suggestion:
    """Builds the spec's layer, emits its proposal (or why none), its verdict."""
    try:
        layer = _build.build(spec)
    except Exception as e:  # noqa: BLE001 - report and continue
        emit(f"\n{spec.name}: not built on a CPU: {type(e).__name__}: {e}")
        return Suggestion.NOT_BUILT
    found = propose(layer)
    emit(f"\n{spec.name} ({spec.module}.{spec.layer_class})")
    if isinstance(found, str):
        emit(f"  {found}")
        return Suggestion.DECLINES
    _, n = found
    emit(f"  @AttnThenMoE.register({spec.layer_class})")
    emit(f"  def block(layer: {spec.layer_class}) -> AttnThenMoE.Roles:")
    emit("      return AttnThenMoE.Roles(")
    emit(f"          in_norm=layer.{n.input_norm},")
    role = (
        "LinearAttention"
        if linear_attention(layer.get_submodule(n.attention))
        else "Attention"
    )
    emit(f"          attn={role}.from_module(layer.{n.attention}),")
    emit(f"          post_norm=layer.{n.post_norm},")
    emit(f"          moe=MoE(module=layer.{n.moe}, router=layer.{n.moe}.{n.router},")
    emit(f"                  experts=layer.{n.moe}.{n.experts}),")
    emit("          layer_call=...,")
    emit("      )")
    return Suggestion.CONFORMS


def report(emit: Emit, *, as_json: bool) -> None:
    """Emits every surveyed layer's proposal; ``as_json``: then the verdicts."""
    verdicts: dict[str, str] = {}
    emit("=== proposed attn_then_moe declarations (not applied)")
    for spec in _build.SPECS:
        verdicts[spec.name] = suggest(spec, emit).value
    if as_json:
        emit(json.dumps(verdicts))
