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
layer per surveyed model (prototype/_build.py), it looks for children of the
role kinds (two RMSNorm-family norms, an attention child, a MoE child owning
a router and a MoE runner), tries every binding of them and every forward
convention (``blocks.stream``), and runs the same conformance check the
install runs. A layer that conforms gets the binding function its model owner
could add to ``models/<family>/<model>/registration.py``; one that does not
gets the reason.
"""

import argparse
import dataclasses
import itertools
import json
from collections.abc import Callable

from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.linear import RowParallelLinear

from prototype import _build
from vllm_torchtpu.blocks import BlockDeclaration
from vllm_torchtpu.blocks.attn_moe import check_moe, conform
from vllm_torchtpu.blocks.attn_moe.pattern import (
    AttnMoEParts,
    attn_then_moe,
    holds_attention,
    is_residual_norm,
)
from vllm_torchtpu.blocks.conformance import BlockConformanceError, ExportError
from vllm_torchtpu.blocks.parts import as_kind, attribute_path
from vllm_torchtpu.blocks.stream import ATTENDS, STREAMS, Attend, Stream


@dataclasses.dataclass(frozen=True)
class Proposal:
    """A conforming binding, and the helpers it uses."""

    parts: AttnMoEParts
    stream: Callable[..., Stream]
    attend: Callable[..., Attend]


def _one[T](module: nn.Module, kind: type[T]) -> T | None:
    found = [c for c in module.children() if isinstance(c, kind)]
    return found[0] if len(found) == 1 else None


def _out_proj(attention: nn.Module, hidden: int) -> RowParallelLinear | None:
    """The last RowParallelLinear to ``hidden`` in attention's subtree."""
    found = [
        m
        for m in attention.modules()
        if isinstance(m, RowParallelLinear) and m.output_size == hidden
    ]
    return found[-1] if found else None


def _error(told: list[str], untraced: list[str]) -> str:
    """The most telling error: one that got past calling the layer, if any."""
    return (told or untraced or ["no binding tried"])[0]


def propose(layer: nn.Module) -> Proposal | str:
    """A conforming binding for ``layer``, or why there is none."""
    children = list(layer.children())
    norms = [c for c in children if is_residual_norm(c)]
    attns = [c for c in children if holds_attention(c)]
    moes = [
        c
        for c in children
        if _one(c, MoERunnerInterface) is not None and _one(c, GateLinear) is not None
    ]
    if len(norms) != 2 or len(attns) != 1 or len(moes) != 1:
        return (
            f"no binding: {len(norms)} norms called with a residual, "
            f"{len(attns)} attention children, {len(moes)} MoE children"
        )
    (attention,), (moe,) = attns, moes
    router, experts = _one(moe, GateLinear), _one(moe, MoERunnerInterface)
    if router is None or experts is None:
        return "no binding: the MoE child has no single router and runner"
    out_proj = _out_proj(attention, router.input_size)
    if out_proj is None:
        return "does not conform: attention has no RowParallelLinear to the hidden size"
    try:  # The MoE child's own forward, whichever binding the layer gets.
        with set_current_vllm_config(_build.current_config()):
            check_moe(layer, moe, router, as_kind(experts, MoERunner))
    except BlockConformanceError as e:
        return f"does not conform: {e}"
    told: list[str] = []
    untraced: list[str] = []
    trials = itertools.product(itertools.permutations(norms), STREAMS, ATTENDS)
    for (in_norm, post_norm), stream, attend in trials:
        try:
            parts = AttnMoEParts(
                input_norm=in_norm,
                attention=attention,
                out_proj=out_proj,
                post_norm=post_norm,
                moe=moe,
                router=router,
                experts=experts,
                stream=stream(layer),
                attend=attend(attention),
            )
            declaration = BlockDeclaration(
                type(layer), attn_then_moe, lambda _, p=parts: p
            )
            with set_current_vllm_config(_build.current_config()):
                conform(layer, declaration)
        except ExportError as e:  # This convention cannot call the layer.
            untraced.append(str(e))
            continue
        except BlockConformanceError as e:
            told.append(str(e))
            continue
        return Proposal(parts, stream, attend)
    return f"does not conform: {_error(told, untraced)}"


def _attr(layer: nn.Module, part: nn.Module) -> str:
    return attribute_path(layer, part) or "?"


def binding_source(layer: nn.Module, found: Proposal) -> list[str]:
    """The binding function ``found`` stands for, as source lines."""
    parts = found.parts
    cls = type(layer).__name__
    attention = _attr(layer, parts.attention)
    moe = _attr(layer, parts.moe)
    return [
        f"@declare_block({cls})",
        f"def binding(layer: {cls}) -> AttnMoEParts | None:",
        f"    attention, moe = layer.{attention}, layer.{moe}",
        "    return AttnMoEParts(",
        f"        input_norm=layer.{_attr(layer, parts.input_norm)},",
        "        attention=attention,",
        f"        out_proj=attention.{_attr(parts.attention, parts.out_proj)},",
        f"        post_norm=layer.{_attr(layer, parts.post_norm)},",
        "        moe=moe,",
        f"        router=moe.{_attr(parts.moe, parts.router)},",
        f"        experts=moe.{_attr(parts.moe, parts.experts)},",
        f"        stream={found.stream.__name__}(layer),",
        f"        attend={found.attend.__name__}(attention),",
        "    )",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the verdicts")
    args = parser.parse_args()
    verdicts = {}
    print("=== proposed attn_then_moe declarations (not applied)")
    for spec in _build.SPECS:
        try:
            layer = _build.build(spec)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"\n{spec.name}: not built on a CPU: {type(e).__name__}: {e}")
            verdicts[spec.name] = "not built"
            continue
        found = propose(layer)
        where = f"{spec.layer_class.__module__}.{spec.layer_class.__qualname__}"
        print(f"\n{spec.name} ({where})")
        if isinstance(found, str):
            print(f"  {found}")
            verdicts[spec.name] = "declines"
            continue
        verdicts[spec.name] = "conforms"
        for line in binding_source(layer, found):
            print(f"  {line}")
    if args.json:
        print(json.dumps(verdicts))


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
