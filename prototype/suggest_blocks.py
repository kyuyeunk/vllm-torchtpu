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
role kinds (two RMSNorm-family norms, an attention child, a MoE child), tries
the binding, and runs the same conformance check the install runs. A layer
that conforms gets the binding function its model owner could add to
``models/<family>/<model>/registration.py``; one that does not gets the
reason.
"""

import inspect
import itertools
import json
import sys

from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner_interface import (
    MoERunnerInterface,
)
from vllm.model_executor.layers.linear import RowParallelLinear

from prototype import _build
from prototype.subblock_reach import guess_stream
from vllm_torchtpu.blocks import BlockDeclaration
from vllm_torchtpu.blocks.attn_moe.install import conform
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEBinding
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.norm_attn.pattern import is_residual_norm
from vllm_torchtpu.blocks.stream import attend_with_positions, attend_without_positions


def _moe_parts(child: nn.Module) -> tuple[str, str] | None:
    """The router's and the MoE runner's names in ``child``, if it has one each."""
    routers = [n for n, c in child.named_children() if isinstance(c, GateLinear)]
    runners = [
        n for n, c in child.named_children() if isinstance(c, MoERunnerInterface)
    ]
    return (routers[0], runners[0]) if len(routers) == len(runners) == 1 else None


def _binding(names: dict[str, str]):  # noqa: ANN202 - a binding function
    def guessed(layer: nn.Module) -> AttnMoEBinding:
        moe = layer.get_submodule(names["moe"])
        attention = layer.get_submodule(names["attention"])
        projections = [
            m for m in attention.modules() if isinstance(m, RowParallelLinear)
        ]
        reads_positions = "positions" in inspect.signature(attention.forward).parameters
        return AttnMoEBinding(
            input_norm=layer.get_submodule(names["input_norm"]),
            attention=attention,
            out_proj=projections[-1],
            post_norm=layer.get_submodule(names["post_norm"]),
            moe=moe,
            router=moe.get_submodule(names["router"]),
            experts=moe.get_submodule(names["experts"]),
            stream=guess_stream(layer),
            attend=(
                attend_with_positions(attention)  # pyright: ignore[reportArgumentType]
                if reads_positions
                else attend_without_positions(attention)  # pyright: ignore[reportArgumentType]
            ),
        )

    return guessed


def propose(layer: nn.Module) -> tuple[BlockDeclaration, dict[str, str]] | str:
    """A conforming declaration for ``layer`` and its names, or why there is none."""
    children = dict(layer.named_children())
    norms = [n for n, c in children.items() if is_residual_norm(c)]
    attns = [
        n
        for n, c in children.items()
        if any(isinstance(m, AttentionLayerBase) for m in c.modules())
    ]
    moes = [n for n, c in children.items() if _moe_parts(c) is not None]
    if len(norms) != 2 or len(attns) != 1 or len(moes) != 1:
        return (
            f"no binding: {len(norms)} norms called with a residual, "
            f"{len(attns)} attention children, {len(moes)} MoE children"
        )
    error = ""
    found = _moe_parts(children[moes[0]])
    assert found is not None
    for input_norm, post_norm in itertools.permutations(norms):
        names = {
            "input_norm": input_norm,
            "attention": attns[0],
            "post_norm": post_norm,
            "moe": moes[0],
            "router": found[0],
            "experts": found[1],
        }
        declaration = BlockDeclaration.of(type(layer), _binding(names))
        try:
            with set_current_vllm_config(_build.current_config()):
                conform(layer, declaration)
        except BlockConformanceError as e:
            error = error or str(e)
            continue
        return declaration, names
    return f"does not conform: {error}"


def main() -> None:
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
        print(f"\n{spec.name} ({spec.module}.{spec.layer_class})")
        if isinstance(found, str):
            print(f"  {found}")
            verdicts[spec.name] = "declines"
            continue
        verdicts[spec.name] = "conforms"
        _, n = found
        print(f"  @declare_block({spec.layer_class})")
        print(f"  def block(layer: {spec.layer_class}) -> AttnMoEBinding:")
        print("      return AttnMoEBinding(")
        for field in ("input_norm", "attention", "post_norm", "moe"):
            print(f"          {field}=layer.{n[field]},")
        print(f"          router=layer.{n['moe']}.{n['router']},")
        print(f"          experts=layer.{n['moe']}.{n['experts']},")
        print("      )")
    if "--json" in sys.argv:
        print(json.dumps(verdicts))


if __name__ == "__main__":
    main()
