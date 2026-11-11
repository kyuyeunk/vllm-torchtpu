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
that conforms gets the ``declare_block`` call its model owner could add to
``models/<family>/<model>/registration.py``; one that does not gets the
reason.
"""

import itertools
import json
import sys

from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from prototype import _build
from vllm_torchtpu.blocks import BlockDeclaration
from vllm_torchtpu.blocks.attn_moe import conform
from vllm_torchtpu.blocks.attn_moe.pattern import _is_norm, attn_then_moe, moe_parts
from vllm_torchtpu.blocks.conformance import BlockConformanceError


def _moe_child(child: nn.Module) -> bool:
    try:
        moe_parts(child)
    except StopIteration:
        return False
    return True


def propose(layer: nn.Module) -> BlockDeclaration | str:
    """A conforming declaration for ``layer``, or why there is none."""
    children = dict(layer.named_children())
    norms = [n for n, c in children.items() if _is_norm(c)]
    attns = [
        n
        for n, c in children.items()
        if any(isinstance(m, AttentionLayerBase) for m in c.modules())
    ]
    moes = [n for n, c in children.items() if _moe_child(c)]
    if len(norms) != 2 or len(attns) != 1 or len(moes) != 1:
        return (
            f"no binding: {len(norms)} norms called with a residual, "
            f"{len(attns)} attention children, {len(moes)} MoE children"
        )
    error = ""
    for in_norm, post_norm in itertools.permutations(norms):
        roles = {"in_norm": in_norm, "attn": attns[0], "post_norm": post_norm}
        declaration = BlockDeclaration(
            type(layer), attn_then_moe, roles | {"moe": moes[0]}, lambda _: True
        )
        try:
            with set_current_vllm_config(_build.current_config()):
                conform(layer, declaration)
        except BlockConformanceError as e:
            error = error or str(e)
            continue
        return declaration
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
        roles = ", ".join(f'"{k}": "{v}"' for k, v in found.roles.items())
        print(f"  declare_block({spec.layer_class}, pattern=attn_then_moe,")
        print(f"                roles={{{roles}}})")
    if "--json" in sys.argv:
        print(json.dumps(verdicts))


if __name__ == "__main__":
    main()
