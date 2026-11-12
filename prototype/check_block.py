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
"""Checks the attn_then_moe block on a CPU: conformance, plan, outside writes.

1. Conformance: the declared Qwen3.5 layers (full and linear attention)
   conform; so do two planted fixtures (renamed attributes declared with the
   new names; a dead branch and a same-shape view). Qwen3.5 with
   ``layer_scale`` and a fixture that calls its norms in the wrong order get a
   conformance error naming the role edge and the source line.
2. Plan per bucket: the declared Qwen3.5 block at its real size (hidden
   2048, bf16, TP 4) under the rule, the overrides and the cross-check.
3. Outside write: after binding, writing a child decision the plan owns
   raises (the MoE runner's ``skip_final_all_reduce``; attention's contract
   is a constructor argument of the nested runner, not a field).
4. A mismatch declines at install: the misordered fixture, declared, keeps its
   upstream forward and the report names the edge. CI still fails on it: set
   ``PLANT_MISMATCH = True`` and check 1 treats it as a declared model that
   must conform, so the command exits 1.
"""

import dataclasses
import sys
from collections.abc import Callable

import torch
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from prototype import _build, fixtures
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import BlockDeclaration, installer
from vllm_torchtpu.blocks.attn_moe import contract_adapter
from vllm_torchtpu.blocks.attn_moe import plan as plan_mod
from vllm_torchtpu.blocks.attn_moe.install import apply_block, conform
from vllm_torchtpu.blocks.attn_moe.pattern import (
    AttnMoEBinding,
    attn_then_moe,
    block_key,
)
from vllm_torchtpu.blocks.binding import as_kind
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.context import InstallContext

_BUCKETS = (16, 256, 1024, 4096, 8192, 16384)

# Comment in to see CI fail: the misordered fixture becomes a declared model.
PLANT_MISMATCH = False
# PLANT_MISMATCH = True


def _declared(layer: nn.Module) -> BlockDeclaration:
    blocks.load_declarations()
    return blocks.DECLARATIONS[(type(layer), attn_then_moe)]


def _cases() -> list[tuple[str, nn.Module, BlockDeclaration, bool]]:
    """(label, layer, declaration, expected to conform)."""
    full, linear = _build.SPECS[:2]
    cases = []
    for spec in (full, linear):
        layer = _build.build(spec)
        cases.append((f"declared {spec.name}", layer, _declared(layer), True))
    hf = full.config()
    hf.layer_scale = True
    scaled = _build.build(dataclasses.replace(full, config=lambda: hf))
    cases.append(("Qwen3.5 with layer_scale", scaled, _declared(scaled), False))
    for cls, expect in (
        (fixtures.Renamed, True),
        (fixtures.ReorderedNorms, False),
        (fixtures.DeadBranchAndView, True),
    ):
        layer = cls(_build.build(full))
        declaration = BlockDeclaration.of(cls, fixtures.renamed_binding)
        cases.append((f"fixture {cls.__name__}", layer, declaration, expect))
    if PLANT_MISMATCH:
        layer = fixtures.ReorderedNorms(_build.build(full))
        declaration = BlockDeclaration.of(
            fixtures.ReorderedNorms, fixtures.renamed_binding
        )
        cases.append(("declared ReorderedNorms (planted)", layer, declaration, True))
    return cases


def conformance() -> list[str]:
    failures = []
    for label, layer, declaration, expect in _cases():
        try:
            with set_current_vllm_config(_build.current_config()):
                conform(layer, declaration)
            got, detail = True, "conforms"
        except BlockConformanceError as e:
            got, detail = False, str(e)
        print(f"    {'ok  ' if got == expect else 'FAIL'} {label}: {detail}")
        if got != expect:
            failures.append(f"{label}: expected {'conforms' if expect else 'an error'}")
    return failures


def plan_table() -> list[str]:
    layer = _build.build(_build.SPECS[0])
    binding = _declared(layer).binding(layer)
    assert isinstance(binding, AttnMoEBinding)
    key = block_key(binding, tp=4, dtype=torch.bfloat16)
    key = dataclasses.replace(key, hidden=2048)  # Qwen3.5-35B-A3B
    fused = dataclasses.replace(key, fused_output_reduced=True)
    sp_moe = dataclasses.replace(key, sp_moe=True)
    print(f"    rule floor: {plan_mod.min_tokens(key)} tokens (8 MiB per chip)")
    overlap = "collectives=reduce_scatter_all_gather"
    reduce = "collectives=all_reduce"
    cases = (
        ("rule", key, "", "", False),
        ("fused EP on", fused, "", "", False),
        ("model plan pin", key, "", reduce, False),
        ("user over model", key, overlap, reduce, False),
        ("user pin, SP-MoE", sp_moe, overlap, "", True),
        ("contract pin", key, "dtype=float32", "", True),
    )
    failures = []
    for label, k, user, model, should_raise in cases:
        try:
            table, fell_back = plan_mod.resolve_table(
                k,
                _BUCKETS,
                plan_mod.parse_pin(user, "TPU_BLOCK_PLAN"),
                plan_mod.parse_pin(model, "tpu_block_plan"),
            )
        except ValueError as err:
            print(f"    {label + ':':17} raises: {err}")
            failures += [] if should_raise else [f"{label} raised"]
            continue
        cells = " ".join(
            f"{b}:{p.collectives.value[:2]}({s.value})" for b, p, s in table.entries
        )
        tail = f" [rule fell back: {'; '.join(fell_back)}]" if fell_back else ""
        print(f"    {label + ':':17} {cells}{tail}")
        failures += [f"{label} did not raise"] if should_raise else []
    return failures


def outside_write() -> list[str]:
    layer = _build.build(_build.SPECS[0])
    declaration = _declared(layer)
    with set_current_vllm_config(_build.current_config()):
        binding = conform(layer, declaration)
    table = plan_mod.BlockPlanTable(((8192, plan_mod.OVERLAP, plan_mod.Source.RULE),))
    apply_block("layers.0", layer, binding, table, tp=4)
    try:
        as_kind(binding.experts, MoERunner).moe_config.skip_final_all_reduce = False
    except contract_adapter.BlockContractError as e:
        print(f"    raised: {e}")
        return []
    return ["moe_config.skip_final_all_reduce = False was accepted after binding"]


def mismatch_raises() -> list[str]:
    layer = fixtures.ReorderedNorms(_build.build(_build.SPECS[0]))
    key = (fixtures.ReorderedNorms, attn_then_moe)
    blocks.DECLARATIONS[key] = BlockDeclaration.of(
        fixtures.ReorderedNorms, fixtures.renamed_binding
    )
    context = InstallContext(
        tp=4,
        dtype=torch.bfloat16,
        buckets=_BUCKETS,
    )
    try:
        with set_current_vllm_config(_build.current_config()):
            installer.install(nn.ModuleDict({"layer": layer}), context)
    except BlockConformanceError as e:
        print(f"    raised: {e}")
        return []
    finally:
        del blocks.DECLARATIONS[key]
    return ["the mismatched declaration did not stop startup"]


def _run(check: Callable[[], list[str]], title: str) -> bool:
    print(title)
    failures = check()
    for failure in failures:
        print(f"    FAIL {failure}")
    print("  -> " + ("FAIL" if failures else "PASS"))
    return not failures


def main() -> None:
    ok = _run(conformance, "1 conformance: declared layers and planted fixtures")
    ok &= _run(plan_table, "2 plan per bucket (al = all_reduce, re = overlap)")
    ok &= _run(outside_write, "3 outside write raises after binding")
    ok &= _run(mismatch_raises, "4 a declaration that does not conform stops startup")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
