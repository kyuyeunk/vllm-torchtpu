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
"""Plan invariance for the attention layer, on a CPU.

For every value of the layer's plan, on every case, this forces the value,
exports the layer's forward with `torch.export` (an ATen graph, the same for
every backend) and compares the outcome with the layer's declared capability
on a TPU:

- a value declared capable must export, and its output must have the
  contract's shape and dtype ("fails" / "mismatch" otherwise);
- a value declared incapable that still exports is reported ("exports"):
  only a run on the target shows whether it fails loudly;
- a plan field whose single-field flips never change the graph is unread
  ("unread"); a field with one value in every plan is a handed-down contract,
  not a choice ("fixed").

Then a small numerics check: each exportable value's output against
upstream's own forward on the same weights, within the layer contract's
tolerance.

The MoE combine's structural tier compiles JAX for a TPU topology described
without a device. This layer is PyTorch, whose TPU compile needs the device,
so its structural tier is the exported graph.

Run: PYTHONPATH=.:src:<vllm-v0.33.0> python -m prototype.check_attention_plans
"""

import dataclasses
import enum
import itertools
import re
from typing import Any

# isort: off
from prototype import _cpu  # first: keeps the TPU runtime out

# isort: on
import torch
from vllm.model_executor.models import qwen3_next
from vllm.platforms.interface import PlatformEnum

from prototype import _layer
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.layers.core import plan_marker
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod

# (name, config overrides).
CONFIGS = (("gated", {}), ("no_gate", {"attn_output_gate": False}))
TOKENS = (1, 7, 64)
CHIP = alp.ChipDesc(PlatformEnum.TPU)
KNOWN: set[str] = set()  # Findings kept as ratchets; empty means none open.


class Verdict(enum.Enum):
    """How a forced value's outcome compares with its declaration."""

    AGREES = "agrees"
    FAILS = "FAILS though declared capable"
    EXPORTS = "EXPORTS though declined"


class _LayerField:
    """The ``layer`` field: which plan type a value is."""

    name = "layer"


_LAYER = _LayerField()
# A plan field: the plan type, or one field of the plan's dataclass.
_Field = _LayerField | dataclasses.Field[Any]


def _fields(plan: alp.AttentionLayerPlan) -> dict[_Field, object]:
    values: dict[_Field, object] = {_LAYER: type(plan)}
    values.update((f, getattr(plan, f.name)) for f in dataclasses.fields(plan))
    return values


def _graph(layer, args) -> str:
    code = torch.export.export(layer, args).graph_module.code
    return re.sub(r"\s+#.*", "", code)  # drop source comments


def check_config(name: str, overrides: dict) -> list[str]:
    config = _layer.tiny_config(**overrides)
    probe = _cpu.build_real(config)
    caps = layer_mod.describe_children(probe)
    values = alp.enumerate_plans(probe.desc)
    owners = {plan_marker.PLAN_OWNERS[type(v)] for v in values}
    print(f"== {name}: {len(values)} plan values, owner {sorted(owners)}")
    findings: list[str] = []
    graphs: dict[tuple[alp.AttentionLayerPlan, int], str] = {}
    for value, t in itertools.product(values, TOKENS):
        declines = alp.plan_declines(value, probe.desc, caps, CHIP)
        layer = _cpu.build_real(config, value)
        args = _cpu.inputs(config, t)
        exports = False
        try:
            graphs[value, t] = _graph(layer, args)
            with torch.no_grad():
                out = layer(*args)
            spec = layer.contract.output
            exports = tuple(out.shape) == (t, spec.shape[1]) and (
                out.dtype == spec.dtype
            )
            outcome = (
                "exports" if exports else f"mismatch {tuple(out.shape)} {out.dtype}"
            )
        except Exception as e:  # noqa: BLE001 -- the outcome is the finding.
            outcome = (
                f"fails ({type(e).__name__}: {(str(e).splitlines() or [''])[0][:70]})"
            )
        verdict = Verdict.AGREES
        if not declines and not exports:
            verdict = Verdict.FAILS
        elif declines and exports:
            verdict = Verdict.EXPORTS
        if t == TOKENS[0] or verdict is not Verdict.AGREES:
            print(f"  t={t:<3} {alp.plan_name(value)}")
            print(
                f"        declared {'capable' if not declines else 'declines'};"
                f" {outcome}; {verdict.value}"
            )
        if verdict is not Verdict.AGREES:
            findings.append(f"{name} t={t} {alp.plan_name(value)}: {verdict.value}")
    # Field reads: flip one field between two exportable values.
    exported = {p for p, _ in graphs}
    plans = [v for v in values if v in exported]
    all_fields = set().union(*(_fields(v) for v in values))
    for field in sorted(all_fields, key=lambda f: f.name):
        seen = {repr(_fields(v)[field]) for v in values if field in _fields(v)}
        if len(seen) == 1:
            print(
                f"  field {field.name}: fixed"
                f" (one value wherever it exists: {seen.pop()})"
            )
            continue
        pairs = [
            (a, b)
            for a, b in itertools.combinations(plans, 2)
            if [
                k
                for k in set(_fields(a)) | set(_fields(b))
                if _fields(a).get(k) != _fields(b).get(k)
            ]
            == [field]
        ]
        if not pairs:
            print(f"  field {field.name}: no exportable single-field flip to compare")
            continue
        changed = any(graphs[a, t] != graphs[b, t] for a, b in pairs for t in TOKENS)
        print(f"  field {field.name}: {'read' if changed else 'UNREAD'}")
        if not changed:
            findings.append(f"{name} unread {field.name}")
    # Numerics: each exportable value against upstream's forward.
    for value in plans:
        layer = _cpu.build_real(config, value)
        atol, rtol = layer.contract.atol, layer.contract.rtol
        worst = 0.0
        for t in TOKENS:
            args = _cpu.inputs(config, t)
            with torch.no_grad():
                got = layer(*args).float()
                ref = qwen3_next.Qwen3NextAttention.forward(layer, *args).float()
            worst = max(worst, (got - ref).abs().max().item())
            if not torch.allclose(got, ref, atol=atol, rtol=rtol):
                findings.append(f"{name} {alp.plan_name(value)} outside tolerance")
        print(
            f"  numerics {alp.plan_name(value)} vs upstream forward:"
            f" max |diff| {worst:.1e} (atol {atol}, rtol {rtol})"
        )
    return findings


def main() -> int:
    findings = []
    with _layer.runtime():
        for name, overrides in CONFIGS:
            findings += check_config(name, overrides)
    for finding in findings:
        print(f"finding: {finding}")
    new = sorted(set(findings) - KNOWN)
    print(f"findings: {len(findings)} ({len(new)} new)")
    return 1 if new else 0


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    raise SystemExit(main())
