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

`test_attention_layer_plan_invariance.py` asserts each case's findings equal
its recorded known set; `prototype/check_attention_plans.py` prints them.
"""

# isort: off
from tests.layers.attention_layer_plan import cpu_layer  # first: no TPU runtime

# isort: on
import dataclasses
import enum
import itertools
import re

import torch
from transformers import Qwen3NextConfig
from vllm.model_executor.models import qwen3_next
from vllm.platforms.interface import PlatformEnum

from tests.layers.attention_layer_plan import layer_builder
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.layers.core import plan_marker
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod


@dataclasses.dataclass(frozen=True)
class LayerCase:
    """One layer configuration the plan is checked on.

    Attributes:
        name: The printed name.
        attn_output_gate: Whether the layer gates its output.
        known: Findings kept as ratchets; empty means none open.
    """

    name: str
    attn_output_gate: bool
    known: frozenset[str] = frozenset()

    def config(self) -> Qwen3NextConfig:
        """The case's tiny config.

        Returns:
            The config.
        """
        return layer_builder.tiny_config(attn_output_gate=self.attn_output_gate)


CASES = (
    LayerCase("gated", attn_output_gate=True),
    LayerCase("no_gate", attn_output_gate=False),
)
TOKENS = (1, 7, 64)
CHIP = alp.ChipDesc(PlatformEnum.TPU)


class Verdict(enum.Enum):
    """How a forced value's outcome compares with its declaration."""

    AGREES = "agrees"
    FAILS = "FAILS though declared capable"
    EXPORTS = "EXPORTS though declined"


class _LayerField:
    """The ``layer`` field: which plan type a value is."""

    name: str = "layer"


_LAYER = _LayerField()
# A plan field: the plan type, or one field of the plan's dataclass.
_Field = _LayerField | dataclasses.Field[object]


def _fields(plan: alp.AttentionLayerPlan) -> dict[_Field, object]:
    values: dict[_Field, object] = {_LAYER: type(plan)}
    values.update((f, getattr(plan, f.name)) for f in dataclasses.fields(plan))
    return values


def _graph(layer: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> str:
    code = torch.export.export(layer, args).graph_module.code
    return re.sub(r"\s+#.*", "", code)  # drop source comments


Graphs = dict[tuple[alp.AttentionLayerPlan, int], str]


def _check_declarations(
    case: LayerCase,
    values: list[alp.AttentionLayerPlan],
    caps: alp.ChildCapability,
    probe: layer_mod.TpuQwen3NextAttention,
) -> tuple[list[str], Graphs]:
    """Forces each value on each token count and compares with its declaration.

    Returns:
        The findings, and each exported value's graph per token count.
    """
    config = case.config()
    findings: list[str] = []
    graphs: Graphs = {}
    for value, t in itertools.product(values, TOKENS):
        declines = alp.plan_declines(value, probe.desc, caps, CHIP)
        layer = cpu_layer.build_real(config, value)
        args = cpu_layer.inputs(config, t)
        exports = False
        try:
            graphs[value, t] = _graph(layer, args)
            with torch.no_grad():
                out: torch.Tensor = layer(*args)
            spec = layer.contract.output
            exports = tuple(out.shape) == (t, spec.shape[1]) and (
                out.dtype == spec.dtype
            )
            outcome = (
                "exports" if exports else f"mismatch {tuple(out.shape)} {out.dtype}"
            )
        except Exception as e:  # noqa: BLE001  # the outcome is the finding
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
            findings.append(
                f"{case.name} t={t} {alp.plan_name(value)}: {verdict.value}"
            )
    return findings, graphs


def _single_field_flips(
    plans: list[alp.AttentionLayerPlan], field: _Field
) -> list[tuple[alp.AttentionLayerPlan, alp.AttentionLayerPlan]]:
    """Pairs of plans that differ in `field` alone."""
    return [
        (a, b)
        for a, b in itertools.combinations(plans, 2)
        if [
            k
            for k in set(_fields(a)) | set(_fields(b))
            if _fields(a).get(k) != _fields(b).get(k)
        ]
        == [field]
    ]


def _check_field_reads(
    case: LayerCase, values: list[alp.AttentionLayerPlan], graphs: Graphs
) -> tuple[list[str], list[alp.AttentionLayerPlan]]:
    """Flips one field between two exportable values and compares graphs.

    Returns:
        The findings, and the exportable values.
    """
    findings: list[str] = []
    exported = {p for p, _ in graphs}
    plans = [v for v in values if v in exported]
    all_fields: set[_Field] = set().union(*(_fields(v) for v in values))
    for field in sorted(all_fields, key=lambda f: f.name):
        seen = {repr(_fields(v)[field]) for v in values if field in _fields(v)}
        if len(seen) == 1:
            print(
                f"  field {field.name}: fixed"
                f" (one value wherever it exists: {seen.pop()})"
            )
            continue
        pairs = _single_field_flips(plans, field)
        if not pairs:
            print(f"  field {field.name}: no exportable single-field flip to compare")
            continue
        changed = any(graphs[a, t] != graphs[b, t] for a, b in pairs for t in TOKENS)
        print(f"  field {field.name}: {'read' if changed else 'UNREAD'}")
        if not changed:
            findings.append(f"{case.name} unread {field.name}")
    return findings, plans


def _check_numerics(case: LayerCase, plans: list[alp.AttentionLayerPlan]) -> list[str]:
    """Compares each exportable value with upstream's forward.

    Returns:
        One finding per value and token count outside the contract's tolerance.
    """
    config = case.config()
    findings: list[str] = []
    for value in plans:
        layer = cpu_layer.build_real(config, value)
        atol, rtol = layer.contract.atol, layer.contract.rtol
        worst = 0.0
        for t in TOKENS:
            args = cpu_layer.inputs(config, t)
            with torch.no_grad():
                got: torch.Tensor = layer(*args).float()
                ref = qwen3_next.Qwen3NextAttention.forward(layer, *args).float()
            worst = max(worst, (got - ref).abs().max().item())
            if not torch.allclose(got, ref, atol=atol, rtol=rtol):
                findings.append(f"{case.name} {alp.plan_name(value)} outside tolerance")
        print(
            f"  numerics {alp.plan_name(value)} vs upstream forward:"
            f" max |diff| {worst:.1e} (atol {atol}, rtol {rtol})"
        )
    return findings


def check_case(case: LayerCase) -> list[str]:
    """Runs the declaration, field-read and numerics checks on one case.

    Args:
        case: The layer configuration.

    Returns:
        The findings, in the order they were found.
    """
    probe = cpu_layer.build_real(case.config())
    caps = layer_mod.describe_children(probe)
    values = alp.enumerate_plans(probe.desc)
    owners = {plan_marker.PLAN_OWNERS[type(v)] for v in values}
    print(f"== {case.name}: {len(values)} plan values, owner {sorted(owners)}")
    findings, graphs = _check_declarations(case, values, caps, probe)
    field_findings, plans = _check_field_reads(case, values, graphs)
    return findings + field_findings + _check_numerics(case, plans)


def main() -> int:
    """Prints every case's checks and findings.

    Returns:
        The exit code: 1 if a finding is not in its case's known set.
    """
    findings: list[str] = []
    new: set[str] = set()
    with layer_builder.runtime():
        for case in CASES:
            case_findings = check_case(case)
            findings += case_findings
            new |= set(case_findings) - case.known
    for finding in findings:
        print(f"finding: {finding}")
    print(f"findings: {len(findings)} ({len(new)} new)")
    return 1 if new else 0
