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
"""Plan invariance, compile tier: every capable plan compiles on every chip.

The value tier (`plan_invariance`) runs on the host's chip only. This tier
compiles ahead of time for TPU topologies JAX describes without a device
(`jax.experimental.topologies`), so it runs on a CPU host and reaches chips
the host lacks. It cannot compare values; it checks what a compiler can:

- every plan the capability predicate admits compiles for the chip;
- every plan's output has the resolved plan's shape, dtype and sharding;
- a field whose single-field flips never change the optimized program
  (modulo source locations) is unconsumed on that chip.

Kernels read the chip through `pltpu.get_tpu_info()`, which answers from the
abstract mesh's device when one is set, so tracing under the topology mesh's
abstract mesh targets that chip without any source change.

Mosaic kernels cannot be partitioned automatically, so the entry runs under
`shard_map`: each device holds one copy of the case's arguments, stacked
along one mesh axis, and computes its own output rows.
"""

import dataclasses
import functools
import itertools
import time
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tests.layers.core.plan_invariance import PlanCase, changed_fields, plan_values
from vllm_torchtpu.layers.core.plan_marker import PLAN_OWNERS

_AXIS = "devices"

# Sections of the compiled module text that index source locations.
_LOCATION_SECTIONS = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


@dataclasses.dataclass(frozen=True, kw_only=True)
class AotChip:
    """A TPU target to compile for.

    Attributes:
        name: Label for the report.
        topology: A `topologies.get_topology_desc` name, e.g. "v6e:2x2".
    """

    name: str
    topology: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class AotOutcome:
    """What one plan value compiled to on one case and chip.

    Attributes:
        chip: The chip name.
        case: The case name.
        plan: The plan value.
        status: One of "resolved", "compiled" (a distinct program),
            "same_program" (optimized HLO equal to an earlier plan's),
            "shape_mismatch", "error" (a capable plan failed to compile),
            "rejected_loud", "rejected_compiles" (prefixes the status the
            rejected plan got; a run-time failure cannot be seen here).
        detail: Error text, the plan whose program this repeats, or the
            output and the arguments the program never reads.
    """

    chip: str
    case: str
    plan: Any
    status: str
    detail: str = ""


@dataclasses.dataclass(kw_only=True)
class AotReport:
    """The compile tier's findings for one plan type.

    Attributes:
        owner: The plan type's owner.
        outcomes: One entry per (chip, case, constructible plan value).
        unconsumed: Chip name to the fields whose single-field flips
            compiled and never changed the optimized program on that chip.
        untraced: Chip name to the fields no compiled pair of plans differed
            in alone.
        out_shardings: Chip name to the resolved plans' output sharding.
        seconds: Wall time.
    """

    owner: str
    outcomes: list[AotOutcome]
    unconsumed: dict[str, list[str]]
    untraced: dict[str, list[str]]
    out_shardings: dict[str, str]
    seconds: float

    def failures(self) -> list[AotOutcome]:
        """Outcomes that refute the capability predicate or the output spec."""
        return [o for o in self.outcomes if o.status in ("shape_mismatch", "error")]

    def summary(self) -> str:
        """A readable table of the report."""
        lines = [f"plan owner {self.owner}: {self.seconds:.1f}s"]
        lines += [
            f"  {o.chip:6} {o.case:22} {o.status:18} {o.plan} {o.detail}"
            for o in self.outcomes
        ]
        lines.append(f"  unconsumed fields: {self.unconsumed}")
        lines.append(f"  untraced fields: {self.untraced}")
        lines.append(f"  output shardings: {self.out_shardings}")
        return "\n".join(lines)


def normalized_hlo(text: str) -> str:
    """Compiled module text without source locations.

    Drops each instruction's `metadata={...}` and the location tables, so two
    programs compare equal when only the Python lines that built them differ.

    Args:
        text: `Compiled.as_text()`.

    Returns:
        The text to compare.
    """
    kept, skipping = [], False
    for line in text.splitlines():
        if line in _LOCATION_SECTIONS:
            skipping = True
        elif skipping and not line:
            skipping = False
        elif not skipping:
            kept.append(_strip_metadata(line))
    return "\n".join(kept)


def _strip_metadata(line: str) -> str:
    start = line.find(", metadata={")
    if start < 0:
        return line
    # Metadata values are quoted strings; skip braces inside them.
    depth, quoted = 0, False
    for i in range(start + len(", metadata="), len(line)):
        c = line[i]
        if c == '"' and line[i - 1] != "\\":
            quoted = not quoted
        elif not quoted and c == "{":
            depth += 1
        elif not quoted and c == "}":
            depth -= 1
            if depth == 0:
                return line[:start] + line[i + 1 :]
    return line


@dataclasses.dataclass(frozen=True)
class _Program:
    hlo: str
    shape: tuple[int, ...]
    dtype: Any
    sharding: str
    unread: tuple[int, ...]


def _compile(
    entry: Callable[..., jax.Array], case: PlanCase, plan: Any, mesh: Mesh
) -> tuple[str, Callable[[], _Program]]:
    """The traced program's text, and a thunk that compiles it."""
    spec = PartitionSpec(_AXIS)
    sharding = NamedSharding(mesh, spec)
    fn = jax.shard_map(
        functools.partial(entry, case, plan),
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        # Pallas calls carry no varying-axis annotations.
        check_vma=False,
    )
    args = [
        jax.ShapeDtypeStruct(
            (mesh.size * a.shape[0], *a.shape[1:]), a.dtype, sharding=sharding
        )
        for a in case.args
    ]
    traced = jax.jit(fn).trace(*args)

    def build() -> _Program:
        lowered = traced.lower()
        compiled = lowered.compile()
        out = lowered.out_info
        # jit drops arguments the program never reads; they get no sharding.
        unread = tuple(
            i for i, s in enumerate(compiled.input_shardings[0]) if s is None
        )
        return _Program(
            hlo=normalized_hlo(compiled.as_text()),
            shape=tuple(out.shape),
            dtype=out.dtype,
            sharding=str(compiled.output_shardings.spec),
            unread=unread,
        )

    return str(traced.jaxpr), build


def _check_case(
    entry: Callable[..., jax.Array],
    case: PlanCase,
    values: Sequence[Any],
    chip: AotChip,
    mesh: Mesh,
    *,
    resolve: Callable[[PlanCase], Any],
    capable: Callable[[PlanCase, Any], bool],
) -> tuple[list[AotOutcome], dict[Any, str], str | None]:
    """Compiles every plan on one case.

    Returns:
        The outcomes, each compiled plan's optimized HLO, and the resolved
        plan's output sharding (None if it failed to compile).
    """
    resolved = resolve(case)
    assert capable(case, resolved), (
        f"{chip.name} {case.name}: resolved {resolved} is not capable"
    )
    outcomes: list[AotOutcome] = []
    # Traced text to its compiled program: plans that trace alike compile
    # alike, so they compile once.
    built: dict[str, _Program] = {}
    first: dict[str, Any] = {}  # Optimized HLO to the first plan with it.
    hlos: dict[Any, str] = {}
    ref: _Program | None = None
    for plan in [resolved, *(v for v in values if v != resolved)]:
        accepted = capable(case, plan)
        outcome = functools.partial(
            AotOutcome, chip=chip.name, case=case.name, plan=plan
        )
        try:
            key, build = _compile(entry, case, plan, mesh)
            if key not in built:
                built[key] = build()
        except Exception as e:  # noqa: BLE001 - any failure is loud; report it.
            status = "error" if accepted else "rejected_loud"
            outcomes.append(
                outcome(status=status, detail=f"{type(e).__name__}: {str(e)[:120]}")
            )
            if plan == resolved:
                break  # Nothing to compare against.
            continue
        prog = built[key]
        hlos[plan] = prog.hlo
        out = (prog.shape, prog.dtype, prog.sharding)
        detail = f"{prog.shape} {prog.dtype} unread={list(prog.unread)}"
        if ref is None:
            ref, status = prog, "resolved"
        elif out != (ref.shape, ref.dtype, ref.sharding):
            status = "shape_mismatch"
            detail = f"{out} vs {(ref.shape, ref.dtype, ref.sharding)}"
        elif prog.hlo in first:
            status, detail = "same_program", f"as {first[prog.hlo]}"
        else:
            status = "compiled"
        first.setdefault(prog.hlo, plan)
        if not accepted:
            status, detail = "rejected_compiles", f"{status} {detail}"
        outcomes.append(outcome(status=status, detail=detail))
    return outcomes, hlos, None if ref is None else ref.sharding


def check_plan_compiles(
    entry: Callable[..., jax.Array],
    cases: Callable[[], Sequence[PlanCase]],
    plan_type: type,
    chips: Sequence[AotChip],
    *,
    resolve: Callable[[PlanCase], Any],
    capable: Callable[[PlanCase, Any], bool],
    domains: dict[str, Sequence[Any]] | None = None,
) -> AotReport:
    """Compiles every plan value for every chip and reports what differs.

    Args:
        entry: `entry(case, plan, *case.args)` runs the module under `plan`
            on one device's arguments.
        cases: Builds the contract-satisfying inputs (`jax.ShapeDtypeStruct`
            args, leading dimension at least one); called once per chip
            inside that chip's mesh, so it may read the chip.
        plan_type: The marked plan type.
        chips: The targets.
        resolve: The plan the module would choose for a case.
        capable: The module's capability predicate; every plan it admits
            must compile.
        domains: Values for bounded-integer fields.

    Returns:
        The report.
    """
    start = time.perf_counter()
    values, _ = plan_values(plan_type, domains or {})
    fields = [f.name for f in dataclasses.fields(plan_type)]
    outcomes: list[AotOutcome] = []
    unconsumed: dict[str, list[str]] = {}
    untraced: dict[str, list[str]] = {}
    shardings: dict[str, str] = {}
    for chip in chips:
        topology = topologies.get_topology_desc(chip.topology, "tpu")
        mesh = Mesh(np.array(topology.devices), (_AXIS,))
        # Per field: None until a compiled pair differs in it alone, then
        # whether any such pair compiled to different programs.
        consumed: dict[str, bool | None] = dict.fromkeys(fields)
        with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            for case in cases():
                found, hlos, sharding = _check_case(
                    entry, case, values, chip, mesh, resolve=resolve, capable=capable
                )
                outcomes += found
                if sharding is not None:
                    shardings[chip.name] = sharding
                for a, b in itertools.combinations(hlos, 2):
                    changed = list(changed_fields(a, b))
                    if len(changed) == 1:
                        name = changed[0]
                        consumed[name] = bool(consumed[name]) or hlos[a] != hlos[b]
        unconsumed[chip.name] = [n for n, used in consumed.items() if used is False]
        untraced[chip.name] = [n for n, used in consumed.items() if used is None]
    return AotReport(
        owner=PLAN_OWNERS[plan_type],
        outcomes=outcomes,
        unconsumed=unconsumed,
        untraced=untraced,
        out_shardings=shardings,
        seconds=time.perf_counter() - start,
    )
