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
"""The structural tier: a module's declared capability against the compiler.

`check_structure` compiles a module under every value of its plan type, on
every case and chip, ahead of time, for a TPU topology JAX describes without a
device. It needs no TPU, only libtpu to compile.

Inputs:

- `entry(case, plan, *args)` runs the module under one plan value.
- `cases(chip)` builds the inputs: one `Case` per shape, its arrays abstract.
- `resolve(case)` is the plan value the module itself would pick.
- `declared(case, plan, chip)` is the module's own capability predicate.
- `chips` are the fake chips; `plan_type` and `domains` give the values.

How the chip reaches the code: `cases` and `declared` receive the chip's
`pltpu.TpuInfo` as an argument. `entry` and `resolve` are the module under
test, which reads the chip itself through `pltpu.get_tpu_info()`; that call
answers from the abstract mesh in context, so every callback runs inside
`_on_chip`, which sets the fake chip's mesh as that context.

Worked example, from `prototype/check_plans.py`: `entry` runs `moe_combine`
under one `MoeCombinePlan`; `cases` gives abstract arrays for seven combine
shapes; `resolve` is `resolve_combine_plan`; `declared` is `combine_capable`.
On TPU7x, for the FP32 shape, `combine_capable` declines the EP v3 plan (v3
is BF16-only), yet that plan compiles: a COMPILES finding.

Findings, per chip and case:

- FAILS: a value declared capable does not compile.
- COMPILES: a value declined does compile; only a run can show whether it
  fails loudly or computes something wrong.
- MISMATCH: a compiled value's output differs in shape, dtype or sharding
  from the output of the value `resolve` picks.
- UNREAD (per chip, over all cases): no single-field flip between two
  compiled values changes the optimized program, so the chip never reads that
  field.
"""

import contextlib
import dataclasses
import enum
import functools
import itertools
import re
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np
from jax.experimental import topologies
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from vllm_torchtpu.layers.core import plan_marker

_AXIS = "devices"


class Case(NamedTuple):
    """One contract-satisfying input of the module."""

    name: str
    args: tuple[jax.ShapeDtypeStruct, ...]  # The arrays on one device.
    meta: Any  # Whatever the entry needs besides the arrays.


class Chip(NamedTuple):
    """A fake chip JAX describes without a device."""

    name: str
    topology: str  # A `topologies.get_topology_desc` name.


class Kind(enum.Enum):
    """What a finding reports; each value is how it prints."""

    FAILS = "fails"  # Declared capable, but does not compile.
    COMPILES = "compiles"  # Declined, yet compiles.
    MISMATCH = "mismatch"  # Output shape, dtype or sharding differs.
    UNREAD = "unread"  # A field whose flips never change the program.


class Finding(NamedTuple):
    """One finding of the structural tier.

    `case` and `plan` are None for `Kind.UNREAD`, which names `field`
    instead; `field` is None otherwise.
    """

    kind: Kind
    chip: Chip
    case: str | None = None
    plan: Any = None
    field: str | None = None

    def columns(self) -> tuple[str, ...]:
        """Returns the finding as printed, one column per part."""
        if self.kind is Kind.UNREAD:
            return (self.kind.value, self.chip.name, str(self.field))
        return (
            self.kind.value,
            self.chip.name,
            str(self.case),
            plan_marker.plan_name(self.plan),
        )


class _Compiled(NamedTuple):
    """What one compile of the module produced."""

    program: str  # The optimized program, without source locations.
    output: tuple[Any, Any, str]  # Output shape, dtype and sharding.


class _Trial(NamedTuple):
    """One plan value on one case: what the module declares, what compiled."""

    capable: bool
    compiled: _Compiled | None  # None: the compile raised.


type Entry[P] = Callable[[Case, P, *tuple[jax.Array, ...]], jax.Array]


def check_structure[P](  # noqa: PLR0913 - one argument per input the docstring names
    entry: Entry[P],
    cases: Callable[[pltpu.TpuInfo], Sequence[Case]],
    plan_type: type[P],
    chips: Sequence[Chip],
    *,
    resolve: Callable[[Case], P],
    declared: Callable[[Case, P, pltpu.TpuInfo], bool],
    domains: Mapping[type, Sequence[Any]] | None = None,
) -> set[Finding]:
    """Compiles every plan value on every case and chip; returns the findings.

    Args:
        entry: `entry(case, plan, *args)` runs the module under `plan`.
        cases: Builds the cases for a chip.
        plan_type: A type marked with `plan_marker.plan_of`.
        chips: The fake chips.
        resolve: The plan the module picks for a case.
        declared: Whether the module declares a plan able to run a case on
            a chip.
        domains: Values for the plan type's non-finite field types (`int`).

    Returns:
        The findings.
    """
    values: list[P] = plan_marker.plan_values(plan_type, domains)
    findings: set[Finding] = set()
    for chip in chips:
        mesh = _mesh(chip)
        reads: list[dict[dataclasses.Field[Any], bool]] = []
        with _on_chip(mesh) as info:
            for case in cases(info):
                picked = resolve(case)
                trials = {
                    plan: _Trial(
                        capable=declared(case, plan, info),
                        compiled=_compile(entry, case, plan, mesh),
                    )
                    for plan in values
                }
                findings |= _check_capability(chip, case, trials)
                findings |= _check_output(chip, case, trials, picked)
                reads.append(_fields_read(trials))
        findings |= _check_unread(chip, reads)
    return findings


def _mesh(chip: Chip) -> Mesh:
    """Returns a one-axis mesh over every device of the chip's topology."""
    devices = topologies.get_topology_desc(chip.topology, "tpu").devices
    return Mesh(np.array(devices), (_AXIS,))


@contextlib.contextmanager
def _on_chip(mesh: Mesh) -> Generator[pltpu.TpuInfo]:
    """Makes the mesh's chip the one the module under test reads.

    The module reads the chip through `pltpu.get_tpu_info()`, never as an
    argument; with no TPU attached, that call answers from the abstract mesh
    in context. So the module, and every callback, must run inside this.

    Args:
        mesh: A mesh over a fake chip's devices.

    Yields:
        The chip's `pltpu.TpuInfo`, for the callbacks that take it explicitly.
    """
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        yield pltpu.get_tpu_info()


def _compile[P](entry: Entry[P], case: Case, plan: P, mesh: Mesh) -> _Compiled | None:
    """Returns `_program`'s result, or None if compiling raised anything."""
    try:
        return _program(entry, case, plan, mesh)
    except Exception:  # noqa: BLE001 - any failure to compile is an outcome
        return None


def _program[P](entry: Entry[P], case: Case, plan: P, mesh: Mesh) -> _Compiled:
    """Compiles `entry` on the case under `plan`, once per device of the mesh.

    Mosaic kernels cannot be partitioned automatically, so each device runs
    the module on its own copy of the case under `shard_map`: the case's
    arrays are stacked once per device along the leading axis.

    Raises:
        ValueError: The compiled module has no text.
    """
    spec = PartitionSpec(_AXIS)
    fn = jax.shard_map(
        functools.partial(entry, case, plan),
        mesh=mesh,
        in_specs=spec,
        out_specs=spec,
        check_vma=False,
    )
    args = [
        jax.ShapeDtypeStruct(
            (mesh.size * a.shape[0], *a.shape[1:]),
            a.dtype,
            sharding=NamedSharding(mesh, spec),
        )
        for a in case.args
    ]
    compiled = jax.jit(fn).trace(*args).lower().compile()
    output = (
        compiled.out_info.shape,
        compiled.out_info.dtype,
        str(compiled.output_shardings),
    )
    text = compiled.as_text()
    if text is None:
        msg = "the compiled module has no text"
        raise ValueError(msg)
    return _Compiled(_without_locations(text), output)


def _without_locations(text: str) -> str:
    """Returns the program text without the Python source locations in it.

    Two plans that compile to the same program would otherwise differ in
    text, because their plan-dependent Python branches sit on different
    source lines; the UNREAD check compares programs, not lines. Drops each
    instruction's metadata and the location tables, each a titled block that
    ends at a blank line.
    """
    kept: list[str] = []
    skipping = False
    for line in re.sub(r", metadata=\{[^{}]*\}", "", text).splitlines():
        if line in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            skipping = True
        elif skipping and not line:
            skipping = False
        elif not skipping:
            kept.append(line)
    return "\n".join(kept)


def _check_capability[P](
    chip: Chip, case: Case, trials: Mapping[P, _Trial]
) -> set[Finding]:
    """Returns FAILS and COMPILES: the declared capability against the compile."""
    findings: set[Finding] = set()
    for plan, trial in trials.items():
        if trial.capable and trial.compiled is None:
            findings.add(Finding(Kind.FAILS, chip, case.name, plan))
        if not trial.capable and trial.compiled is not None:
            findings.add(Finding(Kind.COMPILES, chip, case.name, plan))
    return findings


def _check_output[P](
    chip: Chip, case: Case, trials: Mapping[P, _Trial], picked: P
) -> set[Finding]:
    """Returns MISMATCH: compiled outputs that differ from the picked plan's.

    Nothing to compare against when the picked plan did not compile.
    """
    ref = trials[picked].compiled if picked in trials else None
    if ref is None:
        return set()
    return {
        Finding(Kind.MISMATCH, chip, case.name, plan)
        for plan, trial in trials.items()
        if trial.compiled is not None and trial.compiled.output != ref.output
    }


def _fields_read[P](trials: Mapping[P, _Trial]) -> dict[dataclasses.Field[Any], bool]:
    """Returns, per field, whether a single-field flip changed the program.

    Covers only fields that some pair of compiled values differs in alone.
    """
    compiled = {
        plan: trial.compiled
        for plan, trial in trials.items()
        if trial.compiled is not None
    }
    read: dict[dataclasses.Field[Any], bool] = {}
    for a, b in itertools.combinations(compiled, 2):
        changed = plan_marker.changed_fields(a, b)
        if len(changed) == 1:
            field = changed[0]
            flipped = compiled[a].program != compiled[b].program
            read[field] = read.get(field, False) or flipped
    return read


def _check_unread(
    chip: Chip, reads: Iterable[Mapping[dataclasses.Field[Any], bool]]
) -> set[Finding]:
    """Returns UNREAD: fields no flip changed the program of, on any case."""
    read: dict[dataclasses.Field[Any], bool] = {}
    for case_read in reads:
        for field, flipped in case_read.items():
            read[field] = read.get(field, False) or flipped
    return {
        Finding(Kind.UNREAD, chip, field=field.name)
        for field, used in read.items()
        if not used
    }
