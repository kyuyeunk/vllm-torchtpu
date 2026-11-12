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

For every value of a plan type, on every case and chip, `check_structure`
compiles the module under that value ahead of time, for a TPU topology JAX
describes without a device, and compares the outcome with the module's own
capability predicate:

- a value it declares capable must compile ("fails" otherwise);
- a value it declines should not compile ("compiles" otherwise: only a run
  can show whether it fails loudly);
- every compiled value's output must have the picked value's shape, dtype
  and sharding ("mismatch" otherwise);
- a field whose single-field flips never change the optimized program on a
  chip is unread there ("unread").
"""

import dataclasses
import enum
import functools
import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import jax
import numpy as np
from jax.experimental import topologies
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
        """The finding as printed, one column per part."""
        if self.kind is Kind.UNREAD:
            return (self.kind.value, self.chip.name, str(self.field))
        return (
            self.kind.value,
            self.chip.name,
            str(self.case),
            plan_marker.plan_name(self.plan),
        )


def _program(entry, case: Case, plan: Any, mesh: Mesh) -> tuple[str, Any]:
    # Mosaic kernels cannot be partitioned automatically, so each device runs
    # the module on its own copy of the case under shard_map.
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
    out = (
        compiled.out_info.shape,
        compiled.out_info.dtype,
        str(compiled.output_shardings),
    )
    return _without_locations(compiled.as_text()), out


def _without_locations(text: str) -> str:
    # Two programs are the same when only the Python lines that built them
    # differ: drop each instruction's metadata and the location tables, each
    # a titled block that ends at a blank line.
    kept, skipping = [], False
    for line in re.sub(r", metadata=\{[^{}]*\}", "", text).splitlines():
        if line in ("FileNames", "FunctionNames", "FileLocations", "StackFrames"):
            skipping = True
        elif skipping and not line:
            skipping = False
        elif not skipping:
            kept.append(line)
    return "\n".join(kept)


def check_structure(
    entry: Callable[..., jax.Array],
    cases: Callable[[], Sequence[Case]],
    plan_type: type,
    chips: Sequence[Chip],
    *,
    resolve: Callable[[Case], Any],
    declared: Callable[[Case, Any], bool],
    domains: Mapping[type, Sequence[Any]] | None = None,
) -> set[Finding]:
    """Compiles every plan value on every case and chip; returns the findings.

    Args:
        entry: `entry(case, plan, *args)` runs the module under `plan`.
        cases: Builds the cases; called inside each chip's mesh, so the
            module and the predicates read that chip.
        plan_type: A type marked with `plan_marker.plan_of`.
        chips: The fake chips.
        resolve: The plan the module picks for a case.
        declared: Whether the module declares a plan able to run a case.
        domains: Values for the plan type's non-finite field types (`int`).

    Returns:
        The findings.
    """
    values = plan_marker.plan_values(plan_type, domains)
    findings: set[Finding] = set()
    for chip in chips:
        mesh = Mesh(
            np.array(topologies.get_topology_desc(chip.topology, "tpu").devices),
            (_AXIS,),
        )
        # Field to: some single-field flip changed the program.
        read: dict[dataclasses.Field[Any], bool] = {}
        with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            for case in cases():
                picked = resolve(case)
                programs: dict[Any, tuple[str, Any]] = {}
                for plan in values:
                    capable = declared(case, plan)
                    try:
                        programs[plan] = _program(entry, case, plan, mesh)
                    except Exception:  # Any failure is an outcome.
                        if capable:
                            findings.add(Finding(Kind.FAILS, chip, case.name, plan))
                        continue
                    if not capable:
                        findings.add(Finding(Kind.COMPILES, chip, case.name, plan))
                ref = programs.get(picked)
                for plan, (_, out) in programs.items():
                    if ref is not None and out != ref[1]:
                        findings.add(Finding(Kind.MISMATCH, chip, case.name, plan))
                for a, b in itertools.combinations(programs, 2):
                    changed = plan_marker.changed_fields(a, b)
                    if len(changed) == 1:
                        field = changed[0]
                        read[field] = (
                            read.get(field, False) or programs[a][0] != programs[b][0]
                        )
        findings |= {
            Finding(Kind.UNREAD, chip, field=field.name)
            for field, used in read.items()
            if not used
        }
    return findings
