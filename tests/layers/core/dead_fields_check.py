# Copyright 2025 Google LLC
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
"""A child does not ignore what it was handed: every plan field is read.

Traces `moe_combine` once per `MoeCombinePlan` value for a TPU chip that JAX
describes without a device, and reports each field whose flips, with every
other field held, never change the traced program. Such a field is a decision
the child received and ignored, or re-made from somewhere else. One expert
parallel (EP) shape on v6e; plans that fail to trace there are skipped. A
smaller form of the compile check of DATA_CONTRACTS section 5.8, which
compiles for several chips and shapes and compares the optimized programs.

Needs libtpu (`jax.experimental.topologies`), not a device. Shared by
`test_dead_fields.py` and the `prototype.check_dead_fields` demo.
"""

import contextlib
import dataclasses
import enum
import itertools
import typing

import jax
import ml_dtypes
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from vllm_torchtpu.layers.core import fused_moe_gmm, moe_plan

TOKENS, TOPK, HIDDEN, LOCAL_EXPERTS = 512, 8, 4096, 32


def domain(hint: object) -> list[object]:
    """The values of a field typed with enums and None."""
    if hint is type(None):
        return [None]
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return list(hint)
    args: tuple[object, ...] = typing.get_args(hint)
    return [v for arg in args for v in domain(arg)]


def field_names() -> list[str]:
    """The `MoeCombinePlan` field names, in declaration order."""
    return [f.name for f in dataclasses.fields(moe_plan.MoeCombinePlan)]


def plan_values() -> list[moe_plan.MoeCombinePlan]:
    """Every value the plan type accepts."""
    hints: dict[str, object] = typing.get_type_hints(moe_plan.MoeCombinePlan)
    names = field_names()
    domains = [domain(hints[n]) for n in names]
    # Built from each field's own domain, so the values fit; the plain callable
    # type lets the generic product pass them by name.
    make = typing.cast(
        "typing.Callable[..., moe_plan.MoeCombinePlan]", moe_plan.MoeCombinePlan
    )
    values: list[moe_plan.MoeCombinePlan] = []
    for combo in itertools.product(*domains):
        # The plan type requires a version on ep_ragged and rejects it elsewhere.
        with contextlib.suppress(ValueError):
            values.append(make(**dict(zip(names, combo, strict=True))))
    return values


def traced(plan: moe_plan.MoeCombinePlan, mesh: Mesh) -> str | None:
    """The traced program of `moe_combine` under `plan`, or None if it fails."""
    spec = PartitionSpec("devices")
    sharding = NamedSharding(mesh, spec)
    routes, n = TOKENS * TOPK, mesh.size
    args = [
        jax.ShapeDtypeStruct((n * s[0], *s[1:]), d, sharding=sharding)
        for s, d in (
            ((routes, HIDDEN), ml_dtypes.bfloat16),
            ((LOCAL_EXPERTS,), np.int32),
            ((routes,), np.int32),
            ((routes,), ml_dtypes.bfloat16),
            ((routes,), np.bool_),
            ((routes,), np.int32),
        )
    ]

    def combine(*a: jax.Array) -> jax.Array:
        return fused_moe_gmm.moe_combine(
            *a,
            plan=plan,
            num_tokens=TOKENS,
            topk=TOPK,
            out_dtype=np.dtype(ml_dtypes.bfloat16),
        )

    fn = jax.shard_map(
        combine, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
    )
    try:
        program: object = jax.jit(fn).trace(*args).jaxpr  # pyright: ignore[reportUnknownMemberType, reportAny]  # jax.jit overloads and Traced.jaxpr are untyped.
        return str(program)
    except Exception:  # noqa: BLE001 - a plan this chip cannot run is skipped.
        return None


def programs_on_v6e() -> dict[moe_plan.MoeCombinePlan, str]:
    """The traced program of every plan that traces on a v6e 2x2 topology."""
    desc = topologies.get_topology_desc("v6e:2x2", "tpu")  # pyright: ignore[reportUnknownMemberType]  # untyped **kwargs.
    mesh = Mesh(np.array(desc.devices), ("devices",))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped list.
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        programs = {p: traced(p, mesh) for p in plan_values()}
    return {p: t for p, t in programs.items() if t is not None}


def _differing(a: moe_plan.MoeCombinePlan, b: moe_plan.MoeCombinePlan) -> list[str]:
    return [n for n in field_names() if getattr(a, n) != getattr(b, n)]


def fields_read(programs: dict[moe_plan.MoeCombinePlan, str]) -> set[str]:
    """The fields whose single flip changes the traced program somewhere."""
    return {
        changed[0]
        for a, b in itertools.combinations(programs, 2)
        if len(changed := _differing(a, b)) == 1 and programs[a] != programs[b]
    }


def fields_never_read(programs: dict[moe_plan.MoeCombinePlan, str]) -> list[str]:
    """The fields no single flip ever changes the program for, in field order."""
    read = fields_read(programs)
    return [n for n in field_names() if n not in read]
