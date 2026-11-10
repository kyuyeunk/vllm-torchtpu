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
"""

import contextlib
import dataclasses
import enum
import itertools
import typing

import jax
import numpy as np
from jax import numpy as jnp
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
    return [v for arg in typing.get_args(hint) for v in domain(arg)]


def plan_values() -> list[moe_plan.MoeCombinePlan]:
    """Every value the plan type accepts."""
    hints = typing.get_type_hints(moe_plan.MoeCombinePlan)
    names = [f.name for f in dataclasses.fields(moe_plan.MoeCombinePlan)]
    domains = [domain(hints[n]) for n in names]
    values = []
    for combo in itertools.product(*domains):
        # The plan type requires a version on ep_ragged and rejects it elsewhere.
        with contextlib.suppress(ValueError):
            values.append(moe_plan.MoeCombinePlan(**dict(zip(names, combo))))
    return values


def traced(plan: moe_plan.MoeCombinePlan, mesh: Mesh) -> str | None:
    """The traced program of `moe_combine` under `plan`, or None if it fails."""
    spec = PartitionSpec("devices")
    sharding = NamedSharding(mesh, spec)
    routes, n = TOKENS * TOPK, mesh.size
    args = [
        jax.ShapeDtypeStruct((n * s[0], *s[1:]), d, sharding=sharding)
        for s, d in (
            ((routes, HIDDEN), jnp.bfloat16),
            ((LOCAL_EXPERTS,), jnp.int32),
            ((routes,), jnp.int32),
            ((routes,), jnp.bfloat16),
            ((routes,), jnp.bool_),
            ((routes,), jnp.int32),
        )
    ]

    def combine(*a: jax.Array) -> jax.Array:
        return fused_moe_gmm.moe_combine(
            *a, plan=plan, num_tokens=TOKENS, topk=TOPK, out_dtype=jnp.bfloat16
        )

    fn = jax.shard_map(
        combine, mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False
    )
    try:
        return str(jax.jit(fn).trace(*args).jaxpr)
    except Exception:  # noqa: BLE001 - a plan this chip cannot run is skipped.
        return None


def main() -> None:
    devices = topologies.get_topology_desc("v6e:2x2", "tpu").devices
    mesh = Mesh(np.array(devices), ("devices",))
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        programs = {p: traced(p, mesh) for p in plan_values()}
    programs = {p: t for p, t in programs.items() if t is not None}
    print(f"plans traced on v6e: {len(programs)}")
    names = [f.name for f in dataclasses.fields(moe_plan.MoeCombinePlan)]
    read = dict.fromkeys(names, False)
    for a, b in itertools.combinations(programs, 2):
        changed = [n for n in names if getattr(a, n) != getattr(b, n)]
        if len(changed) == 1 and programs[a] != programs[b]:
            read[changed[0]] = True
    print(f"fields never read: {[n for n, r in read.items() if not r]}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
