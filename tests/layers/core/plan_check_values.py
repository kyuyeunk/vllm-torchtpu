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
"""The numerics tier: a plan only chooses how a module computes, by value.

`check_values` runs the module on the host's chip under every value of its
plan type that the module declares capable, and compares each output with the
output under the value it picks, within a tolerance the caller supplies.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import jax
import numpy as np

from vllm_torchtpu.layers.core import plan_marker


class Case[M](NamedTuple):
    """One contract-satisfying input of the module."""

    name: str
    args: tuple[jax.Array, ...]
    meta: M  # Whatever the entry needs besides the arrays.


class Finding[P](NamedTuple):
    """A capable plan value whose output differs from the picked value's."""

    case: str
    plan: P
    error: float  # The largest absolute difference.

    def columns(self) -> tuple[str, ...]:
        """The finding as printed, one column per part."""
        return (
            "value",
            self.case,
            plan_marker.plan_name(self.plan),
            f"{self.error:.1e}",
        )


def check_values[P, M](
    entry: Callable[[Case[M], P, *tuple[jax.Array, ...]], jax.Array],
    cases: Sequence[Case[M]],
    plan_type: type[P],
    tolerance: Callable[[Case[M]], tuple[float, float]],
    *,
    resolve: Callable[[Case[M]], P],
    declared: Callable[[Case[M], P], bool],
    domains: Mapping[type, Sequence[object]] | None = None,
) -> set[Finding[P]]:
    """Runs every capable plan value and returns the ones whose values differ.

    Args:
        entry: `entry(case, plan, *case.args)` runs the module under `plan`.
        cases: The inputs, as concrete arrays.
        plan_type: A type marked with `plan_marker.plan_of`.
        tolerance: The caller's (rtol, atol) for a case.
        resolve: The plan the module picks for a case.
        declared: Whether the module declares a plan able to run a case.
        domains: Values for the plan type's non-finite field types (`int`).

    Returns:
        A finding for every capable value whose output differs from the
        picked value's beyond the tolerance.
    """
    findings: set[Finding[P]] = set()
    for case in cases:
        picked = resolve(case)
        ref = np.asarray(entry(case, picked, *case.args), np.float32)
        rtol, atol = tolerance(case)
        values: list[P] = plan_marker.plan_values(plan_type, domains)
        for plan in values:
            if plan == picked or not declared(case, plan):
                continue
            got = np.asarray(entry(case, plan, *case.args), np.float32)
            if not np.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=True):
                # numpy types nanmax's result as Any.
                error = float(np.nanmax(np.abs(got - ref)))  # pyright: ignore[reportAny]
                findings.add(Finding(case.name, plan, error))
    return findings
