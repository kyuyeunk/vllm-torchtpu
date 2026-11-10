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
"""Plan invariance: flipping a plan's fields must not change the output.

A plan type (marked with `plan_of`) chooses how a module computes. For every
input that satisfies the module's contract and every plan value the module
can run, the output must match the output under the plan the module would
resolve itself, within the contract's tolerance. A field whose flip breaks
that is contract misfiled as plan.

`check_plan_invariance` enumerates plan values from the field types, traces
every value on every case (shapes and dtypes, no compilation), runs the
distinct programs on the cases marked for values, and returns a report;
callers assert on the report.
"""

import dataclasses
import enum
import functools
import itertools
import time
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal

import jax
import numpy as np

from vllm_torchtpu.layers.core.plan_marker import PLAN_OWNERS


@dataclasses.dataclass(frozen=True, kw_only=True)
class PlanCase:
    """One contract-satisfying input.

    Attributes:
        name: Label for the report.
        args: The entry's array arguments.
        check_values: Whether to run and compare values, not only shapes.
        meta: Whatever the entry, resolver and predicate need besides `args`.
    """

    name: str
    args: tuple[jax.Array, ...]
    check_values: bool
    meta: Any


@dataclasses.dataclass(frozen=True, kw_only=True)
class Outcome:
    """What one plan value did on one case.

    Attributes:
        case: The case name.
        plan: The plan value.
        status: One of "resolved", "same_program" (traces to a program
            already seen), "match", "mismatch", "shape_mismatch", "error"
            (a capable plan failed), "rejected_loud", "rejected_silent"
            (prefixes the status the rejected plan got), "traced" (shape
            matched; values not checked on this case).
        detail: Error text, the plan whose program this repeats, or the
            maximum absolute difference.
    """

    case: str
    plan: Any
    status: str
    detail: str = ""


@dataclasses.dataclass(kw_only=True)
class InvarianceReport:
    """The checker's findings for one plan type.

    Attributes:
        owner: The plan type's owner.
        outcomes: One entry per (case, constructible plan value).
        unconstructible: Field combinations the plan type itself refuses.
        misclassified: Field name to the cases where flipping it (alone or
            with others) moved the output beyond tolerance.
        unconsumed: Fields whose single-field flips were traced and never
            changed the program.
        untraced: Fields no traced pair of plans differed in alone, so
            neither consumed nor unconsumed could be shown.
        seconds: Wall time.
    """

    owner: str
    outcomes: list[Outcome]
    unconstructible: int
    misclassified: dict[str, list[str]]
    unconsumed: list[str]
    untraced: list[str]
    seconds: float

    def failures(self) -> list[Outcome]:
        """Outcomes that break invariance or hide a rejection."""
        bad = ("mismatch", "shape_mismatch", "error", "rejected_silent")
        return [o for o in self.outcomes if o.status in bad]

    def summary(self) -> str:
        """A readable table of the report."""
        lines = [f"plan owner {self.owner}: {self.seconds:.1f}s"]
        lines += [
            f"  {o.case:28} {o.status:16} {o.plan} {o.detail}" for o in self.outcomes
        ]
        lines.append(f"  unconstructible combinations: {self.unconstructible}")
        lines.append(f"  misclassified fields: {self.misclassified}")
        lines.append(f"  unconsumed fields: {self.unconsumed}")
        lines.append(f"  untraced fields: {self.untraced}")
        return "\n".join(lines)


def _field_domain(name: str, hint: Any, domains: Mapping[str, Sequence[Any]]) -> tuple:
    if name in domains:
        return tuple(domains[name])
    origin = typing.get_origin(hint)
    if origin is Literal:
        return typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        return tuple(
            itertools.chain.from_iterable(
                _field_domain(name, arg, domains) for arg in typing.get_args(hint)
            )
        )
    if hint is type(None):
        return (None,)
    if hint is bool:
        return (False, True)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return tuple(hint)
    raise TypeError(
        f"plan field {name!r} has unbounded type {hint!r}; give it a domain"
    )


def plan_values(
    plan_type: type, domains: Mapping[str, Sequence[Any]]
) -> tuple[list[Any], int]:
    """Every value of a plan type the type itself accepts.

    Args:
        plan_type: A marked frozen dataclass.
        domains: Values for fields whose type is not finite (bounded ints).

    Returns:
        The constructible values, and how many combinations the type's own
        validation refused.
    """
    hints = typing.get_type_hints(plan_type)
    fields = [f.name for f in dataclasses.fields(plan_type)]
    axes = [_field_domain(name, hints[name], domains) for name in fields]
    values, refused = [], 0
    for combo in itertools.product(*axes):
        try:
            values.append(plan_type(**dict(zip(fields, combo, strict=True))))
        except (ValueError, TypeError):
            refused += 1
    return values, refused


def changed_fields(a: Any, b: Any) -> Iterator[str]:
    """Names of the fields two values of one plan type differ in."""
    for field in dataclasses.fields(a):
        if getattr(a, field.name) != getattr(b, field.name):
            yield field.name


def check_plan_invariance(
    entry: Callable[..., jax.Array],
    cases: Sequence[PlanCase],
    plan_type: type,
    tolerance: Callable[[PlanCase], tuple[float, float]],
    *,
    resolve: Callable[[PlanCase], Any],
    capable: Callable[[PlanCase, Any], bool],
    domains: Mapping[str, Sequence[Any]] | None = None,
) -> InvarianceReport:
    """Checks that a module's output does not depend on its plan.

    Args:
        entry: `entry(case, plan, *case.args)` runs the module under `plan`.
        cases: Contract-satisfying inputs.
        plan_type: The marked plan type.
        tolerance: `(rtol, atol)` of the output contract for a case.
        resolve: The plan the module would choose for a case.
        capable: The module's capability predicate; a plan it rejects must
            fail loudly when forced.
        domains: Values for bounded-integer fields.

    Returns:
        The report.
    """
    start = time.perf_counter()
    owner = PLAN_OWNERS[plan_type]
    values, refused = plan_values(plan_type, domains or {})
    outcomes: list[Outcome] = []
    misclassified: dict[str, set[str]] = {}
    # Per field: None until a traced pair differs in it alone, then whether
    # any such pair traced to different programs.
    consumed: dict[str, bool | None] = {
        f.name: None for f in dataclasses.fields(plan_type)
    }
    for case in cases:
        resolved = resolve(case)
        assert capable(case, resolved), (
            f"{case.name}: resolved {resolved} is not capable"
        )
        rtol, atol = tolerance(case)
        # Traced program text to the first plan that produced it, and that
        # program's output: plans that trace alike need not run again.
        programs: dict[str, Any] = {}
        outputs: dict[str, np.ndarray] = {}
        texts: dict[Any, str] = {}
        ref_avals = None
        ref_out = None
        for plan in [resolved, *(v for v in values if v != resolved)]:
            accepted = capable(case, plan)
            fn = functools.partial(entry, case, plan)
            try:
                jaxpr = jax.make_jaxpr(fn)(*case.args)
                text = str(jaxpr)
                # A rejected plan runs even on shape-only cases: a failure at
                # compile or run time is loud too.
                if (case.check_values or not accepted) and text not in outputs:
                    outputs[text] = np.asarray(jax.jit(fn)(*case.args), np.float32)
            except Exception as e:  # noqa: BLE001 - any failure is loud; report it.
                status = "rejected_loud" if not accepted else "error"
                detail = f"{type(e).__name__}: {str(e)[:120]}"
                outcomes.append(
                    Outcome(case=case.name, plan=plan, status=status, detail=detail)
                )
                if plan == resolved:
                    break  # Nothing to compare against.
                continue
            texts[plan] = text
            if plan == resolved:
                ref_avals, ref_out = jaxpr.out_avals, outputs.get(text)
                programs[text] = plan
                outcomes.append(Outcome(case=case.name, plan=plan, status="resolved"))
                continue
            out = outputs.get(text)
            if jaxpr.out_avals != ref_avals:
                status, detail = "shape_mismatch", f"{jaxpr.out_avals} vs {ref_avals}"
            elif programs.get(text) == resolved:
                status, detail = "same_program", f"as {resolved}"
            elif out is None or ref_out is None:
                status, detail = "traced", ""
            else:
                diff = np.abs(out - ref_out)
                within = np.isclose(out, ref_out, rtol=rtol, atol=atol, equal_nan=True)
                status = "match" if bool(np.all(within)) else "mismatch"
                detail = f"max|diff|={float(np.nanmax(diff)):.3g}"
                if not np.all(np.isfinite(out)):
                    detail += " non-finite"
            programs.setdefault(text, plan)
            if not accepted:
                status, detail = "rejected_silent", f"{status} {detail}"
            elif status in ("mismatch", "shape_mismatch"):
                for name in changed_fields(plan, resolved):
                    misclassified.setdefault(name, set()).add(case.name)
            outcomes.append(
                Outcome(case=case.name, plan=plan, status=status, detail=detail)
            )
        for a, b in itertools.combinations(texts, 2):
            changed = list(changed_fields(a, b))
            if len(changed) == 1:
                consumed[changed[0]] = (
                    bool(consumed[changed[0]]) or texts[a] != texts[b]
                )
    return InvarianceReport(
        owner=owner,
        outcomes=outcomes,
        unconstructible=refused,
        misclassified={name: sorted(c) for name, c in misclassified.items()},
        unconsumed=[name for name, used in consumed.items() if used is False],
        untraced=[name for name, used in consumed.items() if used is None],
        seconds=time.perf_counter() - start,
    )
