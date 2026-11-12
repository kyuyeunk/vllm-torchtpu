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
"""Marks plan types and enumerates their values.

A plan type is a frozen dataclass whose fields choose how a module computes,
never what it computes. Its fields are enums, booleans or optional versions
of them, or integers given a finite domain, so every value can be listed, and
one checker can run the owner under each value and compare.

Imports nothing outside the standard library, so plan modules stay pure.
"""

import dataclasses
import enum
import itertools
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any

PLAN_OWNERS: dict[type, str] = {}
"""Every marked plan type, mapped to the module that owns its decision."""


def plan_of[T: type](owner: str) -> Callable[[T], T]:
    """Marks a frozen dataclass as the plan type of `owner`.

    Args:
        owner: The module whose decision this is, e.g. "moe.combine".

    Returns:
        A class decorator that records the class and returns it unchanged.
    """

    def mark(cls: T) -> T:
        if not _frozen_dataclass(cls):
            raise TypeError(f"plan type {cls.__name__} must be a frozen dataclass")
        PLAN_OWNERS[cls] = owner
        return cls

    return mark


def _frozen_dataclass(cls: type) -> bool:
    if not dataclasses.is_dataclass(cls):
        return False
    params = cls.__dataclass_params__  # pyright: ignore[reportAttributeAccessIssue]
    return params.frozen


def plan_values(
    plan_type: type, domains: Mapping[str, Sequence[Any]] | None = None
) -> list[Any]:
    """Every value of a plan type that the type itself accepts.

    Args:
        plan_type: A type marked with `plan_of`.
        domains: Values for fields whose type is not finite (integers).

    Returns:
        The values, in field-declaration order; combinations the type's own
        checks reject are left out.
    """
    hints = typing.get_type_hints(plan_type)
    fields = [f.name for f in dataclasses.fields(plan_type)]
    axes = [_domain(name, hints[name], domains or {}) for name in fields]
    values = []
    for combo in itertools.product(*axes):
        try:
            values.append(plan_type(**dict(zip(fields, combo, strict=True))))
        except ValueError:
            continue
    return values


def changed_fields(a: Any, b: Any) -> list[str]:
    """The fields two values of one plan type differ in."""
    return [
        f.name
        for f in dataclasses.fields(a)
        if getattr(a, f.name) != getattr(b, f.name)
    ]


def _domain(name: str, hint: Any, domains: Mapping[str, Sequence[Any]]) -> tuple:
    if name in domains:
        return tuple(domains[name])
    if hint is bool:
        return (False, True)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return tuple(hint)
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        return tuple(
            v for arg in typing.get_args(hint) for v in _domain(name, arg, domains)
        )
    if hint is type(None):
        return (None,)
    raise TypeError(f"plan field {name!r} has no finite domain; give it one")
