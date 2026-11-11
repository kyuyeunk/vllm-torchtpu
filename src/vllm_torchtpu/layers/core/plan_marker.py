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
"""Marks plan types with the module that owns their decision.

A plan type is a frozen dataclass whose fields choose how a module computes,
never what it computes. The marker records the owner so one checker
(`tests/layers/core/plan_invariance.py`) can find every plan type and test
that flipping its fields leaves the owner's output within contract.

Imports nothing outside the standard library, so plan modules stay pure.
"""

import dataclasses
from collections.abc import Callable

PLAN_OWNERS: dict[type, str] = {}
"""Every marked plan type, mapped to its owner."""


def plan_of[T: type](owner: str) -> Callable[[T], T]:
    """Marks a frozen dataclass as the plan type of `owner`.

    Args:
        owner: Dotted name of the module whose decision this is, e.g.
            "moe.combine".

    Returns:
        A class decorator that registers the class and returns it unchanged.
    """

    def mark(cls: T) -> T:
        if not _frozen_dataclass(cls):
            raise TypeError(f"plan type {cls.__name__} must be a frozen dataclass")
        if cls in PLAN_OWNERS:
            raise TypeError(f"plan type {cls.__name__} is already marked")
        PLAN_OWNERS[cls] = owner
        return cls

    return mark


def _frozen_dataclass(cls: type) -> bool:
    """Whether ``cls`` is a frozen dataclass."""
    if not dataclasses.is_dataclass(cls):
        return False
    return cls.__dataclass_params__.frozen  # pyright: ignore[reportAttributeAccessIssue]
