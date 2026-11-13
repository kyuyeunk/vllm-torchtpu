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
"""Numbered checks that print their table and return their failures.

A check takes an ``Emit`` (``print`` in a demo script, a no-op in a test) and
returns the failures it found; an empty list is a pass. ``run_checks`` prints
each check's title, its failures and its verdict, as the demo scripts in
``prototype/`` show them. The pytest wrappers call each check directly and
assert it returns no failures.
"""

import dataclasses
from collections.abc import Callable, Sequence

type Emit = Callable[[str], None]
"""Where a check writes its table lines."""


def discard(line: str) -> None:
    """An ``Emit`` that drops every line (the pytest wrappers use it)."""
    del line


@dataclasses.dataclass(frozen=True)
class Check:
    """One numbered check: its title line and the function that runs it."""

    title: str
    run: Callable[[Emit], list[str]]

    @property
    def name(self) -> str:
        """The check function's name, the pytest id of its wrapper."""
        return self.run.__name__


def run_checks(checks: Sequence[Check], emit: Emit) -> bool:
    """Runs ``checks`` in order, emitting each table; True if all passed."""
    ok = True
    for check in checks:
        emit(check.title)
        failures = check.run(emit)
        for failure in failures:
            emit(f"    FAIL {failure}")
        emit("  -> " + ("FAIL" if failures else "PASS"))
        ok &= not failures
    return ok
