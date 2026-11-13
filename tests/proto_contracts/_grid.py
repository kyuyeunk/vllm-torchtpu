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
"""A collector for one family of an oracle grid.

A grid holds thousands of cases, too many for one test id each. Each test
covers one family of the grid (a chip, a routing setting, a cap pair), checks
every case in it, and fails once with every mismatching case listed, so one
bad case does not hide the others. Each family prints its case count, so the
counts can be summed and compared across versions of the test.
"""

import dataclasses


@dataclasses.dataclass
class GridFamily:
    """The case count and the mismatches of one grid family.

    Attributes:
        label: The family's name in the printed count.
        cases: Cases checked so far.
        mismatches: One line per failed check, with its context.
    """

    label: str
    cases: int = 0
    mismatches: list[str] = dataclasses.field(default_factory=list[str])

    def expect(self, ok: bool, context: object) -> None:  # noqa: FBT001 - the check's outcome, like assert
        """Records a mismatch with its context unless `ok`."""
        if not ok:
            self.mismatches.append(repr(context))

    def expect_equal(self, actual: object, expected: object, context: object) -> None:
        """Records a mismatch unless `actual == expected`."""
        if actual != expected:
            self.mismatches.append(f"{actual!r} != {expected!r}: {context!r}")

    def assert_no_mismatches(self) -> None:
        """Prints the case count, then fails listing every mismatch."""
        print(f"\n[{self.label}] {self.cases} cases")
        assert not self.mismatches, (
            f"{len(self.mismatches)} of {self.cases} cases mismatch:\n"
            + "\n".join(self.mismatches)
        )
