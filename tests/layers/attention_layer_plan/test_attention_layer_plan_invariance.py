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
"""Plan invariance for the attention layer: findings equal the known set.

The check itself is `plan_check.check_case`; this asserts, per case, that its
findings are exactly the recorded known set (empty today).
"""

import pytest

from tests.layers.attention_layer_plan import plan_check


@pytest.mark.parametrize(
    "case", plan_check.CASES, ids=[c.name for c in plan_check.CASES]
)
def test_findings_equal_known(case: plan_check.LayerCase) -> None:
    """Every forced value agrees with its declaration, fields and numerics."""
    assert frozenset(plan_check.check_case(case)) == case.known
