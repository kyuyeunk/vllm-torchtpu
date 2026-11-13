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

Compile tier: traces for a v6e topology through libtpu, without a device. Run
with `--run-compile-tier` under `flock /tmp/vt-tpu.lock`.
"""

import pytest

from tests.layers.core import dead_fields_check
from vllm_torchtpu.layers.core import moe_plan

pytestmark = pytest.mark.compile_tier


@pytest.fixture(scope="module")
def programs() -> dict[moe_plan.MoeCombinePlan, str]:
    """The traced program of every plan that traces on v6e."""
    return dead_fields_check.programs_on_v6e()


def test_plans_traced(programs: dict[moe_plan.MoeCombinePlan, str]) -> None:
    """Seven plan values trace on v6e, as recorded in the README."""
    assert len(programs) == 7  # noqa: PLR2004 - the recorded count.


@pytest.mark.parametrize("field", dead_fields_check.field_names())
def test_field_is_read(
    programs: dict[moe_plan.MoeCombinePlan, str], field: str
) -> None:
    """Flipping the field alone changes the traced program for some plan."""
    assert field in dead_fields_check.fields_read(programs)
