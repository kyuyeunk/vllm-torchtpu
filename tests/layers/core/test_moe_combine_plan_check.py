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
r"""The MoE combine's plan check: each tier's findings equal its known set.

The structural tier compiles for fake chips and needs libtpu, so it is marked
`compile_tier`; the numerics tier runs on the host's chip, so it is marked
`tpu`. A default CPU run skips both:

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m pytest --run-compile-tier \
        tests/layers/core/test_moe_combine_plan_check.py
    flock /tmp/vt-tpu.lock env PYTHONPATH=src \
        python -m pytest --run-tpu \
        tests/layers/core/test_moe_combine_plan_check.py
"""

import contextlib

import pytest

from tests.layers.core import moe_combine_plan_check, moe_combine_plan_values_check


@pytest.mark.compile_tier
@pytest.mark.parametrize(
    "replay_import_time_kernel",
    [False, True],
    ids=["ep_kernel_by_plan", "ep_kernel_at_import_time"],
)
def test_structure_findings_are_known(*, replay_import_time_kernel: bool) -> None:
    """Every structural finding is known, and every known one still appears."""
    with (
        moe_combine_plan_check.import_time_kernel()
        if replay_import_time_kernel
        else contextlib.nullcontext()
    ):
        findings = moe_combine_plan_check.structure_findings()
    known = moe_combine_plan_check.known(
        replay_import_time_kernel=replay_import_time_kernel
    )
    new = moe_combine_plan_check.lines(findings - known)
    gone = moe_combine_plan_check.lines(known - findings)
    assert findings == known, f"new: {new}, gone: {gone}"


@pytest.mark.tpu
def test_value_findings_are_known() -> None:
    """Every numerics finding is known, and every known one still appears.

    The error size is printed by the demo, not ratcheted.
    """
    found = moe_combine_plan_values_check.found(
        moe_combine_plan_values_check.value_findings()
    )
    known = moe_combine_plan_values_check.KNOWN
    new, gone = sorted(map(str, found - known)), sorted(map(str, known - found))
    assert found == known, f"new: {new}, gone: {gone}"
