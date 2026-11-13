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
"""Prints the structural tier's findings for the MoE combine.

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.check_plans

Loads libtpu to compile, so run it under the TPU lock, never beside a TPU job.
The shapes, callbacks and known findings are in
`tests/layers/core/moe_combine_plan_check.py`; the test that asserts the
findings equal the known set is `tests/layers/core/test_moe_combine_plan_check.py`.
"""

import contextlib
import time

from tests.layers.core import moe_combine_plan_check

# Comment in to replay the EP kernel chosen at import time, not by the plan.
REPLAY_IMPORT_TIME_KERNEL = False
# REPLAY_IMPORT_TIME_KERNEL = True  # noqa: ERA001 - a comment-in switch


def main() -> None:
    """Prints the findings, one per line, and their count."""
    start = time.perf_counter()
    with (
        moe_combine_plan_check.import_time_kernel()
        if REPLAY_IMPORT_TIME_KERNEL
        else contextlib.nullcontext()
    ):
        findings = moe_combine_plan_check.structure_findings()
    for line in moe_combine_plan_check.lines(findings):
        print(line)
    print(f"findings: {len(findings)} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
