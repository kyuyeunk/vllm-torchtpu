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
"""Runs ``tools/check_step_plan_writers.py`` on this branch's package."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_check(check: str) -> subprocess.CompletedProcess[str]:
    """Run one check of the tool in a fresh interpreter (it imports nothing)."""
    return subprocess.run(  # noqa: S603 - a fixed argv: this repo's own checker
        [sys.executable, str(ROOT / "tools" / "check_step_plan_writers.py"), check],
        capture_output=True,
        text=True,
        check=False,
    )


def test_dp_step_has_one_writer() -> None:
    """Only the DP agreement writes the runner's `DpStep`, and only whole."""
    result = _run_check("dp-step")
    assert result.returncode == 0, result.stdout + result.stderr


def test_builder_context_stash_ratchet() -> None:
    """What is left on the stash: the chunk loop's reads after the forward."""
    result = _run_check("stash")
    assert result.returncode == 0, result.stdout + result.stderr
