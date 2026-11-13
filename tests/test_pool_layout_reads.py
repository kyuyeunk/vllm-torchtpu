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
"""Runs ``tools/check_pool_layout_reads.py`` on this branch's package."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run_check(check: str) -> subprocess.CompletedProcess[str]:
    """Run one check of the tool in a fresh interpreter (it imports nothing)."""
    return subprocess.run(  # noqa: S603 - a fixed argv: this repo's own checker
        [sys.executable, str(ROOT / "tools" / "check_pool_layout_reads.py"), check],
        capture_output=True,
        text=True,
        check=False,
    )


def test_direct_pool_layout_reads_only_shrink() -> None:
    """The direct reads equal the known set: none new, none moved."""
    result = _run_check("reads")
    assert result.returncode == 0, result.stdout + result.stderr


def test_only_pool_layout_writes_the_record() -> None:
    """Only ``pool_layout.py`` names the stage marker."""
    result = _run_check("writers")
    assert result.returncode == 0, result.stdout + result.stderr
