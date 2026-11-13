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
"""The prototype decides nothing by string: the shared checker finds no site.

``tools/check_no_string_logic.py`` is the rule every prototype branch shares.
A site it should allow carries ``# str-ok: <reason>`` (the parse points of
the user's pins, torch.fx's node kinds, stack-frame paths, the surveys'
reads of config.json and of undeclared classes). It runs on the framework's
``blocks/``, the demo scripts in ``prototype/`` and the check code they call
in ``tests/blocks/``.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The prototype's own code outside src/: its demos and the check code they call.
_PROTOTYPE = (
    "prototype",
    "tests/blocks/_build.py",
    "tests/blocks/_platform.py",
    "tests/blocks/_report.py",
    "tests/blocks/block_check.py",
    "tests/blocks/fixtures.py",
    "tests/blocks/subblock_check.py",
    "tests/blocks/subblock_fixtures.py",
    "tests/blocks/subblock_reach.py",
    "tests/blocks/suggest_blocks.py",
)


def _check(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - a fixed argv: this repo's own checker
        [
            sys.executable,
            str(ROOT / "tools" / "check_no_string_logic.py"),
            str(ROOT / path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_blocks_decide_nothing_by_string() -> None:
    """Every comparison, key and lookup in blocks/ is typed."""
    result = _check("src/vllm_torchtpu/blocks")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("path", _PROTOTYPE)
def test_prototype_decides_nothing_by_string(path: str) -> None:
    """Every comparison, key and lookup under ``path`` is typed."""
    result = _check(path)
    assert result.returncode == 0, result.stdout + result.stderr
