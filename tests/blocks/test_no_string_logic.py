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
"""The framework decides nothing by string: the shared checker finds no site.

``tools/check_no_string_logic.py`` is the rule every prototype branch shares.
A site it should allow carries ``# str-ok: <reason>`` (the parse points of
the user's pins, torch.fx's node kinds, stack-frame paths).
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_blocks_decide_nothing_by_string() -> None:
    """Every comparison, key and lookup in blocks/ is typed."""
    result = subprocess.run(  # noqa: S603 - a fixed argv: this repo's own checker
        [
            sys.executable,
            str(ROOT / "tools" / "check_no_string_logic.py"),
            str(ROOT / "src" / "vllm_torchtpu" / "blocks"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
