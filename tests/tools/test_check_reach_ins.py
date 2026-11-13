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
"""Runs tools/check_reach_ins.py: the tree passes, and planted reach-ins show.

The ratchet run holds the package to its two baselines
(`tools/reach_in_writes_baseline.txt`, `tools/model_identity_baseline.txt`):
no new entry, no stale one. The `--report` runs plant the README's examples in
a temporary file, so the tree is never touched.
"""

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "check_reach_ins.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - a fixed argv: this repo's own checker.
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )


def test_tree_matches_baselines() -> None:
    """No new reach-in and no stale baseline entry in src/vllm_torchtpu."""
    result = _run()
    assert (result.returncode, result.stdout) == (0, ""), result.stdout + result.stderr


@pytest.mark.parametrize(
    ("source", "hit"),
    [
        pytest.param(
            "def f(layer):\n    layer.mlp.experts.gate = None\n",
            "f:layer.mlp.experts.gate",
            id="write-through-child",
        ),
        pytest.param(
            'def g(cfg):\n    return cfg.model_type == "qwen3_5_moe_text"\n',
            "model_type:qwen3_5_moe_text",
            id="model-type-literal",
        ),
        pytest.param(
            "def h(self):\n    self.x = 1\n",
            None,
            id="own-state-not-flagged",
        ),
    ],
)
def test_report_flags_planted_reach_in(
    tmp_path: pathlib.Path, source: str, hit: str | None
) -> None:
    """`--report` lists exactly the planted reach-in, or nothing for own state."""
    planted = tmp_path / "planted_reach_in.py"
    _ = planted.write_text(source)
    result = _run("--report", str(planted))
    assert result.returncode == 0, result.stderr
    expected = [] if hit is None else [f"{planted}:{hit}"]
    assert result.stdout.splitlines() == expected
