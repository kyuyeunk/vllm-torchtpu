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
"""What two parallel changes to the combine table move, from git history.

Person A adds `gather_reduce_v4` and person B adds `segment_sum`, both from
the same commit; the branch merges them. For the base, A, B and the merge,
`parallel_changes` extracts that commit's `src/` and plans the grid of
`moe_combine_moves` under every `TPU_MOE_COMBINE` setting, and collects the
files each change touched and where git's merge conflicts. Shared by
`test_moe_combine_moves.py` and `prototype/parallel_changes.py`.
"""

import dataclasses
import enum
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from typing import cast

from tests.layers.core.moe_combine_moves import (
    Call,
    Chip,
    Outcome,
    Raises,
    Runs,
    describe,
    grid,
    outcome,
)
from vllm_torchtpu.layers.core import moe_plan

# The repository root: `probe` imports this module from it.
_ROOT = pathlib.Path(__file__).resolve().parents[3]

type Setting = moe_plan.Combine | None
# `probe`'s JSON: the grid, then per setting each call's variant or reason.
type _Probed = tuple[
    list[tuple[str, int, int]],
    list[tuple[str | None, list[tuple[str | None, str | None]]]],
]


class Side(enum.Enum):
    """A commit of the parallel change; the value is its name in the output."""

    BASE = "base"
    A = "A"
    B = "B"
    MERGE = "A+B"


@dataclasses.dataclass(frozen=True)
class CommitPlan:
    """What every setting runs on every call of the grid, at one commit."""

    grid: list[Call]
    runs: dict[Setting, list[Outcome]]


def probe() -> None:
    """Prints, as JSON, what each setting runs on each call of the grid.

    Runs in a subprocess against one commit's `src/`. Variants and chips go
    by name, since that commit's `Combine` is not this checkout's.
    """
    calls = grid()
    runs: list[tuple[str | None, list[tuple[str | None, str | None]]]] = []
    for setting in (None, *moe_plan.VARIANTS):
        override = None if setting is None else moe_plan.MoeCombinePlan(combine=setting)
        process = moe_plan.MoeProcess(use_sparse_core=True, override=override)
        row: list[tuple[str | None, str | None]] = []
        for chip, tokens, topk in calls:
            what = outcome(process, *describe(tokens, topk), chip.info)
            row.append(
                (what.combine.value, None)
                if isinstance(what, Runs)
                else (None, what.reason)
            )
        runs.append((None if setting is None else setting.value, row))
    print(json.dumps([[[chip.name, t, k] for chip, t, k in calls], runs]))


def _parse(probed: str) -> CommitPlan:
    """Parses `probe`'s JSON into this checkout's types."""
    calls, runs = cast(_Probed, json.loads(probed))

    def setting(name: str | None) -> Setting:
        return None if name is None else moe_plan.Combine(name)

    def cell(name: str | None, reason: str | None) -> Outcome:
        return Runs(moe_plan.Combine(name)) if name is not None else Raises(str(reason))

    return CommitPlan(
        grid=[(Chip[chip], tokens, topk) for chip, tokens, topk in calls],
        runs={setting(name): [cell(*c) for c in row] for name, row in runs},
    )


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603  # fixed git subcommands, no shell
        ["git", *args],  # noqa: S607  # git from PATH, as a developer runs it
        check=True,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    ).stdout.strip()


def _plan_at(commit: str) -> CommitPlan:
    """Runs `probe` against `commit`'s `src/`, with this checkout's tests."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(  # noqa: S603  # fixed git subcommand, no shell
            ["git", "archive", commit, "src"],  # noqa: S607  # git from PATH
            check=True,
            capture_output=True,
            cwd=_ROOT,
        ).stdout
        archive_path = pathlib.Path(tmp, "src.tar")
        _ = archive_path.write_bytes(archive)
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp, filter="data")
        env = {**os.environ, "PYTHONPATH": f"{tmp}/src{os.pathsep}{_ROOT}"}
        out = subprocess.run(  # noqa: S603  # this interpreter, fixed code
            [
                sys.executable,
                "-c",
                "import tests.layers.core.moe_parallel_changes as p; p.probe()",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            cwd=_ROOT,
        ).stdout
    return _parse(out.splitlines()[-1])


def _conflicts(base: str, ours: str, theirs: str) -> dict[str, list[str]]:
    """Each conflicted file, with the first line each side wrote per hunk."""
    hunks: dict[str, list[str]] = {}
    path, lines = "", _git("merge-tree", base, ours, theirs).splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  our "):
            path = line.split()[-1]
        elif line.startswith("+<<<<<<< "):
            split = lines.index("+=======", i)
            ours_line, theirs_line = (
                lines[i + 1][1:].strip(),
                lines[split + 1][1:].strip(),
            )
            hunks.setdefault(path, []).append(f"{ours_line}  |  {theirs_line}")
    return hunks


@dataclasses.dataclass(frozen=True)
class Moves:
    """What one setting does to the grid, against what ran before.

    Attributes:
        moved: Calls that now run a different variant.
        raised: Calls the setting raises on.
        reasons: The distinct reasons it raises with, sorted.
    """

    moved: int
    raised: int
    reasons: list[str]


def moves(row: list[Outcome], before: list[Outcome]) -> Moves:
    """Compares one setting's outcomes with the outcomes before it.

    Args:
        row: What each call does under the setting.
        before: What each call ran before, in the same order.

    Returns:
        The calls moved, the calls raised on, and the reasons.
    """
    return Moves(
        moved=sum(
            new != old and isinstance(new, Runs)
            for new, old in zip(row, before, strict=True)
        ),
        raised=sum(isinstance(r, Raises) for r in row),
        reasons=sorted({r.reason for r in row if isinstance(r, Raises)}),
    )


@dataclasses.dataclass(frozen=True)
class ParallelChanges:
    """The two changes, their merge, and what each setting runs at each.

    Attributes:
        commits: Each side's commit.
        touched: The files A and B each changed since the base.
        conflicts: Each file git's merge conflicts in, with its hunks.
        runs: At each side, what each setting runs on each call.
        grid: The calls, in the order of `runs`.
    """

    commits: dict[Side, str]
    touched: dict[Side, set[str]]
    conflicts: dict[str, list[str]]
    runs: dict[Side, dict[Setting, list[Outcome]]]
    grid: list[Call]

    @property
    def default(self) -> list[Outcome]:
        """What each call runs at the base with no setting."""
        return self.runs[Side.BASE][None]

    @property
    def existing(self) -> list[Setting]:
        """The settings the base has, no setting first."""
        return list(self.runs[Side.BASE])

    @property
    def changed(self) -> list[Setting]:
        """The base's settings that run anything differently at some side."""
        base = self.runs[Side.BASE]
        return [
            s
            for s in self.existing
            if any(runs[s] != base[s] for runs in self.runs.values())
        ]

    @property
    def new(self) -> dict[Side, list[moe_plan.Combine]]:
        """The settings A and B each add."""
        old = self.existing
        return {
            who: [s for s in moe_plan.Combine if s in self.runs[who] and s not in old]
            for who in (Side.A, Side.B)
        }

    @property
    def overlap(self) -> list[Call]:
        """The calls both new variants run at the merge."""
        merged = self.runs[Side.MERGE]
        added = [s for settings in self.new.values() for s in settings]
        return [
            call
            for i, call in enumerate(self.grid)
            if not any(isinstance(merged[s][i], Raises) for s in added)
        ]


def parallel_changes() -> ParallelChanges:
    """Collects the last merge in HEAD's history, its parents and their base.

    Returns:
        The commits, the files and conflicts, and each commit's plan.
    """
    merge = _git("rev-list", "--merges", "-1", "HEAD")
    a, b = _git("rev-parse", f"{merge}^1"), _git("rev-parse", f"{merge}^2")
    base = _git("merge-base", a, b)
    commits = {Side.BASE: base, Side.A: a, Side.B: b, Side.MERGE: merge}
    plans = {who: _plan_at(c) for who, c in commits.items()}
    return ParallelChanges(
        commits=commits,
        touched={
            who: set(_git("diff", "--name-only", base, commits[who]).splitlines())
            for who in (Side.A, Side.B)
        },
        conflicts=_conflicts(base, a, b),
        runs={who: p.runs for who, p in plans.items()},
        grid=plans[Side.MERGE].grid,
    )
