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
"""Shows what two parallel changes to the combine table move, on CPU.

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.parallel_changes

Person A adds `gather_reduce_v4` and person B adds `segment_sum`, both from
the same commit; the branch merges them. For the base, A, B and the merge,
this extracts that commit's `src/` and plans the same grid of calls under
every `TPU_MOE_COMBINE` setting, then prints the files each change touched,
where git's merge conflicts, what each setting does under each commit, and
the calls both new variants run: the overlap the merger reviews.
"""

import dataclasses
import enum
import itertools
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable

from prototype.combine_plan import Chip, Raises, Runs, describe, outcome
from vllm_torchtpu.layers.core import moe_plan

TOKENS = (16, 64, 256, 512, 2048, 8192)
TOPK = (1, 2, 4, 8)

type Call = tuple[Chip, int, int]
type Setting = moe_plan.Combine | None
type Outcome = Runs | Raises


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
    grid = list(itertools.product(Chip, TOKENS, TOPK))
    runs = []
    for setting in (None, *moe_plan.VARIANTS):
        override = None if setting is None else moe_plan.MoeCombinePlan(combine=setting)
        process = moe_plan.MoeProcess(use_sparse_core=True, override=override)
        row = []
        for chip, tokens, topk in grid:
            what = outcome(process, *describe(tokens, topk), chip.info)
            row.append(
                [what.combine.value, None]
                if isinstance(what, Runs)
                else [None, what.reason]
            )
        runs.append([None if setting is None else setting.value, row])
    print(json.dumps([[[chip.name, t, k] for chip, t, k in grid], runs]))


def _parse(probed: str) -> CommitPlan:
    """Parses `probe`'s JSON into this checkout's types."""
    grid, runs = json.loads(probed)

    def setting(name: str | None) -> Setting:
        return None if name is None else moe_plan.Combine(name)

    def cell(name: str | None, reason: str | None) -> Outcome:
        return Runs(moe_plan.Combine(name)) if name is not None else Raises(str(reason))

    return CommitPlan(
        grid=[(Chip[chip], tokens, topk) for chip, tokens, topk in grid],
        runs={setting(name): [cell(*c) for c in row] for name, row in runs},
    )


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _plan_at(commit: str) -> CommitPlan:
    """Runs `probe` against `commit`'s `src/`, with this checkout's prototype."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "archive", commit, "src"], check=True, capture_output=True
        ).stdout
        archive_path = pathlib.Path(tmp, "src.tar")
        archive_path.write_bytes(archive)
        with tarfile.open(archive_path) as tar:
            tar.extractall(tmp, filter="data")
        env = {**os.environ, "PYTHONPATH": f"{tmp}/src{os.pathsep}{os.getcwd()}"}
        out = subprocess.run(
            [sys.executable, "-c", "import prototype.parallel_changes as p; p.probe()"],
            check=True,
            capture_output=True,
            text=True,
            env=env,
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


def _names(paths: Iterable[str]) -> str:
    return ", ".join(p.removeprefix("src/vllm_torchtpu/") for p in sorted(paths))


def _outcome(row: list[Outcome], before: list[Outcome]) -> str:
    moved = sum(
        new != old and isinstance(new, Runs)
        for new, old in zip(row, before, strict=True)
    )
    reasons = sorted({r.reason for r in row if isinstance(r, Raises)})
    raised = sum(isinstance(r, Raises) for r in row)
    return f"moves {moved}, raises on {raised}" + (
        f" ({'; '.join(reasons)})" if reasons else ""
    )


def main() -> None:
    """Prints the files, the conflict, each setting's calls, and the overlap."""
    merge = _git("rev-list", "--merges", "-1", "HEAD")
    a, b = _git("rev-parse", f"{merge}^1"), _git("rev-parse", f"{merge}^2")
    base = _git("merge-base", a, b)
    touched = {
        who: set(_git("diff", "--name-only", base, commit).splitlines())
        for who, commit in ((Side.A, a), (Side.B, b))
    }
    print(f"base {base[:9]}, A {a[:9]}, B {b[:9]}, merge {merge[:9]}")
    print(f"A touched: {_names(touched[Side.A])}")
    print(f"B touched: {_names(touched[Side.B])}")
    print(f"both touched: {_names(touched[Side.A] & touched[Side.B])}")
    conflicts = _conflicts(base, a, b)
    print(f"git merge: {len(conflicts)} conflicted file(s)")
    for path, hunks in conflicts.items():
        print(f"  {_names([path])}, {len(hunks)} hunks, ours  |  theirs:")
        for hunk in hunks:
            print(f"    {hunk}")

    commits = {Side.BASE: base, Side.A: a, Side.B: b, Side.MERGE: merge}
    plans = {who: _plan_at(c) for who, c in commits.items()}
    grid = plans[Side.MERGE].grid
    runs = {who: p.runs for who, p in plans.items()}
    default = runs[Side.BASE][None]
    print(f"{len(grid)} calls: v6e and TPU7x, {len(TOKENS)} token counts, top-k {TOPK}")
    old = list(runs[Side.BASE])
    changed = [s for s in old if any(runs[w][s] != runs[Side.BASE][s] for w in runs)]
    print(
        f"same under base, A, B and A+B: no setting ({default[0]} on "
        f"{default.count(default[0])}) and {len(old) - 1 - len(changed)} of "
        f"{len(old) - 1} existing settings; changed: {_settings(changed)}"
    )
    new = {
        who: [s for s in moe_plan.Combine if s in runs[who] and s not in old]
        for who in (Side.A, Side.B)
    }
    for who, settings in new.items():
        for setting in settings:
            same = runs[who][setting] == runs[Side.MERGE][setting]
            print(
                f"{who.value}'s TPU_MOE_COMBINE={setting.value}: "
                f"{_outcome(runs[who][setting], default)}; "
                f"{'same' if same else 'CHANGED'} under A+B"
            )
    both = [
        call
        for i, call in enumerate(grid)
        if not any(
            isinstance(runs[Side.MERGE][s][i], Raises)
            for s in new[Side.A] + new[Side.B]
        )
    ]
    print(f"calls both new variants run, the overlap the merger reviews: {len(both)}")
    for chip in dict.fromkeys(c for c, _, _ in both):
        topks = sorted({k for c, _, k in both if c is chip})
        tokens = sorted({t for c, t, _ in both if c is chip})
        print(f"  {chip.label:6} top-k {topks} x tokens {tokens}")


def _settings(settings: list[Setting]) -> str:
    """The settings as the output names them, or "none"."""
    names = ["no setting" if s is None else s.value for s in settings]
    return ", ".join(names) or "none"


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
