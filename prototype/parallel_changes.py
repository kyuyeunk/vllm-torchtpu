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
the calls both new variants run: the overlap the merger reviews. The
collecting is `tests/layers/core/moe_parallel_changes.py`'s, which
`test_moe_combine_moves.py` checks.
"""

from collections.abc import Iterable

from tests.layers.core.moe_combine_moves import TOKENS, TOPK, Outcome
from tests.layers.core.moe_parallel_changes import (
    Setting,
    Side,
    moves,
    parallel_changes,
)


def _names(paths: Iterable[str]) -> str:
    return ", ".join(p.removeprefix("src/vllm_torchtpu/") for p in sorted(paths))


def _outcome(row: list[Outcome], before: list[Outcome]) -> str:
    m = moves(row, before)
    return f"moves {m.moved}, raises on {m.raised}" + (
        f" ({'; '.join(m.reasons)})" if m.reasons else ""
    )


def _settings(settings: list[Setting]) -> str:
    """The settings as the output names them, or "none"."""
    names = ["no setting" if s is None else s.value for s in settings]
    return ", ".join(names) or "none"


def main() -> None:
    """Prints the files, the conflict, each setting's calls, and the overlap."""
    report = parallel_changes()
    base, a, b, merge = (report.commits[s] for s in Side)
    touched = report.touched
    print(f"base {base[:9]}, A {a[:9]}, B {b[:9]}, merge {merge[:9]}")
    print(f"A touched: {_names(touched[Side.A])}")
    print(f"B touched: {_names(touched[Side.B])}")
    print(f"both touched: {_names(touched[Side.A] & touched[Side.B])}")
    print(f"git merge: {len(report.conflicts)} conflicted file(s)")
    for path, hunks in report.conflicts.items():
        print(f"  {_names([path])}, {len(hunks)} hunks, ours  |  theirs:")
        for hunk in hunks:
            print(f"    {hunk}")

    runs, default, old = report.runs, report.default, report.existing
    print(
        f"{len(report.grid)} calls: v6e and TPU7x, {len(TOKENS)} token counts, "
        f"top-k {TOPK}"
    )
    print(
        f"same under base, A, B and A+B: no setting ({default[0]} on "
        f"{default.count(default[0])}) and {len(old) - 1 - len(report.changed)} of "
        f"{len(old) - 1} existing settings; changed: {_settings(report.changed)}"
    )
    for who, settings in report.new.items():
        for setting in settings:
            same = runs[who][setting] == runs[Side.MERGE][setting]
            print(
                f"{who.value}'s TPU_MOE_COMBINE={setting.value}: "
                f"{_outcome(runs[who][setting], default)}; "
                f"{'same' if same else 'CHANGED'} under A+B"
            )
    both = report.overlap
    print(f"calls both new variants run, the overlap the merger reviews: {len(both)}")
    for chip in dict.fromkeys(c for c, _, _ in both):
        topks = sorted({k for c, _, k in both if c is chip})
        tokens = sorted({t for c, t, _ in both if c is chip})
        print(f"  {chip.label:6} top-k {topks} x tokens {tokens}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
