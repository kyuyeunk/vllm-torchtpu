#!/usr/bin/env python3
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
"""Fail when the step's DP agreement gains a writer, or the context stash grows.

Two checks over ``src/vllm_torchtpu``, both by AST, importing nothing:

- ``dp-step``: only ``__init__`` and ``_dp_coordinated_step`` assign the
  runner's ``_dp_step``.
- ``stash``: a ratchet of the accesses to ``_attn_metadata_builder_ctx``
  that remain (the chunk loop's reads after the forward). Every metadata
  builder receives its context as an argument; a new access fails here, and
  moving one fails until its entry is removed from ``known_stash_uses``.

    tools/check_step_plan_writers.py           # both checks
    tools/check_step_plan_writers.py dp-step   # the single writer only
    tools/check_step_plan_writers.py stash     # the stash ratchet only
"""

from __future__ import annotations

import argparse
import ast
import collections
import enum
import pathlib
import sys
from typing import NamedTuple

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "vllm_torchtpu"
_RUNNER = "runner/tpu_runner.py"


class Check(enum.Enum):
    """One of the two checks; the value is its name on the command line."""

    DP_STEP = "dp-step"
    STASH = "stash"


class Use(enum.Enum):
    """How a function touches an attribute."""

    READ = enum.auto()
    WRITE = enum.auto()


class Access(NamedTuple):
    """A function's access to an attribute, by file and function name."""

    file: str
    function: str
    use: Use


def known_dp_step_writers() -> frozenset[tuple[str, str]]:
    """The functions that may assign the runner's `_dp_step`."""
    return frozenset({(_RUNNER, "__init__"), (_RUNNER, "_dp_coordinated_step")})


def known_stash_uses() -> collections.Counter[Access]:
    """The accesses to the context stash that remain, with their counts."""
    return collections.Counter(
        {
            Access(_RUNNER, "_prepare_inputs", Use.WRITE): 1,
            Access(_RUNNER, "_dummy_run", Use.WRITE): 1,
            Access(_RUNNER, "execute_model", Use.READ): 2,
            Access(_RUNNER, "_mamba_state_index_groups", Use.READ): 1,
        }
    )


def find_accesses(package: pathlib.Path, attr: str) -> collections.Counter[Access]:
    """Count every access to `.attr` per (file, function, use) in `package`."""
    uses: collections.Counter[Access] = collections.Counter()
    for path in sorted(package.rglob("*.py")):
        rel = str(path.relative_to(package))
        for func in ast.walk(ast.parse(path.read_text())):
            if not isinstance(func, ast.FunctionDef):
                continue
            for node in ast.walk(func):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == attr  # str-ok: matches the scanned source name
                ):
                    use = Use.WRITE if isinstance(node.ctx, ast.Store) else Use.READ
                    uses[Access(rel, func.name, use)] += 1
    return uses


def _report(line: str) -> None:
    """Write one line of the report to stdout."""
    print(line)  # noqa: T201 - the CLI's report is its output


def check_dp_step(package: pathlib.Path) -> bool:
    """Print the `_dp_step` writers; return whether they are the known two."""
    writers = frozenset(
        (a.file, a.function)
        for a in find_accesses(package, "_dp_step")
        if a.use is Use.WRITE
    )
    for file, function in sorted(writers):
        _report(f"_dp_step written in {file}:{function}")
    return writers == known_dp_step_writers()


def check_stash(package: pathlib.Path) -> bool:
    """Print the stash accesses; return whether they equal the known set."""
    found = find_accesses(package, "_attn_metadata_builder_ctx")
    for access, count in sorted(found.items()):
        _report(f"{access.file}:{access.function}: {access.use.name} x{count}")
    _report(f"stash: {found.total()} accesses in {len(found)} functions")
    return found == known_stash_uses()


def run(check: Check, package: pathlib.Path) -> bool:
    """Run one check over `package`; return whether it passed."""
    match check:
        case Check.DP_STEP:
            return check_dp_step(package)
        case Check.STASH:
            return check_stash(package)


class _Args(argparse.Namespace):
    """The parsed command line."""

    def __init__(self) -> None:
        """Start from no checks named and this branch's package."""
        super().__init__()
        self.checks: list[Check] = []
        self.package: pathlib.Path = PACKAGE


def main(argv: list[str]) -> int:
    """Run the requested checks; return 1 if any failed."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    _ = parser.add_argument("checks", nargs="*", type=Check, metavar="{dp-step,stash}")
    _ = parser.add_argument("--package", type=pathlib.Path, default=PACKAGE)
    args = parser.parse_args(argv, namespace=_Args())
    results = [run(check, args.package) for check in args.checks or list(Check)]
    return 0 if all(results) else 1


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(main(sys.argv[1:]))
