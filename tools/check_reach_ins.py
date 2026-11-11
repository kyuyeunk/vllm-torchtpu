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
"""Fail when new code reaches into another module's decisions or names a model.

Two ratchets, each against its own baseline:

- ``writes``: an assignment to an attribute of an object reached through
  another object (``self.attn.o_proj.reduce_results = False``,
  ``layer.mlp.experts.gate = None``, or a local bound from such a chain and
  then written). A child's decision set from outside its own ``__init__``
  bypasses the plan that should hand it down. ``self.x = ...`` is the
  module's own state and does not count.
- ``identity``: outside ``models/``, an import of a model module
  (``vllm.model_executor.models``, ``vllm.models``, ``vllm_torchtpu.models``)
  or a comparison against a ``model_type`` / ``architectures`` string. Shared
  code chooses by structure and declared capability, never by which model it
  is.

The existing entries stay until someone migrates them; a new one fails, and a
fixed one must be deleted from the baseline so it cannot come back.

    tools/check_reach_ins.py                  # check both
    tools/check_reach_ins.py --update         # rewrite both baselines
    tools/check_reach_ins.py --report PATH... # list hits in any files
"""

from __future__ import annotations

import argparse
import ast
import enum
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "vllm_torchtpu"


class Ratchet(enum.Enum):
    """The two ratchets; the value names one in messages and its baseline."""

    WRITES = "writes"
    IDENTITY = "identity"


BASELINES = {
    Ratchet.WRITES: REPO_ROOT / "tools" / "reach_in_writes_baseline.txt",
    Ratchet.IDENTITY: REPO_ROOT / "tools" / "model_identity_baseline.txt",
}
_MODEL_MODULES = ("vllm.model_executor.models", "vllm.models", "vllm_torchtpu.models")
_IDENTITY_ATTRS = {"model_type", "architectures"}


def _chain(node: ast.AST) -> list[str]:
    """Names along an attribute chain, root first; empty if not rooted at a name."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return []
    return [node.id, *reversed(parts)]


def _reached_locals(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Locals bound to an object reached through another: ``x = a.b``."""
    reached = set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            values = [node.value]
            if isinstance(node.value, ast.IfExp):
                values = [node.value.body, node.value.orelse]
            if all(len(_chain(v)) >= 2 for v in values):
                reached |= {t.id for t in targets if isinstance(t, ast.Name)}
    return reached


def _imported(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


def _patches(chain: list[str], imported: set[str]) -> bool:
    # Writes onto a module or class object are monkeypatches, which the patch
    # registry owns; this check is about one object's instance decisions.
    return (
        chain[0] in imported
        or chain[0] == "cls"  # str-ok: the lint matches source names
        or any(part[:1].isupper() for part in chain[:-1])
    )


def find_writes(path: pathlib.Path, rel: str) -> list[str]:
    """``rel:function:target`` for each write into another object's field."""
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = _imported(tree)
    found = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        reached = _reached_locals(fn)
        for node in ast.walk(fn):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                chain = _chain(target)
                if len(chain) < 2 or _patches(chain, imported):
                    continue
                through_other = len(chain) >= 3 or chain[0] in reached
                if through_other:
                    found.add(f"{rel}:{fn.name}:{'.'.join(chain)}")
    return sorted(found)


def find_identity(path: pathlib.Path, rel: str) -> list[str]:
    """``rel:kind:detail`` for each model import or model-identity literal."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            names = []
        for name in names:
            if name.startswith(_MODEL_MODULES):
                found.add(f"{rel}:import:{name}")
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            attrs = {c[-1] for s in sides if (c := _chain(_strip(s)))}
            literals = [v for s in sides for v in _strings(s)]
            if attrs & _IDENTITY_ATTRS and literals:
                field = sorted(attrs & _IDENTITY_ATTRS)[0]
                found.add(f"{rel}:{field}:{','.join(sorted(literals))}")
    return sorted(found)


def _strings(node: ast.AST) -> list[str]:
    """String literals in ``node`` or in the tuple/list/set it spells."""
    elts = node.elts if isinstance(node, (ast.Tuple, ast.List, ast.Set)) else [node]
    return [
        e.value
        for e in elts
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    ]


def _strip(node: ast.AST) -> ast.AST:
    # architectures[0] and getattr-free subscripts read the same field.
    return node.value if isinstance(node, ast.Subscript) else node


def scan(ratchet: Ratchet) -> list[str]:
    """The package's hits for one ratchet."""
    hits: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if ratchet is Ratchet.WRITES:
            hits += find_writes(path, rel)
        elif not path.is_relative_to(PACKAGE / "models"):
            hits += find_identity(path, rel)
    return hits


def _load(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite baselines")
    parser.add_argument("--report", nargs="+", type=pathlib.Path, help="files to scan")
    args = parser.parse_args()
    if args.report:
        for path in args.report:
            for hit in find_writes(path, str(path)) + find_identity(path, str(path)):
                print(hit)
        return 0
    failed = False
    for ratchet, baseline in BASELINES.items():
        kind = ratchet.value
        hits = scan(ratchet)
        if args.update:
            baseline.write_text(
                f"# {kind} reach-ins, as of the last update. Generated by\n"
                "# tools/check_reach_ins.py --update. Shrink it; never grow it.\n"
                + "\n".join(hits)
                + "\n"
            )
            continue
        known = _load(baseline)
        for hit in sorted(set(hits) - set(known)):
            print(f"new {kind} reach-in: {hit}")
            failed = True
        for entry in sorted(set(known) - set(hits)):
            print(f"stale {kind} baseline entry (delete it): {entry}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(main())
