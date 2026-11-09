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
"""Fail when a new ``envs.X`` read appears outside the producer modules.

``tools/check_env_reads.py`` keeps raw ``os.environ`` reads inside
``envs.py``. That does not stop a layer, kernel or the runner from reading
``envs.SOME_KNOB`` at its own call site, which hides a configuration choice
inside the code that acts on it. The rule this enforces is that the producer
modules read the environment and hand the result over as a value; everything
else takes the value.

This is a ratchet, as ``check_env_reads.py`` is. ``ENVS_USE_SITES_BASELINE``
records the reads that already existed as ``relpath:NAME`` entries, a new
``(file, name)`` pair fails the hook, and an entry that no longer matches a
read fails until the baseline is shrunk.

A read is any attribute of ``vllm_torchtpu.envs`` used through a name bound
to it (``from vllm_torchtpu import envs``, ``import vllm_torchtpu.envs as
tpu_envs``), the full dotted path, ``from vllm_torchtpu.envs import NAME``,
and ``getattr(envs, ...)``, recorded as ``<dynamic>`` when the name is not a
literal. ``vllm.envs`` is upstream vLLM's registry and is not counted.

Run directly to see the current state:

    tools/check_envs_use_sites.py            # check
    tools/check_envs_use_sites.py --update   # rewrite the baseline
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "vllm_torchtpu"
BASELINE_PATH = REPO_ROOT / "tools" / "envs_use_sites_baseline.txt"
ENVS_MODULE = "vllm_torchtpu.envs"

# Modules allowed to read the registry: the registry itself, the derived
# defaults layered on it, and the decision and plan modules that turn knobs
# into values. Matched by full path so a same-named file elsewhere does not
# inherit the exemption.
PRODUCERS = {
    PACKAGE / "envs.py",
    PACKAGE / "env_override.py",
    PACKAGE / "layers" / "core" / "moe_plan.py",
    PACKAGE / "layers" / "core" / "moe_collective_plan.py",
    PACKAGE / "layers" / "core" / "tpu_target.py",
}


def _envs_aliases(tree: ast.Module) -> set[str]:
    """Returns local names bound to the ``vllm_torchtpu.envs`` module.

    Args:
        tree: The parsed module.

    Returns:
        Names such as ``envs`` or ``tpu_envs``.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "vllm_torchtpu":
            aliases.update(a.asname or a.name for a in node.names if a.name == "envs")
        elif isinstance(node, ast.Import):
            # A bare `import vllm_torchtpu.envs` binds `vllm_torchtpu`, which
            # the dotted-path match in `_read_name` covers.
            aliases.update(
                a.asname for a in node.names if a.name == ENVS_MODULE and a.asname
            )
    return aliases


def _dotted(node: ast.expr) -> str:
    """Returns the dotted name of an attribute chain, or "" if it has none."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _read_name(node: ast.AST, aliases: set[str]) -> str | None:
    """Returns the registry name a node reads, or None if it reads none.

    Args:
        node: Any AST node.
        aliases: Local names bound to ``vllm_torchtpu.envs``.

    Returns:
        The name read, ``<dynamic>`` for a non-literal ``getattr``, or None.
    """
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        owner = _dotted(node.value)
        if owner in aliases or owner == ENVS_MODULE:
            return node.attr
        return None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
        and _dotted(node.args[0]) in aliases | {ENVS_MODULE}
    ):
        name = node.args[1] if len(node.args) > 1 else None
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            return name.value
        return "<dynamic>"
    return None


def reads_in(tree: ast.Module) -> set[str]:
    """Returns every registry name a module reads.

    Args:
        tree: The parsed module.

    Returns:
        Names read through an alias, the dotted path, ``getattr`` or a
        ``from vllm_torchtpu.envs import NAME``.
    """
    aliases = _envs_aliases(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == ENVS_MODULE:
            names.update(a.name for a in node.names)
            continue
        name = _read_name(node, aliases)
        if name is not None:
            names.add(name)
    return names


def find_reads() -> list[str]:
    """Returns every read outside the producers as ``relpath:NAME`` entries."""
    found: set[str] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if path in PRODUCERS:
            continue
        source = path.read_text()
        # Every spelling counted names `envs`; skip parsing the rest.
        if "envs" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        rel = path.relative_to(REPO_ROOT).as_posix()
        found.update(f"{rel}:{name}" for name in reads_in(tree))
    return sorted(found)


def load_baseline() -> list[str]:
    """Returns the baseline entries, sorted, without comments."""
    if not BASELINE_PATH.exists():
        return []
    return sorted(
        line.strip()
        for line in BASELINE_PATH.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def write_baseline(entries: list[str]) -> None:
    """Rewrites the baseline file.

    Args:
        entries: Sorted ``relpath:NAME`` entries.
    """
    header = (
        "# envs.X reads outside the producer modules, as of the last update.\n"
        "# Generated by tools/check_envs_use_sites.py --update.\n"
        "# Shrink this file by passing the value in from a producer; never "
        "grow it.\n"
    )
    BASELINE_PATH.write_text(header + "\n".join(entries) + "\n")


def main() -> int:
    """Checks the tree against the baseline.

    Returns:
        0 when the tree matches the baseline, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the current tree",
    )
    # pre-commit passes the staged filenames; this hook always scans the
    # whole package, so accept and ignore them.
    _ = parser.add_argument("filenames", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()

    found = find_reads()
    if args.update:
        write_baseline(found)
        print(f"baseline updated: {len(found)} entries")
        return 0

    baseline = load_baseline()
    added = sorted(set(found) - set(baseline))
    removed = sorted(set(baseline) - set(found))

    if added:
        print("New envs.X reads outside the producer modules:\n")
        for entry in added:
            path, _, name = entry.rpartition(":")
            print(f"  {path}  reads envs.{name}")
        print(
            "\nRead the knob in a producer (envs.py, env_override.py or the "
            "decision or plan module that owns the choice) and pass the "
            "resulting value in.\nIf this module is itself the producer of "
            "that choice, add it to PRODUCERS in "
            "tools/check_envs_use_sites.py."
        )
        return 1

    if removed:
        print("These baseline entries no longer exist (nice):\n")
        for entry in removed:
            print(f"  {entry}")
        print(
            "\nRun `tools/check_envs_use_sites.py --update` to shrink the "
            "baseline, and commit it."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
