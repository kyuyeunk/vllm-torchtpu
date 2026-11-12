r"""Flags string-based logic in Python code.

Framework code must not decide anything by comparing or indexing with string
literals. A string is parsed once, at the user boundary, into a typed value
(an enum, a dataclass, a plan-value object). This check flags:

* a comparison with a string constant (``x == "a"``, ``"a" in names``);
* a subscript or ``.get``/``.pop``/``.setdefault`` with a string-constant key
  (``parts["qkv_proj"]``, ``cfg.get("plan")``);
* ``getattr`` with a default (a missing attribute becomes a guess);
* ``Literal[...]`` made of strings (use an enum or typed objects);
* a ``match`` case on a string constant;
* a tuple of strings used as candidate parameter names
  (``next(p for p in ("hidden_states", "x") if p in params)``).

A line may opt out with ``# str-ok: <reason>``; the reason is required. Log
and error text is not flagged (string constants that are not compared or
used as keys are fine).

Usage:
    check_no_string_logic.py [--base REV] PATH...

With ``--base``, only lines added since ``REV`` (``git diff -U0``) are
checked, so untouched upstream code in a changed file is not reported.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_OPT_OUT = re.compile(r"#\s*str-ok:\s*\S")
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_MIN_NAME_TUPLE = 2
_KEYED_METHODS = frozenset({"get", "pop", "setdefault"})


@dataclass(frozen=True)
class Finding:
    """One flagged site."""

    path: str
    line: int
    rule: str


def _is_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _literal_of_strings(node: ast.Subscript) -> bool:
    name = node.value
    is_literal = (
        isinstance(name, ast.Name)
        and name.id == "Literal"  # str-ok: matching source names is the lint's job
    ) or (
        isinstance(name, ast.Attribute)
        and name.attr == "Literal"  # str-ok: matching source names is the lint's job
    )
    if not is_literal:
        return False
    elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    return any(_is_str(e) for e in elts)


def _name_tuple_in_generator(node: ast.comprehension) -> bool:
    it = node.iter
    return (
        isinstance(it, ast.Tuple)
        and len(it.elts) >= _MIN_NAME_TUPLE
        and all(_is_str(e) for e in it.elts)
        and bool(node.ifs)
    )


def _rule(node: ast.AST) -> str | None:  # noqa: PLR0911 - one return per rule
    if isinstance(node, ast.Compare) and any(
        _is_str(o) for o in (node.left, *node.comparators)
    ):
        return "compare-with-string"
    if isinstance(node, ast.Subscript):
        if _is_str(node.slice):
            return "string-key-subscript"
        if _literal_of_strings(node):
            return "string-literal-type"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"  # str-ok: matching source names is the lint's job
        and len(node.args) == 3  # noqa: PLR2004 - getattr(obj, name, default)
    ):
        return "getattr-default"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _KEYED_METHODS
        and node.args
        and _is_str(node.args[0])
    ):
        return "string-key-subscript"
    if isinstance(node, ast.MatchValue) and _is_str(node.value):
        return "match-on-string"
    if isinstance(node, ast.comprehension) and _name_tuple_in_generator(node):
        return "name-sniffing"
    return None


def _added_lines(path: Path, base: str) -> set[int]:
    diff = subprocess.run(  # noqa: S603 - fixed argv, path from the caller
        ["git", "diff", "-U0", base, "--", str(path)],  # noqa: S607 - git on PATH
        capture_output=True,
        text=True,
        check=True,
        cwd=path.parent,
    ).stdout
    lines: set[int] = set()
    for m in _HUNK.finditer(diff):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def check_file(path: Path, base: str | None) -> list[Finding]:
    """Returns the findings in one file (added lines only if ``base``)."""
    path = path.resolve()
    src = path.read_text()
    source_lines = src.splitlines()
    tree = ast.parse(src, filename=str(path))
    scope = _added_lines(path, base) if base else None
    findings: list[Finding] = []
    for node in ast.walk(tree):
        rule = _rule(node)
        line = getattr(node, "lineno", 0)  # str-ok: ast nodes vary in lineno
        if rule is None or line == 0:
            continue
        if scope is not None and line not in scope:
            continue
        if _OPT_OUT.search(source_lines[line - 1]):
            continue
        findings.append(Finding(str(path), line, rule))
    return findings


def main(argv: list[str]) -> int:
    """Command-line entry point; nonzero when anything is flagged."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--base", default=None)
    _ = parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    base: str | None = args.base  # pyright: ignore[reportAny] - argparse
    paths: list[Path] = args.paths  # pyright: ignore[reportAny] - argparse
    files = [
        p for path in paths for p in (path.rglob("*.py") if path.is_dir() else [path])
    ]
    files = [p for p in files if p.exists()]  # A diff may list deleted files.
    findings = [f for p in sorted(files) for f in check_file(p, base)]
    for f in findings:
        _ = sys.stdout.write(f"{f.path}:{f.line}: {f.rule}\n")
    _ = sys.stdout.write(f"{len(findings)} finding(s) in {len(files)} file(s)\n")
    return 1 if findings else 0


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(main(sys.argv[1:]))
