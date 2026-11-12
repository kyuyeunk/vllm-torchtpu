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
"""Does a module's forward compute its declared pattern? Graph comparison.

Framework-owned. Each bound role is replaced by an opaque custom op
(``blockrole::*``, with a fake implementation), so a role's internals never
enter the graph. The module's forward and the pattern's reference forward
are exported with non-strict ``torch.export`` on meta tensors, functionalized,
and canonicalized:

- names and metadata go; a node is its op, constant arguments (the role name
  of a ``blockrole`` op is one) and output shape and dtype;
- dead nodes go, so a branch that computes nothing returned does not count;
- shape-only ops whose output has their input's shape and dtype (a view to
  the same shape, a contiguous, a clone) fold into their input. A slice that
  happens to take the whole tensor (``aten.alias``) does not: on another
  config it truncates;
- every node gets a forward hash over its ancestry, with the operands of
  commutative ops sorted, and a backward hash over its paths to the output.

The graphs conform when their outputs' hashes agree. Otherwise the first
node of the module's graph that the pattern lacks is reported with the roles
around it and its source line, like a compiler error.
"""

import contextlib
import dataclasses
import hashlib
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch import fx, nn

_LIB = torch.library.Library("blockrole", "DEF")
_LIB.define("pair(Tensor x, Tensor residual, str role) -> (Tensor, Tensor)")
_LIB.define("unary(Tensor x, str role) -> Tensor")
_LIB.define("binary(Tensor x, Tensor y, str role) -> Tensor")
_LIB.define("project(Tensor x, int width, str role) -> Tensor")


@torch.library.impl(_LIB, "pair", "CompositeExplicitAutograd")
def _pair(
    x: torch.Tensor, residual: torch.Tensor, role: str
) -> tuple[torch.Tensor, torch.Tensor]:
    del role
    return x.clone(), residual.clone()


@torch.library.impl(_LIB, "unary", "CompositeExplicitAutograd")
def _unary(x: torch.Tensor, role: str) -> torch.Tensor:
    del role
    return x.clone()


@torch.library.impl(_LIB, "binary", "CompositeExplicitAutograd")
def _binary(x: torch.Tensor, y: torch.Tensor, role: str) -> torch.Tensor:
    del y, role
    return x.clone()


@torch.library.impl(_LIB, "project", "CompositeExplicitAutograd")
def _project(x: torch.Tensor, width: int, role: str) -> torch.Tensor:
    del role
    return x.new_empty((*x.shape[:-1], width))


torch.library.register_fake("blockrole::pair")(_pair)
torch.library.register_fake("blockrole::unary")(_unary)
torch.library.register_fake("blockrole::binary")(_binary)
torch.library.register_fake("blockrole::project")(_project)

ops = torch.ops.blockrole


class BlockConformanceError(ValueError):
    """A declared block does not compute its pattern."""


@contextlib.contextmanager
def opaque(module: nn.Module, stub: Callable[..., Any]) -> Iterator[None]:
    """Replaces ``module.forward`` with ``stub`` while exporting."""
    own = module.__dict__.get("forward")
    module.forward = stub
    try:
        yield
    finally:
        if own is None:
            del module.forward
        else:
            module.forward = own


class _Call(nn.Module):
    def __init__(self, fn: Callable[..., Any], owner: nn.Module | None) -> None:
        super().__init__()
        self.fn = fn
        self.owner = owner  # registers the module's parameters, if any

    def forward(self, *args: torch.Tensor) -> Any:  # noqa: ANN401
        return self.fn(*args)


def export(
    fn: Callable[..., Any], args: tuple[torch.Tensor, ...], owner: nn.Module | None
) -> torch.export.ExportedProgram:
    """Exports ``fn(*args)`` non-strictly and functionalizes it."""
    try:
        with torch.no_grad():
            program = torch.export.export(_Call(fn, owner), args, strict=False)
    except BlockConformanceError:
        raise
    except Exception as e:  # noqa: BLE001 - any trace failure is a non-conformance
        message = str(e).splitlines()[0][:200]
        raise BlockConformanceError(f"the forward does not export: {message}") from e
    return program.run_decompositions({})


# ---------------------------------------------------------------------------
# Canonical graphs.

_SHAPE_ONLY = {
    "aten.view.default",
    "aten.reshape.default",
    "aten._unsafe_view.default",
    "aten.clone.default",
    "aten.contiguous.default",
}
_COMMUTATIVE = {"aten.mul.Tensor", "aten.add.Tensor"}
_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)')


def _h(*parts: object) -> str:
    return hashlib.blake2b(repr(parts).encode(), digest_size=10).hexdigest()


def _meta(node: fx.Node) -> object:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return (tuple(int(d) for d in val.shape), str(val.dtype))
    if isinstance(val, (tuple, list)):
        return tuple(_meta_of(v) for v in val)
    return None


def _meta_of(val: Any) -> object:  # noqa: ANN401
    if isinstance(val, torch.Tensor):
        return (tuple(int(d) for d in val.shape), str(val.dtype))
    return None


def _const(arg: Any) -> object:  # noqa: ANN401
    if isinstance(arg, fx.Node):
        return "%"
    if isinstance(arg, (tuple, list)):
        return tuple(_const(a) for a in arg)
    if isinstance(arg, (torch.dtype, torch.device)):
        return str(arg)
    return arg


def _operands(node: fx.Node) -> list[fx.Node]:
    found: list[fx.Node] = []
    fx.node.map_arg((node.args, node.kwargs), found.append)
    return found


def role_of(node: fx.Node) -> str | None:
    """The role a ``blockrole`` op stands for, else None."""
    if str(node.target).startswith("blockrole."):
        return str(node.args[-1])
    return None


@dataclasses.dataclass
class Canon:
    nodes: list[fx.Node]
    fwd: dict[fx.Node, str]
    bwd: dict[fx.Node, str]
    output: str
    inputs: set[str]


def canonicalize(
    program: torch.export.ExportedProgram, input_names: Sequence[str]
) -> Canon:
    """The canonical graph of an exported program (see the module docstring)."""
    sig = program.graph_signature
    roles = dict(zip(sig.user_inputs, input_names, strict=True))
    graph = program.graph
    out = next(n for n in graph.nodes if n.op == "output")
    live: set[fx.Node] = set()
    stack = [out]
    while stack:
        n = stack.pop()
        if n not in live:
            live.add(n)
            stack.extend(_operands(n))
    rep: dict[fx.Node, fx.Node] = {}
    fwd: dict[fx.Node, str] = {}
    kept: list[fx.Node] = []
    first: dict[str, fx.Node] = {}
    for n in graph.nodes:
        if n not in live or n.op == "output":
            continue
        target = str(n.target)
        ins = [rep[a] for a in _operands(n) if a in rep]
        if target in _SHAPE_ONLY and ins and _meta(ins[0]) == _meta(n):
            rep[n] = ins[0]
            continue
        if n.op == "placeholder":
            label = f"input:{roles[n.name]}" if n.name in roles else f"state:{n.name}"
        elif getattr(n.target, "__name__", "") == "getitem":
            label = f"getitem[{n.args[1]}]"
        elif target in _SHAPE_ONLY:
            # A reshape is its result's shape: view(t, 4, d) and view(-1, 4, d)
            # are the same op.
            label = _h(target, _meta(n))
        else:
            label = _h(target, _const(n.args), _const(n.kwargs), _meta(n))
        hashes = [fwd[a] for a in ins]
        if target in _COMMUTATIVE:
            hashes = sorted(hashes)
        fwd[n] = _h(label, tuple(hashes))
        rep[n] = first.setdefault(fwd[n], n)
        if rep[n] is n:
            kept.append(n)
    users: dict[fx.Node, list[tuple[fx.Node, int]]] = {n: [] for n in kept}
    for n in kept:
        for i, a in enumerate(_operands(n)):
            if a in rep and rep[a] in users:
                users[rep[a]].append((n, -1 if str(n.target) in _COMMUTATIVE else i))
    out_args = [rep[a] for a in _operands(out)]
    bwd: dict[fx.Node, str] = {}
    for n in reversed(kept):
        outs = tuple(i for i, a in enumerate(out_args) if a is n)
        bwd[n] = _h(fwd[n], outs, tuple(sorted((bwd[u], i) for u, i in users[n])))
    output = _h("output", tuple(fwd[a] for a in out_args))
    return Canon(kept, {n: fwd[n] for n in kept}, bwd, output, set(roles))


def site(node: fx.Node) -> str:
    """``file:line`` of the innermost frame outside torch and this module."""
    frames = _FRAME.findall(node.meta.get("stack_trace") or "")
    for file, line, _, _ in reversed(frames):
        if "/torch/" not in file and not file.endswith("conformance.py"):
            return f"{file.rsplit('/', 1)[-1]}:{line}"
    return "unknown site"


def _near(node: fx.Node, up: bool, inputs: set[str]) -> str | None:
    """The nearest role before (``up``) or after the node, by graph distance.

    Returns "the block input" or "the block output" at the boundary, and None
    when, going up, only parameters and buffers are reached (the node is not
    on the stream).
    """
    seen, frontier = {node}, [node]
    found_input = False
    while frontier:
        nxt = []
        for n in frontier:
            for m in _operands(n) if up else list(n.users):
                if m in seen:
                    continue
                seen.add(m)
                if (r := role_of(m)) is not None:
                    return r
                if m.op == "placeholder":
                    found_input |= m.name in inputs
                elif m.op == "output":
                    return "the block output"
                else:
                    nxt.append(m)
        frontier = nxt
    if up:
        return "the block input" if found_input else None
    return "the block output"


def compare(
    pattern_name: str, module: Canon, pattern: Canon, where: str = ""
) -> str | None:
    """None when the graphs conform, else a compiler-style error message."""
    if module.output == pattern.output and len(module.nodes) == len(pattern.nodes):
        return None
    pf, pb = set(pattern.fwd.values()), set(pattern.bwd.values())
    extra = [
        n
        for n in module.nodes
        if module.fwd[n] not in pf and module.bwd[n] not in pb and n.op != "placeholder"
    ]
    prefix = f"declared {pattern_name}{where}, but"
    # Report the first extra node on the stream; a parameter's own reshaping
    # (``scale[0] + 1``) only matters where it meets the stream.
    on_stream = [n for n in extra if _near(n, True, module.inputs) is not None]
    if on_stream:
        n = on_stream[0]
        before = _near(n, True, module.inputs)
        if (role := role_of(n)) is not None:
            return (
                f"{prefix} role {role} runs after {before} at {site(n)}, "
                "where the pattern does not call it"
            )
        name = ".".join(str(n.target).split(".")[:2])
        return (
            f"{prefix} extra op {name} between {before} and "
            f"{_near(n, False, module.inputs)} at {site(n)}"
        )
    mf, mb = set(module.fwd.values()), set(module.bwd.values())
    missing = [
        n
        for n in pattern.nodes
        if pattern.fwd[n] not in mf and pattern.bwd[n] not in mb
    ]
    what = role_of(missing[0]) if missing else None
    return f"{prefix} the forward never calls {what or 'what the pattern computes'}"
