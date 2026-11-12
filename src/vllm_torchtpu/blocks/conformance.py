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

import dataclasses
import hashlib
import inspect
import operator
import re
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import override

import torch
from torch import fx, nn

Tensor = torch.Tensor

_LIB = torch.library.Library("blockrole", "DEF")
_LIB.define("pair(Tensor x, Tensor residual, str role) -> (Tensor, Tensor)")  # pyright: ignore[reportUnknownMemberType] - torch.library is untyped
_LIB.define("unary(Tensor x, str role) -> Tensor")  # pyright: ignore[reportUnknownMemberType] - torch.library is untyped
_LIB.define("binary(Tensor x, Tensor y, str role) -> Tensor")  # pyright: ignore[reportUnknownMemberType] - torch.library is untyped
_LIB.define("project(Tensor x, int width, str role) -> Tensor")  # pyright: ignore[reportUnknownMemberType] - torch.library is untyped


@torch.library.impl(_LIB, "pair", "CompositeExplicitAutograd")
def _pair(x: Tensor, residual: Tensor, role: str) -> tuple[Tensor, Tensor]:
    del role
    return x.clone(), residual.clone()


@torch.library.impl(_LIB, "unary", "CompositeExplicitAutograd")
def _unary(x: Tensor, role: str) -> Tensor:
    del role
    return x.clone()


@torch.library.impl(_LIB, "binary", "CompositeExplicitAutograd")
def _binary(x: Tensor, y: Tensor, role: str) -> Tensor:
    del y, role
    return x.clone()


@torch.library.impl(_LIB, "project", "CompositeExplicitAutograd")
def _project(x: Tensor, width: int, role: str) -> Tensor:
    del role
    return x.new_empty((*x.shape[:-1], width))


for _name, _fake in (
    ("blockrole::pair", _pair),
    ("blockrole::unary", _unary),
    ("blockrole::binary", _binary),
    ("blockrole::project", _project),
):
    _ = torch.library.register_fake(_name)(_fake)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] - torch.library is untyped

# torch.ops is untyped; these are the typed boundary over the four ops.
_UNARY: Callable[[Tensor, str], object] = torch.ops.blockrole.unary
_BINARY: Callable[[Tensor, Tensor, str], object] = torch.ops.blockrole.binary
_PAIR: Callable[[Tensor, Tensor, str], tuple[object, object]] = torch.ops.blockrole.pair
_PROJECT: Callable[[Tensor, int, str], object] = torch.ops.blockrole.project
_ROLE_OPS: frozenset[object] = frozenset(
    {
        torch.ops.blockrole.pair.default,
        torch.ops.blockrole.unary.default,
        torch.ops.blockrole.binary.default,
        torch.ops.blockrole.project.default,
    }
)


def _tensor(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(type(value).__name__)
    return value


def unary(x: Tensor, label: str) -> Tensor:
    """An opaque op labelled ``label``: ``x -> x``'s shape."""
    return _tensor(_UNARY(x, label))


def binary(x: Tensor, y: Tensor, label: str) -> Tensor:
    """An opaque op of two inputs labelled ``label``, ``x``'s shape."""
    return _tensor(_BINARY(x, y, label))


def pair(x: Tensor, residual: Tensor, label: str) -> tuple[Tensor, Tensor]:
    """An opaque ``(x, residual) -> (x, residual)`` labelled ``label``."""
    first, second = _PAIR(x, residual, label)
    return _tensor(first), _tensor(second)


def project(x: Tensor, width: int, label: str) -> Tensor:
    """An opaque projection to ``width`` labelled ``label``."""
    return _tensor(_PROJECT(x, width, label))


class BlockConformanceError(ValueError):
    """A declared block does not compute its pattern."""


class Opaque:
    """Replaces ``module.forward`` with ``stub`` while exporting."""

    def __init__(self, module: object, stub: Callable[..., object]) -> None:
        """Holds the module and the stub; ``__enter__`` swaps them."""
        if not isinstance(module, nn.Module):
            msg = f"an opaque part must be a module, not a {type(module).__name__}"
            raise BlockConformanceError(msg)
        self.module: nn.Module = module
        self.stub: Callable[..., object] = stub
        self.previous: Callable[..., object] = module.forward

    def __enter__(self) -> None:
        """Installs the stub as the instance's forward."""
        self.previous = self.module.forward
        self.module.forward = self.stub

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Restores what the instance's forward was: its class's, or its own."""
        previous = self.previous
        if inspect.ismethod(previous) and previous.__self__ is self.module:
            del self.module.forward  # The class's forward again.
        else:
            self.module.forward = previous


def opaque(module: object, stub: Callable[..., object]) -> Opaque:
    """A context in which ``stub`` runs as ``module``'s forward."""
    return Opaque(module, stub)


class _Call(nn.Module):
    def __init__(self, fn: Callable[..., object], owner: nn.Module | None) -> None:
        super().__init__()
        self.fn: Callable[..., object] = fn
        self.owner: nn.Module | None = owner  # registers its parameters, if any

    @override
    def forward(self, *args: Tensor) -> object:
        return self.fn(*args)


def export(
    fn: Callable[..., object], args: tuple[Tensor, ...], owner: nn.Module | None
) -> torch.export.ExportedProgram:
    """Exports ``fn(*args)`` non-strictly and functionalizes it."""
    try:
        with torch.no_grad():
            program = torch.export.export(_Call(fn, owner), args, strict=False)
    except BlockConformanceError:
        raise
    except Exception as e:
        message = str(e).splitlines()[0][:200]
        msg = f"the forward does not export: {message}"
        raise BlockConformanceError(msg) from e
    decomposed: torch.export.ExportedProgram = program.run_decompositions({})  # pyright: ignore[reportUnknownMemberType] - typed by the annotation
    return decomposed


# ---------------------------------------------------------------------------
# Canonical graphs.

_SHAPE_ONLY: frozenset[object] = frozenset(
    {
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,  # noqa: SLF001 - an aten op's own name
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
    }
)
_COMMUTATIVE: frozenset[object] = frozenset(
    {torch.ops.aten.mul.Tensor, torch.ops.aten.add.Tensor}
)
_FRAME = re.compile(r'File "([^"]+)", line (\d+), in (\S+)\n\s*(.*)')


def _h(*binding: object) -> str:
    return hashlib.blake2b(repr(binding).encode(), digest_size=10).hexdigest()


def _items(seq: tuple[object, ...] | list[object]) -> list[object]:
    return list(seq)


def _meta_of(val: object) -> object:
    if isinstance(val, Tensor):
        return (tuple(int(d) for d in val.shape), str(val.dtype))
    if isinstance(val, (tuple, list)):
        return tuple(_meta_of(v) for v in _items(val))  # pyright: ignore[reportUnknownArgumentType] - any items
    return None


def _meta(node: fx.Node) -> object:
    val: object = node.meta.get("val")  # str-ok: torch.fx's meta key
    return _meta_of(val)


def _const(arg: object) -> object:
    if isinstance(arg, fx.Node):
        return "%"
    if isinstance(arg, (tuple, list)):
        return tuple(_const(a) for a in _items(arg))  # pyright: ignore[reportUnknownArgumentType] - any items
    if isinstance(arg, (torch.dtype, torch.device)):
        return str(arg)
    return arg


def _operands(node: fx.Node) -> list[fx.Node]:
    found: list[fx.Node] = []
    _ = fx.map_arg((node.args, node.kwargs), found.append)
    return found


def _nodes(graph: fx.Graph) -> list[fx.Node]:
    nodes: list[fx.Node] = list(graph.nodes)
    return nodes


def role_of(node: fx.Node) -> str | None:
    """The role a ``blockrole`` op stands for, else None."""
    if node.target in _ROLE_OPS:
        return str(node.args[-1])
    return None


def _is_input(node: fx.Node) -> bool:
    return node.op == "placeholder"  # str-ok: torch.fx names node kinds by string


def _is_output(node: fx.Node) -> bool:
    return node.op == "output"  # str-ok: torch.fx names node kinds by string


@dataclasses.dataclass
class Canon:
    """A canonical graph: kept nodes, their hashes and the output's."""

    nodes: list[fx.Node]
    fwd: dict[fx.Node, str]
    bwd: dict[fx.Node, str]
    output: str
    inputs: set[str]


def canonicalize(  # noqa: C901, PLR0912 - one pass over the graph; each branch is a node kind
    program: torch.export.ExportedProgram, input_names: Sequence[str]
) -> Canon:
    """The canonical graph of an exported program (see the module docstring)."""
    user_inputs = [str(name) for name in program.graph_signature.user_inputs]
    roles = dict(zip(user_inputs, input_names, strict=True))
    nodes = _nodes(program.graph)
    out = next(n for n in nodes if _is_output(n))
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
    for n in nodes:
        if n not in live or _is_output(n):
            continue
        target: object = n.target
        ins = [rep[a] for a in _operands(n) if a in rep]
        if target in _SHAPE_ONLY and ins and _meta(ins[0]) == _meta(n):
            rep[n] = ins[0]
            continue
        if _is_input(n):
            label = f"input:{roles[n.name]}" if n.name in roles else f"state:{n.name}"
        elif target is operator.getitem:
            label = f"getitem[{n.args[1]}]"
        elif target in _SHAPE_ONLY:
            # A reshape is its result's shape: view(t, 4, d) and view(-1, 4, d)
            # are the same op.
            label = _h(str(target), _meta(n))
        else:
            label = _h(str(target), _const(n.args), _const(n.kwargs), _meta(n))
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
                users[rep[a]].append((n, -1 if n.target in _COMMUTATIVE else i))
    out_args = [rep[a] for a in _operands(out)]
    bwd: dict[fx.Node, str] = {}
    for n in reversed(kept):
        outs = tuple(i for i, a in enumerate(out_args) if a is n)
        bwd[n] = _h(fwd[n], outs, tuple(sorted((bwd[u], i) for u, i in users[n])))
    output = _h("output", tuple(fwd[a] for a in out_args))
    return Canon(kept, {n: fwd[n] for n in kept}, bwd, output, set(roles))


def site(node: fx.Node) -> str:
    """``file:line`` of the innermost frame outside torch and this module."""
    trace: object = node.meta.get("stack_trace")  # str-ok: torch.fx's meta key
    frames: list[tuple[str, str, str, str]] = _FRAME.findall(
        trace if isinstance(trace, str) else ""
    )
    for file, line, _, _ in reversed(frames):
        own = file.endswith("conformance.py")  # str-ok: a stack frame's file path
        if "/torch/" not in file and not own:  # str-ok: a stack frame's file path
            return f"{file.rsplit('/', 1)[-1]}:{line}"
    return "unknown site"


def _near(node: fx.Node, *, up: bool, inputs: set[str]) -> str | None:
    """The nearest role before (``up``) or after the node, by graph distance.

    Returns "the block input" or "the block output" at the boundary, and None
    when, going up, only parameters and buffers are reached (the node is not
    on the stream).
    """
    seen, frontier = {node}, [node]
    found_input = False
    while frontier:
        nxt: list[fx.Node] = []
        for n in frontier:
            for m in _operands(n) if up else list(n.users):
                if m in seen:
                    continue
                seen.add(m)
                if (r := role_of(m)) is not None:
                    return r
                if _is_input(m):
                    found_input |= m.name in inputs
                elif _is_output(m):
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
        if module.fwd[n] not in pf and module.bwd[n] not in pb and not _is_input(n)
    ]
    prefix = f"declared {pattern_name}{where}, but"
    # Report the first extra node on the stream; a parameter's own reshaping
    # (``scale[0] + 1``) only matters where it meets the stream.
    on_stream = [
        n for n in extra if _near(n, up=True, inputs=module.inputs) is not None
    ]
    if on_stream:
        n = on_stream[0]
        before = _near(n, up=True, inputs=module.inputs)
        if (role := role_of(n)) is not None:
            return (
                f"{prefix} role {role} runs after {before} at {site(n)}, "
                "where the pattern does not call it"
            )
        name = ".".join(str(n.target).split(".")[:2])
        return (
            f"{prefix} extra op {name} between {before} and "
            f"{_near(n, up=False, inputs=module.inputs)} at {site(n)}"
        )
    mf, mb = set(module.fwd.values()), set(module.bwd.values())
    missing = [
        n
        for n in pattern.nodes
        if pattern.fwd[n] not in mf and pattern.bwd[n] not in mb
    ]
    what = role_of(missing[0]) if missing else None
    return f"{prefix} the forward never calls {what or 'what the pattern computes'}"
