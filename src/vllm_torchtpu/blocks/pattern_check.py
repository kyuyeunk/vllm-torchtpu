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
"""Does a declared layer's forward compute its pattern? One trace for any pattern.

Everything comes from the pattern and the declaration, so no pattern is
named here:

- **Roles.** The pattern's keyword arguments are its roles; the declared
  ``Roles`` has one field per keyword.
- **Opaque ops.** Each role's module runs as one opaque op on every tensor
  it is called with, in the order of its forward's parameters, so a call
  spelled by position and one spelled by keyword are the same op. The op
  returns as many tensors as the pattern's annotation for the role does
  (``Norm`` two, ``Op`` one).
- **Two traces, one set of ops.** The layer's forward runs through the
  declared ``layer_call``; the pattern runs on the same inputs, calling each
  role as the declaration says the layer calls it (``CalledRole.call``,
  which passes the positions). The graphs must agree, so an extra input, a
  missing positions argument or a call another way than declared is an edge
  the comparison names, with its source line.

The inputs are a decoder layer's, ``(hidden, residual, positions)``, as
``Stream`` and ``Attend`` fix them: the pattern takes the first two, and a
role's declared call adds the positions.
"""

import contextlib
import inspect
import typing
from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.roles import BlockRoles, CalledRole, ModuleRole

Tensor = torch.Tensor
TOKENS = 8
INPUTS = ("hidden", "residual", "positions")


def layer_key(layer: nn.Module, roles_of: object) -> tuple[object, ...]:
    """What one check covers: layers this key equates check alike.

    Layers of one class differ in flags their forward branches on
    (layer_type, layer_scale) and, built at other sizes, in their shapes.
    """
    state: dict[str, object] = vars(layer)
    flags = tuple(
        sorted((k, v) for k, v in state.items() if isinstance(v, (bool, str)))
    )
    shapes = tuple(tuple(t.shape) for t in layer.parameters())
    return (type(layer), roles_of, flags, shapes)


def _role_names(pattern: Callable[..., object]) -> list[str]:
    """The pattern's roles: its keyword-only parameters."""
    params = inspect.signature(pattern).parameters.values()
    return [p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]


def _outputs(annotation: object) -> int:
    """How many tensors a role's annotated callable returns."""
    params_and_returns: tuple[object, ...] = typing.get_args(annotation)
    returns = params_and_returns[-1]
    if typing.get_origin(returns) is tuple:
        items: tuple[object, ...] = typing.get_args(returns)
        return len(items)
    return 1


def _module_of(role: object) -> nn.Module:
    if isinstance(role, ModuleRole):
        return role.module
    if isinstance(role, nn.Module):
        return role
    msg = f"a role must be a module or carry one, not a {type(role).__name__}"
    raise conf.BlockConformanceError(msg)


def _tensors(
    module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> list[Tensor]:
    """The call's tensors in the order of ``module``'s forward parameters."""
    signature = inspect.signature(type(module).forward)
    bound: list[object] = list(
        signature.bind(module, *args, **kwargs).arguments.values()
    )
    found: list[Tensor] = []
    for value in bound[1:]:  # Past ``self``.
        if isinstance(value, tuple):  # ``*args``
            items = cast("tuple[object, ...]", value)
        elif isinstance(value, dict):  # ``**kwargs``
            items = tuple(cast("dict[str, object]", value).values())
        else:
            items = (value,)
        found += [v for v in items if isinstance(v, Tensor)]
    return found


def _stub(module: nn.Module, label: str, outputs: int) -> Callable[..., object]:
    """The role's opaque op, as ``module``'s forward."""

    def stub(*args: object, **kwargs: object) -> object:
        xs = _tensors(module, args, kwargs)
        out = tuple(conf.output(xs, i, label) for i in range(outputs))
        return out[0] if outputs == 1 else out

    return stub


def _as_called(role: object, positions: Tensor) -> object:
    """The role as the pattern calls it: through the declared call, if any."""
    if not isinstance(role, CalledRole):
        return _module_of(role)
    call = role.call

    def called(x: Tensor) -> Tensor:
        return call(x, positions)

    return called


def check_pattern(
    layer: nn.Module, pattern: Callable[..., object], roles: BlockRoles, hidden: int
) -> None:
    """Checks that ``layer``'s forward computes ``pattern`` over ``roles``.

    Args:
        layer: The declared layer.
        pattern: The pattern's reference forward, ``(hidden, residual,
            *, role...) -> (hidden, residual)``.
        roles: The declaration's roles for ``layer``.
        hidden: The residual stream's width.

    Raises:
        BlockConformanceError: Naming the edge and its source line.
    """
    names = _role_names(pattern)
    hints: dict[str, object] = typing.get_type_hints(pattern)
    values: dict[str, object] = {name: getattr(roles, name) for name in names}
    stubs = [
        (module, _stub(module, name, _outputs(hints[name])))
        for name, module in ((n, _module_of(v)) for n, v in values.items())
    ]

    def layer_fn(h: Tensor, r: Tensor, p: Tensor) -> object:
        return roles.layer_call.call(h, r, p)

    def reference_fn(h: Tensor, r: Tensor, p: Tensor) -> object:
        return pattern(h, r, **{n: _as_called(v, p) for n, v in values.items()})

    x = torch.empty(TOKENS, hidden, device="meta", dtype=torch.bfloat16)
    args = (
        x,
        torch.empty_like(x),
        torch.empty(TOKENS, device="meta", dtype=torch.long),
    )
    with contextlib.ExitStack() as stack:
        for module, stub in stubs:
            stack.enter_context(conf.opaque(module, stub))
        got = conf.canonicalize(conf.export(layer_fn, args, layer), INPUTS)
        want = conf.canonicalize(conf.export(reference_fn, args, None), INPUTS)
    if error := conf.compare(pattern.__name__, got, want):
        raise conf.BlockConformanceError(error)
