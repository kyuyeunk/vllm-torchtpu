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
"""Register a `pallas.jax_op` together with a fake built from an output spec.

`pallas.jax_op`'s default fake exports the JAX function to read its output
shapes, and refuses inputs with a symbolic dimension. vLLM traces the token
dimension symbolically, so every op on the forward path needs a hand-written
fake, and each one restates the kernel's output-shape rule in torch.

`jax_op_with_fake` takes that rule as a declared spec instead, for example::

    _op = jax_op_with_fake(
        "pallas::moe_router_topk",
        router_topk,
        out=(
            spec(shape=(dim("scores", 0), "topk"), dtype=torch.float32),
            spec(shape=(dim("scores", 0), "topk"), dtype=torch.int32),
        ),
    )

Specs name the JAX function's parameters, so they read like the kernel's own
signature. A spec can be wrong in the same way a hand-written fake can; tests
catch that with `check_fake_matches_kernel`, which compares the spec with
`jax.eval_shape` of the kernel on concrete shapes.
"""

import dataclasses
import inspect
import operator
from collections.abc import Callable, Mapping
from typing import Any, overload

import torch
from torch._library.custom_ops import CustomOpDef
from torch_tpu._internal import pallas


class Expr:
    """A value computed from the bound arguments of one op call.

    Supports ``+``, ``-``, ``*`` and ``//`` with ints, argument names and
    other expressions, so a shape entry can be ``dim("x", 0) * "topk"``.
    """

    def __init__(self, fn: Callable[[Mapping[str, Any]], Any], text: str) -> None:
        self._fn = fn
        self._text = text

    def evaluate(self, bound: Mapping[str, Any]) -> Any:
        """Returns the value for one call, given its arguments by name."""
        return self._fn(bound)

    def __repr__(self) -> str:
        return self._text

    def _binary(
        self,
        other: "Expr | int | str",
        op: Callable[[Any, Any], Any],
        symbol: str,
        *,
        reflected: bool = False,
    ) -> "Expr":
        rhs = _as_expr(other)
        lhs, rhs = (rhs, self) if reflected else (self, rhs)
        return Expr(
            lambda b: op(lhs.evaluate(b), rhs.evaluate(b)), f"({lhs} {symbol} {rhs})"
        )

    def __add__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.add, "+")

    def __radd__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.add, "+", reflected=True)

    def __sub__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.sub, "-")

    def __rsub__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.sub, "-", reflected=True)

    def __mul__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.mul, "*")

    def __rmul__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.mul, "*", reflected=True)

    def __floordiv__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.floordiv, "//")

    def __rfloordiv__(self, other: "Expr | int | str") -> "Expr":
        return self._binary(other, operator.floordiv, "//", reflected=True)


def _as_expr(value: "Expr | int | str") -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return static(value)
    # bool is an int subclass, and a bool in a shape is always a mistake.
    if isinstance(value, int) and not isinstance(value, bool):
        return Expr(lambda _: value, str(value))
    raise TypeError(f"Shape entries must be int, str or Expr, got {value!r}.")


def static(name: str) -> Expr:
    """The value of argument `name`, usually a static int.

    A bare string in a `spec` shape or in arithmetic means the same thing.
    """
    return Expr(lambda b: b[name], name)


def dim(arg: str, i: int) -> Expr:
    """Dimension `i` of tensor argument `arg`; negative `i` counts from the end."""
    return Expr(lambda b: b[arg].shape[i], f"{arg}.shape[{i}]")


def like_dtype(arg: str) -> Expr:
    """The dtype of tensor argument `arg`."""
    return Expr(lambda b: b[arg].dtype, f"{arg}.dtype")


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    """One output tensor of any rank.

    Attributes:
        shape: Either one entry per dimension (an int, the name of an
            argument, or an `Expr`), or a single `Expr` giving the whole shape.
        dtype: A torch dtype, or an `Expr` such as `like_dtype`.
    """

    shape: tuple[Expr | int | str, ...] | Expr
    dtype: torch.dtype | Expr

    def evaluate(self, bound: Mapping[str, Any]) -> tuple[tuple[Any, ...], Any]:
        """Returns ``(shape, dtype)`` for one call."""
        if isinstance(self.shape, Expr):
            shape = tuple(self.shape.evaluate(bound))
        else:
            shape = tuple(_as_expr(d).evaluate(bound) for d in self.shape)
        dtype = (
            self.dtype.evaluate(bound) if isinstance(self.dtype, Expr) else self.dtype
        )
        return shape, dtype


def spec(
    *, shape: tuple[Expr | int | str, ...], dtype: torch.dtype | Expr
) -> TensorSpec:
    """An output of the given shape and dtype; see `TensorSpec`."""
    return TensorSpec(shape=tuple(shape), dtype=dtype)


def like(arg: str) -> TensorSpec:
    """An output with the shape and dtype of tensor argument `arg`."""
    return TensorSpec(
        shape=Expr(lambda b: b[arg].shape, f"{arg}.shape"), dtype=like_dtype(arg)
    )


# One tensor, a tuple of specs (nested as the kernel returns its outputs), or a
# plain fake function, called with the op's arguments, for rules the helpers
# cannot express.
OutSpec = TensorSpec | tuple["OutSpec", ...] | Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class RegisteredOp:
    """What `jax_op_with_fake` recorded about one op.

    Attributes:
        fn: The JAX function the op wraps.
        signature: `fn`'s signature, which op calls are bound to.
        out: The declared output spec.
        fake: The fake registered for the op.
    """

    fn: Callable[..., Any]
    signature: inspect.Signature
    out: OutSpec
    fake: Callable[..., Any]


_REGISTERED: dict[CustomOpDef, RegisteredOp] = {}


def registered_op(op: CustomOpDef) -> RegisteredOp:
    """Returns the record of an op built with `jax_op_with_fake`."""
    try:
        return _REGISTERED[op]
    except KeyError:
        raise ValueError(f"{op} was not registered by jax_op_with_fake.") from None


def _signature(fn: Callable[..., Any]) -> inspect.Signature:
    # The same signature pallas.jax_op derives the op schema from, so the
    # fake's arguments bind to it.
    return inspect.signature(inspect.unwrap(fn), eval_str=True)


def _first_tensor_device(bound: Mapping[str, Any]) -> torch.device:
    for value in bound.values():
        for t in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(t, torch.Tensor):
                return t.device
    raise ValueError("A spec-built fake needs at least one tensor argument.")


def _build(
    out: OutSpec, bound: Mapping[str, Any], device: torch.device
) -> torch.Tensor | tuple[Any, ...]:
    if isinstance(out, TensorSpec):
        shape, dtype = out.evaluate(bound)
        return torch.empty(shape, dtype=dtype, device=device)
    if isinstance(out, tuple):
        return tuple(_build(o, bound, device) for o in out)
    raise TypeError(f"Unsupported output spec {out!r}.")


def make_fake(fn: Callable[..., Any], out: OutSpec) -> Callable[..., Any]:
    """Builds a fake kernel for `fn` from the output spec `out`.

    Args:
        fn: The JAX function; its parameter names are what specs refer to.
        out: The output spec. A plain callable is returned unchanged.

    Returns:
        A function taking the op's arguments and returning empty tensors of the
        declared shapes, on the device of the first tensor argument.
    """
    if not isinstance(out, (TensorSpec, tuple)):
        return out
    signature = _signature(fn)

    def fake(*args: Any, **kwargs: Any) -> torch.Tensor | tuple[Any, ...]:
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        return _build(out, call.arguments, _first_tensor_device(call.arguments))

    return fake


@overload
def jax_op_with_fake(
    name: str, fn: None = None, /, *, out: OutSpec, **jax_op_kwargs: Any
) -> Callable[[Callable[..., Any]], CustomOpDef]: ...


@overload
def jax_op_with_fake(
    name: str, fn: Callable[..., Any], /, *, out: OutSpec, **jax_op_kwargs: Any
) -> CustomOpDef: ...


def jax_op_with_fake(
    name: str,
    fn: Callable[..., Any] | None = None,
    /,
    *,
    out: OutSpec,
    **jax_op_kwargs: Any,
) -> CustomOpDef | Callable[[Callable[..., Any]], CustomOpDef]:
    """Registers `fn` as `pallas.jax_op(name, ...)` with a fake built from `out`.

    Usable as a call, ``jax_op_with_fake(name, fn, out=...)``, or as a
    decorator, ``@jax_op_with_fake(name, out=...)``.

    Args:
        name: The op name, ``"namespace::op"``.
        fn: The JAX function, or None to return a decorator.
        out: The output spec; see `TensorSpec`, `like`, `spec` and `OutSpec`.
        **jax_op_kwargs: Passed to `pallas.jax_op`, such as `mesh`.

    Returns:
        The registered op, or a decorator that registers and returns it.
    """

    def register(jax_fn: Callable[..., Any]) -> CustomOpDef:
        op = pallas.jax_op(name, jax_fn, **jax_op_kwargs)
        fake = make_fake(jax_fn, out)
        op.register_fake(fake)
        _REGISTERED[op] = RegisteredOp(
            fn=jax_fn,
            signature=_signature(jax_fn),
            out=out,
            fake=fake,
        )
        return op

    return register if fn is None else register(fn)
