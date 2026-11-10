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
"""Registers a plugin subclass as the class an upstream layer builds as.

An expansion point turns an upstream layer into a decision boundary: the
layer's (inputs) -> output stays the contract, and everything inside it becomes
the replacement's plan. The hook mirrors vLLM's ``PluggableLayer.register_oot``
and takes no dotted-string targets: the decorated class names its upstream by
subclassing it, so the target is ``replacement.__mro__[1]``.

Each replacement declares its ``expansion_key`` (a
``patch_registry.ExpansionKey``), which names its owner entry. A guard checks
the upstream class the replacement was written against. Each replacement
declares the upstream ``__init__`` and ``forward`` signatures it expects; a
mismatch raises at registration. Parameter names are checked by
``check_parameter_names`` on an instance built on the meta device.
"""

import inspect
from collections.abc import Callable, Iterable
from typing import Any, overload

from torch import nn
from vllm.model_executor.custom_op import PluggableLayer

from vllm_torchtpu.patch_registry import ExpansionKey

# Upstream class -> replacement, for upstream layers that are not
# PluggableLayers. PluggableLayer upstreams register in vLLM's own
# op_registry_oot instead.
_expansion_registry: dict[type, type] = {}
# Every key registered, PluggableLayer or not; each needs an owner entry in
# patch_registry.EXPANSION_POINTS.
registered_keys: set[ExpansionKey] = set()


class ExpansionGuardError(RuntimeError):
    """The upstream layer no longer matches what the replacement expects."""


def _signature(fn: Callable[..., Any]) -> str:
    # Names, kinds and defaults; annotations are left out because they render
    # with the installed transformers/vLLM module paths.
    sig = inspect.signature(fn)
    empty = inspect.Parameter.empty
    params = [p.replace(annotation=empty) for p in sig.parameters.values()]
    return str(sig.replace(parameters=params, return_annotation=empty))


def _guard_signatures(upstream: type, replacement: type) -> None:
    for attr, expected in (
        ("__init__", replacement.upstream_init_signature),
        ("forward", replacement.upstream_forward_signature),
    ):
        actual = _signature(getattr(upstream, attr))
        if actual != expected:
            raise ExpansionGuardError(
                f"{upstream.__qualname__}.{attr} changed upstream; "
                f"{replacement.__qualname__} expects {expected}, found {actual}"
            )


def check_parameter_names(layer: nn.Module, expected: Iterable[str]) -> None:
    """Raises if an upstream-built layer lacks a parameter the plan relies on."""
    missing = set(expected) - {name for name, _ in layer.named_parameters()}
    if missing:
        raise ExpansionGuardError(
            f"{type(layer).__qualname__} lacks parameters {sorted(missing)}"
        )


def _declared_key(replacement: type) -> ExpansionKey:
    """The ``expansion_key`` the replacement class declares."""
    return replacement.expansion_key


def _install_new(upstream: type) -> None:
    # Same lookup as PluggableLayer.__new__, but only when building upstream
    # itself; other subclasses of upstream build as themselves.
    base_new = upstream.__new__

    def __new__(cls, *args, **kwargs):
        target = _expansion_registry[upstream] if cls is upstream else cls
        if base_new is object.__new__:
            return object.__new__(target)
        return base_new(target, *args, **kwargs)

    upstream.__new__ = __new__


def _register[T: type[nn.Module]](replacement: T) -> T:
    upstream = replacement.__mro__[1]
    key = _declared_key(replacement)
    _guard_signatures(upstream, replacement)
    registered_keys.add(key)
    if issubclass(upstream, PluggableLayer):
        # vLLM's registry is keyed by name: the key's value, at that boundary.
        return PluggableLayer.register_oot(name=key.value)(replacement)
    if upstream in _expansion_registry:
        raise ValueError(f"Duplicate expansion point: {key.value}")
    _expansion_registry[upstream] = replacement
    _install_new(upstream)
    return replacement


@overload
def expansion_point[T: type[nn.Module]](_replacement: T) -> T: ...
@overload
def expansion_point[T: type[nn.Module]](
    _replacement: None = None,
) -> Callable[[T], T]: ...
def expansion_point(_replacement=None) -> Any:
    """Registers a subclass of an upstream layer as that layer's replacement.

    Works as ``@expansion_point`` and ``@expansion_point()``. The replacement
    declares ``expansion_key``, which must have an owner entry in
    ``patch_registry.EXPANSION_POINTS``.
    """
    if _replacement is None:
        return _register
    if isinstance(_replacement, type):
        return _register(_replacement)
    raise TypeError("expansion_point can only decorate classes.")
