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
"""Block scope: decisions that span several children of one decoder layer.

Ownership is split at the declaration:

- **The model owner declares.** In the model's own directory
  (``models/<family>/<model>/registration.py``, the only place a model's
  identity appears) a binding function decorated with ``declare_block``
  takes a decoder layer and returns the pattern's parts dataclass, built by
  typed attribute access, or None where the layer does not conform. The
  return type names the pattern; the type checker checks the attributes.
- **The framework runs what is declared.** The pattern, the conformance
  check, the block plan, the child-contract adapter and the block forward
  live here, in a shared package. A declaration that does not conform fails
  the merge-time check (``tests/blocks``) and the startup install, with the
  role edge and source line.

``install`` runs once after the weights load. For each module whose class is
declared, it checks conformance, resolves the block's plan per compiled
bucket, and installs the block's forward where the plan needs it. Layers
whose plan is the default are left as upstream built them.

Scopes nest, so one decoder class may declare several patterns: a block
(``attn_then_moe``) and a sub-block inside it (``norm_then_attention``). An
implementation lists the patterns it nests (``nests``); where a layer declares
both, the outer one receives the inner declaration and builds the inner
runner itself, with the contract it needs, so the inner installer never runs
on that layer.
"""

import dataclasses
import importlib
import pathlib
import pkgutil
import typing
from collections.abc import Callable
from typing import Any, Protocol

from torch import nn

from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.parts import BlockParts, PartsKindError, attribute_of
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


Binding = Callable[[Any], BlockParts | None]
"""A model owner's binding function: layer -> the pattern's parts, or None."""


@dataclasses.dataclass(frozen=True)
class BlockDeclaration:
    """A model owner's statement that ``layer_class`` conforms to ``pattern``.

    Attributes:
        layer_class: The decoder layer class, matched exactly.
        pattern: The pattern's reference forward, read off the binding
            function's parts type.
        parts_of: The binding function.
    """

    layer_class: type
    pattern: Callable[..., Any]
    parts_of: Binding

    @classmethod
    def of(cls, layer_class: type, parts_of: Binding) -> "BlockDeclaration":
        """The declaration ``parts_of`` makes; its return type names the pattern."""
        returns = typing.get_type_hints(parts_of).get("return")
        kinds = [
            t
            for t in typing.get_args(returns) or (returns,)
            if isinstance(t, type) and issubclass(t, BlockParts)
        ]
        if len(kinds) != 1:
            raise TypeError(
                f"{parts_of.__qualname__} must return one BlockParts dataclass "
                f"(or None), not {returns}"
            )
        return cls(layer_class, kinds[0].pattern, parts_of)

    def parts(self, layer: nn.Module) -> BlockParts | None:
        """The layer's parts, or None where the declaration does not cover it.

        Raises:
            BlockConformanceError: A part is not the kind its field needs,
                named by its attribute on the layer.
        """
        try:
            return self.parts_of(layer)
        except PartsKindError as e:
            raise BlockConformanceError(e.describe(layer)) from e

    def when(self, layer: nn.Module) -> bool:
        """Whether the declaration covers this instance."""
        return self.parts(layer) is not None

    def bind(self, layer: nn.Module) -> dict[str, str]:
        """Each layer-level field's attribute name, recovered by identity.

        Raises:
            BlockConformanceError: The layer is not covered, or a part is not
                one of its children.
        """
        parts = self.parts(layer)
        if parts is None:
            raise BlockConformanceError(
                f"{self.parts_of.__qualname__} returns None for this layer"
            )
        return {
            field: attribute_of(layer, module, field)
            for field, module in parts.layer_children().items()
        }


DECLARATIONS: dict[tuple[type, Callable[..., Any]], BlockDeclaration] = {}


def declarations_for(layer_class: type) -> list[BlockDeclaration]:
    """Every pattern ``layer_class`` is declared to conform to."""
    return [d for (cls, _), d in DECLARATIONS.items() if cls is layer_class]


def declare_block[L: nn.Module, P: BlockParts](
    layer_class: type[L],
) -> Callable[[Callable[[L], P | None]], Callable[[L], P | None]]:
    """Declares a binding function for ``layer_class`` (the model owner's API).

    The function takes the layer, typed as ``layer_class``, so its attribute
    reads are type-checked against the model's class, and returns the
    pattern's parts dataclass, or None where the layer does not conform.
    """

    def register(parts_of: Callable[[L], P | None]) -> Callable[[L], P | None]:
        declaration = BlockDeclaration.of(layer_class, parts_of)
        key = (layer_class, declaration.pattern)
        if key in DECLARATIONS:
            raise ValueError(
                f"{layer_class.__qualname__} is already declared for "
                f"{declaration.pattern.__name__}"
            )
        DECLARATIONS[key] = declaration
        return parts_of

    return register


def load_declarations() -> None:
    """Imports every ``models/vllm/<model>/registration.py``, without naming one.

    Looks for the file on disk first, so model packages without declarations
    are not imported.
    """
    package = importlib.import_module("vllm_torchtpu.models.vllm")
    for info in pkgutil.iter_modules(package.__path__):
        root = pathlib.Path(info.module_finder.path) / info.name  # type: ignore[union-attr]
        if info.ispkg and (root / "registration.py").exists():
            importlib.import_module(f"{package.__name__}.{info.name}.registration")


@dataclasses.dataclass(frozen=True)
class InstallContext:
    """What every implementation reads at install time, passed as data."""

    tp: int
    dtype: str
    buckets: tuple[int, ...]
    override: str
    model_plan: str
    subblock_plan: str = ""
    # The chip's `pltpu.ChipVersion` value; empty off-TPU (no chip facts).
    chip: str = ""
    # vLLM's KV cache dtype (``cache_dtype``); empty when unknown. The TPU
    # runner leaves it empty while the fused value is the plain-torch
    # reference, which writes no KV cache; a kernel build passes it.
    kv_cache_dtype: str = ""


@dataclasses.dataclass(frozen=True)
class Installed:
    name: str
    layer_class: str
    outcome: str


class BlockImplementation(Protocol):
    nests: tuple[Callable[..., Any], ...]
    """Patterns this one builds inside itself when a layer declares both."""

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
        nested: dict[Callable[..., Any], BlockDeclaration],
    ) -> Installed: ...


IMPLEMENTATIONS: dict[Callable[..., Any], BlockImplementation] = {}


def block_implementation(
    pattern: Callable[..., Any],
) -> Callable[[type[Any]], type[Any]]:
    """Registers the framework's implementation of ``pattern``."""

    def register(cls: type[Any]) -> type[Any]:
        IMPLEMENTATIONS[pattern] = cls()
        return cls

    return register


def install(model: nn.Module, context: InstallContext) -> list[Installed]:
    """Checks and installs every declared block in ``model``.

    A declaration is the model owner's explicit claim, so a layer that does
    not conform to its declared pattern stops startup, naming the edge and
    line. A pinned plan value whose capability does not cover a conforming
    layer stops startup too; a value the rule picked falls back to the next
    capable one (the default) quietly.
    """
    from vllm_torchtpu.blocks import attn_moe, norm_attn  # noqa: F401 - register

    load_declarations()
    report = []
    for name, module in list(model.named_modules()):
        cls = type(module)
        try:
            declared = [d for d in declarations_for(cls) if d.when(module)]
        except BlockConformanceError as e:
            raise BlockConformanceError(f"{name} ({cls.__qualname__}): {e}") from e
        inner = {p for d in declared for p in IMPLEMENTATIONS[d.pattern].nests}
        nested = {d.pattern: d for d in declared if d.pattern in inner}
        for declaration in declared:
            if declaration.pattern in inner:
                continue  # The enclosing pattern's installer builds it.
            impl = IMPLEMENTATIONS[declaration.pattern]
            try:
                report.append(impl.install(name, module, declaration, context, nested))
            except BlockConformanceError as e:
                raise BlockConformanceError(
                    f"{name} ({cls.__qualname__}) is declared "
                    f"{declaration.pattern.__name__}: {e}"
                ) from e
    counts: dict[str, int] = {}
    for item in report:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    for outcome, count in sorted(counts.items()):
        logger.info("block install: %d layer(s): %s", count, outcome)
    return report
