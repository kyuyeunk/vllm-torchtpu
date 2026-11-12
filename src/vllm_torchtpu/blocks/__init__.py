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

``installer.install`` runs once after the weights load. For each module whose class is
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
import importlib.machinery
import inspect
import pathlib
import pkgutil
import typing
from collections.abc import Callable
from typing import cast

from torch import nn

from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.parts import BlockParts, PartsKindError, attribute_of

Pattern = Callable[..., object]
"""A pattern's reference forward, e.g. ``attn_then_moe``; it names the pattern."""

Binding = Callable[[nn.Module], BlockParts | None]
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
    pattern: Pattern
    parts_of: Binding

    @classmethod
    def of[L: nn.Module](
        cls, layer_class: type[L], parts_of: Callable[[L], BlockParts | None]
    ) -> "BlockDeclaration":
        """The declaration ``parts_of`` makes; its return type names the pattern."""
        signature = inspect.signature(parts_of, eval_str=True)
        returns = cast("object", signature.return_annotation)  # inspect types it Any
        kinds = [
            t
            for t in typing.get_args(returns) or (returns,)
            if isinstance(t, type) and issubclass(t, BlockParts)
        ]
        if len(kinds) != 1:
            msg = (
                f"{parts_of.__qualname__} must return one BlockParts dataclass "
                f"(or None), not {returns}"
            )
            raise TypeError(msg)
        # Called only on instances of ``layer_class``.
        return cls(layer_class, kinds[0].pattern, cast("Binding", parts_of))

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
            msg = f"{self.parts_of.__qualname__} returns None for this layer"
            raise BlockConformanceError(msg)
        return {
            field: attribute_of(layer, module, field)
            for field, module in parts.layer_children().items()
        }


DECLARATIONS: dict[tuple[type, Pattern], BlockDeclaration] = {}


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
            msg = (
                f"{layer_class.__qualname__} is already declared for "
                f"{declaration.pattern.__name__}"
            )
            raise ValueError(msg)
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
        finder = info.module_finder
        if not isinstance(finder, importlib.machinery.FileFinder):
            continue
        root = pathlib.Path(finder.path) / info.name
        if info.ispkg and (root / "registration.py").exists():
            _ = importlib.import_module(f"{package.__name__}.{info.name}.registration")
