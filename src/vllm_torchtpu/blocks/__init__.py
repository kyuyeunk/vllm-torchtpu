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
  identity appears) a function decorated with ``@Pattern.register(Layer)``
  (``BlockPattern.register``) takes a decoder layer and returns the
  pattern's ``Roles``, one keyword per role of the pattern's reference
  forward, built by typed attribute access, or None where the layer does not
  conform. The type checker checks the attributes.
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
(``AttnThenMoE``) and a sub-block over a span of its roles
(``NormThenAttention``, the span ``in_norm, attn``). Where a layer is
declared for both, the block builds the sub-block from that span of its own
roles (``BlockPattern.spans``), with the contract it needs, so the inner
installer never runs on that layer.
"""

import dataclasses
import functools
import importlib
import importlib.machinery
import pathlib
import pkgutil
from collections.abc import Callable
from typing import ClassVar, cast

from torch import nn

from vllm_torchtpu.blocks.binding import BindingKindError, BlockBinding, attribute_of
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.roles import BlockRoles

Pattern = Callable[..., object]
"""A pattern's reference forward, e.g. ``attn_then_moe``; it names the pattern."""

Binding = Callable[[nn.Module], BlockBinding | None]
"""A model owner's binding function: layer -> the pattern's binding, or None."""


@dataclasses.dataclass(frozen=True)
class BlockDeclaration:
    """A model owner's statement that ``layer_class`` conforms to ``pattern``.

    Attributes:
        layer_class: The decoder layer class, matched exactly.
        pattern: The pattern's reference forward (``BlockPattern.reference``).
        binding_of: The model owner's function, turned into the runner's
            binding (``BlockPattern.declaration``).
        roles_of: The model owner's function itself: layer -> the pattern's
            ``Roles``, or None.
        spans: Declarations of sub-blocks over a span of this pattern's roles,
            derived from the same declaration (``BlockPattern.spans``).
    """

    layer_class: type
    pattern: Pattern
    binding_of: Binding
    roles_of: Callable[[nn.Module], BlockRoles | None]
    spans: tuple["BlockDeclaration", ...] = ()

    def binding(self, layer: nn.Module) -> BlockBinding | None:
        """The layer's binding, or None where the declaration does not cover it.

        Raises:
            BlockConformanceError: A part is not the kind its field needs,
                named by its attribute on the layer.
        """
        try:
            return self.binding_of(layer)
        except BindingKindError as e:
            raise BlockConformanceError(e.describe(layer)) from e

    def roles(self, layer: nn.Module) -> BlockRoles:
        """The layer's declared roles.

        Raises:
            BlockConformanceError: The declaration does not cover the layer.
        """
        roles = self.roles_of(layer)
        if roles is None:
            msg = f"{self.binding_of.__qualname__} returns None for this layer"
            raise BlockConformanceError(msg)
        return roles

    def when(self, layer: nn.Module) -> bool:
        """Whether the declaration covers this instance."""
        return self.binding(layer) is not None

    def bind(self, layer: nn.Module) -> dict[str, str]:
        """Each layer-level field's attribute name, recovered by identity.

        Raises:
            BlockConformanceError: The layer is not covered, or a part is not
                one of its children.
        """
        binding = self.binding(layer)
        if binding is None:
            msg = f"{self.binding_of.__qualname__} returns None for this layer"
            raise BlockConformanceError(msg)
        return {
            field: attribute_of(layer, module, field)
            for field, module in binding.layer_children().items()
        }


DECLARATIONS: dict[tuple[type, Pattern], BlockDeclaration] = {}


def declarations_for(layer_class: type) -> list[BlockDeclaration]:
    """Every pattern ``layer_class`` is declared to conform to."""
    return [d for (cls, _), d in DECLARATIONS.items() if cls is layer_class]


class BlockPattern:
    """A pattern: its reference forward, and the roles a model declares for it.

    Each pattern is a subclass that sets ``reference`` and defines a nested
    ``Roles`` dataclass whose top-level fields are the reference's role names.
    A model owner declares a layer class against it, as vLLM's
    ``PluggableLayer.register_oot`` registers a layer::

        @NormThenAttention.register(LlamaDecoderLayer)
        def llama(layer: LlamaDecoderLayer) -> NormThenAttention.Roles:
            return NormThenAttention.Roles(in_norm=..., attn=..., ...)

    The function takes the layer, typed as its class, so the type checker
    checks its attribute reads, and returns the pattern's ``Roles``, or None
    where the layer does not conform.
    """

    reference: ClassVar[Pattern]
    """The pattern's reference forward, with one keyword argument per role."""

    Roles: ClassVar[type[BlockRoles]]
    """The dataclass a declaration returns; its fields are the roles."""

    @classmethod
    def declaration[L: nn.Module](
        cls, layer_class: type[L], roles_of: Callable[[L], BlockRoles | None]
    ) -> BlockDeclaration:
        """The declaration ``roles_of`` makes, without registering it."""
        roles_type = cls.Roles
        # Called only on instances of ``layer_class``.
        read = cast("Callable[[nn.Module], BlockRoles | None]", roles_of)

        @functools.wraps(roles_of)
        def checked(layer: nn.Module) -> BlockRoles | None:
            roles = read(layer)
            if roles is not None and not isinstance(roles, roles_type):
                msg = (
                    f"{roles_of.__qualname__} returns {type(roles).__name__}, "
                    f"not {cls.__name__}.Roles"
                )
                raise TypeError(msg)
            return roles

        @functools.wraps(roles_of)
        def binding_of(layer: nn.Module) -> BlockBinding | None:
            roles = checked(layer)
            return None if roles is None else roles.binding()

        spans = cls.spans(layer_class, read)
        return BlockDeclaration(layer_class, cls.reference, binding_of, checked, spans)

    @classmethod
    def spans(
        cls,
        layer_class: type,
        roles_of: Callable[[nn.Module], BlockRoles | None],
    ) -> tuple[BlockDeclaration, ...]:
        """Sub-blocks this pattern derives from a span of its roles; none by default."""
        del layer_class, roles_of
        return ()

    @classmethod
    def register[L: nn.Module, R: BlockRoles](
        cls, layer_class: type[L]
    ) -> Callable[[Callable[[L], R | None]], Callable[[L], R | None]]:
        """Declares that ``layer_class`` conforms to this pattern (model owner API).

        Raises:
            ValueError: ``layer_class`` is already declared for this pattern.
        """

        def decorate(roles_of: Callable[[L], R | None]) -> Callable[[L], R | None]:
            declaration = cls.declaration(layer_class, roles_of)
            key = (layer_class, declaration.pattern)
            if key in DECLARATIONS:
                msg = (
                    f"{layer_class.__qualname__} is already declared for {cls.__name__}"
                )
                raise ValueError(msg)
            DECLARATIONS[key] = declaration
            return roles_of

        return decorate


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
