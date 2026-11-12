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
  the merge-time check (``tests/blocks``), naming the role edge and source
  line; at startup the layer is declined and keeps its upstream forward.

``install`` runs once after the weights load. For each module whose class is
declared, it checks conformance, resolves the block's plan per compiled
bucket, and installs the block's forward where the plan needs it. Layers
whose plan is the default are left as upstream built them.
"""

import dataclasses
import importlib
import inspect
import pathlib
import pkgutil
import typing
from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import nn

from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.parts import BlockParts, PartsKindError
from vllm_torchtpu.logger import init_logger

if typing.TYPE_CHECKING:  # attn_moe imports this package.
    from vllm_torchtpu.blocks.attn_moe.plan import Collectives

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
        returns = inspect.signature(parts_of, eval_str=True).return_annotation
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


DECLARATIONS: dict[type, BlockDeclaration] = {}


def declare_block[L: nn.Module, P: BlockParts](
    layer_class: type[L],
) -> Callable[[Callable[[L], P | None]], Callable[[L], P | None]]:
    """Declares a binding function for ``layer_class`` (the model owner's API).

    The function takes the layer, typed as ``layer_class``, so its attribute
    reads are type-checked against the model's class, and returns the
    pattern's parts dataclass, or None where the layer does not conform.
    """

    def register(parts_of: Callable[[L], P | None]) -> Callable[[L], P | None]:
        if layer_class in DECLARATIONS:
            raise ValueError(f"{layer_class.__qualname__} is already declared")
        DECLARATIONS[layer_class] = BlockDeclaration.of(layer_class, parts_of)
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
    """What every implementation reads at install time, passed as typed data.

    The user's and the model config's strings are parsed before this is built
    (``attn_moe.plan.parse_pin``).

    Attributes:
        block_pin, block_model_pin: The user's (``TPU_BLOCK_PLAN``) and the
            model plan's (``tpu_block_plan``) pin of the block's collectives,
            or None.
    """

    tp: int
    dtype: torch.dtype
    buckets: tuple[int, ...]
    block_pin: "Collectives | None" = None
    block_model_pin: "Collectives | None" = None


@dataclasses.dataclass(frozen=True)
class Installed:
    name: str
    layer_class: str
    outcome: str


class BlockImplementation(Protocol):
    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
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

    A layer that does not conform to its declared pattern is declined: it
    keeps its upstream forward and its children keep their contracts, so
    serving loses only the block's optimization. CI fails on the same
    mismatch (``prototype.check_block`` and the conformance test).
    """
    from vllm_torchtpu.blocks import attn_moe  # noqa: F401 - registers itself

    load_declarations()
    report = []
    declined: set[type[Any]] = set()
    for name, module in list(model.named_modules()):
        declaration = DECLARATIONS.get(type(module))
        if declaration is None:
            continue
        impl = IMPLEMENTATIONS[declaration.pattern]
        try:
            if declaration.parts(module) is None:
                continue
            report.append(impl.install(name, module, declaration, context))
        except BlockConformanceError as e:
            cls = type(module)
            if cls not in declined:
                declined.add(cls)
                logger.warning("block declined for %s: %s", cls.__qualname__, e)
            report.append(Installed(name, cls.__qualname__, f"declined: {e}"))
    counts: dict[str, int] = {}
    for item in report:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    for outcome, count in sorted(counts.items()):
        logger.info("block install: %d layer(s): %s", count, outcome)
    return report
