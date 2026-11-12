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
  identity appears) ``declare_block`` states that a decoder class conforms to a
  pattern and binds its children to the pattern's roles.
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
(``attn_then_moe``) and a sub-block inside it (``norm_then_attention``).
"""

import dataclasses
import importlib
import pathlib
import pkgutil
from collections.abc import Callable
from typing import Any, Protocol

from torch import nn

from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class BlockDeclaration:
    """A model owner's statement that ``layer_class`` conforms to ``pattern``.

    Attributes:
        layer_class: The decoder layer class, matched exactly.
        pattern: The pattern's reference forward, e.g. ``attn_then_moe``.
        roles: Role name -> child attribute; a tuple lists alternatives, of
            which the first present on the layer is bound.
        when: Which instances of the class the declaration covers.
    """

    layer_class: type
    pattern: Callable[..., Any]
    roles: dict[str, str | tuple[str, ...]]
    when: Callable[[nn.Module], bool]

    def bind(self, layer: nn.Module) -> dict[str, str]:
        """The child bound to each role on this instance."""
        bound = {}
        for role, names in self.roles.items():
            options = (names,) if isinstance(names, str) else names
            present = [n for n in options if getattr(layer, n, None) is not None]
            if not present:
                raise AttributeError(f"{role}: the layer has none of {options}")
            bound[role] = present[0]
        return bound


DECLARATIONS: dict[tuple[type, Callable[..., Any]], BlockDeclaration] = {}


def declarations_for(layer_class: type) -> list[BlockDeclaration]:
    """Every pattern ``layer_class`` is declared to conform to."""
    return [d for (cls, _), d in DECLARATIONS.items() if cls is layer_class]


def declare_block(
    layer_class: type,
    *,
    pattern: Callable[..., Any],
    roles: dict[str, str | tuple[str, ...]],
    when: Callable[[nn.Module], bool] = lambda layer: True,
) -> None:
    """Declares that ``layer_class`` conforms to ``pattern`` (model owner's API)."""
    key = (layer_class, pattern)
    if key in DECLARATIONS:
        raise ValueError(
            f"{layer_class.__qualname__} is already declared for {pattern.__name__}"
        )
    DECLARATIONS[key] = BlockDeclaration(layer_class, pattern, roles, when)


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
    from vllm_torchtpu.blocks import attn_moe, norm_attn  # noqa: F401 - register

    load_declarations()
    report = []
    declined: set[tuple[type[Any], str]] = set()
    for name, module in list(model.named_modules()):
        cls = type(module)
        for declaration in declarations_for(cls):
            if not declaration.when(module):
                continue
            impl = IMPLEMENTATIONS[declaration.pattern]
            try:
                report.append(impl.install(name, module, declaration, context))
            except BlockConformanceError as e:
                if (cls, declaration.pattern.__name__) not in declined:
                    declined.add((cls, declaration.pattern.__name__))
                    logger.warning("block declined for %s: %s", cls.__qualname__, e)
                report.append(Installed(name, cls.__qualname__, f"declined: {e}"))
    counts: dict[str, int] = {}
    for item in report:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    for outcome, count in sorted(counts.items()):
        logger.info("block install: %d layer(s): %s", count, outcome)
    return report
