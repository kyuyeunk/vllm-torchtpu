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
"""What a model's binding function returns: a pattern's parts, typed.

A model owner declares a block by writing a function from its decoder layer
to the pattern's parts dataclass (``declare_block``). The dataclass is the
declaration's whole content:

- **Which pattern.** Each parts class names its pattern (``pattern``), so the
  binding function's return type says which pattern the layer conforms to.
- **Which children.** Each field is a module of the layer, reached by
  attribute access the type checker sees, so a misspelled attribute is a type
  error, not a string that fails at startup.
- **What kind each is.** ``problems`` checks every field's kind when the
  dataclass is built; a field of the wrong kind raises there.
- **What the runner takes.** Each parts class extends the dataclass its
  runner's ``__init__`` takes, so the declaration produces the runner's
  constructor arguments.

Error messages still name attributes: ``attribute_of`` recovers a part's
attribute name on the layer by identity, and ``describe`` uses it to word a
kind error as ``field=attribute is ...``.
"""

from collections.abc import Callable
from typing import Any, ClassVar

from torch import nn

from vllm_torchtpu.blocks.conformance import BlockConformanceError


class BlockParts:
    """Base of every pattern's parts dataclass.

    Subclasses are frozen keyword-only dataclasses that set ``pattern`` and
    implement ``layer_children`` and ``problems``.
    """

    pattern: ClassVar[Callable[..., Any]]

    def layer_children(self) -> dict[str, nn.Module]:
        """The parts that are direct children of the decoder layer, by field."""
        raise NotImplementedError

    def problems(self) -> list[tuple[str, str]]:
        """(field, why) for each part that is not its field's kind."""
        raise NotImplementedError

    def __post_init__(self) -> None:
        if problems := self.problems():
            raise PartsKindError(self, problems)


class PartsKindError(BlockConformanceError):
    """A part of the wrong kind, raised as the parts dataclass is built."""

    def __init__(self, parts: BlockParts, problems: list[tuple[str, str]]) -> None:
        self.parts = parts
        self.problems = problems
        super().__init__(self.describe(None))

    def describe(self, layer: nn.Module | None) -> str:
        """The problems, each part named by its attribute on ``layer`` if given."""
        words = []
        for field, why in self.problems:
            part = getattr(self.parts, field, None)
            where = field
            if layer is not None and isinstance(part, nn.Module):
                found = [n for n, m in layer.named_modules() if m is part and n]
                where = f"{field}={found[0]}" if found else field
            words.append(f"{where} {why}")
        return f"declared {self.parts.pattern.__name__}, but " + "; ".join(words)


def attribute_of(parent: nn.Module, part: nn.Module, field: str) -> str:
    """The dotted attribute path of ``part`` under ``parent``, found by identity.

    Raises:
        BlockConformanceError: ``part`` is not a module of ``parent``.
    """
    for name, module in parent.named_modules():
        if module is part and name:
            return name
    raise BlockConformanceError(
        f"{field} is a {type(part).__name__} that is not a module of "
        f"{type(parent).__qualname__}"
    )


def as_kind[T](module: object, kind: type[T]) -> T:
    """``module``, narrowed to ``kind`` for the type checker, checked at run time.

    For a child a model builds from a class parameter (Llama's
    ``attn_layer_type``, gpt-oss's ``attention_cls``), which the type checker
    sees only as ``nn.Module``: the binding function states the class it
    expects, and the attribute reads after it are type-checked against it.

    Raises:
        BlockConformanceError: ``module`` is not a ``kind``.
    """
    if not isinstance(module, kind):
        raise BlockConformanceError(
            f"expected a {kind.__qualname__}, found a {type(module).__qualname__}"
        )
    return module


def require(obj: object, attribute: str, field: str) -> Any:  # noqa: ANN401 - the attribute's own type
    """``obj.attribute``, raising a conformance error naming ``field`` if absent.

    For a property read off an untyped part, where a default would turn a
    missing attribute into a silent guess.

    Raises:
        BlockConformanceError: ``obj`` has no ``attribute``.
    """
    try:
        return getattr(obj, attribute)
    except AttributeError:
        raise BlockConformanceError(
            f"{field} is a {type(obj).__qualname__} without {attribute}"
        ) from None
