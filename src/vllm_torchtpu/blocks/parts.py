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
  dataclass is built; a part of the wrong kind raises there.

Error messages still name attributes: ``attribute_of`` recovers a part's
attribute path on the layer by identity.
"""

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar

from torch import nn

from vllm_torchtpu.blocks.conformance import BlockConformanceError


@dataclasses.dataclass(frozen=True)
class Problem:
    """One part that is not its field's kind.

    Attributes:
        field: The parts field, for the message.
        part: The part itself, so the message can name its attribute.
        why: What is wrong, e.g. "is not an RMSNorm called with a residual".
    """

    field: str
    part: object
    why: str


class BlockParts:
    """Base of every pattern's parts dataclass.

    Subclasses are frozen keyword-only dataclasses that set ``pattern`` and
    implement ``problems``.
    """

    pattern: ClassVar[Callable[..., Any]]

    def problems(self) -> list[Problem]:
        """Each part that is not its field's kind."""
        raise NotImplementedError

    def __post_init__(self) -> None:
        if problems := self.problems():
            raise PartsKindError(self, problems)


class PartsKindError(BlockConformanceError):
    """A part of the wrong kind, raised as the parts dataclass is built."""

    def __init__(self, parts: BlockParts, problems: list[Problem]) -> None:
        self.parts = parts
        self.problems = problems
        super().__init__(self.describe(None))

    def describe(self, layer: nn.Module | None) -> str:
        """The problems, each part named by its attribute on ``layer`` if given."""
        words = []
        for p in self.problems:
            where = p.field
            if layer is not None and isinstance(p.part, nn.Module):
                found = attribute_path(layer, p.part)
                where = f"{p.field}={found}" if found else p.field
            words.append(f"{where} {p.why}")
        return f"declared {self.parts.pattern.__name__}, but " + "; ".join(words)


def attribute_path(parent: nn.Module, part: nn.Module) -> str | None:
    """The dotted attribute path of ``part`` under ``parent`` by identity, or None."""
    return next((n for n, m in parent.named_modules() if m is part and n), None)


def as_kind[T](module: object, kind: type[T]) -> T:
    """``module``, narrowed to ``kind`` for the type checker, checked at run time.

    For a child a model types loosely (``nn.Module``): the caller states the
    class it expects, and the attribute reads after it are type-checked
    against that class.

    Raises:
        BlockConformanceError: ``module`` is not a ``kind``.
    """
    if not isinstance(module, kind):
        raise BlockConformanceError(
            f"expected a {kind.__qualname__}, found a {type(module).__qualname__}"
        )
    return module
