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
"""The residual stream a decoder layer's forward receives, as named parameters.

Models order the same three inputs differently (GLM-4.5 takes
``(positions, hidden_states, residual)``, Qwen3.5 ``(hidden_states, residual,
positions)``), so a runner that replaces a layer's forward binds them by name,
from the parameter names conformance read off the model's own forward.
"""

import dataclasses
import inspect
from typing import Any

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class StreamParams:
    """Where the stream sits in a layer's forward.

    Attributes:
        order: The forward's parameter names, for positional calls.
        hidden, residual: The stream's parameter names.
        positions: The positions parameter's name, "" when there is none.
    """

    order: tuple[str, ...]
    hidden: str
    residual: str
    positions: str

    @classmethod
    def of(
        cls, layer: nn.Module, hidden: str, residual: str, positions: str
    ) -> "StreamParams":
        """Reads the order from the layer class's own forward."""
        order = tuple(inspect.signature(type(layer).forward).parameters)[1:]
        return cls(order, hidden, residual, positions)

    def bind(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """(hidden, residual, positions) from one call's arguments."""
        return (
            self._get(args, kwargs, self.hidden),
            self._get(args, kwargs, self.residual),
            self._get(args, kwargs, self.positions) if self.positions else None,
        )

    def _get(self, args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:  # noqa: ANN401 - whatever the caller passed
        if name in kwargs:
            return kwargs[name]
        index = self.order.index(name)
        return args[index] if index < len(args) else None


def route(layer: nn.Module, runner: nn.Module) -> None:
    """Makes ``runner`` the layer's forward, on this instance only.

    The runner is stored as the instance's ``forward`` attribute, which
    ``nn.Module.__call__`` reads, so neither the layer's class nor its module
    tree changes: no mixin, no ``__class__`` rebinding, and the runner's parts
    stay registered where the model put them. ``type(layer).forward`` stays
    the model's forward, which conformance reads.
    """
    object.__setattr__(layer, "forward", runner)


def routed_runner(layer: nn.Module) -> nn.Module | None:
    """The runner ``route`` installed on ``layer``, or None."""
    runner = layer.__dict__.get("forward")
    return runner if isinstance(runner, nn.Module) else None
