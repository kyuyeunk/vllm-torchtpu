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
"""How a runner calls a decoder layer's own forward and stands in for it.

Models order the same three inputs differently (GLM-4.5 takes
``(positions, hidden_states, residual)``, gpt-oss ``(hidden_states,
positions, residual)``, Qwen3-Next ``(hidden_states, residual, positions)``).
The binding function picks the helper whose Protocol matches its class, so
the type checker checks the order and the names against the model's forward,
and the framework never reads a parameter name.
"""

import dataclasses
from collections.abc import Callable
from typing import Protocol, cast

import torch
from torch import nn

Tensor = torch.Tensor
Triple = tuple[Tensor, Tensor | None, Tensor]


@dataclasses.dataclass(frozen=True)
class Stream:
    """A layer's forward convention.

    Attributes:
        call: Calls the layer class's own forward with ``(hidden, residual,
            positions)``; conformance traces it.
        unpack: Takes a call in the layer's own convention (positional or
            keyword) and returns ``(hidden, residual, positions)``; a runner
            standing in for the forward reads its arguments with it.
    """

    call: Callable[[Tensor, Tensor | None, Tensor], tuple[Tensor, Tensor]]
    unpack: Callable[..., Triple]


class PositionsFirst(Protocol):
    """``forward(positions, hidden_states, residual)``: Llama, Qwen3, GLM-4.5."""

    def forward(
        self, positions: Tensor, hidden_states: Tensor, residual: Tensor | None
    ) -> object:
        """The call the runner makes."""


class HiddenPositionsResidual(Protocol):
    """``forward(hidden_states, positions, residual)``: gpt-oss."""

    def forward(
        self, hidden_states: Tensor, positions: Tensor, residual: Tensor | None
    ) -> object:
        """The call the runner makes."""


class HiddenResidualPositions(Protocol):
    """``forward(hidden_states, residual, positions)``: Qwen3-Next, Qwen3.5."""

    def forward(
        self, hidden_states: Tensor, residual: Tensor | None, positions: Tensor
    ) -> object:
        """The call the runner makes."""


def _pair(out: object) -> tuple[Tensor, Tensor]:
    """The typed boundary over a layer's output.

    vLLM annotates its decoder forwards loosely (gpt-oss says ``Tensor``,
    Qwen3-Next allows a None residual), so the Protocols above check the
    parameters and leave the return to this check.
    """
    if not isinstance(out, tuple):
        msg = f"a decoder layer returned {type(out).__name__}, not a pair"
        raise TypeError(msg)
    items = cast("tuple[object, ...]", out)
    if len(items) != len(("hidden", "residual")):
        msg = f"a decoder layer returned {len(items)} values, not a pair"
        raise TypeError(msg)
    hidden, residual = items
    if not (isinstance(hidden, Tensor) and isinstance(residual, Tensor)):
        msg = "a decoder layer returned a pair that is not two tensors"
        raise TypeError(msg)
    return hidden, residual


def positions_first(layer: PositionsFirst) -> Stream:
    """The stream of a ``forward(positions, hidden_states, residual)`` layer."""
    forward = type(layer).forward

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> tuple[Tensor, Tensor]:
        return _pair(forward(layer, positions=p, hidden_states=h, residual=r))

    def unpack(
        positions: Tensor, hidden_states: Tensor, residual: Tensor | None
    ) -> Triple:
        return hidden_states, residual, positions

    return Stream(call, unpack)


def hidden_positions_residual(layer: HiddenPositionsResidual) -> Stream:
    """The stream of a ``forward(hidden_states, positions, residual)`` layer."""
    forward = type(layer).forward

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> tuple[Tensor, Tensor]:
        return _pair(forward(layer, hidden_states=h, positions=p, residual=r))

    def unpack(
        hidden_states: Tensor, positions: Tensor, residual: Tensor | None
    ) -> Triple:
        return hidden_states, residual, positions

    return Stream(call, unpack)


def hidden_residual_positions(layer: HiddenResidualPositions) -> Stream:
    """The stream of a ``forward(hidden_states, residual, positions)`` layer."""
    forward = type(layer).forward

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> tuple[Tensor, Tensor]:
        return _pair(forward(layer, hidden_states=h, residual=r, positions=p))

    def unpack(
        hidden_states: Tensor, residual: Tensor | None, positions: Tensor
    ) -> Triple:
        return hidden_states, residual, positions

    return Stream(call, unpack)


Attend = Callable[[Tensor, Tensor], Tensor]
"""``attend(normed, positions) -> attention's output``: how the layer calls it."""


class AttentionWithPositions(Protocol):
    """An attention whose forward takes ``positions`` and ``hidden_states``."""

    def forward(self, *, positions: Tensor, hidden_states: Tensor) -> Tensor:
        """The call the runner makes."""
        ...


class AttentionWithoutPositions(Protocol):
    """An attention whose forward takes ``hidden_states`` only (Gated DeltaNet)."""

    def forward(self, *, hidden_states: Tensor) -> Tensor:
        """The call the runner makes."""
        ...


def attend_with_positions(attention: AttentionWithPositions) -> Attend:
    """Calls ``attention`` with the normed tokens and the positions, by keyword."""

    def attend(normed: Tensor, positions: Tensor) -> Tensor:
        return attention.forward(positions=positions, hidden_states=normed)

    return attend


def attend_without_positions(attention: AttentionWithoutPositions) -> Attend:
    """Calls ``attention`` with the normed tokens only."""

    def attend(normed: Tensor, positions: Tensor) -> Tensor:
        del positions
        return attention.forward(hidden_states=normed)

    return attend


def route(layer: nn.Module, runner: nn.Module) -> None:
    """Makes ``runner`` the layer's forward, on this instance only.

    The runner is stored as the instance's ``forward`` attribute, which
    ``nn.Module.__call__`` reads, so neither the layer's class nor its module
    tree changes: no mixin, no ``__class__`` rebinding, and the runner's parts
    stay registered where the model put them. ``type(layer).forward`` stays
    the model's forward, which ``Stream.call`` calls.
    """
    object.__setattr__(layer, "forward", runner)


def routed_runner(layer: nn.Module) -> nn.Module | None:
    """The runner ``route`` installed on ``layer``, or None."""
    runner = vars(layer).get("forward")  # str-ok: the attribute route() sets
    return runner if isinstance(runner, nn.Module) else None
