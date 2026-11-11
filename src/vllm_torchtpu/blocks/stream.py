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
"""How the block calls a decoder layer's forward and reads its arguments.

Models order the same three inputs differently (GLM-4 MoE takes
``(positions, hidden_states, residual)``, Bailing-MoE ``(hidden_states,
position_ids, residual)``, Qwen3-Next ``(hidden_states, residual,
positions)``). The binding function picks the helper whose Protocol matches
its class, so the type checker checks the order and the names against the
model's forward, and the framework never reads a parameter name. The same
goes for how the layer calls its attention child (``Attend``).
"""

import dataclasses
from collections.abc import Callable
from typing import Protocol

import torch

Tensor = torch.Tensor
Triple = tuple[Tensor, Tensor | None, Tensor | None]
"""``(hidden, residual, positions)``; positions is None for a layer without."""


@dataclasses.dataclass(frozen=True)
class Stream:
    """A layer's forward convention.

    Attributes:
        call: Calls the layer's forward with ``(hidden, residual, positions)``
            in the layer's positional order, as the model calls it;
            conformance traces it, so a helper of the wrong order computes
            another graph and fails. It dispatches through ``layer.forward``,
            so once the block forward is installed it runs that.
        unpack: Takes a call in the layer's own convention (positional or
            keyword) and returns ``(hidden, residual, positions)``; the block
            forward reads its arguments with it.
    """

    call: Callable[[Tensor, Tensor | None, Tensor], object]
    unpack: Callable[..., Triple]


class PositionsFirst(Protocol):
    """``forward(positions, hidden_states, residual)``: Qwen3-MoE, GLM-4 MoE."""

    def forward(
        self, positions: Tensor, hidden_states: Tensor, residual: Tensor | None
    ) -> object: ...


class PositionsFirstScaled(Protocol):
    """``forward(positions, hidden_states, residual, llama_4_scaling)``: DeepSeek-V2."""

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        residual: Tensor | None,
        llama_4_scaling: Tensor | None = None,
    ) -> object: ...


class HiddenResidualPositions(Protocol):
    """``forward(hidden_states, residual, positions)``: Qwen3-Next, Qwen3.5."""

    def forward(
        self, hidden_states: Tensor, residual: Tensor | None, positions: Tensor
    ) -> object: ...


class HiddenPositionsResidual(Protocol):
    """``forward(hidden_states, positions, residual)``: gpt-oss."""

    def forward(
        self, hidden_states: Tensor, positions: Tensor, residual: Tensor | None
    ) -> object: ...


class HiddenPositionIdsResidual(Protocol):
    """``forward(hidden_states, position_ids, residual)``: Bailing-MoE."""

    def forward(
        self, hidden_states: Tensor, position_ids: Tensor, residual: Tensor | None
    ) -> object: ...


class HiddenResidual(Protocol):
    """``forward(hidden_states, residual)``: a layer that reads no positions."""

    def forward(self, hidden_states: Tensor, residual: Tensor | None) -> object: ...


def positions_first(layer: PositionsFirst) -> Stream:
    """The stream of a ``forward(positions, hidden_states, residual)`` layer."""

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> object:
        return layer.forward(p, h, r)

    def unpack(
        positions: Tensor, hidden_states: Tensor, residual: Tensor | None
    ) -> Triple:
        return hidden_states, residual, positions

    return Stream(call, unpack)


def positions_first_unscaled(layer: PositionsFirstScaled) -> Stream:
    """The stream of a layer that also takes ``llama_4_scaling``, called without.

    The block does not carry the scaling: a call that passes one is refused.
    """

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> object:
        return layer.forward(p, h, r)

    def unpack(
        positions: Tensor,
        hidden_states: Tensor,
        residual: Tensor | None,
        llama_4_scaling: Tensor | None = None,
    ) -> Triple:
        if llama_4_scaling is not None:
            raise NotImplementedError("the block does not carry llama_4_scaling")
        return hidden_states, residual, positions

    return Stream(call, unpack)


def hidden_residual_positions(layer: HiddenResidualPositions) -> Stream:
    """The stream of a ``forward(hidden_states, residual, positions)`` layer."""

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> object:
        return layer.forward(h, r, p)

    def unpack(
        hidden_states: Tensor, residual: Tensor | None, positions: Tensor
    ) -> Triple:
        return hidden_states, residual, positions

    return Stream(call, unpack)


def hidden_positions_residual(layer: HiddenPositionsResidual) -> Stream:
    """The stream of a ``forward(hidden_states, positions, residual)`` layer."""

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> object:
        return layer.forward(h, p, r)

    def unpack(
        hidden_states: Tensor, positions: Tensor, residual: Tensor | None
    ) -> Triple:
        return hidden_states, residual, positions

    return Stream(call, unpack)


def hidden_position_ids_residual(layer: HiddenPositionIdsResidual) -> Stream:
    """The stream of a ``forward(hidden_states, position_ids, residual)`` layer."""

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> object:
        return layer.forward(h, p, r)

    def unpack(
        hidden_states: Tensor, position_ids: Tensor, residual: Tensor | None
    ) -> Triple:
        return hidden_states, residual, position_ids

    return Stream(call, unpack)


def hidden_residual(layer: HiddenResidual) -> Stream:
    """The stream of a ``forward(hidden_states, residual)`` layer."""

    def call(h: Tensor, r: Tensor | None, p: Tensor) -> object:
        del p
        return layer.forward(h, r)

    def unpack(hidden_states: Tensor, residual: Tensor | None) -> Triple:
        return hidden_states, residual, None

    return Stream(call, unpack)


STREAMS: tuple[Callable[..., Stream], ...] = (
    positions_first,
    positions_first_unscaled,
    hidden_residual_positions,
    hidden_positions_residual,
    hidden_position_ids_residual,
    hidden_residual,
)
"""Every helper above, for a tool that tries each (``suggest_blocks``)."""


Attend = Callable[[Tensor, Tensor | None], Tensor]
"""``attend(normed, positions) -> attention's output``: how the layer calls it."""


class AttentionWithPositions(Protocol):
    """An attention whose forward takes ``positions`` and ``hidden_states``."""

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor: ...


class AttentionWithPositionIds(Protocol):
    """An attention whose forward takes ``hidden_states`` and ``position_ids``."""

    def forward(self, hidden_states: Tensor, position_ids: Tensor) -> Tensor: ...


class AttentionWithScaling(Protocol):
    """An attention taking ``positions``, ``hidden_states`` and ``llama_4_scaling``.

    DeepSeek-V2/V3's attention; the block passes no scaling.
    """

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        llama_4_scaling: Tensor | None,
    ) -> Tensor: ...


class AttentionWithoutPositions(Protocol):
    """An attention whose forward takes ``hidden_states`` only (Gated DeltaNet)."""

    def forward(self, hidden_states: Tensor) -> Tensor: ...


def require_positions(positions: Tensor | None) -> Tensor:
    """``positions``, raising where an attention needs them and got none."""
    if positions is None:
        raise TypeError("this attention reads positions; the layer passed none")
    return positions


def attend_with_positions(attention: AttentionWithPositions) -> Attend:
    """Calls ``attention`` with the normed tokens and the positions, by keyword."""

    def attend(normed: Tensor, positions: Tensor | None) -> Tensor:
        return attention.forward(
            positions=require_positions(positions), hidden_states=normed
        )

    return attend


def attend_with_position_ids(attention: AttentionWithPositionIds) -> Attend:
    """Calls ``attention`` with the normed tokens and the position ids."""

    def attend(normed: Tensor, positions: Tensor | None) -> Tensor:
        return attention.forward(
            hidden_states=normed, position_ids=require_positions(positions)
        )

    return attend


def attend_unscaled(attention: AttentionWithScaling) -> Attend:
    """Calls ``attention`` with the normed tokens, the positions and no scaling."""

    def attend(normed: Tensor, positions: Tensor | None) -> Tensor:
        return attention.forward(
            positions=require_positions(positions),
            hidden_states=normed,
            llama_4_scaling=None,
        )

    return attend


def attend_without_positions(attention: AttentionWithoutPositions) -> Attend:
    """Calls ``attention`` with the normed tokens only."""

    def attend(normed: Tensor, positions: Tensor | None) -> Tensor:
        del positions
        return attention.forward(hidden_states=normed)

    return attend


ATTENDS: tuple[Callable[..., Attend], ...] = (
    attend_with_positions,
    attend_with_position_ids,
    attend_unscaled,
    attend_without_positions,
)
"""Every attend helper above, for a tool that tries each (``suggest_blocks``)."""
