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
"""The layer-level conformance check every pre-norm pattern shares.

With the declared input norm, attention, post norm and the rest of the layer
opaque, the layer's own forward (called through the declared ``Stream``) must
compute the pattern's graph, and must call attention exactly as the declared
``attend`` does: the same objects (the normed tokens, the positions) in the
same parameters of attention's forward. Everything is tracked by object
identity; no parameter name is read.
"""

import contextlib
import dataclasses
import enum
import inspect
from collections.abc import Callable
from typing import overload

import torch
from torch import nn

from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.stream import Attend, Stream

Tensor = torch.Tensor
TOKENS = 8


class Arg(enum.Enum):
    """What one tensor argument of attention's call is, by identity."""

    NORMED = enum.auto()
    POSITIONS = enum.auto()
    OTHER = enum.auto()


def bound_arguments(
    module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[object, ...]:
    """A call's arguments in the parameter order of ``module``'s class forward.

    Two calls that pass the same objects to the same parameters give the same
    tuple, however each spelled the call (positional or keyword).
    """
    signature = inspect.signature(type(module).forward)
    return tuple(signature.bind(module, *args, **kwargs).arguments.values())[1:]


def classify(
    call: tuple[object, ...], normed: object, positions: object
) -> tuple[Arg, ...]:
    """Each tensor argument of a call as normed, positions or other."""
    return tuple(
        Arg.NORMED if a is normed else Arg.POSITIONS if a is positions else Arg.OTHER
        for a in call
        if isinstance(a, Tensor)
    )


class NormStub:
    """An opaque RMSNorm: ``x -> normed`` or ``(x, residual) -> (normed, sum)``."""

    def __init__(self, label: str) -> None:
        """Labels the op in the exported graph."""
        self.label: str = label

    @overload
    def __call__(self, x: Tensor) -> Tensor: ...
    @overload
    def __call__(self, x: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]: ...
    def __call__(
        self, x: Tensor, residual: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """The part's call contract."""
        if residual is None:
            return conf.unary(x, self.label)
        return conf.pair(x, residual, self.label)


def norm_stub(label: str) -> NormStub:
    """An opaque RMSNorm labelled ``label`` in the exported graph."""
    return NormStub(label)


@dataclasses.dataclass
class _Seen:
    normed: Tensor | None = None
    positions: Tensor | None = None
    call: tuple[object, ...] = ()


@dataclasses.dataclass(frozen=True)
class LayerParts:
    """The layer-level parts the check makes opaque, and how to call them."""

    input_norm: nn.Module
    attention: nn.Module
    post_norm: nn.Module
    rest: nn.Module
    stream: Stream
    attend: Attend


def check_layer(  # noqa: C901 - the opaque stubs are closures over one trace
    layer: nn.Module,
    parts: LayerParts,
    hidden: int,
    pattern: Callable[[Tensor, Tensor], object],
    pattern_name: str,
) -> tuple[Arg, ...]:
    """Checks the layer's wiring against ``pattern`` and attention's call.

    Args:
        layer: The declared layer.
        parts: Its declared layer-level parts.
        hidden: The residual stream's width.
        pattern: ``pattern(hidden, residual)`` over the opaque labels
            ``in_norm``, ``attn``, ``post_norm`` and the rest's label.
        pattern_name: For error messages.

    Returns:
        Attention's tensor arguments, classified (normed, positions).

    Raises:
        BlockConformanceError: Naming the edge and source line, or how the
            attention call differs.
    """
    seen = _Seen()
    in_norm = norm_stub("in_norm")

    def recording_in_norm(
        x: Tensor, residual: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        if residual is None:
            seen.normed = in_norm(x)
            return seen.normed
        normed, summed = in_norm(x, residual)
        seen.normed = normed
        return normed, summed

    def attention_stub(*a: object, **k: object) -> Tensor:
        seen.call = bound_arguments(parts.attention, a, k)
        normed = next((t for t in seen.call if t is seen.normed), None)
        if not isinstance(normed, Tensor):
            msg = (
                f"declared {pattern_name} on {type(layer).__qualname__}, but "
                "attention does not read the normed tokens"
            )
            raise conf.BlockConformanceError(msg)
        return conf.unary(normed, "attn")

    def rest_stub(x: Tensor, *a: object, **k: object) -> Tensor:
        del a, k
        return conf.unary(x, "rest")

    def layer_fn(h: Tensor, r: Tensor, p: Tensor) -> object:
        seen.positions = p
        return parts.stream.call(h, r, p)

    def pattern_fn(h: Tensor, r: Tensor, p: Tensor) -> object:
        del p
        return pattern(h, r)

    x = torch.empty(TOKENS, hidden, device="meta", dtype=torch.bfloat16)
    args = (
        x,
        torch.empty_like(x),
        torch.empty(TOKENS, device="meta", dtype=torch.long),
    )
    names = ("hidden", "residual", "positions")
    stubs: tuple[tuple[nn.Module, Callable[..., object]], ...] = (
        (parts.input_norm, recording_in_norm),
        (parts.post_norm, norm_stub("post_norm")),
        (parts.attention, attention_stub),
        (parts.rest, rest_stub),
    )
    with contextlib.ExitStack() as stack:
        for module, stub in stubs:
            stack.enter_context(conf.opaque(module, stub))
        got = conf.canonicalize(conf.export(layer_fn, args, layer), names)
        layer_call = classify(seen.call, seen.normed, seen.positions)
        # The declared adapter, on fresh tensors, must make the same call.
        normed, positions = torch.empty_like(x), torch.empty_like(args[2])
        seen.normed = normed
        _ = parts.attend(normed, positions)
        attend_call = classify(seen.call, normed, positions)
    want = conf.canonicalize(conf.export(pattern_fn, args, None), names)
    if error := conf.compare(pattern_name, got, want):
        raise conf.BlockConformanceError(error)
    where = f"declared {pattern_name} on {type(layer).__qualname__}, but"
    if Arg.OTHER in layer_call:
        msg = f"{where} attention reads more than the normed tokens and the positions"
        raise conf.BlockConformanceError(msg)
    if attend_call != layer_call:
        msg = f"{where} the declared attend calls attention another way than the layer"
        raise conf.BlockConformanceError(msg)
    return layer_call
