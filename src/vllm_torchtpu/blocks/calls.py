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
"""The call contracts of the parts the runners call, as Protocols.

``nn.Module.__call__`` returns ``Any``, so a runner that called its parts as
plain modules would let ``Any`` spread through its forward. Each runner
instead annotates its references with these Protocols: the one typed
boundary between vLLM's untyped modules and the framework's composition.
Conformance has checked that each declared part is the kind whose contract
this states.
"""

from typing import Protocol, overload

import torch

Tensor = torch.Tensor


class ResidualNorm(Protocol):
    """vLLM's RMSNorm: ``x -> normed`` or ``(x, residual) -> (normed, sum)``."""

    @overload
    def __call__(self, x: Tensor, /) -> Tensor: ...
    @overload
    def __call__(self, x: Tensor, residual: Tensor, /) -> tuple[Tensor, Tensor]: ...


class HeadNorm(Protocol):
    """A per-head norm over ``(..., heads, head_dim)``."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """The part's call contract."""
        ...


class Linear(Protocol):
    """A vLLM linear built with ``return_bias=True``: ``x -> (out, bias)``."""

    def __call__(self, x: Tensor, /) -> tuple[Tensor, Tensor | None]:
        """The part's call contract."""
        ...


class QuantMethod(Protocol):
    """A linear's quant method: the GEMM alone, no reduction over ranks."""

    def apply(self, layer: object, x: Tensor, bias: Tensor | None = None) -> Tensor:
        """The GEMM alone, without a reduction over ranks."""
        ...


class RowLinear(Linear, Protocol):
    """A ``RowParallelLinear``: its forward, and its GEMM without the all-reduce."""

    @property
    def quant_method(self) -> QuantMethod:
        """The part's ``quant_method``."""
        ...


class Rope(Protocol):
    """RoPE on flat q and k: ``(positions, q, k) -> (q, k)``."""

    def __call__(
        self, positions: Tensor, q: Tensor, k: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        """The part's call contract."""
        ...


class AttentionCore(Protocol):
    """The attention core on flat ``(q, k, v)``."""

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, /) -> Tensor:
        """The part's call contract."""
        ...


class Router(Protocol):
    """A router: ``x -> (logits, bias)`` or ``x -> logits``, per ``return_bias``."""

    @property
    def return_bias(self) -> bool:
        """The part's ``return_bias``."""
        ...

    def __call__(self, x: Tensor, /) -> Tensor | tuple[Tensor, Tensor | None]:
        """The part's call contract."""
        ...


class Experts(Protocol):
    """A MoE runner taking ``(hidden_states, router_logits)``."""

    def __call__(self, *, hidden_states: Tensor, router_logits: Tensor) -> Tensor:
        """The part's call contract."""
        ...


class AttentionPart(Protocol):
    """``(hidden, residual, positions) -> (attention's output, residual stream)``."""

    def __call__(
        self, hidden: Tensor, residual: Tensor | None, positions: Tensor, /
    ) -> tuple[Tensor, Tensor]:
        """The part's call contract."""
        ...


class Mlp(Protocol):
    """The rest of a layer after the post norm: ``x -> out``."""

    def __call__(self, x: Tensor, /) -> Tensor:
        """The part's call contract."""
        ...
