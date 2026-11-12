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
"""AttnMoERunner: the framework's one forward for an attention-then-MoE block.

Shaped like vLLM's ``MoERunner`` (``layers/fused_moe/runner/moe_runner.py``)
and like ``AttentionRunner``, one scope up:

- ``__init__`` receives parts already built: the attention part (the nested
  ``AttentionRunner``, or the model's own input norm and attention), the post
  norm, the router, the experts, and the per-bucket plan table. It keeps
  references; the parts stay registered where the model put them.
- ``forward`` calls the parts. The runner owns the composition only: which
  collectives join attention to the MoE (PR #1492's overlap or upstream's
  all-reduce).
- ``_select_forward`` picks the implementation once.

The attention part's contract is stated when the block builds it: the block
reduces attention's output itself, so its ``AttentionRunner`` is constructed
with ``output_reduced=False``. Nothing writes ``o_proj.reduce_results``.

Its contract is the residual stream ``hidden + residual``: the overlap
returns the pre-attention residual and folds attention into the output, so
the two returned tensors differ from upstream's while their sum does not.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu.blocks.attn_moe.plan import OVERLAP, BlockPlanTable
from vllm_torchtpu.blocks.stream import StreamParams
from vllm_torchtpu.compilation import shape_variants

CALLS: dict[str, dict[str, int]] = {}
"""Eager runs per layer and plan value, for evidence that a value ran."""


class ModelAttention(nn.Module):
    """The attention part for a layer with no nested sub-block: the model's own parts.

    Calls the model's input norm, then its attention module with the
    arguments conformance recorded. The model's attention calls its output
    projection itself, so here the unreduced contract is the interim field
    write (``contract_adapter``), not a constructor argument.
    """

    input_norm: nn.Module
    attention: nn.Module

    def __init__(
        self,
        *,
        input_norm: nn.Module,
        attention: nn.Module,
        attention_args: tuple[tuple[str, str], ...],
        positions_param: str,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "input_norm", input_norm)  # a reference
        object.__setattr__(self, "attention", attention)
        self.attention_args = attention_args
        self.positions_param = positions_param

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream: torch.Tensor
        if residual is None:  # The first layer: the stream is its own residual.
            normed, stream = self.input_norm(hidden), hidden
        else:
            normed, stream = self.input_norm(hidden, residual)
        sources = {"normed": normed, self.positions_param: positions}
        return self.attention(
            **{k: sources[src] for k, src in self.attention_args}
        ), stream


@dataclasses.dataclass(frozen=True, kw_only=True)
class MoEParts:
    """What ``AttnMoERunner`` calls after attention: the model's own objects.

    Execution needs only each part's call contract, so tests build this from
    synthetic modules. A model's declaration builds ``AttnMoEParts``, which
    extends it with the layer's other roles and checks every kind.

    Attributes:
        post_norm: The norm between attention and the MoE.
        router: The router, returning logits (``(out, bias)`` unless built
            with ``return_bias=False``).
        experts: The MoE runner, taking ``(hidden_states, router_logits)``.
    """

    post_norm: nn.Module
    router: nn.Module
    experts: nn.Module


class AttnMoERunner(nn.Module):
    """The block's composition; the parts stay the model's.

    Args:
        layer_name: The decoder layer's module path.
        parts: The post norm, the router and the experts.
        attention: The attention part: input norm -> attention -> out
            projection, returning the unreduced attention output and the
            residual stream. The nested ``AttentionRunner`` (built with
            ``output_reduced=False``) or a ``ModelAttention``.
        table: The plan per compiled bucket.
        tp: The tensor-parallel world size.
        stream: Where the stream sits in the layer's forward, so the runner
            can stand in for it.
    """

    attention: nn.Module
    post_norm: nn.Module
    router: nn.Module
    experts: nn.Module

    def __init__(
        self,
        layer_name: str,
        parts: MoEParts,
        *,
        attention: nn.Module,
        table: BlockPlanTable,
        tp: int,
        stream: StreamParams,
    ) -> None:
        super().__init__()
        refs = {
            "attention": attention,
            "post_norm": parts.post_norm,
            "router": parts.router,
            "experts": parts.experts,
        }
        for name, part in refs.items():
            object.__setattr__(self, name, part)  # a reference, not a child
        self.layer_name = layer_name
        self.table = table
        self.tp = tp
        self.stream = stream
        self._forward_entry = self._select_forward()

    def _select_forward(self) -> Callable[..., Any]:
        """One plan everywhere, or the per-bucket lookup."""
        if not self.table.uses(OVERLAP.collectives):
            return self._all_reduce
        return self._by_bucket

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
        """Takes the layer's own arguments: the runner stands in for its forward."""
        hidden, residual, positions = self.stream.bind(args, kwargs)
        return self._forward_entry(hidden, residual, positions)

    def _by_bucket(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.shape[-1] if positions is not None else hidden.shape[0]
        # Only the backbone warmup gives each bucket its own trace; a compile
        # outside it (the profile run, the MTP drafter) keeps the all-reduce
        # path. The gate comes first so such a trace never reads num_tokens,
        # which it holds as a dynamic size.
        if (
            not torch.compiler.is_compiling()
            or shape_variants.aot_cache_tag() is not None
        ) and self.table.plan_for(num_tokens) == OVERLAP:
            return self._overlap(hidden, residual, positions, num_tokens)
        return self._all_reduce(hidden, residual, positions)

    # The plan values.

    def _all_reduce(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Upstream's collectives: all-reduce attention, all-reduce the MoE."""
        self._count("all_reduce")
        stream: torch.Tensor
        attn, stream = self.attention(hidden, residual, positions)
        h, stream = self.post_norm(tensor_model_parallel_all_reduce(attn), stream)
        out = self.experts(hidden_states=h, router_logits=self._logits(h))
        return tensor_model_parallel_all_reduce(out), stream

    def _overlap(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor | None,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PR #1492: norm and route a token shard; fold attention into the MoE sum."""
        self._count("reduce_scatter_all_gather")
        stream: torch.Tensor
        attn, stream = self.attention(hidden, residual, positions)
        x_loc = tensor_model_parallel_reduce_scatter(attn + stream / self.tp, 0)
        h_loc = self.post_norm(x_loc)
        logits = tensor_model_parallel_all_gather(self._logits(h_loc), 0)
        h = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, num_tokens)
        out = self.experts(
            hidden_states=h, router_logits=logits.narrow(0, 0, num_tokens)
        )
        return tensor_model_parallel_all_reduce(out + attn), stream

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        out = self.router(h)
        # A vLLM linear returns (output, bias) unless built with return_bias=False.
        return out[0] if getattr(self.router, "return_bias", True) else out

    def _count(self, value: str) -> None:
        if not torch.compiler.is_compiling():
            calls = CALLS.setdefault(self.layer_name, {})
            calls[value] = calls.get(value, 0) + 1
