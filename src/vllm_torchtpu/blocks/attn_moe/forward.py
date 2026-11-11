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
"""The attention-then-MoE block's forward, written against the declared parts.

One forward serves every matched layer: it reaches the children through the
parts the model's binding function returned, so it never names a model's
attributes, and reads its arguments through the declared ``Stream``. Its
contract is the residual stream ``hidden + residual``: the overlap plan
returns the pre-attention residual and folds attention into the output, so
the two returned tensors differ from upstream's while their sum does not.
"""

import torch
from torch import nn
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu.blocks.attn_moe.pattern import ForwardParts
from vllm_torchtpu.blocks.attn_moe.plan import OVERLAP, BlockPlanTable
from vllm_torchtpu.compilation import shape_variants


class AttentionMoEBlock(nn.Module):
    """Mixin placed before a matched decoder layer's class.

    The installer sets ``_block_parts``, ``_block_table`` and ``_block_tp``
    (``bind_block``).
    """

    _block_parts: ForwardParts
    _block_table: BlockPlanTable
    _block_tp: int

    def bind_block(self, parts: ForwardParts, table: BlockPlanTable, tp: int) -> None:
        """Sets what the block forward reads: the parts, the plan table, the TP size."""
        self._block_parts = parts
        self._block_table = table
        self._block_tp = tp

    def forward(
        self, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self._block_parts
        hidden, residual, positions = parts.stream.unpack(*args, **kwargs)
        num_tokens = positions.shape[-1] if positions is not None else hidden.shape[0]

        if residual is None:
            residual = hidden
            normed = parts.input_norm(hidden)
        else:
            normed, residual = parts.input_norm(hidden, residual)
        # Partial sum over TP: the output projection does not reduce.
        attn = parts.attend(normed, positions)

        # Only the backbone warmup gives each bucket its own trace; a compile
        # outside it (the profile run, the MTP drafter) keeps the all-reduce
        # path. The gate comes first so such a trace never reads num_tokens,
        # which it holds as a dynamic size.
        overlap = (
            not torch.compiler.is_compiling()
            or shape_variants.aot_cache_tag() is not None
        ) and self._block_table.plan_for(num_tokens) == OVERLAP
        router, experts = parts.router, parts.experts
        if overlap:
            x_loc = tensor_model_parallel_reduce_scatter(
                attn + residual / self._block_tp, 0
            )
            h_loc = parts.post_norm(x_loc)
            logits_loc = _logits(router(h_loc))
            logits = tensor_model_parallel_all_gather(logits_loc, 0).narrow(
                0, 0, num_tokens
            )
            h = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, num_tokens)
        else:
            h, residual = parts.post_norm(
                tensor_model_parallel_all_reduce(attn), residual
            )
            logits = _logits(router(h))
        out = experts(hidden_states=h, router_logits=logits)
        if overlap:
            out = out + attn
        return tensor_model_parallel_all_reduce(out), residual


def _logits(out: torch.Tensor | tuple[torch.Tensor, object]) -> torch.Tensor:
    # A vLLM linear returns (output, bias) unless built with return_bias=False.
    return out[0] if isinstance(out, tuple) else out
