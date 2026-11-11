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
"""The attention-then-MoE block's forward, written against roles.

One forward serves every matched layer: it reaches the children through the
roles bound by the model's declaration, so it never names a model's attributes. Its
contract is the residual stream ``hidden + residual``: the overlap plan
returns the pre-attention residual and folds attention into the output, so
the two returned tensors differ from upstream's while their sum does not.
"""

from typing import Any

import torch
from torch import nn
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)

from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoERoles
from vllm_torchtpu.blocks.attn_moe.plan import OVERLAP, BlockPlanTable
from vllm_torchtpu.compilation import shape_variants


def _arg(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    order: tuple[str, ...],
    name: str,
) -> Any:  # noqa: ANN401 - whatever the caller passed
    if name in kwargs:
        return kwargs[name]
    index = order.index(name)
    return args[index] if index < len(args) else None


class AttentionMoEBlock(nn.Module):
    """Mixin placed before a matched decoder layer's class.

    The installer sets ``_block_roles``, ``_block_table``, ``_block_order``
    (the forward's parameter names, for positional calls) and ``_block_tp``.
    """

    _block_roles: AttnMoERoles
    _block_table: BlockPlanTable
    _block_order: tuple[str, ...]
    _block_tp: int

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
        roles = self._block_roles
        hidden = _arg(args, kwargs, self._block_order, roles.hidden_param)
        residual = _arg(args, kwargs, self._block_order, roles.residual_param)
        stream = {roles.hidden_param: hidden}
        if roles.positions_param:
            stream[roles.positions_param] = _arg(
                args, kwargs, self._block_order, roles.positions_param
            )
        num_tokens = (
            stream[roles.positions_param].shape[-1]
            if roles.positions_param
            else hidden.shape[0]
        )

        input_norm = self.get_submodule(roles.input_norm)
        post_norm = self.get_submodule(roles.post_norm)
        attention = self.get_submodule(roles.attention)
        router = self.get_submodule(roles.router)
        experts = self.get_submodule(roles.experts)

        if residual is None:
            residual = hidden
            normed = input_norm(hidden)
        else:
            normed, residual = input_norm(hidden, residual)
        stream["normed"] = normed
        # Partial sum over TP: the output projection does not reduce.
        attn = attention(**{k: stream[src] for k, src in roles.attention_args})

        # Only the backbone warmup gives each bucket its own trace; a compile
        # outside it (the profile run, the MTP drafter) keeps the all-reduce
        # path. The gate comes first so such a trace never reads num_tokens,
        # which it holds as a dynamic size.
        overlap = (
            not torch.compiler.is_compiling()
            or shape_variants.aot_cache_tag() is not None
        ) and self._block_table.plan_for(num_tokens) == OVERLAP
        if overlap:
            x_loc = tensor_model_parallel_reduce_scatter(
                attn + residual / self._block_tp, 0
            )
            h_loc = post_norm(x_loc)
            logits_loc = _logits(router, router(h_loc))
            logits = tensor_model_parallel_all_gather(logits_loc, 0).narrow(
                0, 0, num_tokens
            )
            h = tensor_model_parallel_all_gather(h_loc, 0).narrow(0, 0, num_tokens)
        else:
            h, residual = post_norm(tensor_model_parallel_all_reduce(attn), residual)
            logits = _logits(router, router(h))
        out = experts(hidden_states=h, router_logits=logits)
        if overlap:
            out = out + attn
        return tensor_model_parallel_all_reduce(out), residual


def _logits(router: nn.Module, out: Any) -> torch.Tensor:  # noqa: ANN401
    # A vLLM linear returns (output, bias) unless built with return_bias=False.
    return out[0] if getattr(router, "return_bias", True) else out
