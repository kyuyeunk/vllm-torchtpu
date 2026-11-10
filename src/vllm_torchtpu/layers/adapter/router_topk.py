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
"""Torch bridge for the sort-free MoE router top-k.

On TPU the selection runs as the Pallas kernel in ``kernels/router_topk.py``,
elsewhere as the identical plain-torch algorithm in
``layers/core/rowmax_topk.py``.
"""

import torch

from vllm_torchtpu.kernels.router_topk import router_topk
from vllm_torchtpu.layers.adapter.jax_op_spec import dim, jax_op_with_fake, spec
from vllm_torchtpu.layers.core.rowmax_topk import rowmax_topk

# Registered at import, not lazily: this op is called from inside the MoE
# forward, and Dynamo cannot trace a lock-guarded registration there.
_op = jax_op_with_fake(
    "pallas::moe_router_topk",
    router_topk,
    out=(
        spec(shape=(dim("scores", 0), "topk"), dtype=torch.float32),
        spec(shape=(dim("scores", 0), "topk"), dtype=torch.int32),
    ),
)


def rowmax_select(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _op(scores.float(), topk)
