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
"""A hypothetical EP SparseCore combine that skips tokens with no local route.

A stub for the prototype: it delegates to the v2 kernel, whose mask already
zeroes those tokens, and declares the limits a real kernel would.
"""

import jax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
    ragged_gather_reduce_v2,
)
from vllm_torchtpu.kernels.sparse_core.support import get_sparse_core_info


def ragged_gather_reduce(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    token_local_routes: jax.Array,
    reduce_group_size: int,
) -> jax.Array:
    """Like v2, but writes zeros without gathering where `token_local_routes` is 0."""
    del token_local_routes  # The stub gathers every token.
    return ragged_gather_reduce_v2(
        x, indices, topk_weights, valid_rows_mask, reduce_group_size=reduce_group_size
    )


def ragged_gather_reduce_supports(
    x: jax.ShapeDtypeStruct,
    indices: jax.ShapeDtypeStruct,
    topk_weights: jax.ShapeDtypeStruct,
    valid_rows_mask: jax.ShapeDtypeStruct,
    token_local_routes: jax.ShapeDtypeStruct | None,
    reduce_group_size: int,
    *,
    chip: pltpu.TpuInfo | None,
) -> str | None:
    """Why the kernel cannot run these arguments, or None if it can."""
    del x, topk_weights, valid_rows_mask
    tokens = indices.size // reduce_group_size
    if token_local_routes is None or token_local_routes.shape != (tokens,):
        return "needs routing's per-token count of local routes"
    if chip is None or get_sparse_core_info(chip) is None:
        return "needs a SparseCore these kernels are validated on"
    return None
