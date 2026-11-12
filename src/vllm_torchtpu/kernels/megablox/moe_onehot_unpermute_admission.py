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
"""Which calls the blockwise one-hot unpermute kernel accepts.

Plain Python over shapes, dtypes and the chip's `pltpu.TpuInfo`. The kernel checks its
inputs with it, and the MoE plan (`layers/core/moe_plan.py`) uses it to choose
the owner-output combine.
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.experimental.pallas import tpu as pltpu

ROUTE_BLOCK_SIZE = 128
VMEM_WORKSPACE_MARGIN_BYTES = 4 * 1024 * 1024


def explicit_vmem_bytes(route_capacity: int, hidden_size: int, num_tokens: int) -> int:
    """Size of the explicitly allocated VMEM buffers used by the kernel."""
    owner_and_store = num_tokens * hidden_size * (4 + 2)
    double_buffered_routes = 2 * ROUTE_BLOCK_SIZE * hidden_size * 2
    route_metadata = route_capacity * (4 + 4)
    return owner_and_store + double_buffered_routes + route_metadata


def shape_admits(
    *,
    route_shape: tuple[int, ...],
    route_dtype: jnp.dtype,
    token_indices_shape: tuple[int, ...],
    token_indices_dtype: jnp.dtype,
    weights_shape: tuple[int, ...],
    weights_dtype: jnp.dtype,
    valid_count_shape: tuple[int, ...],
    valid_count_dtype: jnp.dtype,
) -> bool:
    """The kernel's shape and dtype requirements.

    Args:
        route_shape: `route_output.shape`.
        route_dtype: `route_output.dtype`.
        token_indices_shape: `token_indices_sorted.shape`.
        token_indices_dtype: `token_indices_sorted.dtype`.
        weights_shape: `topk_weights_sorted.shape`.
        weights_dtype: `topk_weights_sorted.dtype`.
        valid_count_shape: `valid_count.shape`.
        valid_count_dtype: `valid_count.dtype`.

    Returns:
        Whether the inputs have the shapes and dtypes the kernel takes.
    """
    if len(route_shape) != 2 or route_dtype != jnp.bfloat16:
        return False
    route_capacity = route_shape[0]
    return (
        tuple(token_indices_shape) == (route_capacity,)
        and token_indices_dtype == jnp.int32
        and tuple(weights_shape) == (route_capacity,)
        and weights_dtype == jnp.bfloat16
        and tuple(valid_count_shape) == (1,)
        and valid_count_dtype == jnp.int32
    )


def chip_admits(
    *,
    route_capacity: int,
    hidden_size: int,
    num_tokens: int,
    dtype: jnp.dtype,
    chip: "pltpu.TpuInfo",
) -> bool:
    """The kernel's alignment and VMEM requirements on a chip.

    Args:
        route_capacity: Routed rows.
        hidden_size: Route width.
        num_tokens: Destination tokens.
        dtype: The route dtype.
        chip: The chip. Its sublane tiling is asked only when the earlier
            checks pass (it raises past the generations JAX describes).

    Returns:
        Whether the kernel fits.
    """
    if route_capacity % ROUTE_BLOCK_SIZE:
        return False
    if hidden_size % chip.mxu_column_size:
        return False
    if num_tokens % chip.get_sublane_tiling(dtype):
        return False
    vmem_limit_bytes = int(chip.vmem_capacity_bytes * 0.9)
    required_vmem_bytes = (
        explicit_vmem_bytes(route_capacity, hidden_size, num_tokens)
        + VMEM_WORKSPACE_MARGIN_BYTES
    )
    return required_vmem_bytes <= vmem_limit_bytes
