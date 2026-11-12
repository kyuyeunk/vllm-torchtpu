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
"""Work partitioning and capability of `ragged_gather_reduce_v3`.

Shared by the kernel and the MoE plan, which needs the destination-token
alignment and what the kernel can run to decide whether it is worth running.
Plain Python over shapes, a dtype and the chip's `pltpu.TpuInfo`.
"""

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.experimental.pallas import tpu as pltpu

TOKEN_SUBCHUNK = 16
NUM_TOKEN_SUBCHUNKS = 4
TOKEN_BLOCK = TOKEN_SUBCHUNK * NUM_TOKEN_SUBCHUNKS
# Source indices are packed into this many bits of the route metadata.
ROUTE_INDEX_BITS = 20
# The kernel's SparseCore SIMD width.
SPARSE_CORE_LANES = 16


def input_supported(*, hidden: int, dtype: jnp.dtype, chip: "pltpu.TpuInfo") -> bool:
    """The kernel's own entry check; it takes its TensorCore path otherwise.

    Args:
        hidden: Width of the combine input.
        dtype: Its dtype.
        chip: The chip.

    Returns:
        True for a BF16 input of a lane-aligned width.
    """
    return dtype == jnp.bfloat16 and hidden % chip.num_lanes == 0


def rows_supported(rows: int) -> bool:
    """Whether a source row index fits the packed route metadata."""
    return rows <= 1 << ROUTE_INDEX_BITS


def v3_capable(
    *, rows: int, hidden: int, dtype: jnp.dtype, chip: "pltpu.TpuInfo"
) -> bool:
    """Whether the SparseCore kernel can run this input at all.

    A 16-lane SparseCore, the kernel's entry check and the index capacity;
    the VMEM preference for the TensorCore path is not part of it.

    Args:
        rows: Rows of the combine input.
        hidden: Its width.
        dtype: Its dtype.
        chip: The chip.

    Returns:
        True when the kernel's SparseCore path can run.
    """
    sc = chip.sparse_core
    return (
        sc is not None
        and sc.num_lanes == SPARSE_CORE_LANES
        and input_supported(hidden=hidden, dtype=dtype, chip=chip)
        and rows_supported(rows)
    )


def num_column_partitions(hidden_size: int, num_cores: int, num_lanes: int) -> int:
    """Column partitions of the SparseCore subcores for a width.

    Args:
        hidden_size: Combine input width.
        num_cores: SparseCore subcores in total (cores x subcores).
        num_lanes: TensorCore vector lanes.

    Returns:
        The number of column partitions.
    """
    preferred_num_stages = 4
    partitions = 1
    while (
        num_cores % (partitions * 2) == 0
        and hidden_size % (num_lanes * partitions * 2) == 0
        and hidden_size // (partitions * 2 * num_lanes) >= preferred_num_stages
    ):
        partitions *= 2
    # Destination-major accumulation keeps a dense FP32 token block in VMEM.
    # For columns up to 1024, using one fewer column partition amortizes the
    # per-block metadata, clear, and output-DMA overhead while still leaving
    # enough room for the double-buffered indirect gather.  Wider columns stay
    # with the conservative partitioning selected above.
    if partitions >= 2 and hidden_size <= 1024 * (partitions // 2):
        partitions //= 2
    return partitions


def partition_counts(hidden_size: int, chip: "pltpu.TpuInfo") -> tuple[int, int]:
    """Column and row partition counts of the chip's SparseCore subcores.

    Args:
        hidden_size: Combine input width.
        chip: The chip; it must have a SparseCore.

    Returns:
        (column partitions, row partitions).
    """
    sc = chip.sparse_core
    assert sc is not None, "v3 partitions the SparseCore subcores"
    num_cores = sc.num_cores * sc.num_subcores
    columns = num_column_partitions(hidden_size, num_cores, chip.num_lanes)
    return columns, num_cores // columns


def token_block_alignment(hidden_size: int, chip: "pltpu.TpuInfo") -> int:
    """Destination-token granularity: one token block per row partition.

    Args:
        hidden_size: Combine input width.
        chip: The chip; it must have a SparseCore.

    Returns:
        The token count destinations are padded to a multiple of.
    """
    return partition_counts(hidden_size, chip)[1] * TOKEN_BLOCK
