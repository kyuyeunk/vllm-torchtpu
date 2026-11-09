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
"""Work partitioning of `ragged_gather_reduce_v3`, as pure arithmetic.

Shared by the kernel and the MoE plan, which needs the destination-token
alignment to decide whether the kernel is worth running. No JAX import.
"""

TOKEN_SUBCHUNK = 16
NUM_TOKEN_SUBCHUNKS = 4
TOKEN_BLOCK = TOKEN_SUBCHUNK * NUM_TOKEN_SUBCHUNKS
# Source indices are packed into this many bits of the route metadata.
ROUTE_INDEX_BITS = 20


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


def partition_counts(
    hidden_size: int, num_cores: int, num_lanes: int
) -> tuple[int, int]:
    """Column and row partition counts of the SparseCore subcores.

    Args:
        hidden_size: Combine input width.
        num_cores: SparseCore subcores in total (cores x subcores).
        num_lanes: TensorCore vector lanes.

    Returns:
        (column partitions, row partitions).
    """
    columns = num_column_partitions(hidden_size, num_cores, num_lanes)
    return columns, num_cores // columns


def token_block_alignment(hidden_size: int, num_cores: int, num_lanes: int) -> int:
    """Destination-token granularity: one token block per row partition.

    Args:
        hidden_size: Combine input width.
        num_cores: SparseCore subcores in total (cores x subcores).
        num_lanes: TensorCore vector lanes.

    Returns:
        The token count destinations are padded to a multiple of.
    """
    return partition_counts(hidden_size, num_cores, num_lanes)[1] * TOKEN_BLOCK
