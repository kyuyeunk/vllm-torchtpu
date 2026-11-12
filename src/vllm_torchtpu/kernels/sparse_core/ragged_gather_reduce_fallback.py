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
"""When the SparseCore gather-reduce kernels take their TensorCore path.

The one rule for v1, v2 and v3. Plain Python over shapes, a dtype and the
chip's `pltpu.TpuInfo`: the kernels call it when the caller leaves the choice
to them, and the MoE plan (`layers/core/moe_plan.py`) calls it to decide and
passes the answer in.
"""

import enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v3_partition as v3

if TYPE_CHECKING:
    from jax.experimental.pallas import tpu as pltpu

# Fraction of TensorCore VMEM under which input plus output likely fit there,
# where a plain TensorCore gather-reduce beats a round trip through SparseCore
# and HBM.
VMEM_FRACTION = 0.6


class GatherReduceVersion(enum.Enum):
    """The SparseCore gather-reduce kernel versions.

    Each value is the version's RAGGED_GATHER_REDUCE_VERSION spelling.
    """

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


def takes_tensorcore_path(
    *,
    version: GatherReduceVersion,
    size: int,
    hidden: int,
    dtype: jnp.dtype,
    chip: "pltpu.TpuInfo",
) -> bool:
    """Whether a gather-reduce call runs its TensorCore fallback.

    Args:
        version: The kernel version.
        size: Elements of the gathered source `x`.
        hidden: Width of `x` (its last dimension).
        dtype: `x.dtype`.
        chip: The chip.

    Returns:
        True to run the TensorCore fallback.
    """
    if chip.sparse_core is None:
        return True
    # v3 is BF16-only and needs a lane-aligned width.
    if version is GatherReduceVersion.V3 and not v3.input_supported(
        hidden=hidden, dtype=dtype, chip=chip
    ):
        return True
    # TODO(kyuyeunk): Improve fallback calculation logic.
    dtype_bytes = jax.dtypes.itemsize_bits(dtype) // 8
    return size * dtype_bytes * 2 < chip.vmem_capacity_bytes * VMEM_FRACTION
