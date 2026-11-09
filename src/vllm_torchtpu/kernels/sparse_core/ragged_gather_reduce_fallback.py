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

The one rule for v1, v2 and v3. Pure Python, no JAX import: the kernels call
it when the caller leaves the choice to them, and the MoE plan
(`layers/core/moe_plan.py`) calls it to decide and passes the answer in.
"""

# Fraction of TensorCore VMEM under which input plus output likely fit there,
# where a plain TensorCore gather-reduce beats a round trip through SparseCore
# and HBM.
VMEM_FRACTION = 0.6


def takes_tensorcore_path(
    *,
    version: str,
    size: int,
    hidden: int,
    dtype: str,
    dtype_bytes: int,
    vmem_capacity_bytes: int,
    has_sparse_core: bool,
    num_lanes: int,
) -> bool:
    """Whether a gather-reduce call runs its TensorCore fallback.

    Args:
        version: "v1", "v2" or "v3".
        size: Elements of the gathered source `x`.
        hidden: Width of `x` (its last dimension).
        dtype: `x` dtype name.
        dtype_bytes: `jax.dtypes.itemsize_bits(x.dtype) // 8`.
        vmem_capacity_bytes: TensorCore VMEM per core.
        has_sparse_core: The chip has a SparseCore.
        num_lanes: TensorCore vector lanes.

    Returns:
        True to run the TensorCore fallback.
    """
    if not has_sparse_core:
        return True
    # v3 is BF16-only and needs a lane-aligned width.
    if version == "v3" and (dtype != "bfloat16" or hidden % num_lanes != 0):
        return True
    # TODO(kyuyeunk): Improve fallback calculation logic.
    return size * dtype_bytes * 2 < vmem_capacity_bytes * VMEM_FRACTION
