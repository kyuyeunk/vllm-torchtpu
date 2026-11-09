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
"""Pure MoE path decisions (data contract C7).

Imports no JAX or TPU runtime, so every rule here runs on CPU.
"""

from vllm_torchtpu.layers.core.tpu_target import TpuTarget

# Cap on the auto one-hot threshold.
ONEHOT_AUTO_CAP = 512


def auto_onehot_permute_threshold(target: TpuTarget | None) -> int:
    """The auto one-hot routed-row threshold for a chip.

    `min(SparseCore block rows - 1, ONEHOT_AUTO_CAP)` on SparseCore chips, 0
    on chips without a SparseCore and on hosts with no TPU description.

    Args:
        target: The chip, or None where `get_tpu_info` raised ValueError.

    Returns:
        The routed-row count at or below which MoE permute and combine run as
        TensorCore one-hot matmuls.
    """
    if target is None or target.sparse_core is None:
        return 0
    return min(target.sparse_core.block_rows - 1, ONEHOT_AUTO_CAP)
