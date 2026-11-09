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
"""Synthetic chips for the CPU table tests, shaped like `pltpu.TpuInfo`.

The first eleven are `jax._src.tpu_info.get_tpu_info_for_chip(cv, 1)` from
jax 0.11.2, dumped once with JAX_PLATFORMS=cpu (no device touched) so the
tests themselves need no JAX. The rest are made up to reach branches no real
chip does (no SparseCore with little VMEM, tiny VMEM with a 16-lane
SparseCore, 256 TensorCore lanes, a generation JAX does not describe).
"""

import dataclasses

_BITS = {"bfloat16": 16, "float32": 32, "float16": 16, "int32": 32}


@dataclasses.dataclass(frozen=True)
class FakeSparseCore:
    """Mirrors `jax._src.tpu_info.SparseCoreInfo` (fields read here)."""

    num_cores: int
    num_subcores: int
    num_lanes: int


@dataclasses.dataclass(frozen=True)
class FakeTpuInfo:
    """Mirrors `jax._src.tpu_info.TpuInfo` (fields read here)."""

    name: str
    generation: int
    num_cores: int
    num_lanes: int
    num_sublanes: int
    mxu_column_size: int
    vmem_capacity_bytes: int
    sparse_core: FakeSparseCore | None

    def get_sublane_tiling(self, dtype: str) -> int:
        """Verbatim from jax 0.11.2 `TpuInfo.get_sublane_tiling`."""
        bitwidth = _BITS[dtype]
        if self.generation < 7:
            if bitwidth == 16 or bitwidth == 32:
                return self.num_sublanes
            else:
                return self.num_sublanes * (32 // bitwidth)
        if self.generation == 7 or self.generation == 8:
            return self.num_sublanes * (32 // bitwidth)
        raise NotImplementedError("TPU generation is not supported")


def _chip(name, gen, lanes, sublanes, mxu, vmem, sc=None):
    return FakeTpuInfo(
        name=name,
        generation=gen,
        num_cores=1,
        num_lanes=lanes,
        num_sublanes=sublanes,
        mxu_column_size=mxu,
        vmem_capacity_bytes=vmem,
        sparse_core=None if sc is None else FakeSparseCore(*sc),
    )


MIB = 1 << 20
CHIPS: tuple[FakeTpuInfo, ...] = (
    _chip("v2", 2, 128, 8, 128, 16 * MIB),
    _chip("v3", 3, 128, 8, 128, 16 * MIB),
    _chip("v4i", 4, 128, 8, 128, 16 * MIB),
    _chip("v4", 4, 128, 8, 128, 16 * MIB),
    _chip("v5e", 5, 128, 8, 128, 128 * MIB),
    _chip("v5p", 5, 128, 8, 128, 64 * MIB, (4, 16, 8)),
    _chip("v6e", 6, 128, 8, 256, 128 * MIB, (2, 16, 8)),
    _chip("7", 7, 128, 8, 256, 64 * MIB, (2, 16, 16)),
    _chip("7x", 7, 128, 8, 256, 64 * MIB, (2, 16, 16)),
    _chip("8i", 8, 128, 8, 256, 192 * MIB, (1, 4, 16)),
    _chip("8t", 8, 128, 16, 256, 128 * MIB, (2, 16, 16)),
    # Synthetic.
    _chip("no_sc_small_vmem", 6, 128, 8, 128, 16 * MIB),
    _chip("sc16_tiny_vmem", 7, 128, 8, 256, 1 * MIB, (2, 16, 16)),
    _chip("sc16_lanes256", 7, 256, 8, 256, 64 * MIB, (2, 16, 16)),
    _chip("sc32_oversized_block", 7, 128, 8, 256, 64 * MIB, (2, 16, 32)),
    _chip("gen9_unknown", 9, 128, 8, 256, 64 * MIB, (2, 16, 16)),
)
