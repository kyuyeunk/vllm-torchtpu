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
"""The chips the CPU table tests enumerate, as `pltpu.TpuInfo` records.

The first eleven are `pltpu.get_tpu_info_for_chip(version, 1)`: the record
JAX builds for each chip it describes, on a CPU, with no device touched. The
rest are `dataclasses.replace` of a real one, to reach branches no real chip
does (no SparseCore with little VMEM, tiny VMEM with a 16-lane SparseCore,
256 TensorCore lanes, a 32-lane SparseCore, a generation JAX does not
describe).
"""

import dataclasses

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.dtype(jnp.bfloat16)
F32 = jnp.dtype(jnp.float32)
I32 = jnp.dtype(jnp.int32)

MIB = 1 << 20
_CV = pltpu.ChipVersion


def _real(version: pltpu.ChipVersion) -> pltpu.TpuInfo:
    return pltpu.get_tpu_info_for_chip(version, 1)


_V6E = _real(_CV.TPU_V6E)
_7X = _real(_CV.TPU_7X)
assert _7X.sparse_core is not None

# (name, chip), the name for messages only.
NAMED_CHIPS: tuple[tuple[str, pltpu.TpuInfo], ...] = (
    ("v2", _real(_CV.TPU_V2)),
    ("v3", _real(_CV.TPU_V3)),
    ("v4i", _real(_CV.TPU_V4I)),
    ("v4", _real(_CV.TPU_V4)),
    ("v5e", _real(_CV.TPU_V5E)),
    ("v5p", _real(_CV.TPU_V5P)),
    ("v6e", _V6E),
    ("7", _real(_CV.TPU_7)),
    ("7x", _7X),
    ("8i", _real(_CV.TPU_8I)),
    ("8t", _real(_CV.TPU_8T)),
    # Planted variants of real chips.
    (
        "no_sc_small_vmem",
        dataclasses.replace(
            _V6E, mxu_column_size=128, vmem_capacity_bytes=16 * MIB, sparse_core=None
        ),
    ),
    ("sc16_tiny_vmem", dataclasses.replace(_7X, vmem_capacity_bytes=1 * MIB)),
    ("sc16_lanes256", dataclasses.replace(_7X, num_lanes=256)),
    (
        "sc32_oversized_block",
        dataclasses.replace(
            _7X, sparse_core=dataclasses.replace(_7X.sparse_core, num_lanes=32)
        ),
    ),
    ("gen9_unknown", dataclasses.replace(_7X, generation=9)),
)
CHIPS: tuple[pltpu.TpuInfo, ...] = tuple(info for _, info in NAMED_CHIPS)


def chip_name(info: pltpu.TpuInfo) -> str:
    """The name a chip of `CHIPS` is listed under."""
    return next(name for name, chip in NAMED_CHIPS if chip is info)
