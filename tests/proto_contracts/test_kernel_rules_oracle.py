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
"""Oracle: the per-kernel pure rules against the d3991ebf9 kernel checks.

Kernels called without a plan decision (`tc_fallback=None`, the direct
kernel tests) decide with these rules, so they must equal what each kernel
decided inline at d3991ebf9. The references are imported from
`test_moe_plan_oracle`, where they are copied verbatim.
"""

import dataclasses
import itertools
from collections.abc import Callable

import _pure_import
import jax.numpy as jnp
import pytest
from _chips import BF16, CHIPS, F32, I32, chip_name
from _grid import GridFamily
from jax.experimental.pallas import tpu as pltpu
from test_moe_plan_oracle import (
    _VMEM_WORKSPACE_MARGIN_BYTES,
    KernelPath,
    _explicit_vmem_bytes,
    _v1_kernel_path,
    _v2_kernel_path,
    _v3_kernel_path,
    can_use_blockwise_onehot_unpermute,
    partition_counts,
    token_block_alignment,
)

from vllm_torchtpu.kernels.megablox import moe_onehot_unpermute_admission as owner
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_fallback as fb
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v3_partition as v3
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_fallback import (
    GatherReduceVersion,
)

_HIDDENS = (100, 128, 256, 512, 1024, 2048, 2880, 4096, 5120, 7168, 8192)
_ROWS = (1, 2, 16, 127, 128, 255, 256, 1000, 1024, 4096, 16384, 65536, 1 << 20)


_SC_CHIPS = tuple(chip for chip in CHIPS if chip.sparse_core is not None)
# The VMEM boundary plants a VMEM size on each chip JAX tiles.
_TILED_CHIPS = tuple(
    chip
    for chip in CHIPS
    if chip.generation <= 8  # noqa: PLR2004 - the last generation JAX tiles
)
_TOKENS_PER_TILE = (1, 64)


def _chip_params(chips: tuple[pltpu.TpuInfo, ...]) -> list[object]:
    """`chips` as pytest params, each named for its chip."""
    return [pytest.param(chip, id=chip_name(chip)) for chip in chips]


def _reference_path(
    version: GatherReduceVersion,
    rows: int,
    hidden: int,
    dtype: jnp.dtype,
    get_tpu_info: Callable[[], pltpu.TpuInfo],
) -> KernelPath:
    """The d3991ebf9 entry checks of the gather-reduce kernel `version`."""
    nbytes = dtype.itemsize
    match version:
        case GatherReduceVersion.V1:
            return _v1_kernel_path(rows, hidden, nbytes, get_tpu_info)
        case GatherReduceVersion.V2:
            return _v2_kernel_path(rows, hidden, dtype, nbytes, get_tpu_info)
        case GatherReduceVersion.V3:
            return _v3_kernel_path(rows, hidden, dtype, nbytes, get_tpu_info)


def test_no_tpu_runtime() -> None:
    """The module imported with no TPU runtime and runs JAX on the CPU."""
    assert _pure_import.loaded_tpu_runtime() == []
    assert _pure_import.jax_backend() == "cpu"  # str-ok: the backend name JAX reports


@pytest.mark.parametrize(
    "version", list(GatherReduceVersion), ids=[v.name for v in GatherReduceVersion]
)
@pytest.mark.parametrize("chip", _chip_params(CHIPS))
def test_gather_reduce_fallback(
    chip: pltpu.TpuInfo, version: GatherReduceVersion
) -> None:
    """Every (rows, width, dtype) of one chip and one kernel version."""
    family = GridFamily(f"gather-reduce fallback {chip_name(chip)} {version.name}")
    for rows, hidden, dtype in itertools.product(_ROWS, _HIDDENS, (BF16, F32)):
        path = _reference_path(version, rows, hidden, dtype, lambda: chip)
        new = fb.takes_tensorcore_path(
            version=version,
            size=rows * hidden,
            hidden=hidden,
            dtype=dtype,
            chip=chip,
        )
        family.expect_equal(
            new, path is KernelPath.TC_FALLBACK, (version, rows, hidden, dtype.name)
        )
        family.cases += 1
    family.assert_no_mismatches()


@pytest.mark.parametrize("chip", _chip_params(_SC_CHIPS))
def test_v3_partition(chip: pltpu.TpuInfo) -> None:
    """Every width on one chip with a SparseCore."""
    family = GridFamily(f"v3 partition {chip_name(chip)}")
    for hidden in (*_HIDDENS, 64, 192, 384, 3072):
        family.expect_equal(
            v3.partition_counts(hidden, chip), partition_counts(hidden, chip), hidden
        )
        family.expect_equal(
            v3.token_block_alignment(hidden, chip),
            token_block_alignment(hidden, chip),
            hidden,
        )
        family.cases += 1
    family.assert_no_mismatches()


@pytest.mark.parametrize("chip", _chip_params(CHIPS))
def test_owner_admission(chip: pltpu.TpuInfo) -> None:
    """Every (routes, width, tokens, dtype, extra index) on one chip."""
    family = GridFamily(f"owner admission {chip_name(chip)}")
    for routes, hidden, tokens, dtype, extra in itertools.product(
        (128, 256, 255, 1024, 4096),
        (256, 512, 2048, 4096, 7168, 100),
        (8, 15, 16, 32, 64, 256, 1024, 8192),
        (BF16, F32),
        (0, 1),
    ):
        route_shape = (routes, hidden)
        ti_shape = (routes + extra,)
        try:
            expected = can_use_blockwise_onehot_unpermute(
                (route_shape, dtype, lambda: chip),
                (ti_shape, I32),
                ((routes,), dtype),
                ((1,), I32),
                num_tokens=tokens,
            )
        except NotImplementedError:
            expected = KernelPath.RAISE
        try:
            actual = owner.shape_admits(
                route_shape=route_shape,
                route_dtype=dtype,
                token_indices_shape=ti_shape,
                token_indices_dtype=I32,
                weights_shape=(routes,),
                weights_dtype=dtype,
                valid_count_shape=(1,),
                valid_count_dtype=I32,
            ) and owner.chip_admits(
                route_capacity=routes,
                hidden_size=hidden,
                num_tokens=tokens,
                dtype=dtype,
                chip=chip,
            )
        except NotImplementedError:
            actual = KernelPath.RAISE
        family.expect_equal(actual, expected, (routes, hidden, tokens, dtype.name))
        family.cases += 1
    family.assert_no_mismatches()


@pytest.mark.parametrize(
    "fits", [True, False], ids=["at_limit", "limit_one_byte_short"]
)
@pytest.mark.parametrize(
    "tokens_per_tile",
    _TOKENS_PER_TILE,
    ids=[f"{n}_tile_rows" for n in _TOKENS_PER_TILE],
)
@pytest.mark.parametrize("chip", _chip_params(_TILED_CHIPS))
def test_owner_admission_vmem_boundary(
    chip: pltpu.TpuInfo, tokens_per_tile: int, *, fits: bool
) -> None:
    """The VMEM check at equality and one byte under, on a planted chip."""
    hidden = chip.mxu_column_size * 4
    tokens = chip.get_sublane_tiling(BF16) * tokens_per_tile
    required = _explicit_vmem_bytes(128, hidden, tokens) + _VMEM_WORKSPACE_MARGIN_BYTES
    limit = required if fits else required - 1
    vmem = -(-limit * 10 // 9)
    while int(vmem * 0.9) > limit:
        vmem -= 1
    planted = dataclasses.replace(chip, vmem_capacity_bytes=vmem)
    assert int(vmem * 0.9) == limit
    expected = can_use_blockwise_onehot_unpermute(
        ((128, hidden), BF16, lambda: planted),
        ((128,), I32),
        ((128,), BF16),
        ((1,), I32),
        num_tokens=tokens,
    )
    actual = owner.chip_admits(
        route_capacity=128,
        hidden_size=hidden,
        num_tokens=tokens,
        dtype=BF16,
        chip=planted,
    )
    assert actual == expected
    assert actual == fits
