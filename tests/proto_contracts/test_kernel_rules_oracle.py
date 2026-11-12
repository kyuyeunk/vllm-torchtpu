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
import unittest

import _pure_import
from _chips import BF16, CHIPS, F32, I32, chip_name
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


class KernelRulesOracleTest(unittest.TestCase):
    """Compatibility paths of the kernels."""

    def test_no_tpu_runtime(self):
        self.assertEqual(_pure_import.loaded_tpu_runtime(), [])
        self.assertEqual(_pure_import.jax_backend(), "cpu")

    def test_gather_reduce_fallback(self):
        cases = 0
        for chip, rows, hidden, dtype in itertools.product(
            CHIPS, _ROWS, _HIDDENS, (BF16, F32)
        ):

            def get_tpu_info(chip=chip):
                return chip

            nbytes = dtype.itemsize
            expected = (
                (
                    GatherReduceVersion.V1,
                    _v1_kernel_path(rows, hidden, nbytes, get_tpu_info),
                ),
                (
                    GatherReduceVersion.V2,
                    _v2_kernel_path(rows, hidden, dtype, nbytes, get_tpu_info),
                ),
                (
                    GatherReduceVersion.V3,
                    _v3_kernel_path(rows, hidden, dtype, nbytes, get_tpu_info),
                ),
            )
            for version, path in expected:
                new = fb.takes_tensorcore_path(
                    version=version,
                    size=rows * hidden,
                    hidden=hidden,
                    dtype=dtype,
                    chip=chip,
                )
                self.assertEqual(
                    new,
                    path is KernelPath.TC_FALLBACK,
                    (version, chip, rows, hidden),
                )
                cases += 1
        print(f"\n[gather-reduce fallback] {cases} cases")

    def test_v3_partition(self):
        cases = 0
        for chip, hidden in itertools.product(CHIPS, _HIDDENS + (64, 192, 384, 3072)):
            if chip.sparse_core is None:
                continue
            self.assertEqual(
                v3.partition_counts(hidden, chip),
                partition_counts(hidden, chip),
            )
            self.assertEqual(
                v3.token_block_alignment(hidden, chip),
                token_block_alignment(hidden, chip),
            )
            cases += 1
        print(f"\n[v3 partition] {cases} cases")

    def test_owner_admission(self):
        cases = 0
        for chip, routes, hidden, tokens, dtype, extra in itertools.product(
            CHIPS,
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
                    (route_shape, dtype, lambda chip=chip: chip),
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
            self.assertEqual(
                actual, expected, (chip_name(chip), routes, hidden, tokens, dtype.name)
            )
            cases += 1
        print(f"\n[owner admission] {cases} cases")

    def test_owner_admission_vmem_boundary(self):
        """The VMEM check at equality and one byte under, on planted chips."""
        cases = 0
        for chip, tokens_per_tile in itertools.product(CHIPS, (1, 64)):
            if chip.generation > 8:  # noqa: PLR2004 - the last generation JAX tiles
                continue
            hidden = chip.mxu_column_size * 4
            tokens = chip.get_sublane_tiling(BF16) * tokens_per_tile
            required = (
                _explicit_vmem_bytes(128, hidden, tokens) + _VMEM_WORKSPACE_MARGIN_BYTES
            )
            for limit in (required, required - 1):
                vmem = -(-limit * 10 // 9)
                while int(vmem * 0.9) > limit:
                    vmem -= 1
                planted = dataclasses.replace(chip, vmem_capacity_bytes=vmem)
                self.assertEqual(int(vmem * 0.9), limit)
                expected = can_use_blockwise_onehot_unpermute(
                    ((128, hidden), BF16, lambda planted=planted: planted),
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
                self.assertEqual(actual, expected, (chip_name(chip), limit))
                self.assertEqual(actual, limit == required)
                cases += 1
        print(f"\n[owner VMEM boundary] {cases} cases")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    unittest.main()
