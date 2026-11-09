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

import itertools
import unittest

import _pure_import
from _chips import CHIPS
from test_moe_plan_oracle import (
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

_HIDDENS = (100, 128, 256, 512, 1024, 2048, 2880, 4096, 5120, 7168, 8192)
_ROWS = (1, 2, 16, 127, 128, 255, 256, 1000, 1024, 4096, 16384, 65536, 1 << 20)


class KernelRulesOracleTest(unittest.TestCase):
    """Compatibility paths of the kernels."""

    def test_no_runtime_imported(self):
        self.assertEqual(_pure_import.loaded_runtime_modules(), [])

    def test_gather_reduce_fallback(self):
        cases = 0
        for chip, rows, hidden, (dtype, nbytes) in itertools.product(
            CHIPS, _ROWS, _HIDDENS, (("bfloat16", 2), ("float32", 4))
        ):

            def get_tpu_info(chip=chip):
                return chip

            expected = {
                "v1": _v1_kernel_path(rows, hidden, nbytes, get_tpu_info),
                "v2": _v2_kernel_path(rows, hidden, dtype, nbytes, get_tpu_info),
                "v3": _v3_kernel_path(rows, hidden, dtype, nbytes, get_tpu_info),
            }
            for version, path in expected.items():
                new = fb.takes_tensorcore_path(
                    version=version,
                    size=rows * hidden,
                    hidden=hidden,
                    dtype=dtype,
                    dtype_bytes=nbytes,
                    vmem_capacity_bytes=chip.vmem_capacity_bytes,
                    has_sparse_core=chip.sparse_core is not None,
                    num_lanes=chip.num_lanes,
                )
                self.assertEqual(
                    new, path == "tc_fallback", (version, chip, rows, hidden)
                )
                cases += 1
        print(f"\n[gather-reduce fallback] {cases} cases")

    def test_v3_partition(self):
        cases = 0
        for chip, hidden in itertools.product(CHIPS, _HIDDENS + (64, 192, 384, 3072)):
            if chip.sparse_core is None:
                continue
            sc = chip.sparse_core
            cores = sc.num_cores * sc.num_subcores
            self.assertEqual(
                v3.partition_counts(hidden, cores, chip.num_lanes),
                partition_counts(hidden, chip),
            )
            self.assertEqual(
                v3.token_block_alignment(hidden, cores, chip.num_lanes),
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
            ("bfloat16", "float32"),
            (0, 1),
        ):
            route_shape = (routes, hidden)
            ti_shape = (routes + extra,)
            try:
                expected = can_use_blockwise_onehot_unpermute(
                    (route_shape, dtype, lambda chip=chip: chip),
                    (ti_shape, "int32"),
                    ((routes,), dtype),
                    ((1,), "int32"),
                    num_tokens=tokens,
                )
            except NotImplementedError:
                expected = "raise"
            try:
                actual = owner.shape_admits(
                    route_shape=route_shape,
                    route_dtype=dtype,
                    token_indices_shape=ti_shape,
                    token_indices_dtype="int32",
                    weights_shape=(routes,),
                    weights_dtype=dtype,
                    valid_count_shape=(1,),
                    valid_count_dtype="int32",
                ) and owner.chip_admits(
                    route_capacity=routes,
                    hidden_size=hidden,
                    num_tokens=tokens,
                    mxu_column_size=chip.mxu_column_size,
                    sublane_tiling=lambda chip=chip,
                    dtype=dtype: chip.get_sublane_tiling(dtype),
                    vmem_capacity_bytes=chip.vmem_capacity_bytes,
                )
            except NotImplementedError:
                actual = "raise"
            self.assertEqual(
                actual, expected, (chip.name, routes, hidden, tokens, dtype)
            )
            cases += 1
        print(f"\n[owner admission] {cases} cases")


if __name__ == "__main__":
    unittest.main()
