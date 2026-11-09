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
"""CPU tests for C0 `TpuTarget` and its first user, the one-hot threshold."""

import types
import unittest

import _pure_import
from _chips import CHIPS

from vllm_torchtpu.layers.core import moe_plan
from vllm_torchtpu.layers.core.tpu_target import SparseCoreInfo, TpuTarget

_ONEHOT_AUTO_CAP = 512


def _reference_resolve(explicit, get_tpu_info):
    """`resolve_onehot_permute_threshold` verbatim from d3991ebf9.

    `envs.ONEHOT_MOE_PERMUTE_THRESHOLD` and `pltpu.get_tpu_info` are
    parameters; nothing else changed.
    """
    if explicit is not None and explicit >= 0:
        return explicit
    try:
        sc_info = get_tpu_info().sparse_core
    except ValueError:
        # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
        return 0
    if sc_info is None:
        return 0
    block_rows = sc_info.num_lanes * sc_info.num_cores * sc_info.num_subcores
    threshold = min(block_rows - 1, _ONEHOT_AUTO_CAP)

    return threshold


def _new_resolve(explicit, get_tpu_info):
    """`resolve_onehot_permute_threshold` after C0, with the same seams."""
    if explicit is not None and explicit >= 0:
        return explicit
    try:
        target = TpuTarget.from_tpu_info(get_tpu_info())
    except ValueError:
        target = None
    return moe_plan.auto_onehot_permute_threshold(target)


class TpuTargetTest(unittest.TestCase):
    """C0 adapter and threshold oracle."""

    def test_no_runtime_imported(self):
        self.assertEqual(_pure_import.loaded_runtime_modules(), [])

    def test_from_tpu_info_round_trip(self):
        for info in CHIPS:
            target = TpuTarget.from_tpu_info(info)
            self.assertEqual(target.generation, info.generation)
            self.assertEqual(target.vmem_bytes_per_core, info.vmem_capacity_bytes)
            self.assertEqual(target.num_lanes, info.num_lanes)
            self.assertEqual(target.mxu_column_size, info.mxu_column_size)
            if info.sparse_core is None:
                self.assertIsNone(target.sparse_core)
            else:
                assert target.sparse_core is not None
                self.assertEqual(
                    target.sparse_core.block_rows,
                    info.sparse_core.num_lanes
                    * info.sparse_core.num_cores
                    * info.sparse_core.num_subcores,
                )

    def test_from_tpu_info_tolerates_partial_fakes(self):
        # The shape existing fused_moe_gmm tests fake get_tpu_info with.
        fake = types.SimpleNamespace(
            sparse_core=types.SimpleNamespace(
                num_lanes=16, num_cores=2, num_subcores=16
            ),
            num_lanes=128,
            vmem_capacity_bytes=64 << 20,
        )
        target = TpuTarget.from_tpu_info(fake)
        self.assertEqual(
            target.sparse_core,
            SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16),
        )
        self.assertEqual(target.generation, 0)

    def test_sublane_tiling_matches_jax(self):
        for info in CHIPS:
            target = TpuTarget.from_tpu_info(info)
            for dtype, bits in (("bfloat16", 16), ("float32", 32)):
                try:
                    expected = info.get_sublane_tiling(dtype)
                except NotImplementedError:
                    with self.assertRaises(NotImplementedError):
                        target.sublane_tiling(bits)
                    continue
                self.assertEqual(target.sublane_tiling(bits), expected)

    def test_frozen_and_hashable(self):
        target = TpuTarget.from_tpu_info(CHIPS[-3])
        self.assertEqual(hash(target), hash(TpuTarget.from_tpu_info(CHIPS[-3])))
        with self.assertRaises(dataclasses_frozen_error()):
            target.generation = 1  # pyright: ignore[reportAttributeAccessIssue]

    def test_onehot_threshold_oracle(self):
        def raising():
            raise ValueError("Unsupported TPU device kind: cpu")

        sources = [raising] + [lambda info=info: info for info in CHIPS]
        explicits = [None, -5, -1, 0, 1, 2, 255, 256, 511, 512, 513, 4096]
        cases = 0
        for explicit in explicits:
            for source in sources:
                self.assertEqual(
                    _new_resolve(explicit, source),
                    _reference_resolve(explicit, source),
                    (explicit, source),
                )
                cases += 1
        self.assertEqual(cases, len(explicits) * len(sources))


def dataclasses_frozen_error() -> type[Exception]:
    import dataclasses

    return dataclasses.FrozenInstanceError


if __name__ == "__main__":
    unittest.main()
