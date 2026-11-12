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
"""CPU tests for the auto one-hot threshold, over every chip JAX describes."""

import unittest

import _pure_import
from _chips import CHIPS

from vllm_torchtpu.layers.core import moe_plan

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
    """`resolve_onehot_permute_threshold` with the same seams."""
    if explicit is not None and explicit >= 0:
        return explicit
    try:
        target = get_tpu_info()
    except ValueError:
        target = None
    return moe_plan.auto_onehot_permute_threshold(target)


class OnehotThresholdTest(unittest.TestCase):
    """The threshold oracle."""

    def test_no_tpu_runtime(self):
        self.assertEqual(_pure_import.loaded_tpu_runtime(), [])
        self.assertEqual(_pure_import.jax_backend(), "cpu")

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


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    unittest.main()
