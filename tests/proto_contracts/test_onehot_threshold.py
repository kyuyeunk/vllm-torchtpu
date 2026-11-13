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

from collections.abc import Callable

import _pure_import
import pytest
from _chips import CHIPS, NAMED_CHIPS
from jax.experimental.pallas import tpu as pltpu

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


_EXPLICITS = (None, -5, -1, 0, 1, 2, 255, 256, 511, 512, 513, 4096)


def _raising() -> pltpu.TpuInfo:
    raise ValueError("Unsupported TPU device kind: cpu")


def _source(chip: pltpu.TpuInfo | None) -> Callable[[], pltpu.TpuInfo]:
    """A `get_tpu_info` stand-in: returns `chip`, or raises for no chip."""
    if chip is None:
        return _raising
    return lambda: chip


def test_no_tpu_runtime() -> None:
    """The module imported with no TPU runtime and runs JAX on the CPU."""
    assert _pure_import.loaded_tpu_runtime() == []
    assert _pure_import.jax_backend() == "cpu"  # str-ok: the backend name JAX reports


@pytest.mark.parametrize(
    "explicit", _EXPLICITS, ids=[f"explicit={explicit}" for explicit in _EXPLICITS]
)
@pytest.mark.parametrize(
    "chip",
    [None, *CHIPS],
    ids=["no_chip", *(name for name, _ in NAMED_CHIPS)],
)
def test_onehot_threshold_oracle(
    chip: pltpu.TpuInfo | None, explicit: int | None
) -> None:
    """The plan's threshold equals main's, for one chip and one env value."""
    source = _source(chip)
    assert _new_resolve(explicit, source) == _reference_resolve(explicit, source)
