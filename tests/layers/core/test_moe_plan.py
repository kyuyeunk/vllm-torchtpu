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
"""Which combines can run a call and which one runs, on CPU, for JAX's chips."""

import dataclasses

import pytest
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.layers.core import moe_plan
from vllm_torchtpu.layers.core.moe_plan import Combine, MoeCombinePlan, MoeProcess, Pick

_TPU7X = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
_V6E = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1)
_BF16 = jnp.dtype(jnp.bfloat16)
_ON = MoeProcess(use_sparse_core=True)
_OFF = MoeProcess(use_sparse_core=False)
_OWNER = MoeProcess(
    use_sparse_core=True, override=MoeCombinePlan(combine=Combine.OWNER_ONEHOT)
)


def _call(
    tokens: int, topk: int, weights: jnp.dtype = _BF16
) -> tuple[moe_plan.MoeOpIn, moe_plan.GmmOut]:
    op = moe_plan.MoeOpIn(num_tokens=tokens, topk=topk, local_num_experts=64)
    out = moe_plan.GmmOut(hidden=4096, dtype=_BF16, weights_dtype=weights)
    return op, out


def _counted(tokens: int, topk: int) -> tuple[moe_plan.MoeOpIn, moe_plan.GmmOut]:
    """A call whose routing passes its per-token count of local routes."""
    op, out = _call(tokens, topk)
    return op, dataclasses.replace(out, token_local_routes_shape=(tokens,))


# (process, chip, call, variant, the reason the variant cannot run or None).
_CAPABLE = {
    "owner output": (_ON, _TPU7X, _call(256, 10), Combine.OWNER_ONEHOT, None),
    "segment sum": (_ON, _V6E, _call(16, 8), Combine.SEGMENT_SUM, None),
    "owner output, fp32 weights": (
        _ON,
        _TPU7X,
        _call(256, 10, jnp.dtype(jnp.float32)),
        Combine.OWNER_ONEHOT,
        "weights must be BF16",
    ),
    "v3 on 8 lanes": (
        _ON,
        _V6E,
        _call(16, 8),
        Combine.GATHER_REDUCE_V3,
        "the SparseCore path needs 16-lane SparseCores",
    ),
    "v3 at top-1": (
        _ON,
        _TPU7X,
        _call(2048, 1),
        Combine.GATHER_REDUCE_V3,
        "the route-pair metadata needs two routes per token",
    ),
    "v4 at top-8": (
        _ON,
        _TPU7X,
        _call(16, 8),
        Combine.GATHER_REDUCE_V4,
        "top-k above 4",
    ),
    "skip_empty with route counts": (
        _ON,
        _V6E,
        _counted(256, 8),
        Combine.SKIP_EMPTY,
        None,
    ),
    "skip_empty without": (
        _ON,
        _V6E,
        _call(256, 8),
        Combine.SKIP_EMPTY,
        "needs routing's per-token count of local routes",
    ),
    "SparseCore off": (
        _OFF,
        _TPU7X,
        _call(16, 8),
        Combine.GATHER_REDUCE_V1,
        "SparseCore is off",
    ),
}


@pytest.mark.parametrize(
    ("process", "chip", "call", "variant", "reason"), _CAPABLE.values(), ids=_CAPABLE
)
def test_combine_capable(process, chip, call, variant, reason) -> None:
    assert moe_plan.combine_capable(process, *call, chip)[variant] == reason


# (process, chip, call, the pick, why).
_PICKS = {
    "v6e decode": (_ON, _V6E, _call(16, 8), Combine.TENSORCORE, Pick.DEFAULT),
    "tpu7x prefill": (_ON, _TPU7X, _call(2048, 4), Combine.TENSORCORE, Pick.DEFAULT),
    "no TPU": (_ON, None, _call(16, 8), Combine.TENSORCORE, Pick.DEFAULT),
    "override": (_OWNER, _TPU7X, _call(256, 10), Combine.OWNER_ONEHOT, Pick.OVERRIDE),
}


@pytest.mark.parametrize(
    ("process", "chip", "call", "pick", "why"), _PICKS.values(), ids=_PICKS
)
def test_pick_combine(process, chip, call, pick, why) -> None:
    assert moe_plan.pick_combine(process, *call, chip) == (
        MoeCombinePlan(combine=pick),
        why,
    )


def test_override_that_cannot_run_raises() -> None:
    with pytest.raises(ValueError, match="weights must be BF16"):
        moe_plan.pick_combine(_OWNER, *_call(256, 10, jnp.dtype(jnp.float32)), _TPU7X)
