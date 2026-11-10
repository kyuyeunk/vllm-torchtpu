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
"""Plan invariance of the MoE combine (`moe_plan.MoeCombinePlan`).

Forces every combine variant on contract-satisfying combine inputs and
compares each output with the output under the plan `moe_gmm` resolves.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.layers.core.plan_invariance import PlanCase, check_plan_invariance
from vllm_torchtpu.layers.core import fused_moe_gmm, moe_plan
from vllm_torchtpu.layers.core.tpu_target import TpuTarget


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Shape:
    num_tokens: int
    topk: int
    hidden: int
    global_experts: int
    local_experts: int
    dtype: jnp.dtype = jnp.bfloat16
    garbage: float = 1e3
    threshold: int | None = None  # None: the chip's auto threshold.

    @property
    def use_ep(self) -> bool:
        return self.local_experts < self.global_experts


# Shapes from test_fused_moe_gmm and the decode/prefill EP tests.
_SHAPES = {
    "ep_t16_k8_h2048": _Shape(
        num_tokens=16, topk=8, hidden=2048, global_experts=128, local_experts=32
    ),
    "ep_t16_k8_h2048_nan": _Shape(
        num_tokens=16,
        topk=8,
        hidden=2048,
        global_experts=128,
        local_experts=32,
        garbage=float("nan"),
    ),
    "ep_t512_k8_h4096": _Shape(
        num_tokens=512, topk=8, hidden=4096, global_experts=128, local_experts=32
    ),
    "tp_t512_k10_h4096": _Shape(
        num_tokens=512,
        topk=10,
        hidden=4096,
        global_experts=64,
        local_experts=64,
        threshold=0,
    ),
    "tp_t300_k10_h5120": _Shape(
        num_tokens=300,
        topk=10,
        hidden=5120,
        global_experts=64,
        local_experts=64,
        threshold=0,
    ),
    "tp_t64_k10_h4096_f32": _Shape(
        num_tokens=64,
        topk=10,
        hidden=4096,
        global_experts=64,
        local_experts=64,
        dtype=jnp.float32,
        threshold=0,
    ),
    "tp_t2048_k1_h4096": _Shape(
        num_tokens=2048,
        topk=1,
        hidden=4096,
        global_experts=64,
        local_experts=64,
        threshold=0,
    ),
}


def _case(name: str, shape: _Shape, threshold: int) -> PlanCase:
    """Combine inputs laid out the way `prepare_routed_gmm_inputs` lays them.

    Routes to experts this shard does not hold sort after the local ones and
    are masked; the second GMM never writes rows past the local prefix, so
    those rows hold `shape.garbage`.
    """
    rng = np.random.default_rng(0)
    routes = shape.num_tokens * shape.topk
    ids = np.stack(
        [
            rng.choice(shape.global_experts, shape.topk, replace=False)
            for _ in range(shape.num_tokens)
        ]
    ).reshape(-1)
    keys = np.minimum(ids, shape.local_experts)
    order = np.argsort(keys, kind="stable")
    revert = np.argsort(order, kind="stable")
    valid = keys < shape.local_experts
    group_sizes = np.bincount(keys, minlength=shape.local_experts + 1)
    computed = int(valid.sum())
    # One second-GMM row per route: the one-hot combine contracts over
    # routes and fails on any other row count, though `GmmOutSpec` keeps the
    # two counts apart.
    gmm2 = np.full((routes, shape.hidden), shape.garbage, np.float32)
    gmm2[:computed] = rng.standard_normal((computed, shape.hidden))
    weights = rng.uniform(0.0, 1.0, routes) / shape.topk
    args = (
        jnp.asarray(gmm2, shape.dtype),
        jnp.asarray(group_sizes[: shape.local_experts], jnp.int32),
        jnp.asarray(revert, jnp.int32),
        jnp.asarray(weights, shape.dtype),
        jnp.asarray(valid),
        jnp.asarray(order // shape.topk, jnp.int32),
    )
    return PlanCase(
        name=name,
        args=args,
        check_values=True,
        meta=dataclasses.replace(shape, threshold=threshold),
    )


def _combine(case: PlanCase, plan: moe_plan.MoeCombinePlan, *args: jax.Array):
    shape: _Shape = case.meta
    return fused_moe_gmm.moe_combine(
        *args,
        plan=plan,
        num_tokens=shape.num_tokens,
        topk=shape.topk,
        out_dtype=shape.dtype,
    )


def _facts_kwargs(case: PlanCase) -> dict:
    shape: _Shape = case.meta
    return {
        "num_tokens": shape.num_tokens,
        "topk": shape.topk,
        "local_num_experts": shape.local_experts,
        "use_ep": shape.use_ep,
    }


def _facts_args(case: PlanCase) -> tuple[jax.Array, ...]:
    gmm2, _, revert, weights, _, token_indices = case.args
    return gmm2, revert, weights, token_indices


def _resolve(case: PlanCase) -> moe_plan.MoeCombinePlan:
    return fused_moe_gmm.resolve_combine_plan(
        *_facts_args(case),
        **_facts_kwargs(case),
        use_sparse_core=True,
        onehot_moe_permute_threshold=case.meta.threshold,
    )


def _capable(case: PlanCase, plan: moe_plan.MoeCombinePlan) -> bool:
    op, out = fused_moe_gmm.combine_facts(*_facts_args(case), **_facts_kwargs(case))
    return moe_plan.combine_capable(plan, op, out, TpuTarget.from_runtime())


def _tolerance(case: PlanCase) -> tuple[float, float]:
    # A BF16 sum of top-k terms of magnitude about one, in any order.
    if case.meta.dtype == jnp.bfloat16:
        return 2.0**-6, 2.0**-5
    return 1e-3, 1e-3


def _plan(
    combine: moe_plan.Combine, version: moe_plan.GatherReduceVersion | None = None
) -> moe_plan.MoeCombinePlan:
    return moe_plan.MoeCombinePlan(combine=combine, ep_gather_reduce_version=version)


# Real findings, kept as a ratchet: each must still fail, and nothing else may.
# With FP32 inputs the TensorCore combine stays in FP32, while the one-hot
# matmul and the v1/v2 TensorCore fallbacks round at about BF16 precision
# (max difference 2e-3 to 3e-3 on outputs of magnitude 1). The output's
# precision depends on the plan, so it is contract filed as plan.
_KNOWN = {
    ("tp_t64_k10_h4096_f32", _plan("ep_onehot")),
    ("tp_t64_k10_h4096_f32", _plan("ep_ragged_tc_fallback", "v1")),
    ("tp_t64_k10_h4096_f32", _plan("ep_ragged_tc_fallback", "v2")),
}


@pytest.mark.tpu_test
def test_combine_plan_invariance():
    if jax.default_backend() != "tpu":
        pytest.fail(f"Expected JAX TPU backend, got {jax.default_backend()}.")
    threshold = moe_plan.auto_onehot_permute_threshold(TpuTarget.from_runtime())
    cases = [
        _case(name, shape, threshold if shape.threshold is None else shape.threshold)
        for name, shape in _SHAPES.items()
    ]
    report = check_plan_invariance(
        _combine,
        cases,
        moe_plan.MoeCombinePlan,
        _tolerance,
        resolve=_resolve,
        capable=_capable,
        # The v3 kernel pads tokens to 64 per SparseCore row partition.
        domains={"local_v3_token_alignment": (None, 64, 128, 256)},
    )
    failures = {(o.case, o.plan) for o in report.failures()}
    assert failures == _KNOWN, report.summary()
