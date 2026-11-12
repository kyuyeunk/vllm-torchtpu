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
"""The numerics tier for the MoE combine, on the host's TPU.

    flock /tmp/vt-tpu.lock env PYTHONPATH=src python -m prototype.check_plan_values

Runs `moe_combine` under every `MoeCombinePlan` value the module declares
capable and compares each output with the picked plan's, within the
tolerance below. Two families of contract-satisfying inputs:

- laid out as routing lays them: routes to experts this shard does not hold
  sort last and are masked, and the second GMM leaves their rows unwritten
  (garbage, or NaN);
- masked at random: a tenth of the slots masked anywhere, which the combine's
  inputs equally allow.

Prints the findings and checks them against the known set.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from prototype.check_plans import (
    DOMAINS,
    EP_T16,
    SHAPES,
    TP_T64_F32,
    TP_T512,
    TP_T2048_K1,
    Shape,
    plan,
)
from prototype.plan_check_values import Case, Finding, check_values
from vllm_torchtpu.layers.core import fused_moe_gmm, moe_plan
from vllm_torchtpu.layers.core.moe_plan import Combine, GatherReduceVersion

# The test author's tolerances: about two BF16 ulps, or FP32 at 1e-3.
TOLERANCE = {
    jnp.dtype(jnp.bfloat16): (2**-6, 2**-5),
    jnp.dtype(jnp.float32): (1e-3, 1e-3),
}

_MASK = "_mask"  # Name suffix of the random-mask cases.

# Findings on v6e, kept as a ratchet: each must still appear, and nothing else
# may. Each is (case name, plan).
KNOWN = {
    # Precision is filed as plan: with FP32 inputs the TensorCore combine stays
    # in FP32, while the one-hot matmul and the v1/v2 TensorCore fallbacks
    # round at about BF16 precision.
    (TP_T64_F32.name, plan(Combine.EP_ONEHOT)),
    *(
        (TP_T64_F32.name, plan(Combine.EP_RAGGED_TC_FALLBACK, v))
        for v in (GatherReduceVersion.V1, GatherReduceVersion.V2)
    ),
    # The owner-output kernel reads the valid count, not the mask: it sums
    # masked slots inside the computed prefix with their weights. Routing
    # masks only non-local routes, which sort past the prefix, but the
    # combine's inputs do not record that.
    *((s.name + _MASK, plan(Combine.EP_OWNER_ONEHOT)) for s in (TP_T512, TP_T2048_K1)),
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class _Inputs:
    """How one case's arrays are made."""

    shape: Shape
    random_mask: bool  # Mask a tenth of the slots anywhere, not by routing.
    garbage: float  # The value of second-GMM rows routing leaves unwritten.


# Routing's layout, plus one NaN-garbage case; random masks on the TP shapes.
_INPUTS = (
    *(
        _Inputs(shape=s, random_mask=False, garbage=1e3)
        for s in SHAPES
        if not s.pad_rows
    ),
    _Inputs(
        shape=dataclasses.replace(EP_T16, name=EP_T16.name + "_nan"),
        random_mask=False,
        garbage=np.nan,
    ),
    *(
        _Inputs(
            shape=dataclasses.replace(s, name=s.name + _MASK),
            random_mask=True,
            garbage=1e3,
        )
        for s in SHAPES
        if not s.use_ep and s.dtype == jnp.bfloat16
    ),
)


def _arrays(inputs: _Inputs, rng: np.random.Generator) -> tuple[jax.Array, ...]:
    s = inputs.shape
    routes = s.num_tokens * s.topk
    ids = np.stack(
        [
            rng.choice(s.global_experts, s.topk, replace=False)
            for _ in range(s.num_tokens)
        ]
    ).reshape(-1)
    keys = np.minimum(ids, s.local_experts)
    order = np.argsort(keys, kind="stable")
    revert = np.argsort(order, kind="stable")
    valid = keys < s.local_experts
    computed = int(valid.sum())
    if inputs.random_mask:
        valid = rng.uniform(0, 1, routes) < 0.9
    gmm2 = np.full((routes, s.hidden), inputs.garbage, np.float32)
    gmm2[:computed] = rng.standard_normal((computed, s.hidden))
    group_sizes = np.bincount(keys, minlength=s.local_experts + 1)[: s.local_experts]
    return (
        jnp.asarray(gmm2, s.dtype),
        jnp.asarray(group_sizes, jnp.int32),
        jnp.asarray(revert, jnp.int32),
        jnp.asarray(rng.uniform(0, 1, routes) / s.topk, s.dtype),
        jnp.asarray(valid),
        jnp.asarray(order // s.topk, jnp.int32),
    )


def cases() -> list[Case]:
    """Both families, at the host chip's one-hot threshold."""
    auto = moe_plan.auto_onehot_permute_threshold(pltpu.get_tpu_info())
    out = []
    for inputs in _INPUTS:
        s = inputs.shape
        meta = dataclasses.replace(
            s, threshold=auto if s.threshold is None else s.threshold
        )
        out.append(Case(s.name, _arrays(inputs, np.random.default_rng(0)), meta))
    return out


def _combine(s: Shape, plan, *args: jax.Array) -> jax.Array:
    return fused_moe_gmm.moe_combine(
        *args, plan=plan, num_tokens=s.num_tokens, topk=s.topk, out_dtype=s.dtype
    )


def entry(case: Case, plan: moe_plan.MoeCombinePlan, *args: jax.Array) -> jax.Array:
    """`moe_combine` on one case under `plan`, compiled."""
    return jax.jit(functools.partial(_combine, case.meta, plan))(*args)


def resolve(case: Case) -> moe_plan.MoeCombinePlan:
    """The plan `moe_gmm` resolves on this chip."""
    gmm2, _, revert, weights, _, token_indices = case.args
    s = case.meta
    return fused_moe_gmm.resolve_combine_plan(
        gmm2,
        revert,
        weights,
        token_indices,
        num_tokens=s.num_tokens,
        topk=s.topk,
        local_num_experts=s.local_experts,
        use_ep=s.use_ep,
        use_sparse_core=True,
        onehot_moe_permute_threshold=s.threshold,
    )


def declared(case: Case, plan: moe_plan.MoeCombinePlan) -> bool:
    """Whether the module declares `plan` able to run the case on this chip."""
    gmm2, _, revert, weights, _, token_indices = case.args
    s = case.meta
    op, out = fused_moe_gmm.combine_facts(
        gmm2,
        revert,
        weights,
        token_indices,
        num_tokens=s.num_tokens,
        topk=s.topk,
        local_num_experts=s.local_experts,
        use_ep=s.use_ep,
    )
    return moe_plan.combine_capable(plan, op, out, pltpu.get_tpu_info())


def tolerance(case: Case) -> tuple[float, float]:
    """(rtol, atol) for the case's dtype."""
    return TOLERANCE[jnp.dtype(case.meta.dtype)]


def _lines(findings: set[Finding]) -> list[str]:
    return [" ".join(c) for c in sorted(f.columns() for f in findings)]


def main() -> None:
    """Prints the findings and checks them against `KNOWN`."""
    if jax.default_backend() != "tpu":  # str-ok: JAX names its backend by string
        raise SystemExit(f"needs a TPU, got {jax.default_backend()}")
    findings = check_values(
        entry,
        cases(),
        moe_plan.MoeCombinePlan,
        tolerance,
        resolve=resolve,
        declared=declared,
        domains=DOMAINS,
    )
    print(f"chip: {jax.devices()[0].device_kind}")
    for line in _lines(findings):
        print(line)
    # The error size is printed, not ratcheted.
    found = {(f.case, f.plan) for f in findings}
    new, gone = sorted(map(str, found - KNOWN)), sorted(map(str, KNOWN - found))
    assert found == KNOWN, f"new: {new}, gone: {gone}"


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
