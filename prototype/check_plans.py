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
"""The structural tier for the MoE combine, on a CPU with fake chips.

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.check_plans

Loads libtpu to compile, so run it under the TPU lock, never beside a TPU job.
Compiles `moe_combine` under every `MoeCombinePlan` value, ahead of time, for
v6e and TPU7x topologies, which JAX describes without a device; `cases` and
`declared` take the fake chip as an argument, and `pltpu.get_tpu_info()`
answers from the abstract mesh, so the planner and the kernels see it too.
Prints every finding and checks them against the known set.
"""

import dataclasses
import time

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from prototype.plan_check import Case, Chip, Finding, Kind, check_structure
from vllm_torchtpu.layers.core import fused_moe_gmm, moe_plan
from vllm_torchtpu.layers.core.moe_plan import Combine, GatherReduceVersion

V6E = Chip("v6e", "v6e:2x2")
TPU7X = Chip("tpu7x", "tpu7x:2x2x2")
CHIPS = (V6E, TPU7X)

# The v3 kernel pads tokens to 64 per SparseCore row partition; None comes
# from the field's optional type.
DOMAINS = {int: (64, 128, 256)}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Shape:
    """One combine call."""

    name: str
    num_tokens: int
    topk: int
    hidden: int
    global_experts: int
    local_experts: int
    dtype: jnp.dtype = jnp.bfloat16
    pad_rows: int = 0  # Second-GMM rows past one per route.
    threshold: int | None = None  # None: the chip's auto one-hot threshold.

    @property
    def use_ep(self) -> bool:
        """Whether the shard holds only some of the experts."""
        return self.local_experts < self.global_experts


# Shapes from the fused-MoE and EP tests. EP holds a quarter of the experts.
EP_T16 = Shape(
    name="ep_t16_k8_h2048",
    num_tokens=16,
    topk=8,
    hidden=2048,
    global_experts=128,
    local_experts=32,
)
EP_T512 = Shape(
    name="ep_t512_k8_h4096",
    num_tokens=512,
    topk=8,
    hidden=4096,
    global_experts=128,
    local_experts=32,
)
TP_T512 = Shape(
    name="tp_t512_k10_h4096",
    num_tokens=512,
    topk=10,
    hidden=4096,
    global_experts=64,
    local_experts=64,
    threshold=0,
)
TP_T300 = Shape(
    name="tp_t300_k10_h5120",
    num_tokens=300,
    topk=10,
    hidden=5120,
    global_experts=64,
    local_experts=64,
    threshold=0,
)
TP_T64_F32 = Shape(
    name="tp_t64_k10_h4096_f32",
    num_tokens=64,
    topk=10,
    hidden=4096,
    global_experts=64,
    local_experts=64,
    dtype=jnp.float32,
    threshold=0,
)
TP_T2048_K1 = Shape(
    name="tp_t2048_k1_h4096",
    num_tokens=2048,
    topk=1,
    hidden=4096,
    global_experts=64,
    local_experts=64,
    threshold=0,
)
# GmmOutSpec records rows apart from routes, so the contract allows a padded
# second-GMM output.
EP_T16_PAD = Shape(
    name="ep_t16_k8_h2048_pad",
    num_tokens=16,
    topk=8,
    hidden=2048,
    global_experts=128,
    local_experts=32,
    pad_rows=128,
)
SHAPES = (EP_T16, EP_T512, TP_T512, TP_T300, TP_T64_F32, TP_T2048_K1, EP_T16_PAD)


def plan(
    combine: Combine,
    version: GatherReduceVersion | None = None,
    alignment: int | None = None,
) -> moe_plan.MoeCombinePlan:
    """One MoeCombinePlan value."""
    return moe_plan.MoeCombinePlan(
        combine=combine,
        ep_gather_reduce_version=version,
        local_v3_token_alignment=alignment,
    )


_V3 = plan(Combine.EP_RAGGED_SC, GatherReduceVersion.V3)
_V1 = plan(Combine.EP_RAGGED_SC, GatherReduceVersion.V1)
_ONEHOT = plan(Combine.EP_ONEHOT)

# Comment in to replay the EP kernel chosen at import time, not by the plan.
REPLAY_IMPORT_TIME_KERNEL = False
# REPLAY_IMPORT_TIME_KERNEL = True

# Findings, kept as a ratchet: each must still appear, and nothing else may.
KNOWN = {
    # EP v3 at top-1: declared capable (local v3 checks topk >= 2, EP v3 does
    # not), but its route-pair metadata fails to trace.
    Finding(Kind.FAILS, TPU7X, TP_T2048_K1.name, _V3),
    # The input facts allow more second-GMM rows than routes, but the one-hot
    # combine contracts over routes, and it and the v1 gather-reduce kernel
    # both fail to compile on a padded output.
    *(
        Finding(Kind.FAILS, chip, EP_T16_PAD.name, p)
        for chip in CHIPS
        for p in (_ONEHOT, _V1)
    ),
    # v3 is declared BF16-only, yet local and EP v3 compile with FP32 input;
    # only a TPU7x run can show whether they fail or round.
    Finding(Kind.COMPILES, TPU7X, TP_T64_F32.name, _V3),
    *(
        Finding(
            Kind.COMPILES,
            TPU7X,
            TP_T64_F32.name,
            plan(Combine.LOCAL_V3, alignment=a),
        )
        for a in (64, 128, 256)
    ),
    # The v3 kernel derives its own token alignment, so on the one chip where
    # local v3 runs, every alignment compiles to the same program.
    Finding(Kind.UNREAD, TPU7X, field="local_v3_token_alignment"),
}

KNOWN_V3_TOP1 = Finding(Kind.FAILS, TPU7X, TP_T2048_K1.name, _V3)
PADDED_V1 = {Finding(Kind.FAILS, chip, EP_T16_PAD.name, _V1) for chip in CHIPS}

# With the replay, the version is unread again, EP v3 runs as the
# import-time kernel even where v3 is declined, and EP v1 is the import-time
# kernel too, so it no longer fails on the padded output.
REPLAYED = {
    *(Finding(Kind.UNREAD, chip, field="ep_gather_reduce_version") for chip in CHIPS),
    *(Finding(Kind.COMPILES, V6E, s.name, _V3) for s in SHAPES),
}


def cases(chip: pltpu.TpuInfo) -> list[Case]:
    """One case per shape, at the chip's one-hot threshold."""
    auto = moe_plan.auto_onehot_permute_threshold(chip)
    out = []
    for s in SHAPES:
        routes = s.num_tokens * s.topk
        sds = jax.ShapeDtypeStruct
        args = (
            sds((routes + s.pad_rows, s.hidden), s.dtype),  # gmm2_res
            sds((s.local_experts,), jnp.int32),  # group_sizes
            sds((routes,), jnp.int32),  # argsort_revert_indices
            sds((routes,), s.dtype),  # topk_weights_flat
            sds((routes,), jnp.bool_),  # valid_mask_flat
            sds((routes,), jnp.int32),  # token_indices_sorted
        )
        threshold = auto if s.threshold is None else s.threshold
        out.append(Case(s.name, args, dataclasses.replace(s, threshold=threshold)))
    return out


def entry(case: Case, plan: moe_plan.MoeCombinePlan, *args: jax.Array) -> jax.Array:
    """`moe_combine` on one case under `plan`."""
    s = case.meta
    return fused_moe_gmm.moe_combine(
        *args, plan=plan, num_tokens=s.num_tokens, topk=s.topk, out_dtype=s.dtype
    )


def resolve(case: Case) -> moe_plan.MoeCombinePlan:
    """The plan `moe_gmm` resolves on the mesh's chip."""
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


def declared(case: Case, plan: moe_plan.MoeCombinePlan, chip: pltpu.TpuInfo) -> bool:
    """Whether the module declares `plan` able to run the case on `chip`."""
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
    return moe_plan.combine_capable(plan, op, out, chip)


def _lines(findings: set[Finding]) -> list[str]:
    return [" ".join(c) for c in sorted(f.columns() for f in findings)]


def main() -> None:
    """Prints the findings and checks them against `KNOWN`."""
    start = time.perf_counter()
    known = KNOWN
    if REPLAY_IMPORT_TIME_KERNEL:
        fused_moe_gmm._ep_gather_reduce = (  # noqa: SLF001 - the replay's seam
            lambda version: fused_moe_gmm.ragged_gather_reduce
        )
        known = (KNOWN - {KNOWN_V3_TOP1, *PADDED_V1}) | REPLAYED
    findings = check_structure(
        entry,
        cases,
        moe_plan.MoeCombinePlan,
        CHIPS,
        resolve=resolve,
        declared=declared,
        domains=DOMAINS,
    )
    for line in _lines(findings):
        print(line)
    print(f"findings: {len(findings)} in {time.perf_counter() - start:.0f} s")
    assert findings == known, (
        f"new: {_lines(findings - known)}, gone: {_lines(known - findings)}"
    )


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
