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
"""Oracles: `moe_plan`'s decisions against main's decision code at d3991ebf9.

Each `_reference_*` function below is the decision code of
`layers/core/fused_moe_gmm.py` (and of the kernels it calls) at d3991ebf9,
copied verbatim except that arrays are replaced by their static shapes and
every array op is replaced by an event appended to a list. Each test runs a
reference and the real plan function on the same inputs and compares the
reference's choice with the plan the function returns: which sort, permute,
entry path and combine, and what each reads. The `fused_moe_gmm` glue that
turns a plan into array ops, environment reads and the runtime chip read is
not restated here; the identity sweep checks it against main.

Each grid holds the decision-relevant values only: both sides of every
threshold and capability condition, the 16 chips (and no chip), and each
dtype and setting that changes a decision.
"""

import collections
import dataclasses
import enum
import functools
import itertools

import _pure_import
import jax.numpy as jnp
import pytest
from _chips import BF16, CHIPS, F32, I32, chip_name
from _grid import GridFamily
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.layers.core import moe_plan
from vllm_torchtpu.layers.core.moe_plan import (
    Combine,
    GatherReduceVersion,
    MoePath,
    OwnerOutputMode,
    Permute,
    SortForm,
    TargetNeed,
)


class KernelPath(enum.Enum):
    """Where a gather-reduce kernel entry goes."""

    TC_FALLBACK = enum.auto()
    SC = enum.auto()
    RAISE = enum.auto()


class _Ev(enum.Enum):
    """What main's code does, in the reference traces."""

    SCRUB_NONFINITE = enum.auto()
    DENSE_EXPERT = enum.auto()
    ROUTED = enum.auto()
    COUNTING_SORT = enum.auto()
    SCATTER_TOKENS = enum.auto()
    ARGSORT_KEYS = enum.auto()
    GROUP_SIZES_FROM_COUNTS = enum.auto()
    SINGLE_SORT = enum.auto()  # With the flat-index bits.
    ARGSORT_REVERT = enum.auto()
    GROUP_SIZES_ONEHOT = enum.auto()
    PACKED_TOKEN_SORT = enum.auto()  # With the token bits.
    GATHER_TOKENS = enum.auto()
    PERMUTE_ONEHOT_FROM_POSITIONS = enum.auto()
    PERMUTE_ONEHOT_FROM_TOKENS = enum.auto()
    PERMUTE_RAGGED_SC = enum.auto()
    PERMUTE_GATHER = enum.auto()


class _Ref(enum.Enum):
    """Reference outcomes that are not one `Combine`."""

    EP_RAGGED = "ep_ragged"  # The configured EP kernel, on either of its paths.
    RAISE = "raise"


# ---------------------------------------------------------------------------
# Reference: d3991ebf9 fused_moe_gmm.py, verbatim decision code.
# ---------------------------------------------------------------------------
_SINGLE_SORT_MAX_ROWS = 2048
_COUNTING_SORT_BLOCK = 512
_COUNTING_SORT_SCATTER_MAX_ROWS = _COUNTING_SORT_BLOCK


def _reference_entry(
    *,
    num_tokens,
    topk,
    use_ep,
    use_sparse_core,
    onehot_moe_permute_threshold,
    _DENSE_EXPERT_THRESHOLD,
    can_use_dense_expert,
):
    """fused_moe_func lines 736-767: the scrub and the dense-expert choice."""
    events = []
    if use_ep and use_sparse_core and num_tokens * topk <= onehot_moe_permute_threshold:
        events.append(_Ev.SCRUB_NONFINITE)
    if 0 < num_tokens <= _DENSE_EXPERT_THRESHOLD and can_use_dense_expert():
        events.append(_Ev.DENSE_EXPERT)
        return events
    events.append(_Ev.ROUTED)
    return events


def _reference_prepare(
    *,
    num_tokens_local,
    topk,
    local_num_experts,
    use_ep,
    use_sparse_core,
    onehot_moe_permute_threshold,
    _SINGLE_SORT,
    _COUNTING_SORT,
    inject_refusal=False,
):
    """prepare_routed_gmm_inputs lines 374-481, arrays replaced by events.

    `inject_refusal` stands in for a test that patches
    `_counting_sort_positions` to return (None, None).
    """
    events = []
    token_indices_flat_rows = num_tokens_local * topk

    def _counting_sort_positions(m, num_buckets):
        if inject_refusal:
            return None, None
        block = min(_COUNTING_SORT_BLOCK, m)
        if num_buckets > block:
            return None, None
        events.append(_Ev.COUNTING_SORT)
        return "positions", "counts"

    ragged_permute = (
        use_ep
        and use_sparse_core
        and token_indices_flat_rows > onehot_moe_permute_threshold
    )
    counting_sort = ragged_permute or (
        _COUNTING_SORT and token_indices_flat_rows <= _SINGLE_SORT_MAX_ROWS
    )
    argsort_revert_indices, bucket_counts = (
        _counting_sort_positions(token_indices_flat_rows, local_num_experts + 1)
        if counting_sort
        else (None, None)
    )
    token_bits = max(1, (num_tokens_local - 1).bit_length())
    packable = (local_num_experts + 1).bit_length() + token_bits < 31
    flat_bits = max(1, (token_indices_flat_rows - 1).bit_length())
    single_sort = (
        _SINGLE_SORT
        and argsort_revert_indices is None
        and not ragged_permute
        and token_indices_flat_rows <= _SINGLE_SORT_MAX_ROWS
        and (local_num_experts + 1).bit_length() + flat_bits < 31
    )
    token_indices_sorted = None
    if (
        argsort_revert_indices is not None
        and not ragged_permute
        and token_indices_flat_rows <= _COUNTING_SORT_SCATTER_MAX_ROWS
    ):
        events.append(_Ev.SCATTER_TOKENS)
        token_indices_sorted = "scattered"
    sorted_indices = None
    if (
        token_indices_sorted is None
        and not single_sort
        and (argsort_revert_indices is None or not packable)
    ):
        events.append(_Ev.ARGSORT_KEYS)
        sorted_indices = "sorted"

    if argsort_revert_indices is not None:
        events.append(_Ev.GROUP_SIZES_FROM_COUNTS)
    else:
        if single_sort:
            events.append((_Ev.SINGLE_SORT, flat_bits))
            token_indices_sorted = "single"
        else:
            assert sorted_indices is not None
            events.append(_Ev.ARGSORT_REVERT)
        events.append(_Ev.GROUP_SIZES_ONEHOT)

    if token_indices_sorted is None:
        if packable:
            events.append((_Ev.PACKED_TOKEN_SORT, token_bits))
        else:
            events.append(_Ev.GATHER_TOKENS)

    if use_ep and use_sparse_core:
        if token_indices_flat_rows <= onehot_moe_permute_threshold:
            if bucket_counts is not None:
                events.append(_Ev.PERMUTE_ONEHOT_FROM_POSITIONS)
            else:
                events.append(_Ev.PERMUTE_ONEHOT_FROM_TOKENS)
        else:
            events.append(_Ev.PERMUTE_RAGGED_SC)
    else:
        events.append(_Ev.PERMUTE_GATHER)
    return events


# Verbatim from kernels/sparse_core/ragged_gather_reduce_v3.py at d3991ebf9.
_TOKEN_SUBCHUNK = 16
_NUM_TOKEN_SUBCHUNKS = 4
_TOKEN_BLOCK = _TOKEN_SUBCHUNK * _NUM_TOKEN_SUBCHUNKS
_ROUTE_INDEX_BITS = 20


def _calculate_num_column_partitions(
    hidden_size: int, num_cores: int, num_lanes: int
) -> int:
    preferred_num_stages = 4
    num_column_partitions = 1
    while (
        num_cores % (num_column_partitions * 2) == 0
        and hidden_size % (num_lanes * num_column_partitions * 2) == 0
        and hidden_size // (num_column_partitions * 2 * num_lanes)
        >= preferred_num_stages
    ):
        num_column_partitions *= 2
    if num_column_partitions >= 2 and hidden_size <= 1024 * (
        num_column_partitions // 2
    ):
        num_column_partitions //= 2
    return num_column_partitions


def partition_counts(hidden_size: int, tpu_info) -> tuple[int, int]:
    sc_info = tpu_info.sparse_core
    num_cores = sc_info.num_cores * sc_info.num_subcores
    num_column_partitions = _calculate_num_column_partitions(
        hidden_size, num_cores, tpu_info.num_lanes
    )
    return num_column_partitions, num_cores // num_column_partitions


def token_block_alignment(hidden_size: int, tpu_info) -> int:
    return partition_counts(hidden_size, tpu_info)[1] * _TOKEN_BLOCK


def _v3_kernel_path(x_rows, x_hidden, x_dtype, x_bytes, get_tpu_info):
    """ragged_gather_reduce_v3.ragged_gather_reduce entry checks."""
    sc_info = get_tpu_info().sparse_core
    if sc_info is None or x_dtype != BF16 or x_hidden % get_tpu_info().num_lanes != 0:
        return KernelPath.TC_FALLBACK
    if x_rows * x_hidden * x_bytes * 2 < get_tpu_info().vmem_capacity_bytes * 0.6:
        return KernelPath.TC_FALLBACK
    if x_rows % 2:
        x_rows += 1
    if x_rows > 1 << _ROUTE_INDEX_BITS:
        return KernelPath.RAISE
    return KernelPath.SC


def _v1_kernel_path(x_rows, x_hidden, x_bytes, get_tpu_info):
    """ragged_gather_reduce.ragged_gather_reduce (v1) entry checks."""
    sc_info = get_tpu_info().sparse_core
    if sc_info is None:
        return KernelPath.TC_FALLBACK
    if x_rows * x_hidden * x_bytes * 2 < get_tpu_info().vmem_capacity_bytes * 0.6:
        return KernelPath.TC_FALLBACK
    return KernelPath.SC


def _v2_kernel_path(x_rows, x_hidden, x_dtype, x_bytes, get_tpu_info):
    """ragged_gather_reduce_v2 wrapper: Config(...) then should_fallback."""
    assert x_dtype in (BF16, F32)
    tpu_info = get_tpu_info()
    if tpu_info.sparse_core is None:
        return KernelPath.TC_FALLBACK
    vmem_capacity_threshold = tpu_info.vmem_capacity_bytes * 0.6
    source_size = x_rows * x_hidden * x_bytes
    if source_size * 2 < vmem_capacity_threshold:
        return KernelPath.TC_FALLBACK
    return KernelPath.SC


# Verbatim from kernels/megablox/moe_onehot_unpermute.py at d3991ebf9.
_ROUTE_BLOCK_SIZE = 128
_VMEM_WORKSPACE_MARGIN_BYTES = 4 * 1024 * 1024


def _explicit_vmem_bytes(route_capacity: int, hidden_size: int, num_tokens: int) -> int:
    owner_and_store = num_tokens * hidden_size * (4 + 2)
    double_buffered_routes = 2 * _ROUTE_BLOCK_SIZE * hidden_size * 2
    route_metadata = route_capacity * (4 + 4)
    return owner_and_store + double_buffered_routes + route_metadata


def can_use_blockwise_onehot_unpermute(
    route_output, token_indices_sorted, topk_weights_sorted, valid_count, *, num_tokens
):
    """Arrays are (shape, dtype) tuples; `get_tpu_info` is the last field."""
    route_output_shape, route_output_dtype, get_tpu_info = route_output
    if len(route_output_shape) != 2 or route_output_dtype != BF16:
        return False
    route_capacity, hidden_size = route_output_shape
    if token_indices_sorted[0] != (route_capacity,):
        return False
    if token_indices_sorted[1] != I32:
        return False
    if topk_weights_sorted[0] != (route_capacity,):
        return False
    if topk_weights_sorted[1] != BF16:
        return False
    if valid_count[0] != (1,) or valid_count[1] != I32:
        return False

    tpu_info = get_tpu_info()
    if route_capacity % _ROUTE_BLOCK_SIZE:
        return False
    if hidden_size % tpu_info.mxu_column_size:
        return False
    if num_tokens % tpu_info.get_sublane_tiling(route_output_dtype):
        return False
    vmem_limit_bytes = int(tpu_info.vmem_capacity_bytes * 0.9)
    required_vmem_bytes = (
        _explicit_vmem_bytes(route_capacity, hidden_size, num_tokens)
        + _VMEM_WORKSPACE_MARGIN_BYTES
    )
    return required_vmem_bytes <= vmem_limit_bytes


def _reference_combine(
    *,
    out,
    dtype,
    num_tokens,
    topk,
    use_ep,
    use_sparse_core,
    onehot_moe_permute_threshold,
    owner_output_mode,
    configured_version,
    get_tpu_info,
):
    """moe_gmm lines 552-658 (+ the kernel entry it reaches).

    `dtype` is the second GMM output's and the top-k weights' dtype; token
    indices are int32. Returns a tuple headed by a `Combine` or a `_Ref`.
    """
    gmm2_rows, gmm2_hidden, gmm2_dtype = out.rows, out.hidden, dtype
    use_local_ragged_combine = (
        use_sparse_core
        and not use_ep
        and topk >= 2
        and topk != 8
        and out.routes > onehot_moe_permute_threshold
        and gmm2_dtype == BF16
        and gmm2_rows <= 1 << 20
    )
    if use_local_ragged_combine:
        try:
            tpu_info = get_tpu_info()
        except ValueError:
            tpu_info = None
        sc_info = tpu_info.sparse_core if tpu_info is not None else None
        # Verbatim, so pyright's d3991ebf9 complaint comes along: it cannot see
        # that sc_info is not None implies tpu_info is not None.
        if (
            sc_info is not None
            and sc_info.num_lanes == 16
            and gmm2_hidden % tpu_info.num_lanes == 0  # pyright: ignore[reportOptionalMemberAccess]
            and gmm2_rows * gmm2_hidden * 4 >= tpu_info.vmem_capacity_bytes * 0.6  # pyright: ignore[reportOptionalMemberAccess]
        ):
            alignment = token_block_alignment(gmm2_hidden, tpu_info)
            padded_tokens = -(-num_tokens // alignment) * alignment
            if num_tokens >= alignment and padded_tokens * 2 <= num_tokens * 3:
                path = _v3_kernel_path(
                    gmm2_rows, gmm2_hidden, gmm2_dtype, out.dtype_bytes, get_tpu_info
                )
                return (Combine.LOCAL_V3, path, alignment)
    if use_ep and use_sparse_core:
        if out.routes <= onehot_moe_permute_threshold:
            # topk_weights_sorted = zeros_like(topk_weights_flat).at[...].set(...)
            # valid_count = group_sizes.sum(dtype=int32)[None]
            owner_on = (
                owner_output_mode.lower() == "on"  # str-ok: verbatim base env check
            )
            if owner_on and can_use_blockwise_onehot_unpermute(
                ((gmm2_rows, gmm2_hidden), gmm2_dtype, get_tpu_info),
                (out.token_indices_shape, I32),
                (out.topk_weights_shape, dtype),
                ((1,), I32),
                num_tokens=num_tokens,
            ):
                return (Combine.EP_OWNER_ONEHOT,)
            return (Combine.EP_ONEHOT,)
        if configured_version is GatherReduceVersion.V1:
            path = _v1_kernel_path(
                gmm2_rows, gmm2_hidden, out.dtype_bytes, get_tpu_info
            )
        elif configured_version is GatherReduceVersion.V2:
            path = _v2_kernel_path(
                gmm2_rows, gmm2_hidden, gmm2_dtype, out.dtype_bytes, get_tpu_info
            )
        else:
            path = _v3_kernel_path(
                gmm2_rows, gmm2_hidden, gmm2_dtype, out.dtype_bytes, get_tpu_info
            )
        return (_Ref.EP_RAGGED, configured_version, path)
    return (Combine.TENSORCORE,)


# ---------------------------------------------------------------------------
# Reading a reference trace as the plan it implies.
# ---------------------------------------------------------------------------
_PERMUTE_OF = {
    _Ev.PERMUTE_ONEHOT_FROM_POSITIONS: Permute.ONEHOT_FROM_POSITIONS,
    _Ev.PERMUTE_ONEHOT_FROM_TOKENS: Permute.ONEHOT_FROM_TOKENS,
    _Ev.PERMUTE_RAGGED_SC: Permute.RAGGED_SC,
    _Ev.PERMUTE_GATHER: Permute.GATHER,
}


def _payload(events, kind):
    """The value carried by the `(kind, value)` event, or None."""
    return next((e[1] for e in events if isinstance(e, tuple) and e[0] is kind), None)


def _expected_routing(events):
    """(sort form, permute, single-sort bits, packed-sort bits) main chose.

    The sort forms are named for what main does: a counting sort or not, and
    then a token scatter, a packed token sort, a token gather or one packed
    single sort.
    """
    counting = _Ev.COUNTING_SORT in events
    single_bits = _payload(events, _Ev.SINGLE_SORT)
    packed_bits = _payload(events, _Ev.PACKED_TOKEN_SORT)
    gathers = _Ev.GATHER_TOKENS in events
    if counting and _Ev.SCATTER_TOKENS in events:
        sort = SortForm.COUNTING_SCATTER
    elif counting:
        sort = (
            SortForm.COUNTING_PACKED
            if packed_bits is not None
            else (SortForm.COUNTING_ARGSORT)
        )
    elif single_bits is not None:
        sort = SortForm.SINGLE_SORT
    else:
        sort = SortForm.ARGSORT_PACKED if not gathers else SortForm.ARGSORT_GATHER
    (permute,) = (_PERMUTE_OF[e] for e in events if e in _PERMUTE_OF)
    return sort, permute, single_bits, packed_bits


def _refusing(op):
    del op
    return False


def _plan_routing(process, op, *, refuse):
    """`moe_plan.plan_routing`; `refuse` stands the counting sort down.

    That is how the TPU tests stand it down (main's tests patched
    `_counting_sort_positions` to return None).
    """
    if not refuse:
        return moe_plan.plan_routing(process, op)
    saved = moe_plan.counting_sort_accepts
    moe_plan.counting_sort_accepts = _refusing
    try:
        return moe_plan.plan_routing(process, op)
    finally:
        moe_plan.counting_sort_accepts = saved


def _routing_case(
    family: GridFamily,
    process: moe_plan.MoeProcessPlan,
    op: moe_plan.MoeOpIn,
    *,
    single: bool,
    counting: bool,
    refuse: bool,
) -> tuple[SortForm, Permute]:
    """Checks one routing case into `family`; returns main's choice."""
    events: list[_Ev | tuple[_Ev, int]] = _reference_prepare(
        num_tokens_local=op.num_tokens,
        topk=op.topk,
        local_num_experts=op.local_num_experts,
        use_ep=op.use_ep,
        use_sparse_core=process.use_sparse_core,
        onehot_moe_permute_threshold=process.onehot_threshold,
        _SINGLE_SORT=single,
        _COUNTING_SORT=counting,
        inject_refusal=refuse,
    )
    routing: tuple[SortForm, Permute, int | None, int | None] = _expected_routing(
        events
    )
    sort, permute, single_bits, packed_bits = routing
    plan = _plan_routing(process, op, refuse=refuse)
    ctx = (op, process, events)
    family.expect(plan.sort is sort, (plan.sort, sort, ctx))
    family.expect(plan.permute is permute, (plan.permute, permute, ctx))
    family.expect_equal(plan.uses_counting_sort, _Ev.COUNTING_SORT in events, ctx)
    family.expect_equal(plan.argsorts_keys, _Ev.ARGSORT_KEYS in events, ctx)
    if single_bits is not None:
        family.expect_equal(plan.flat_bits, single_bits, ctx)
    if packed_bits is not None:
        family.expect_equal(plan.token_bits, packed_bits, ctx)
    family.cases += 1
    return sort, permute


# ---------------------------------------------------------------------------
# Grids: both sides of every boundary, nothing in between.
# ---------------------------------------------------------------------------
_THRESHOLDS = (0, 255, 511)  # Off, and the auto values of v6e and TPU7x.
# Routed rows on both sides of 255/256 and 511/512 (the one-hot thresholds,
# the 512-row counting block and scatter cap) and 2048 (the sort cap).
_ROW_TARGETS = (1, 2, 255, 256, 257, 511, 512, 513, 2047, 2048, 2049, 4096)
# Expert counts whose bucket count (experts + 1) sits on both sides of the
# counting block (512) and of the packed-key widths at very large counts.
_LOCAL_EXPERTS = (1, 7, 255, 510, 511, 512, (1 << 19) - 1, 1 << 19)
_TOPKS = (1, 8)
# Token counts that make the packed (key, token) or (key, row) sort key
# overflow int32 with the largest expert counts.
_HUGE_TOKENS = (1 << 22, (1 << 26) + 3)
# (use_ep, use_sparse_core): routing reads only their conjunction.
_EP_SC = ((False, False), (True, True))


def _routing_shapes():
    """(tokens, topk) pairs whose routed rows hit `_ROW_TARGETS` (or the next
    multiple of topk), plus the huge token counts at top-1."""
    pairs = {(-(-rows // topk), topk) for rows in _ROW_TARGETS for topk in _TOPKS}
    pairs |= {(tokens, 1) for tokens in _HUGE_TOKENS}
    return sorted(pairs)


def _process(*, use_sc, thr, single=True, counting=True, dense_thr=0, **caps):
    return moe_plan.MoeProcessPlan(
        use_sparse_core=use_sc,
        onehot_threshold=thr,
        single_sort=single,
        counting_sort=counting,
        dense_expert_threshold=dense_thr,
        ep_gather_reduce_version=GatherReduceVersion.V2,
        **caps,
    )


@dataclasses.dataclass(frozen=True)
class _RoutingFamily:
    """One process setting of the routing grid; the threshold acts on EP only.

    Attributes:
        use_ep: Expert parallelism.
        use_sc: The SparseCore; routing reads only its conjunction with EP.
        threshold: The one-hot threshold.
    """

    use_ep: bool
    use_sc: bool
    threshold: int

    @property
    def label(self) -> str:
        """The family's test id."""
        path = "ep_sc" if self.use_ep and self.use_sc else "tc"
        return f"{path}-thr{self.threshold}"


# The threshold only acts on the EP SparseCore path.
_ROUTING_FAMILIES = (
    _RoutingFamily(use_ep=False, use_sc=False, threshold=0),
    *(_RoutingFamily(use_ep=True, use_sc=True, threshold=t) for t in _THRESHOLDS),
)


@dataclasses.dataclass(frozen=True)
class _RoutingResult:
    """One routing family's checks and the (sort, permute) choices it reached."""

    family: GridFamily
    outcomes: frozenset[tuple[SortForm, Permute]]


@functools.cache
def _routing_family(family: _RoutingFamily) -> _RoutingResult:
    """Checks every (shape, experts, sort toggles, refusal) of one family."""
    check = GridFamily(f"routing {family.label}")
    outcomes: set[tuple[SortForm, Permute]] = set()
    for (tokens, topk), experts in itertools.product(_routing_shapes(), _LOCAL_EXPERTS):
        op = moe_plan.MoeOpIn(
            num_tokens=tokens,
            topk=topk,
            local_num_experts=experts,
            use_ep=family.use_ep,
        )
        for single, counting, refuse in itertools.product((False, True), repeat=3):
            process = _process(
                use_sc=family.use_sc,
                thr=family.threshold,
                single=single,
                counting=counting,
            )
            outcomes.add(
                _routing_case(
                    check, process, op, single=single, counting=counting, refuse=refuse
                )
            )
    return _RoutingResult(check, frozenset(outcomes))


def test_no_tpu_runtime() -> None:
    """The module imported with no TPU runtime and runs JAX on the CPU."""
    assert _pure_import.loaded_tpu_runtime() == []
    assert _pure_import.jax_backend() == "cpu"  # str-ok: the backend name JAX reports


@pytest.mark.parametrize(
    "family", _ROUTING_FAMILIES, ids=[f.label for f in _ROUTING_FAMILIES]
)
def test_routing_matches_reference(family: _RoutingFamily) -> None:
    """`plan_routing` against `prepare_routed_gmm_inputs` at d3991ebf9.

    Boundaries: rows vs the one-hot threshold, the 512-row counting block and
    scatter cap, the 2048-row sort cap; experts + 1 vs the counting block;
    packed-key width vs 31 bits; the two sort toggles; the counting sort's
    refusal.
    """
    _routing_family(family).family.assert_no_mismatches()


def test_routing_reaches_every_sort_and_permute() -> None:
    """Across the routing families, main picks every sort form and permute."""
    outcomes = {
        outcome
        for family in _ROUTING_FAMILIES
        for outcome in _routing_family(family).outcomes
    }
    print(f"\n[routing] {len(outcomes)} distinct (sort, permute)")
    assert {s for s, _ in outcomes} == set(SortForm)
    assert {p for _, p in outcomes} == set(Permute)


@dataclasses.dataclass(frozen=True)
class _RowCaps:
    """A patched (single-sort cap, counting-scatter cap) pair.

    The existing TPU tests patch these module caps; they are plan inputs now.
    """

    single_sort_max_rows: int
    counting_scatter_max_rows: int

    @property
    def label(self) -> str:
        """The pair's test id."""
        return (
            f"single{self.single_sort_max_rows}-scatter{self.counting_scatter_max_rows}"
        )


_ROW_CAPS = (
    _RowCaps(2048, 0),
    _RowCaps(1024, 512),
    _RowCaps(4096, 2048),
    _RowCaps(0, 0),
)


def _cap_rows() -> list[int]:
    """Rows on both sides of every cap of `_ROW_CAPS`, positive only."""
    caps = {
        cap
        for pair in _ROW_CAPS
        for cap in (pair.single_sort_max_rows, pair.counting_scatter_max_rows)
    }
    rows = sorted({c + d for c in caps for d in (-1, 0, 1)} - {-1})
    return [r for r in rows if r > 0]


@pytest.mark.parametrize("caps", _ROW_CAPS, ids=[c.label for c in _ROW_CAPS])
def test_patched_row_caps_match_reference(caps: _RowCaps) -> None:
    """Rows on both sides of each patched cap, for one cap pair."""
    global _SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS  # noqa: PLW0603 - the reference reads the module caps
    saved = (_SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS)
    check = GridFamily(f"routing caps {caps.label}")
    outcomes: set[tuple[SortForm, Permute]] = set()
    try:
        _SINGLE_SORT_MAX_ROWS = caps.single_sort_max_rows  # pyright: ignore[reportConstantRedefinition] - the reference reads the module caps
        _COUNTING_SORT_SCATTER_MAX_ROWS = caps.counting_scatter_max_rows  # pyright: ignore[reportConstantRedefinition] - the reference reads the module caps
        for tokens, experts, (use_ep, use_sc), refuse in itertools.product(
            _cap_rows(), (7, 255), _EP_SC, (False, True)
        ):
            op = moe_plan.MoeOpIn(
                num_tokens=tokens, topk=1, local_num_experts=experts, use_ep=use_ep
            )
            outcomes.add(
                _routing_case(
                    check,
                    _process(
                        use_sc=use_sc,
                        thr=0,
                        single_sort_max_rows=caps.single_sort_max_rows,
                        counting_scatter_max_rows=caps.counting_scatter_max_rows,
                    ),
                    op,
                    single=True,
                    counting=True,
                    refuse=refuse,
                )
            )
    finally:
        _SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS = saved  # pyright: ignore[reportConstantRedefinition] - the reference reads the module caps
    print(f"\n[routing caps {caps.label}] {len(outcomes)} distinct (sort, permute)")
    check.assert_no_mismatches()


# (use_ep, use_sparse_core): the entry reads each alone.
_ENTRY_EP_SC = ((False, False), (True, False), (False, True), (True, True))
_ENTRY_TOKENS = (0, 1, 15, 16, 17, 31, 32, 63, 64, 255, 256, 511, 512, 1024, 1025)


def _ep_sc_id(ep_sc: tuple[bool, bool]) -> str:
    use_ep, use_sc = ep_sc
    return f"{'ep' if use_ep else 'no_ep'}-{'sc' if use_sc else 'no_sc'}"


@pytest.mark.parametrize("ep_sc", _ENTRY_EP_SC, ids=map(_ep_sc_id, _ENTRY_EP_SC))
def test_entry_matches_reference(ep_sc: tuple[bool, bool]) -> None:
    """The entry plan against `fused_moe_func` at d3991ebf9, for one EP/SC pair.

    Boundaries: tokens on both sides of 0 and of the dense-expert threshold,
    routed rows on both sides of the one-hot threshold; the kernel's own
    admission.
    """
    use_ep, use_sc = ep_sc
    check = GridFamily(f"entry {_ep_sc_id(ep_sc)}")
    outcomes: set[tuple[MoePath, bool]] = set()
    for tokens, topk, thr, dense_thr, admits in itertools.product(
        _ENTRY_TOKENS, _TOPKS, _THRESHOLDS, (0, 16, 1024), (False, True)
    ):
        asked: list[int] = []

        def can_use_dense_expert(
            *, admits: bool = admits, asked: list[int] = asked
        ) -> bool:
            asked.append(1)
            return admits

        events: list[_Ev] = _reference_entry(
            num_tokens=tokens,
            topk=topk,
            use_ep=use_ep,
            use_sparse_core=use_sc,
            onehot_moe_permute_threshold=thr,
            _DENSE_EXPERT_THRESHOLD=dense_thr,
            can_use_dense_expert=can_use_dense_expert,
        )
        process = _process(use_sc=use_sc, thr=thr, dense_thr=dense_thr)
        op = moe_plan.MoeOpIn(
            num_tokens=tokens, topk=topk, local_num_experts=64, use_ep=use_ep
        )
        candidate = moe_plan.dense_expert_candidate(process, op)
        # The kernel's admission is asked exactly where main asked it.
        check.expect_equal(candidate, bool(asked), (op, process))
        entry = moe_plan.plan_entry(
            process, op, dense_expert_admitted=candidate and admits
        )
        expected_path = (
            MoePath.DENSE_EXPERT if _Ev.DENSE_EXPERT in events else MoePath.ROUTED
        )
        check.expect(
            entry.path is expected_path, (entry.path, expected_path, op, process)
        )
        check.expect_equal(
            entry.scrub_nonfinite, _Ev.SCRUB_NONFINITE in events, (op, process)
        )
        outcomes.add((entry.path, entry.scrub_nonfinite))
        check.cases += 1
    print(f"\n[entry {_ep_sc_id(ep_sc)}] {len(outcomes)} distinct (path, scrub)")
    check.assert_no_mismatches()


class _Mode(str):
    """An owner-output mode value that records whether main's code read it."""

    __slots__ = ("reads",)

    def lower(self):
        """Records the read, as main's `.lower() == ...` check does."""
        self.reads.append(1)
        return super().lower()


def _mode(on: bool) -> _Mode:
    mode = _Mode("on" if on else "off")
    mode.reads = []
    return mode


# Widths for mxu (128/256) and lane (128/256) divisibility, an unaligned one,
# and the v3 column-partition steps.
_COMBINE_HIDDENS = (100, 256, 2048, 7168)
# top-1 and top-8 (no local v3) and top-2 (local v3 allowed).
_COMBINE_TOPKS = (1, 2, 8)
# Routed rows on both sides of 511/512 at top-8 (63/64 tokens) and exactly 511
# at top-1, the v3 token
# alignments (64-256 tokens), a large batch for the 0.6 x VMEM cutoffs, and
# 2**20 source rows at top-8 (131072 tokens) and one token past it.
_COMBINE_TOKENS = (16, 63, 64, 100, 511, 8192, 131072, 131073)


def _combine_settings(routes):
    """(use_ep, use_sc, threshold, version, owner_on, extra_rows) that matter.

    TensorCore needs no settings; local v3 reads the threshold; EP reads the
    owner mode only at or under the threshold and the kernel version only
    above it. Extra second-GMM rows only change the owner kernel's admission.
    """
    yield False, False, 0, GatherReduceVersion.V2, False, 0
    for thr in (0, 511):
        yield False, True, thr, GatherReduceVersion.V2, False, 0
    # EP at TPU7x's auto threshold: the token counts already put routes on
    # both sides of it, so a zero threshold adds no new EP decision.
    if routes <= 511:
        for owner_on, extra in itertools.product((False, True), (0, 16)):
            yield True, True, 511, GatherReduceVersion.V2, owner_on, extra
    else:
        for version in GatherReduceVersion:
            yield True, True, 511, version, False, 0


_CombineHead = Combine | _Ref
# (use_ep, use_sc, threshold, version, owner_on, extra_rows).
_CombineSetting = tuple[bool, bool, int, GatherReduceVersion, bool, int]


@dataclasses.dataclass(frozen=True)
class _CombineResult:
    """One chip's combine checks, its cases per head, and its outcomes."""

    family: GridFamily
    by_head: collections.Counter[_CombineHead]
    outcomes: frozenset[tuple[object, ...]]


def _combine_chip_id(chip: pltpu.TpuInfo | None) -> str:
    return "no_chip" if chip is None else chip_name(chip)


_COMBINE_CHIPS = (None, *CHIPS)


@functools.cache
def _combine_family(chip: pltpu.TpuInfo | None) -> _CombineResult:
    """Checks every (dtype, width, top-k, tokens, setting) on one chip."""
    check = GridFamily(f"combine {_combine_chip_id(chip)}")
    by_head: collections.Counter[_CombineHead] = collections.Counter()
    outcomes: set[tuple[object, ...]] = set()
    for dtype, hidden, topk, tokens in itertools.product(
        (BF16, F32), _COMBINE_HIDDENS, _COMBINE_TOPKS, _COMBINE_TOKENS
    ):
        for setting in _combine_settings(tokens * topk):
            head, outcome = _combine_case(
                check, chip, dtype, hidden, topk, tokens, setting
            )
            by_head[head] += 1
            outcomes.add(outcome)
    return _CombineResult(check, by_head, frozenset(outcomes))


@pytest.mark.parametrize(
    "chip", _COMBINE_CHIPS, ids=[_combine_chip_id(c) for c in _COMBINE_CHIPS]
)
def test_combine_matches_reference(chip: pltpu.TpuInfo | None) -> None:
    """`plan_combine` and what it reads against `moe_gmm`'s combine at d3991ebf9.

    One chip (or none) per test. Boundaries: local v3's topk >= 2 and != 8,
    routes vs the threshold, BF16, 2**20 source rows, a 16-lane SparseCore,
    lane alignment, the 0.6 x VMEM cutoffs, the token-alignment and padding
    rule; the EP one-hot threshold and owner admission (rows % 128, width %
    MXU, tokens % sublane tiling, VMEM); each EP kernel version; BF16 and
    FP32.
    """
    result = _combine_family(chip)
    by_head = {head.name: n for head, n in result.by_head.items()}
    print(
        f"\n[combine {_combine_chip_id(chip)}] by path: {dict(sorted(by_head.items()))}"
    )
    result.family.assert_no_mismatches()


def test_combine_reaches_every_path() -> None:
    """Across the chips, the plan picks every combine."""
    by_head: collections.Counter[_CombineHead] = collections.Counter()
    outcomes: set[tuple[object, ...]] = set()
    for chip in _COMBINE_CHIPS:
        result = _combine_family(chip)
        by_head.update(result.by_head)
        outcomes |= result.outcomes
    names = {head.name: n for head, n in by_head.items()}
    print(
        f"\n[combine] {by_head.total()} cases, {len(outcomes)} distinct outcomes; "
        f"by path: {dict(sorted(names.items()))}"
    )
    assert set(by_head) - {_Ref.RAISE} == set(Combine)


def _combine_case(
    check: GridFamily,
    chip: pltpu.TpuInfo | None,
    dtype: jnp.dtype,
    hidden: int,
    topk: int,
    tokens: int,
    setting: _CombineSetting,
) -> tuple[_CombineHead, tuple[object, ...]]:
    """Checks one combine case into `check`; returns its head and outcome."""
    use_ep, use_sc, thr, version, owner_on, extra = setting
    routes = tokens * topk
    out = moe_plan.GmmOutSpec(
        routes=routes,
        rows=routes + extra,
        hidden=hidden,
        dtype=dtype,
        token_indices_shape=(routes,),
        token_indices_dtype=I32,
        topk_weights_shape=(routes,),
        topk_weights_dtype=dtype,
    )
    op = moe_plan.MoeOpIn(
        num_tokens=tokens, topk=topk, local_num_experts=64, use_ep=use_ep
    )
    process = moe_plan.MoeProcessPlan(
        use_sparse_core=use_sc,
        onehot_threshold=thr,
        single_sort=True,
        counting_sort=True,
        dense_expert_threshold=0,
        ep_gather_reduce_version=version,
    )
    chip_reads: list[int] = []

    def get_tpu_info() -> pltpu.TpuInfo:
        chip_reads.append(1)
        if chip is None:
            raise ValueError("Unsupported TPU device kind: cpu")
        return chip

    mode = _mode(owner_on)
    expected: tuple[object, ...]
    try:
        expected = _reference_combine(
            out=out,
            dtype=dtype,
            num_tokens=tokens,
            topk=topk,
            use_ep=use_ep,
            use_sparse_core=use_sc,
            onehot_moe_permute_threshold=thr,
            owner_output_mode=mode,
            configured_version=version,
            get_tpu_info=get_tpu_info,
        )
    except (ValueError, NotImplementedError) as e:
        expected = (_Ref.RAISE, type(e))
    ctx = f"{op} {process} {out} chip={chip and chip.chip_version} -> {expected}"
    # What the plan reads: the owner mode exactly where main read it, and
    # the chip wherever main (or the kernel it called) read it.
    check.expect_equal(
        moe_plan.combine_reads_owner_mode(process, op, out), bool(mode.reads), ctx
    )
    need = moe_plan.combine_target_need(process, op, out, owner_output=owner_on)
    if chip_reads:
        check.expect(need is not TargetNeed.NONE, ("needs no chip", ctx))
    plan: moe_plan.MoeCombinePlan | None
    try:
        plan = moe_plan.plan_combine(process, op, out, chip, owner_output=owner_on)
        actual = (plan.combine, plan.ep_gather_reduce_version)
    except (ValueError, NotImplementedError, AssertionError) as e:
        plan, actual = None, (_Ref.RAISE, type(e))
    check.expect(_combine_agrees(expected, actual, plan, need, chip), (actual, ctx))
    check.cases += 1
    alignment = plan.local_v3_token_alignment if plan is not None else None
    return actual[0], (*actual, alignment)


def _combine_agrees(
    expected: tuple[object, ...],
    actual: tuple[object, ...],
    plan: moe_plan.MoeCombinePlan | None,
    need: TargetNeed,
    chip: pltpu.TpuInfo | None,
) -> bool:
    """Whether the plan's combine is the one main's reference reached."""
    match expected:
        case (Combine.LOCAL_V3, kernel_path, alignment):
            # The plan claims the kernel never falls back on this path.
            return (
                kernel_path is KernelPath.SC
                and actual == (Combine.LOCAL_V3, None)
                and plan is not None
                and plan.local_v3_token_alignment == alignment
            )
        case (Combine.EP_OWNER_ONEHOT | Combine.EP_ONEHOT | Combine.TENSORCORE,):
            return actual == (expected[0], None)
        case (_Ref.EP_RAGGED, ref_version, kernel_path):
            # KernelPath.RAISE: v3 past 2**20 source rows raises after its
            # fallback checks pass; the plan names the SparseCore path it
            # attempts.
            combine = (
                Combine.EP_RAGGED_TC_FALLBACK
                if kernel_path is KernelPath.TC_FALLBACK
                else Combine.EP_RAGGED_SC
            )
            return actual == (combine, ref_version)
        case (_Ref.RAISE, exc):
            # Main raises from get_tpu_info (no chip) inside the EP kernel
            # or the owner admission, or from get_sublane_tiling on an
            # unknown generation. The plan raises the same, or it needs
            # the chip, which the runtime read then fails to give.
            return actual == (_Ref.RAISE, exc) or (
                exc is ValueError and chip is None and need is TargetNeed.REQUIRED
            )
        case _:
            return False


@pytest.mark.parametrize(
    "version", list(GatherReduceVersion), ids=[v.name for v in GatherReduceVersion]
)
def test_gather_reduce_version_parses(version: GatherReduceVersion) -> None:
    """Each version's env value parses to that version."""
    assert moe_plan.gather_reduce_version(version.value) is version


def test_gather_reduce_version_rejects_unknown() -> None:
    """An unknown version names the valid ones."""
    with pytest.raises(ValueError, match="valid: v1, v2, v3"):
        _ = moe_plan.gather_reduce_version("v4")


@pytest.mark.parametrize(
    ("value", "mode"),
    [("ON", OwnerOutputMode.ON), ("off", OwnerOutputMode.OFF)],
    ids=["upper_on", "lower_off"],
)
def test_owner_output_mode_parses(value: str, mode: OwnerOutputMode) -> None:
    """Case-insensitive, like the env getter."""
    assert moe_plan.owner_output_mode(value) is mode


def test_owner_output_mode_rejects_unknown() -> None:
    """An unknown mode names the valid ones."""
    with pytest.raises(ValueError, match="valid: off, on"):
        _ = moe_plan.owner_output_mode("auto")
