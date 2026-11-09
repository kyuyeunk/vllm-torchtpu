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
"""Oracle table test: `moe_plan` against the inline logic at d3991ebf9.

Each `_reference_*` function below is the decision code of
`layers/core/fused_moe_gmm.py` (and of the kernels it calls) at d3991ebf9,
copied verbatim except that arrays are replaced by their static shapes and
every array op is replaced by an event appended to a list. The new code's
side (`_new_*`) mirrors the `match` statements in the scratch build's
`fused_moe_gmm.py` the same way. The test asserts the two event traces are
identical over a dense grid, so both the plan functions and the shape of the
new dispatch are checked against today's control flow.
"""

import itertools
import time
import unittest

import _pure_import
from _chips import CHIPS

from vllm_torchtpu.layers.core import moe_plan
from vllm_torchtpu.layers.core.tpu_target import TpuTarget

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
        events.append("scrub_nonfinite")
    if 0 < num_tokens <= _DENSE_EXPERT_THRESHOLD and can_use_dense_expert():
        events.append("dense_expert")
        return events
    events.append("routed")
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
        events.append("counting_sort")
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
        events.append("scatter_tokens")
        token_indices_sorted = "scattered"
    sorted_indices = None
    if (
        token_indices_sorted is None
        and not single_sort
        and (argsort_revert_indices is None or not packable)
    ):
        events.append("argsort_keys")
        sorted_indices = "sorted"

    if argsort_revert_indices is not None:
        events.append("group_sizes_from_counts")
    else:
        if single_sort:
            events.append(("single_sort", flat_bits))
            token_indices_sorted = "single"
        else:
            assert sorted_indices is not None
            events.append("argsort_revert")
        events.append("group_sizes_onehot")

    if token_indices_sorted is None:
        if packable:
            events.append(("packed_token_sort", token_bits))
        else:
            events.append("gather_tokens")

    if use_ep and use_sparse_core:
        if token_indices_flat_rows <= onehot_moe_permute_threshold:
            if bucket_counts is not None:
                events.append("permute_onehot_from_positions")
            else:
                events.append("permute_onehot_from_tokens")
        else:
            events.append("permute_ragged_sc")
    else:
        events.append("permute_gather")
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
    if (
        sc_info is None
        or x_dtype != "bfloat16"
        or x_hidden % get_tpu_info().num_lanes != 0
    ):
        return "tc_fallback"
    if x_rows * x_hidden * x_bytes * 2 < get_tpu_info().vmem_capacity_bytes * 0.6:
        return "tc_fallback"
    if x_rows % 2:
        x_rows += 1
    if x_rows > 1 << _ROUTE_INDEX_BITS:
        return "raise"
    return "sc"


def _v1_kernel_path(x_rows, x_hidden, x_bytes, get_tpu_info):
    """ragged_gather_reduce.ragged_gather_reduce (v1) entry checks."""
    sc_info = get_tpu_info().sparse_core
    if sc_info is None:
        return "tc_fallback"
    if x_rows * x_hidden * x_bytes * 2 < get_tpu_info().vmem_capacity_bytes * 0.6:
        return "tc_fallback"
    return "sc"


def _v2_kernel_path(x_rows, x_hidden, x_dtype, x_bytes, get_tpu_info):
    """ragged_gather_reduce_v2 wrapper: Config(...) then should_fallback."""
    assert x_dtype in ("bfloat16", "float32")
    tpu_info = get_tpu_info()
    if tpu_info.sparse_core is None:
        return "tc_fallback"
    vmem_capacity_threshold = tpu_info.vmem_capacity_bytes * 0.6
    source_size = x_rows * x_hidden * x_bytes
    if source_size * 2 < vmem_capacity_threshold:
        return "tc_fallback"
    return "sc"


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
    if len(route_output_shape) != 2 or route_output_dtype != "bfloat16":
        return False
    route_capacity, hidden_size = route_output_shape
    if token_indices_sorted[0] != (route_capacity,):
        return False
    if token_indices_sorted[1] != "int32":
        return False
    if topk_weights_sorted[0] != (route_capacity,):
        return False
    if topk_weights_sorted[1] != "bfloat16":
        return False
    if valid_count[0] != (1,) or valid_count[1] != "int32":
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
    num_tokens,
    topk,
    use_ep,
    use_sparse_core,
    onehot_moe_permute_threshold,
    owner_output_mode,
    configured_version,
    get_tpu_info,
):
    """moe_gmm lines 552-658 (+ the kernel entry it reaches)."""
    gmm2_rows, gmm2_hidden, gmm2_dtype = out.rows, out.hidden, out.dtype
    use_local_ragged_combine = (
        use_sparse_core
        and not use_ep
        and topk >= 2
        and topk != 8
        and out.routes > onehot_moe_permute_threshold
        and gmm2_dtype == "bfloat16"
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
                return ("local_v3", path, alignment)
    if use_ep and use_sparse_core:
        if out.routes <= onehot_moe_permute_threshold:
            # topk_weights_sorted = zeros_like(topk_weights_flat).at[...].set(...)
            # valid_count = group_sizes.sum(dtype=int32)[None]
            if owner_output_mode.lower() == "on" and can_use_blockwise_onehot_unpermute(
                ((gmm2_rows, gmm2_hidden), gmm2_dtype, get_tpu_info),
                (out.token_indices_shape, out.token_indices_dtype),
                (out.topk_weights_shape, out.topk_weights_dtype),
                ((1,), "int32"),
                num_tokens=num_tokens,
            ):
                return ("ep_owner_onehot",)
            return ("ep_onehot",)
        if configured_version == "v1":
            path = _v1_kernel_path(
                gmm2_rows, gmm2_hidden, out.dtype_bytes, get_tpu_info
            )
        elif configured_version == "v2":
            path = _v2_kernel_path(
                gmm2_rows, gmm2_hidden, gmm2_dtype, out.dtype_bytes, get_tpu_info
            )
        else:
            path = _v3_kernel_path(
                gmm2_rows, gmm2_hidden, gmm2_dtype, out.dtype_bytes, get_tpu_info
            )
        return ("ep_ragged", configured_version, path)
    return ("tensorcore",)


# ---------------------------------------------------------------------------
# New side: the scratch build's dispatch shape.
# ---------------------------------------------------------------------------
def _new_entry(process, op, can_use_dense_expert):
    admitted = moe_plan.dense_expert_candidate(process, op) and can_use_dense_expert()
    entry = moe_plan.plan_entry(process, op, dense_expert_admitted=admitted)
    events = ["scrub_nonfinite"] if entry.scrub_nonfinite else []
    events.append(entry.path)
    return events


def _new_routing(process, op, inject_refusal):
    """Mirrors the prototype: try the counting sort, then plan on its result."""
    sorted_ok = False
    if moe_plan.counting_sort_attempted(process, op):
        # The kernel's own refusal rule (verbatim), or an injected refusal.
        block = min(_COUNTING_SORT_BLOCK, op.routed_rows)
        sorted_ok = not inject_refusal and not op.local_num_experts + 1 > block
    return moe_plan.plan_routing(process, op, counting_sorted=sorted_ok)


def _new_prepare(routing):
    """Mirrors the matches in the scratch build's prepare_routed_gmm_inputs."""
    events = []
    if routing.uses_counting_sort:
        events.append("counting_sort")
    if routing.sort == "counting_scatter":
        events.append("scatter_tokens")
    if routing.argsorts_keys:
        events.append("argsort_keys")
    match routing.sort:
        case "counting_scatter" | "counting_packed" | "counting_argsort":
            events.append("group_sizes_from_counts")
        case "single_sort":
            events.append(("single_sort", routing.flat_bits))
            events.append("group_sizes_onehot")
        case "argsort_packed" | "argsort_gather":
            events.append("argsort_revert")
            events.append("group_sizes_onehot")
    match routing.sort:
        case "counting_packed" | "argsort_packed":
            events.append(("packed_token_sort", routing.token_bits))
        case "counting_argsort" | "argsort_gather":
            events.append("gather_tokens")
        case "counting_scatter" | "single_sort":
            pass
    events.append(f"permute_{routing.permute}")
    return events


def _new_combine(process, op, out, info_or_none, owner_output_mode):
    """Mirrors the scratch build's moe_gmm combine selection."""
    owner_output = (
        moe_plan.combine_reads_owner_mode(process, op, out)
        and owner_output_mode.lower() == "on"
    )
    needs = moe_plan.combine_needs_target(process, op, out, owner_output=owner_output)
    target = (
        TpuTarget.from_tpu_info(info_or_none)
        if needs and info_or_none is not None
        else None
    )
    return needs, moe_plan.plan_combine(
        process, op, out, target, owner_output=owner_output
    )


# ---------------------------------------------------------------------------
# Grid.
# ---------------------------------------------------------------------------
_TOPKS = tuple(range(1, 11))
_THRESHOLDS = (0, 1, 127, 255, 511, 512)
_ROW_TARGETS = (
    0,
    1,
    2,
    127,
    128,
    255,
    256,
    511,
    512,
    513,
    1024,
    2047,
    2048,
    2049,
    4096,
)


def _token_counts() -> tuple[int, ...]:
    """Token counts whose routed rows land on and beside every boundary."""
    counts = {0, 1, 2, 3, 7, 8, 15, 16, 64, 128, 256, 320, 1000, 4096, 16384}
    for topk in _TOPKS:
        for target in _ROW_TARGETS + tuple(
            t + d for t in _THRESHOLDS for d in (-1, 0, 1)
        ):
            if target >= 0:
                counts.add(target // topk)
                counts.add(target // topk + 1)
    return tuple(sorted(counts))


_LOCAL_EXPERTS = (
    1,
    7,
    8,
    63,
    64,
    127,
    128,
    255,
    256,
    511,
    512,
    1023,
    (1 << 19) - 1,
    1 << 19,
)


class MoeRoutingOracleTest(unittest.TestCase):
    """plan_routing against prepare_routed_gmm_inputs / fused_moe_func."""

    def test_no_runtime_imported(self):
        self.assertEqual(_pure_import.loaded_runtime_modules(), [])

    def test_routing_matches_reference(self):
        start = time.perf_counter()
        cases = 0
        seen_forms: set[str] = set()
        seen_permutes: set[str] = set()
        # Huge token counts reach the non-packable forms with few experts.
        token_counts = _token_counts() + ((1 << 22), (1 << 26) + 3)
        for (
            num_tokens,
            topk,
            experts,
            use_ep,
            use_sc,
            thr,
            single,
            counting,
            inject,
        ) in itertools.product(
            token_counts,
            _TOPKS,
            _LOCAL_EXPERTS,
            (False, True),
            (False, True),
            _THRESHOLDS,
            (False, True),
            (False, True),
            (False, True),
        ):
            process = moe_plan.MoeProcessPlan(
                use_sparse_core=use_sc,
                onehot_threshold=thr,
                single_sort=single,
                counting_sort=counting,
                dense_expert_threshold=0,
                ep_gather_reduce_version="v2",
            )
            op = moe_plan.MoeOpIn(
                num_tokens=num_tokens,
                topk=topk,
                local_num_experts=experts,
                use_ep=use_ep,
            )
            routing = _new_routing(process, op, inject)
            if not inject:
                # Without the counting result the plan predicts the same.
                self.assertEqual(moe_plan.plan_routing(process, op), routing)
            expected = _reference_prepare(
                num_tokens_local=num_tokens,
                topk=topk,
                local_num_experts=experts,
                use_ep=use_ep,
                use_sparse_core=use_sc,
                onehot_moe_permute_threshold=thr,
                _SINGLE_SORT=single,
                _COUNTING_SORT=counting,
                inject_refusal=inject,
            )
            actual = _new_prepare(routing)
            if actual != expected:
                self.fail(f"{op} {process}: {actual} != {expected}")
            seen_forms.add(routing.sort)
            seen_permutes.add(routing.permute)
            cases += 1
        print(
            f"\n[routing] {cases} cases in {time.perf_counter() - start:.1f}s; "
            f"forms={sorted(seen_forms)} permutes={sorted(seen_permutes)}"
        )
        self.assertEqual(seen_forms, set(moe_plan.SortForm.__args__))
        self.assertEqual(seen_permutes, set(moe_plan.Permute.__args__))

    def test_patched_row_caps_match_reference(self):
        # Existing TPU tests patch these module caps; they are plan inputs now.
        global _SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS  # noqa: PLW0603
        saved = (_SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS)
        cases = 0
        try:
            for single_cap, scatter_cap in (
                (2048, 0),
                (1024, 512),
                (4096, 2048),
                (0, 0),
            ):
                _SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS = (
                    single_cap,
                    scatter_cap,
                )
                for (
                    num_tokens,
                    topk,
                    experts,
                    use_ep,
                    use_sc,
                    thr,
                    inject,
                ) in itertools.product(
                    _token_counts(),
                    (1, 2, 8),
                    (7, 64, 255),
                    (False, True),
                    (False, True),
                    (0, 511),
                    (False, True),
                ):
                    process = moe_plan.MoeProcessPlan(
                        use_sparse_core=use_sc,
                        onehot_threshold=thr,
                        single_sort=True,
                        counting_sort=True,
                        dense_expert_threshold=0,
                        ep_gather_reduce_version="v2",
                        single_sort_max_rows=single_cap,
                        counting_scatter_max_rows=scatter_cap,
                    )
                    op = moe_plan.MoeOpIn(
                        num_tokens=num_tokens,
                        topk=topk,
                        local_num_experts=experts,
                        use_ep=use_ep,
                    )
                    expected = _reference_prepare(
                        num_tokens_local=num_tokens,
                        topk=topk,
                        local_num_experts=experts,
                        use_ep=use_ep,
                        use_sparse_core=use_sc,
                        onehot_moe_permute_threshold=thr,
                        _SINGLE_SORT=True,
                        _COUNTING_SORT=True,
                        inject_refusal=inject,
                    )
                    actual = _new_prepare(_new_routing(process, op, inject))
                    self.assertEqual(actual, expected, (op, process))
                    cases += 1
        finally:
            _SINGLE_SORT_MAX_ROWS, _COUNTING_SORT_SCATTER_MAX_ROWS = saved
        print(f"\n[routing caps] {cases} cases")

    def test_entry_matches_reference(self):
        cases = 0
        for (
            num_tokens,
            topk,
            use_ep,
            use_sc,
            thr,
            dense_thr,
            admits,
        ) in itertools.product(
            _token_counts(),
            _TOPKS,
            (False, True),
            (False, True),
            _THRESHOLDS,
            (0, 1, 16, 64, 1024),
            (False, True),
        ):
            asked = []

            def can_use_dense_expert(admits=admits, asked=asked):
                asked.append(1)
                return admits

            process = moe_plan.MoeProcessPlan(
                use_sparse_core=use_sc,
                onehot_threshold=thr,
                single_sort=True,
                counting_sort=True,
                dense_expert_threshold=dense_thr,
                ep_gather_reduce_version="v2",
            )
            op = moe_plan.MoeOpIn(
                num_tokens=num_tokens, topk=topk, local_num_experts=64, use_ep=use_ep
            )
            actual = _new_entry(process, op, can_use_dense_expert)
            asked_new = len(asked)
            asked.clear()
            expected = _reference_entry(
                num_tokens=num_tokens,
                topk=topk,
                use_ep=use_ep,
                use_sparse_core=use_sc,
                onehot_moe_permute_threshold=thr,
                _DENSE_EXPERT_THRESHOLD=dense_thr,
                can_use_dense_expert=can_use_dense_expert,
            )
            self.assertEqual(actual, expected)
            # The kernel admission is asked exactly when it was before.
            self.assertEqual(asked_new, len(asked))
            cases += 1
        print(f"\n[entry] {cases} cases")


def _combine_grid():
    hiddens = (128, 256, 512, 1024, 2048, 2880, 4096, 7168, 100)
    dtypes = (("bfloat16", 16), ("float32", 32))
    token_counts = (
        0,
        1,
        2,
        15,
        16,
        32,
        63,
        64,
        100,
        128,
        255,
        256,
        511,
        512,
        1024,
        2048,
        4096,
        8192,
        32768,
        140000,
    )
    return hiddens, dtypes, token_counts


class MoeCombineOracleTest(unittest.TestCase):
    """plan_combine against moe_gmm's combine and the kernel fallbacks."""

    def test_combine_matches_reference(self):
        start = time.perf_counter()
        hiddens, dtypes, token_counts = _combine_grid()
        chips = (None,) + CHIPS
        counts: dict[str, int] = {}
        cases = 0
        for (
            num_tokens,
            topk,
            hidden,
            (dtype, bits),
            use_ep,
            use_sc,
            thr,
            chip,
            extra_rows,
        ) in itertools.product(
            token_counts,
            _TOPKS,
            hiddens,
            dtypes,
            (False, True),
            (False, True),
            _THRESHOLDS,
            chips,
            (0, 16),
        ):
            routes = num_tokens * topk
            rows = routes + extra_rows
            out = moe_plan.GmmOutSpec(
                routes=routes,
                rows=rows,
                hidden=hidden,
                dtype=dtype,
                dtype_bits=bits,
                token_indices_shape=(routes,),
                token_indices_dtype="int32",
                topk_weights_shape=(routes,),
                topk_weights_dtype=dtype,
            )
            op = moe_plan.MoeOpIn(
                num_tokens=num_tokens, topk=topk, local_num_experts=64, use_ep=use_ep
            )
            versions = ("v1", "v2", "v3") if use_ep and use_sc else ("v2",)
            owner_modes = ("off", "on", "ON") if use_ep and use_sc else ("off",)
            for version, owner_mode in itertools.product(versions, owner_modes):
                process = moe_plan.MoeProcessPlan(
                    use_sparse_core=use_sc,
                    onehot_threshold=thr,
                    single_sort=True,
                    counting_sort=True,
                    dense_expert_threshold=0,
                    ep_gather_reduce_version=version,
                )
                calls: list[int] = []

                def get_tpu_info(chip=chip, calls=calls):
                    calls.append(1)
                    if chip is None:
                        raise ValueError("Unsupported TPU device kind: cpu")
                    return chip

                try:
                    expected = _reference_combine(
                        out=out,
                        num_tokens=num_tokens,
                        topk=topk,
                        use_ep=use_ep,
                        use_sparse_core=use_sc,
                        onehot_moe_permute_threshold=thr,
                        owner_output_mode=owner_mode,
                        configured_version=version,
                        get_tpu_info=get_tpu_info,
                    )
                except (ValueError, NotImplementedError) as e:
                    expected = ("raise", type(e).__name__)
                try:
                    needs, plan = _new_combine(process, op, out, chip, owner_mode)
                    actual = (plan.combine, plan.ep_gather_reduce_version)
                except (ValueError, NotImplementedError) as e:
                    needs, plan = True, None
                    actual = ("raise", type(e).__name__)
                self._check(expected, actual, plan, bool(calls), needs, version)
                key = actual[0] if expected[0] != "raise" else f"raise->{actual[0]}"
                counts[key] = counts.get(key, 0) + 1
                cases += 1
        print(
            f"\n[combine] {cases} cases in {time.perf_counter() - start:.1f}s; "
            f"by path: {dict(sorted(counts.items()))}"
        )
        self.assertEqual(
            {k for k in counts if not k.startswith("raise")},
            set(moe_plan.Combine.__args__),
        )

    def _check(self, expected, actual, plan, reference_read_chip, needs, version):
        ctx = f"expected={expected} actual={actual}"
        # Wherever today's code (or the kernel it calls) reads the chip, the
        # plan must have had it.
        if reference_read_chip:
            self.assertTrue(needs, ctx)
        match expected:
            case ("local_v3", kernel_path, alignment):
                # The plan claims the kernel never falls back on this path.
                self.assertEqual(kernel_path, "sc", ctx)
                self.assertEqual(actual, ("local_v3", None), ctx)
                self.assertEqual(plan.local_v3_token_alignment, alignment, ctx)
            case ("ep_owner_onehot",) | ("ep_onehot",) | ("tensorcore",):
                self.assertEqual(actual, (expected[0], None), ctx)
            case ("ep_ragged", ref_version, kernel_path):
                # "raise": v3 past 2**20 source rows raises after its fallback
                # checks pass; the plan names the SparseCore path it attempts.
                label = (
                    "ep_ragged_tc_fallback"
                    if kernel_path == "tc_fallback"
                    else "ep_ragged_sc"
                )
                self.assertEqual(actual, (label, ref_version), ctx)
            case ("raise", exc):
                # Today raises from get_tpu_info (no TPU description) inside
                # the EP kernel or the owner admission, or from
                # get_sublane_tiling on an unknown generation. The plan either
                # raises the same type itself or names the EP kernel, which
                # then raises at the same point as today.
                self.assertTrue(
                    actual == ("raise", exc)
                    or (
                        exc == "ValueError"
                        and actual[0] in ("ep_ragged_sc", "ep_ragged_tc_fallback")
                        and actual[1] == version
                    ),
                    ctx,
                )
            case _:
                self.fail(ctx)


if __name__ == "__main__":
    unittest.main()
