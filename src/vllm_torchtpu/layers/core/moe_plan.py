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
"""Pure MoE path decisions (data contract C7).

Every path choice `fused_moe_gmm` makes while a bucket compiles lives here as
a pure function of three values:

- `MoeProcessPlan`: process-wide knobs (SparseCore on/off, the one-hot
  threshold, the sort toggles, the EP gather-reduce version);
- `MoeOpIn` and `GmmOutSpec`: the concrete shapes and dtypes of this call;
- `TpuTarget`: the chip.

The rules are today's inline conditions, moved without change. Where a kernel
re-checks a condition and silently runs another path (the SparseCore
gather-reduce kernels' TensorCore fallbacks, the counting sort's refusal,
the owner-output kernel's admission), the rule here computes the same
predicate so the plan names the path that actually runs. The kernels are
untouched, so each such predicate is a duplicate; each is marked
"DUPLICATE of" with the kernel line it mirrors.

The decision resolves in three stages, each where its inputs first exist:
`plan_entry` (non-finite scrub, dense expert) at the top of `fused_moe_func`;
`plan_routing` in `prepare_routed_gmm_inputs`, because the sort form depends
on whether the counting sort returned positions; and `plan_combine` after the
second GMM, because the combine depends on that output's width, which
`gmm_v2` derives internally (it pads N to a lane multiple).

Imports no JAX or TPU runtime, so every rule here runs on CPU.
"""

import dataclasses
from typing import Literal

from vllm_torchtpu.layers.core.tpu_target import TpuTarget

# Cap on the auto one-hot threshold.
ONEHOT_AUTO_CAP = 512
# Routed-row cap for the one-hot EP permute and non-EP paths. Up to here the
# counting sort (or, where it declines, the packed single sort) replaces the
# three-sort form; above it the three sorts stay. Past this point the single
# sort's inverse-permutation scatter costs more than the sorts it replaces,
# and the counting sort's rank contraction does too at 256+ local experts.
SINGLE_SORT_MAX_ROWS = 2048
# Rows per block of the counting sort's rank contraction: the triangular
# matmul is B x B, so this trades that against the number of blocks.
COUNTING_SORT_BLOCK = 512
# Up to one counting-sort block on non-ragged paths, scattering token ids
# through the counting sort's positions is cheaper than the packed value sort
# it replaces and leaves the decode routing prep with no comparison sort.
# Ragged permute keeps the packed sort so SparseCore gather overlap is
# unchanged, and past one block the scatter costs more than that sort.
COUNTING_SORT_SCATTER_MAX_ROWS = COUNTING_SORT_BLOCK
# A packed (key, payload) sort key must fit in a non-negative int32.
_PACKED_SORT_BITS = 31
# The local v3 combine packs source indices into 20 bits
# (`ragged_gather_reduce_v3._ROUTE_INDEX_BITS`).
LOCAL_V3_MAX_SOURCE_ROWS = 1 << 20
# DUPLICATE of `ragged_gather_reduce_v3._TOKEN_BLOCK` (16 x 4).
_V3_TOKEN_BLOCK = 64
# DUPLICATE of `moe_onehot_unpermute._ROUTE_BLOCK_SIZE`.
_OWNER_ROUTE_BLOCK_SIZE = 128
# DUPLICATE of `moe_onehot_unpermute._VMEM_WORKSPACE_MARGIN_BYTES`.
_OWNER_VMEM_WORKSPACE_MARGIN_BYTES = 4 * 1024 * 1024

GatherReduceVersion = Literal["v1", "v2", "v3"]

MoePath = Literal["dense_expert", "routed"]
"""`dense_expert`: the experimental dense-expert kernel; `routed`: GMM."""

SortForm = Literal[
    "counting_scatter",
    "counting_packed",
    "counting_argsort",
    "single_sort",
    "argsort_packed",
    "argsort_gather",
]
"""How the routing permutation and `group_sizes` are built.

- `counting_scatter`: counting-sort positions; token ids scattered through
  them; group sizes from the counting histogram. No comparison sort.
- `counting_packed`: counting-sort positions; token ids from one packed
  (key, token) sort.
- `counting_argsort`: counting-sort positions; token ids gathered through an
  `argsort` of the keys (the packed key does not fit in int32).
- `single_sort`: one packed (key, flat position) sort gives the order, its
  inverse (a scatter) and the token ids.
- `argsort_packed`: `argsort`, `argsort` of that, then a packed token sort --
  the three-sort form.
- `argsort_gather`: two `argsort`s, token ids gathered.
"""

Permute = Literal["onehot_from_positions", "onehot_from_tokens", "ragged_sc", "gather"]
"""How routed rows are gathered from tokens.

- `onehot_from_positions`: one-hot matmul built from the counting-sort
  positions (EP + SparseCore, at or under the one-hot threshold).
- `onehot_from_tokens`: one-hot matmul built from the sorted token ids.
- `ragged_sc`: SparseCore `ragged_gather_v2`.
- `gather`: a plain XLA gather.
"""

Combine = Literal[
    "local_v3",
    "ep_owner_onehot",
    "ep_onehot",
    "ep_ragged_sc",
    "ep_ragged_tc_fallback",
    "tensorcore",
]
"""How the top-k expert outputs are summed back per token.

- `local_v3`: non-EP SparseCore destination-major `ragged_gather_reduce_v3`.
- `ep_owner_onehot`: EP, blockwise one-hot unpermute kernel (owner output).
- `ep_onehot`: EP, dense one-hot combine matmul.
- `ep_ragged_sc`: EP, the configured SparseCore gather-reduce kernel runs on
  the SparseCore.
- `ep_ragged_tc_fallback`: EP, the same kernel entry point is called but it
  takes its own TensorCore fallback (no SparseCore, or the input is small).
- `tensorcore`: plain XLA gather, weight, mask and sum.
"""


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeProcessPlan:
    """Process-wide MoE knobs, fixed before any bucket compiles.

    Attributes:
        use_sparse_core: USE_MOE_SPARSE_CORE, read when the custom op is built.
        onehot_threshold: The resolved one-hot routed-row threshold.
        single_sort: TPU_MOE_SINGLE_SORT, read at import.
        counting_sort: TPU_MOE_COUNTING_SORT, read at import.
        dense_expert_threshold: TPU_MOE_DENSE_EXPERT_THRESHOLD, read at import.
        ep_gather_reduce_version: RAGGED_GATHER_REDUCE_VERSION, read at import.
        single_sort_max_rows: Routed-row cap of the counting and single
            sorts on non-ragged paths.
        counting_scatter_max_rows: Routed-row cap of the token-id scatter.
    """

    use_sparse_core: bool
    onehot_threshold: int
    single_sort: bool
    counting_sort: bool
    dense_expert_threshold: int
    ep_gather_reduce_version: GatherReduceVersion
    single_sort_max_rows: int = SINGLE_SORT_MAX_ROWS
    counting_scatter_max_rows: int = COUNTING_SORT_SCATTER_MAX_ROWS

    def __post_init__(self) -> None:
        if self.ep_gather_reduce_version not in ("v1", "v2", "v3"):
            raise ValueError(f"bad gather-reduce version: {self}")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeOpIn:
    """What one MoE call receives, known when its bucket compiles.

    Attributes:
        num_tokens: Local token rows of `hidden_states`.
        topk: Experts per token.
        local_num_experts: Experts held by this shard (`w1.shape[0]`).
        use_ep: Expert-parallel routing.
    """

    num_tokens: int
    topk: int
    local_num_experts: int
    use_ep: bool

    def __post_init__(self) -> None:
        if min(self.num_tokens, self.topk, self.local_num_experts) < 0:
            raise ValueError(f"bad MoE call shape: {self}")

    @property
    def routed_rows(self) -> int:
        """Routed (token, expert) rows: `num_tokens * topk`."""
        return self.num_tokens * self.topk


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class GmmOutSpec:
    """Static facts of the combine's inputs, read off the traced arrays.

    Attributes:
        routes: `argsort_revert_indices.size`.
        rows: Second GMM output rows (`gmm2_res.shape[0]`).
        hidden: Second GMM output width (`gmm2_res.shape[-1]`).
        dtype: Second GMM output dtype name (e.g. "bfloat16").
        dtype_bits: Second GMM output element width in bits.
        token_indices_shape: `token_indices_sorted.shape`.
        token_indices_dtype: `token_indices_sorted` dtype name.
        topk_weights_shape: `topk_weights_flat.shape`.
        topk_weights_dtype: `topk_weights_flat` dtype name.
    """

    routes: int
    rows: int
    hidden: int
    dtype: str
    dtype_bits: int
    token_indices_shape: tuple[int, ...]
    token_indices_dtype: str
    topk_weights_shape: tuple[int, ...]
    topk_weights_dtype: str

    @property
    def size(self) -> int:
        """Elements of the second GMM output."""
        return self.rows * self.hidden

    @property
    def dtype_bytes(self) -> int:
        """`jax.dtypes.itemsize_bits(dtype) // 8`, as the kernels compute it."""
        return self.dtype_bits // 8


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeEntryPlan:
    """Decisions `fused_moe_func` makes before routing.

    Attributes:
        path: Dense-expert kernel or routed GMM.
        scrub_nonfinite: Zero non-finite hidden states and routing weights
            before a one-hot permute (one bad token must not poison all).
    """

    path: MoePath
    scrub_nonfinite: bool


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeRoutingPlan:
    """How `prepare_routed_gmm_inputs` builds the routed layout.

    Attributes:
        sort: How the permutation and group sizes are built.
        permute: How routed rows are gathered.
        token_bits: Bits for a token id in the packed (key, token) sort.
        flat_bits: Bits for a flat route index in the packed single sort.
    """

    sort: SortForm
    permute: Permute
    token_bits: int
    flat_bits: int

    @property
    def uses_counting_sort(self) -> bool:
        """Whether the counting sort's positions and counts are used."""
        return self.sort in ("counting_scatter", "counting_packed", "counting_argsort")

    @property
    def argsorts_keys(self) -> bool:
        """Whether `jnp.argsort(sort_keys)` runs."""
        return self.sort in ("counting_argsort", "argsort_packed", "argsort_gather")


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeCombinePlan:
    """Decisions made after the second GMM.

    Attributes:
        combine: The combine path that runs.
        ep_gather_reduce_version: The EP kernel version, for `ep_ragged_*`.
        local_v3_token_alignment: Destination-token alignment the local v3
            kernel pads to (`local_v3` only).
    """

    combine: Combine
    ep_gather_reduce_version: GatherReduceVersion | None = None
    local_v3_token_alignment: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeBucketPlan:
    """Every MoE path decision for one compiled bucket.

    Attributes:
        entry: The pre-routing decisions.
        routing: The routing layout; None on the dense-expert path.
        combine: The post-GMM decisions; None on the dense-expert path.
    """

    entry: MoeEntryPlan
    routing: MoeRoutingPlan | None
    combine: MoeCombinePlan | None


def gather_reduce_version(value: str) -> GatherReduceVersion:
    """Narrows a RAGGED_GATHER_REDUCE_VERSION value to the closed set.

    Args:
        value: The env value (already validated by its getter).

    Returns:
        The same value, typed.

    Raises:
        ValueError: For a value outside v1, v2, v3.
    """
    match value:
        case "v1" | "v2" | "v3":
            return value
        case _:
            raise ValueError(f"unknown gather-reduce version {value!r}")


def auto_onehot_permute_threshold(target: TpuTarget | None) -> int:
    """The auto one-hot routed-row threshold for a chip.

    `min(SparseCore block rows - 1, ONEHOT_AUTO_CAP)` on SparseCore chips, 0
    on chips without a SparseCore and on hosts with no TPU description.

    Args:
        target: The chip, or None where `get_tpu_info` raised ValueError.

    Returns:
        The routed-row count at or below which MoE permute and combine run as
        TensorCore one-hot matmuls.
    """
    if target is None or target.sparse_core is None:
        return 0
    return min(target.sparse_core.block_rows - 1, ONEHOT_AUTO_CAP)


def dense_expert_candidate(process: MoeProcessPlan, op: MoeOpIn) -> bool:
    """Whether the token count admits the dense-expert kernel.

    Decode services also compile large prefill buckets; the token threshold
    applies before kernel admission so those buckets keep routed GMM. The
    kernel's own admission (`can_use_dense_expert`, which reads weight
    layouts and dtypes) is asked only when this holds, and its answer is the
    `dense_expert_admitted` input of `plan_routing`.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.

    Returns:
        True when `can_use_dense_expert` should be asked.
    """
    return 0 < op.num_tokens <= process.dense_expert_threshold


def _onehot_permute(process: MoeProcessPlan, op: MoeOpIn) -> bool:
    return (
        op.use_ep
        and process.use_sparse_core
        and op.routed_rows <= process.onehot_threshold
    )


def plan_entry(
    process: MoeProcessPlan, op: MoeOpIn, *, dense_expert_admitted: bool
) -> MoeEntryPlan:
    """The decisions `fused_moe_func` makes before routing.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        dense_expert_admitted: `dense_expert_candidate(...)` and the kernel's
            `can_use_dense_expert(...)` both held. Pass False without asking
            the kernel when the candidate rule fails, as today's short-circuit
            does.

    Returns:
        The entry plan.
    """
    return MoeEntryPlan(
        path="dense_expert" if dense_expert_admitted else "routed",
        scrub_nonfinite=_onehot_permute(process, op),
    )


def _ragged_permute(process: MoeProcessPlan, op: MoeOpIn) -> bool:
    return (
        op.use_ep
        and process.use_sparse_core
        and op.routed_rows > process.onehot_threshold
    )


def counting_sort_attempted(process: MoeProcessPlan, op: MoeOpIn) -> bool:
    """Whether `_counting_sort_positions` is called.

    Always on the ragged permute; elsewhere when TPU_MOE_COUNTING_SORT is on
    and the routed rows are within the single-sort cap.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.

    Returns:
        True when the counting sort should be tried.
    """
    return _ragged_permute(process, op) or (
        process.counting_sort and op.routed_rows <= process.single_sort_max_rows
    )


def counting_sort_accepts(op: MoeOpIn) -> bool:
    """Predicts whether `_counting_sort_positions` returns positions.

    DUPLICATE of its refusal: it returns (None, None), emitting nothing, when
    `local_num_experts + 1 > min(COUNTING_SORT_BLOCK, routed_rows)`. Used by
    `plan_bucket` (logs, tests); `prepare_routed_gmm_inputs` passes the
    actual result instead.

    Args:
        op: This call's shapes.

    Returns:
        True when the counting sort would produce positions.
    """
    return not op.local_num_experts + 1 > min(COUNTING_SORT_BLOCK, op.routed_rows)


def plan_routing(
    process: MoeProcessPlan, op: MoeOpIn, *, counting_sorted: bool | None = None
) -> MoeRoutingPlan:
    """How the routed layout is built for one bucket.

    The sort form depends on whether the counting sort produced positions, a
    value computed while tracing. `prepare_routed_gmm_inputs` calls the
    counting sort when `counting_sort_attempted` holds and passes whether it
    returned positions; callers without that result (logs, tests) leave it
    None and get `counting_sort_accepts`' prediction.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        counting_sorted: The counting sort ran and returned positions, or
            None to predict it.

    Returns:
        The routing plan.

    Raises:
        ValueError: If `counting_sorted` is True but the counting sort would
            not have been attempted.
    """
    rows = op.routed_rows
    num_buckets = op.local_num_experts + 1
    ragged_permute = _ragged_permute(process, op)
    attempted = counting_sort_attempted(process, op)
    if counting_sorted is None:
        counting_sorted = attempted and counting_sort_accepts(op)
    elif counting_sorted and not attempted:
        raise ValueError(f"counting sort reported for a bucket that skips it: {op}")
    counting = counting_sorted
    token_bits = max(1, (op.num_tokens - 1).bit_length())
    packable = num_buckets.bit_length() + token_bits < _PACKED_SORT_BITS
    flat_bits = max(1, (rows - 1).bit_length())
    single_sort = (
        process.single_sort
        and not counting
        and not ragged_permute
        and rows <= process.single_sort_max_rows
        and num_buckets.bit_length() + flat_bits < _PACKED_SORT_BITS
    )
    scatter = (
        counting and not ragged_permute and rows <= process.counting_scatter_max_rows
    )
    sort: SortForm
    if counting:
        if scatter:
            sort = "counting_scatter"
        elif packable:
            sort = "counting_packed"
        else:
            sort = "counting_argsort"
    elif single_sort:
        sort = "single_sort"
    elif packable:
        sort = "argsort_packed"
    else:
        sort = "argsort_gather"
    permute: Permute
    if op.use_ep and process.use_sparse_core:
        if _onehot_permute(process, op):
            permute = "onehot_from_positions" if counting else "onehot_from_tokens"
        else:
            permute = "ragged_sc"
    else:
        permute = "gather"
    return MoeRoutingPlan(
        sort=sort, permute=permute, token_bits=token_bits, flat_bits=flat_bits
    )


def _local_v3_precondition(
    process: MoeProcessPlan, op: MoeOpIn, out: GmmOutSpec
) -> bool:
    return (
        # Local SparseCore path; route-pair metadata needs at least two routes.
        process.use_sparse_core
        and not op.use_ep
        and op.topk >= 2
        # TensorCore combine handles top-8 batches.
        and op.topk != 8
        # Honor the configured SparseCore dispatch cutoff.
        and out.routes > process.onehot_threshold
        # BF16 kernel input.
        and out.dtype == "bfloat16"
        # Packed source-index capacity.
        and out.rows <= LOCAL_V3_MAX_SOURCE_ROWS
    )


def _v3_num_column_partitions(hidden_size: int, num_cores: int, num_lanes: int) -> int:
    # DUPLICATE of `ragged_gather_reduce_v3._calculate_num_column_partitions`.
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


def v3_token_block_alignment(hidden_size: int, target: TpuTarget) -> int:
    """Destination-token granularity of `ragged_gather_reduce_v3`.

    DUPLICATE of `ragged_gather_reduce_v3.token_block_alignment`: one 64-token
    block per SparseCore row partition.

    Args:
        hidden_size: Combine input width.
        target: A chip with a SparseCore.

    Returns:
        The token alignment the kernel pads destinations to.
    """
    sc = target.sparse_core
    assert sc is not None
    num_cores = sc.num_cores * sc.num_subcores
    num_column_partitions = _v3_num_column_partitions(
        hidden_size, num_cores, target.num_lanes
    )
    return (num_cores // num_column_partitions) * _V3_TOKEN_BLOCK


def _local_v3(
    op: MoeOpIn, out: GmmOutSpec, target: TpuTarget | None
) -> tuple[bool, int | None]:
    sc = target.sparse_core if target is not None else None
    # Lane-aligned width, 16-lane SparseCore, and v3's own cutoff (two BF16
    # buffers use at least 60% of VMEM).
    if not (
        target is not None
        and sc is not None
        and sc.num_lanes == 16
        and out.hidden % target.num_lanes == 0
        and out.size * 4 >= target.vmem_bytes_per_core * 0.6
    ):
        return False, None
    # The kernel pads destination tokens to one 64-token block per SparseCore
    # row partition: require at least one full block and padding of at most
    # half the batch.
    alignment = v3_token_block_alignment(out.hidden, target)
    padded_tokens = -(-op.num_tokens // alignment) * alignment
    ok = op.num_tokens >= alignment and padded_tokens * 2 <= op.num_tokens * 3
    return ok, alignment


def _ep_ragged_falls_back(
    version: GatherReduceVersion, out: GmmOutSpec, target: TpuTarget | None
) -> bool:
    """Whether the EP gather-reduce kernel takes its TensorCore fallback.

    With no chip description the kernels raise from `get_tpu_info` instead;
    that host never runs this path, and it is labelled a fallback here.
    """
    if target is None or target.sparse_core is None:
        return True
    small = out.size * out.dtype_bytes * 2 < target.vmem_bytes_per_core * 0.6
    if version == "v3":
        # DUPLICATE of `ragged_gather_reduce_v3.ragged_gather_reduce`'s two
        # fallback checks.
        return out.dtype != "bfloat16" or out.hidden % target.num_lanes != 0 or small
    # DUPLICATE of v1 `ragged_gather_reduce` and of v2 `Config.should_fallback`
    # (source_rows * hidden_size * bytes == size * bytes).
    return small


def owner_onehot_admits(op: MoeOpIn, out: GmmOutSpec, target: TpuTarget | None) -> bool:
    """Whether the blockwise one-hot unpermute kernel accepts this call.

    DUPLICATE of `moe_onehot_unpermute.can_use_blockwise_onehot_unpermute`.
    `topk_weights_sorted` there is `zeros_like(topk_weights_flat)`, and
    `valid_count` is always an int32 `(1,)` array, so their checks reduce to
    the ones below. `fused_moe_gmm` asserts this agrees with the kernel.

    Args:
        op: This call's shapes.
        out: The combine's input facts.
        target: The chip, or None where `get_tpu_info` raised.

    Returns:
        The kernel's admission answer.

    Raises:
        ValueError: When the shape checks pass but there is no chip
            description; the kernel's own `get_tpu_info` call raises there.
    """
    if out.dtype != "bfloat16":
        return False
    route_capacity, hidden_size = out.rows, out.hidden
    if out.token_indices_shape != (route_capacity,):
        return False
    if out.token_indices_dtype != "int32":
        return False
    if out.topk_weights_shape != (route_capacity,):
        return False
    if out.topk_weights_dtype != "bfloat16":
        return False
    if target is None:
        raise ValueError("owner-output admission needs a TPU description")
    if route_capacity % _OWNER_ROUTE_BLOCK_SIZE:
        return False
    if hidden_size % target.mxu_column_size:
        return False
    if op.num_tokens % target.sublane_tiling(out.dtype_bits):
        return False
    vmem_limit_bytes = int(target.vmem_bytes_per_core * 0.9)
    explicit = (
        op.num_tokens * hidden_size * (4 + 2)
        + 2 * _OWNER_ROUTE_BLOCK_SIZE * hidden_size * 2
        + route_capacity * (4 + 4)
    )
    return explicit + _OWNER_VMEM_WORKSPACE_MARGIN_BYTES <= vmem_limit_bytes


def combine_reads_owner_mode(
    process: MoeProcessPlan, op: MoeOpIn, out: GmmOutSpec
) -> bool:
    """Whether TPU_MOE_OWNER_OUTPUT_MODE matters for this bucket.

    The env var is validated on read, so it is read only when this holds,
    exactly where today's code reads it.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        out: The combine's input facts.

    Returns:
        True on the EP one-hot combine.
    """
    # The local v3 rule needs `not use_ep`, so it never pre-empts this one.
    return (
        op.use_ep and process.use_sparse_core and out.routes <= process.onehot_threshold
    )


def combine_needs_target(
    process: MoeProcessPlan, op: MoeOpIn, out: GmmOutSpec, *, owner_output: bool
) -> bool:
    """Whether `plan_combine` reads the chip for this bucket.

    True exactly where today's code (or the kernel it calls) asks
    `get_tpu_info`, so the runtime is not touched on any other path.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        out: The combine's input facts.
        owner_output: TPU_MOE_OWNER_OUTPUT_MODE is "on" (False if not read).

    Returns:
        Whether to pass a `TpuTarget` to `plan_combine`.
    """
    if _local_v3_precondition(process, op, out):
        return True
    if op.use_ep and process.use_sparse_core:
        return out.routes > process.onehot_threshold or owner_output
    return False


def plan_combine(
    process: MoeProcessPlan,
    op: MoeOpIn,
    out: GmmOutSpec,
    target: TpuTarget | None,
    *,
    owner_output: bool,
) -> MoeCombinePlan:
    """The post-GMM combine decision for one bucket.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        out: The combine's input facts.
        target: The chip, or None where it was not needed
            (`combine_needs_target` False) or `get_tpu_info` raised.
        owner_output: TPU_MOE_OWNER_OUTPUT_MODE is "on" (False if not read).

    Returns:
        The combine plan.
    """
    if _local_v3_precondition(process, op, out):
        ok, alignment = _local_v3(op, out, target)
        if ok:
            # The kernel's own fallback checks (no SparseCore, non-BF16, width
            # not lane-aligned, under 60% of VMEM) all fail here by
            # construction, so it runs on the SparseCore.
            return MoeCombinePlan(
                combine="local_v3", local_v3_token_alignment=alignment
            )
    if op.use_ep and process.use_sparse_core:
        if out.routes <= process.onehot_threshold:
            if owner_output and owner_onehot_admits(op, out, target):
                return MoeCombinePlan(combine="ep_owner_onehot")
            return MoeCombinePlan(combine="ep_onehot")
        version = process.ep_gather_reduce_version
        return MoeCombinePlan(
            combine=(
                "ep_ragged_tc_fallback"
                if _ep_ragged_falls_back(version, out, target)
                else "ep_ragged_sc"
            ),
            ep_gather_reduce_version=version,
        )
    return MoeCombinePlan(combine="tensorcore")


def plan_bucket(
    process: MoeProcessPlan,
    op: MoeOpIn,
    out: GmmOutSpec,
    target: TpuTarget | None,
    *,
    dense_expert_admitted: bool,
    owner_output: bool,
) -> MoeBucketPlan:
    """All stages at once, for tests and logs (counting sort predicted).

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        out: The combine's input facts.
        target: The chip.
        dense_expert_admitted: See `plan_entry`.
        owner_output: See `plan_combine`.

    Returns:
        The whole bucket plan.
    """
    entry = plan_entry(process, op, dense_expert_admitted=dense_expert_admitted)
    if entry.path == "dense_expert":
        return MoeBucketPlan(entry=entry, routing=None, combine=None)
    return MoeBucketPlan(
        entry=entry,
        routing=plan_routing(process, op),
        combine=plan_combine(process, op, out, target, owner_output=owner_output),
    )
