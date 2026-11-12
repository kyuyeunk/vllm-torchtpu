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
"""Pure MoE path decisions.

Every path choice `fused_moe_gmm` makes while a bucket compiles lives here as
a pure function of three values:

- `MoeProcessPlan`: process-wide knobs (SparseCore on/off, the one-hot
  threshold, the sort toggles, the EP gather-reduce version);
- `MoeOpIn` and `GmmOutSpec`: the concrete shapes and dtypes of this call;
- `pltpu.TpuInfo`: the chip, as JAX describes it.

The rules are today's inline conditions, moved without change. The plan is
the single decider: where a kernel used to re-decide (the SparseCore
gather-reduce kernels' TensorCore fallbacks, the counting sort's refusal, the
owner-output kernel's admission), the kernel now receives the plan's decision
or asserts its precondition. Kernel-specific rules live next to their kernel
as pure modules (`ragged_gather_reduce_fallback`,
`ragged_gather_reduce_v3_partition`, `moe_onehot_unpermute_admission`) that
both the kernel and this plan import.

The decision resolves in three stages, each where its inputs first exist:
`plan_entry` (non-finite scrub, dense expert) at the top of `fused_moe_func`;
`plan_routing` in `prepare_routed_gmm_inputs`; and `plan_combine` after the
second GMM, because the combine depends on that output's width, which
`gmm_v2` derives internally (it pads N to a lane multiple).

Imports no torch or TPU runtime, and JAX only for dtypes (the chip type is
imported for type checking only), so every rule here runs on CPU.
"""

import dataclasses
import enum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.megablox import moe_onehot_unpermute_admission as owner
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_fallback as fallback
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v3_partition as v3
from vllm_torchtpu.layers.core.plan_marker import plan_of

if TYPE_CHECKING:
    from jax.experimental.pallas import tpu as pltpu

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

GatherReduceVersion = fallback.GatherReduceVersion
"""The SparseCore gather-reduce kernel versions (RAGGED_GATHER_REDUCE_VERSION)."""


class MoePath(enum.Enum):
    """Which kernel family runs the MoE layer.

    - `DENSE_EXPERT`: the experimental dense-expert kernel.
    - `ROUTED`: GMM over routed rows.
    """

    DENSE_EXPERT = "dense_expert"
    ROUTED = "routed"


class SortForm(enum.Enum):
    """How the routing permutation and `group_sizes` are built.

    - `COUNTING_SCATTER`: counting-sort positions; token ids scattered through
      them; group sizes from the counting histogram. No comparison sort.
    - `COUNTING_PACKED`: counting-sort positions; token ids from one packed
      (key, token) sort.
    - `COUNTING_ARGSORT`: counting-sort positions; token ids gathered through
      an `argsort` of the keys (the packed key does not fit in int32).
    - `SINGLE_SORT`: one packed (key, flat position) sort gives the order, its
      inverse (a scatter) and the token ids.
    - `ARGSORT_PACKED`: `argsort`, `argsort` of that, then a packed token sort
      -- the three-sort form.
    - `ARGSORT_GATHER`: two `argsort`s, token ids gathered.
    """

    COUNTING_SCATTER = "counting_scatter"
    COUNTING_PACKED = "counting_packed"
    COUNTING_ARGSORT = "counting_argsort"
    SINGLE_SORT = "single_sort"
    ARGSORT_PACKED = "argsort_packed"
    ARGSORT_GATHER = "argsort_gather"


class Permute(enum.Enum):
    """How routed rows are gathered from tokens.

    - `ONEHOT_FROM_POSITIONS`: one-hot matmul built from the counting-sort
      positions (EP + SparseCore, at or under the one-hot threshold).
    - `ONEHOT_FROM_TOKENS`: one-hot matmul built from the sorted token ids.
    - `RAGGED_SC`: SparseCore `ragged_gather_v2`.
    - `GATHER`: a plain XLA gather.
    """

    ONEHOT_FROM_POSITIONS = "onehot_from_positions"
    ONEHOT_FROM_TOKENS = "onehot_from_tokens"
    RAGGED_SC = "ragged_sc"
    GATHER = "gather"


class Combine(enum.Enum):
    """How the top-k expert outputs are summed back per token.

    - `LOCAL_V3`: non-EP SparseCore destination-major `ragged_gather_reduce_v3`.
    - `EP_OWNER_ONEHOT`: EP, blockwise one-hot unpermute kernel (owner output).
    - `EP_ONEHOT`: EP, dense one-hot combine matmul.
    - `EP_RAGGED_SC`: EP, the configured SparseCore gather-reduce kernel runs on
      the SparseCore.
    - `EP_RAGGED_TC_FALLBACK`: EP, the same kernel entry point runs its
      TensorCore fallback (no SparseCore, or the input is small).
    - `TENSORCORE`: plain XLA gather, weight, mask and sum.
    """

    LOCAL_V3 = "local_v3"
    EP_OWNER_ONEHOT = "ep_owner_onehot"
    EP_ONEHOT = "ep_onehot"
    EP_RAGGED_SC = "ep_ragged_sc"
    EP_RAGGED_TC_FALLBACK = "ep_ragged_tc_fallback"
    TENSORCORE = "tensorcore"


class OwnerOutputMode(enum.Enum):
    """TPU_MOE_OWNER_OUTPUT_MODE; each value is its (lower-case) env spelling."""

    OFF = "off"
    ON = "on"


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
        if not isinstance(self.ep_gather_reduce_version, GatherReduceVersion):
            raise TypeError(f"bad gather-reduce version: {self}")


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
        dtype: Second GMM output dtype.
        token_indices_shape: `token_indices_sorted.shape`.
        token_indices_dtype: `token_indices_sorted.dtype`.
        topk_weights_shape: `topk_weights_flat.shape`.
        topk_weights_dtype: `topk_weights_flat.dtype`.
    """

    routes: int
    rows: int
    hidden: int
    dtype: jnp.dtype
    token_indices_shape: tuple[int, ...]
    token_indices_dtype: jnp.dtype
    topk_weights_shape: tuple[int, ...]
    topk_weights_dtype: jnp.dtype

    @property
    def size(self) -> int:
        """Elements of the second GMM output."""
        return self.rows * self.hidden

    @property
    def dtype_bytes(self) -> int:
        """`jax.dtypes.itemsize_bits(dtype) // 8`, as the kernels compute it."""
        return jax.dtypes.itemsize_bits(self.dtype) // 8


@plan_of("moe.entry")
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


@plan_of("moe.routing")
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
        return self.sort in (
            SortForm.COUNTING_SCATTER,
            SortForm.COUNTING_PACKED,
            SortForm.COUNTING_ARGSORT,
        )

    @property
    def argsorts_keys(self) -> bool:
        """Whether `jnp.argsort(sort_keys)` runs."""
        return self.sort in (
            SortForm.COUNTING_ARGSORT,
            SortForm.ARGSORT_PACKED,
            SortForm.ARGSORT_GATHER,
        )


@plan_of("moe.combine")
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

    def __post_init__(self) -> None:
        # Each optional field belongs to one combine, so a plan cannot carry
        # a choice its combine would silently ignore.
        ep_ragged = self.combine in (
            Combine.EP_RAGGED_SC,
            Combine.EP_RAGGED_TC_FALLBACK,
        )
        if (self.ep_gather_reduce_version is not None) != ep_ragged:
            raise ValueError(f"gather-reduce version is for ep_ragged_* only: {self}")
        if (self.local_v3_token_alignment is not None) != (
            self.combine is Combine.LOCAL_V3
        ):
            raise ValueError(f"token alignment is for local_v3 only: {self}")


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
    """Parses a RAGGED_GATHER_REDUCE_VERSION value; the one place it is read.

    Args:
        value: The env value (already validated by its getter).

    Returns:
        The version.

    Raises:
        ValueError: For a value that names no version.
    """
    try:
        return GatherReduceVersion(value)
    except ValueError:
        valid = ", ".join(v.value for v in GatherReduceVersion)
        raise ValueError(
            f"unknown gather-reduce version {value!r}; valid: {valid}"
        ) from None


def owner_output_mode(value: str) -> OwnerOutputMode:
    """Parses a TPU_MOE_OWNER_OUTPUT_MODE value, case-insensitively.

    Args:
        value: The env value (already validated by its getter).

    Returns:
        The mode.

    Raises:
        ValueError: For a value that names no mode.
    """
    try:
        return OwnerOutputMode(value.lower())
    except ValueError:
        valid = ", ".join(m.value for m in OwnerOutputMode)
        raise ValueError(
            f"unknown TPU_MOE_OWNER_OUTPUT_MODE {value!r}; valid: {valid}"
        ) from None


def auto_onehot_permute_threshold(target: "pltpu.TpuInfo | None") -> int:
    """The auto one-hot routed-row threshold for a chip.

    `min(SparseCore block rows - 1, ONEHOT_AUTO_CAP)` on SparseCore chips, 0
    on chips without a SparseCore and on hosts with no TPU description.

    Args:
        target: The chip, or None where `get_tpu_info` raised ValueError.

    Returns:
        The routed-row count at or below which MoE permute and combine run as
        TensorCore one-hot matmuls.
    """
    sc = target.sparse_core if target is not None else None
    if sc is None:
        return 0
    block_rows = sc.num_lanes * sc.num_cores * sc.num_subcores
    return min(block_rows - 1, ONEHOT_AUTO_CAP)


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
        path=MoePath.DENSE_EXPERT if dense_expert_admitted else MoePath.ROUTED,
        scrub_nonfinite=_onehot_permute(process, op),
    )


def _ragged_permute(process: MoeProcessPlan, op: MoeOpIn) -> bool:
    return (
        op.use_ep
        and process.use_sparse_core
        and op.routed_rows > process.onehot_threshold
    )


def counting_sort_attempted(process: MoeProcessPlan, op: MoeOpIn) -> bool:
    """Whether the counting sort is wanted (it runs if it also accepts).

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
    """Whether the counting sort can serve this bucket.

    The rank contraction runs in blocks of `min(COUNTING_SORT_BLOCK, rows)`;
    XLA miscompiles it on jaxlib 0.10.2 + libtpu 0.0.44.1 when there are more
    buckets than block rows (the positions come back not a permutation), so
    that region uses the comparison sorts. `_counting_sort_positions` asserts
    this precondition.

    Args:
        op: This call's shapes.

    Returns:
        True when the counting sort may run.
    """
    return not op.local_num_experts + 1 > min(COUNTING_SORT_BLOCK, op.routed_rows)


def plan_routing(process: MoeProcessPlan, op: MoeOpIn) -> MoeRoutingPlan:
    """How the routed layout is built for one bucket.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.

    Returns:
        The routing plan.
    """
    rows = op.routed_rows
    num_buckets = op.local_num_experts + 1
    ragged_permute = _ragged_permute(process, op)
    counting = counting_sort_attempted(process, op) and counting_sort_accepts(op)
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
            sort = SortForm.COUNTING_SCATTER
        elif packable:
            sort = SortForm.COUNTING_PACKED
        else:
            sort = SortForm.COUNTING_ARGSORT
    elif single_sort:
        sort = SortForm.SINGLE_SORT
    elif packable:
        sort = SortForm.ARGSORT_PACKED
    else:
        sort = SortForm.ARGSORT_GATHER
    permute: Permute
    if op.use_ep and process.use_sparse_core:
        if _onehot_permute(process, op):
            permute = (
                Permute.ONEHOT_FROM_POSITIONS
                if counting
                else Permute.ONEHOT_FROM_TOKENS
            )
        else:
            permute = Permute.RAGGED_SC
    else:
        permute = Permute.GATHER
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
        and out.dtype == jnp.bfloat16
        # Packed source-index capacity.
        and v3.rows_supported(out.rows)
    )


def _tensorcore_fallback(
    version: GatherReduceVersion, out: GmmOutSpec, target: "pltpu.TpuInfo"
) -> bool:
    return fallback.takes_tensorcore_path(
        version=version,
        size=out.size,
        hidden=out.hidden,
        dtype=out.dtype,
        chip=target,
    )


def _local_v3_alignment(
    op: MoeOpIn, out: GmmOutSpec, target: "pltpu.TpuInfo | None"
) -> int | None:
    """The v3 token alignment when local v3 is worth running, else None."""
    # v3 can run here (a 16-lane SparseCore among its needs), and would not
    # take its own TensorCore path (two BF16 buffers at least 60% of VMEM).
    if target is None or not v3.v3_capable(
        rows=out.rows, hidden=out.hidden, dtype=out.dtype, chip=target
    ):
        return None
    if _tensorcore_fallback(GatherReduceVersion.V3, out, target):
        return None
    # The kernel pads destination tokens to one 64-token block per SparseCore
    # row partition: require at least one full block and padding of at most
    # half the batch.
    alignment = v3.token_block_alignment(out.hidden, target)
    padded_tokens = -(-op.num_tokens // alignment) * alignment
    if op.num_tokens >= alignment and padded_tokens * 2 <= op.num_tokens * 3:
        return alignment
    return None


def owner_onehot_admits(
    op: MoeOpIn, out: GmmOutSpec, target: "pltpu.TpuInfo | None"
) -> bool:
    """Whether the blockwise one-hot unpermute kernel accepts this call.

    `topk_weights_sorted` is `zeros_like(topk_weights_flat)` and `valid_count`
    an int32 `(1,)` array, so their facts come from `out`.

    Args:
        op: This call's shapes.
        out: The combine's input facts.
        target: The chip, or None where `get_tpu_info` raised.

    Returns:
        The kernel's admission answer.

    Raises:
        ValueError: When the shapes pass but there is no chip description,
            where the kernel's own `get_tpu_info` call used to raise.
    """
    if not owner.shape_admits(
        route_shape=(out.rows, out.hidden),
        route_dtype=out.dtype,
        token_indices_shape=out.token_indices_shape,
        token_indices_dtype=out.token_indices_dtype,
        weights_shape=out.topk_weights_shape,
        weights_dtype=out.topk_weights_dtype,
        valid_count_shape=(1,),
        valid_count_dtype=jnp.dtype(jnp.int32),
    ):
        return False
    if target is None:
        raise ValueError("owner-output admission needs a TPU description")
    return owner.chip_admits(
        route_capacity=out.rows,
        hidden_size=out.hidden,
        num_tokens=op.num_tokens,
        dtype=out.dtype,
        chip=target,
    )


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


class TargetNeed(enum.Enum):
    """How `plan_combine` needs the chip.

    - `NONE`: not at all.
    - `OPTIONAL`: the chip, or None where `get_tpu_info` raises.
    - `REQUIRED`: the chip; an exception from `get_tpu_info` propagates.
    """

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


def combine_target_need(
    process: MoeProcessPlan, op: MoeOpIn, out: GmmOutSpec, *, owner_output: bool
) -> TargetNeed:
    """How `plan_combine` reads the chip for this bucket.

    The runtime is read exactly where the old code (or the kernel it called)
    asked `get_tpu_info`, with the same handling of a host JAX has no TPU
    description for: the local v3 check fell back, the EP gather-reduce
    kernels raised, and the owner admission raises after its shape checks.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        out: The combine's input facts.
        owner_output: TPU_MOE_OWNER_OUTPUT_MODE is on (False if not read).

    Returns:
        The need.
    """
    if _local_v3_precondition(process, op, out):
        return TargetNeed.OPTIONAL
    if op.use_ep and process.use_sparse_core:
        if out.routes > process.onehot_threshold:
            return TargetNeed.REQUIRED
        return TargetNeed.OPTIONAL if owner_output else TargetNeed.NONE
    return TargetNeed.NONE


def plan_combine(
    process: MoeProcessPlan,
    op: MoeOpIn,
    out: GmmOutSpec,
    target: "pltpu.TpuInfo | None",
    *,
    owner_output: bool,
) -> MoeCombinePlan:
    """The post-GMM combine decision for one bucket.

    Args:
        process: Process-wide knobs.
        op: This call's shapes.
        out: The combine's input facts.
        target: The chip, read as `combine_target_need` says.
        owner_output: TPU_MOE_OWNER_OUTPUT_MODE is on (False if not read).

    Returns:
        The combine plan.
    """
    if _local_v3_precondition(process, op, out):
        alignment = _local_v3_alignment(op, out, target)
        if alignment is not None:
            return MoeCombinePlan(
                combine=Combine.LOCAL_V3, local_v3_token_alignment=alignment
            )
    if op.use_ep and process.use_sparse_core:
        if out.routes <= process.onehot_threshold:
            if owner_output and owner_onehot_admits(op, out, target):
                return MoeCombinePlan(combine=Combine.EP_OWNER_ONEHOT)
            return MoeCombinePlan(combine=Combine.EP_ONEHOT)
        version = process.ep_gather_reduce_version
        assert target is not None, "the EP gather-reduce choice reads the chip"
        return MoeCombinePlan(
            combine=(
                Combine.EP_RAGGED_TC_FALLBACK
                if _tensorcore_fallback(version, out, target)
                else Combine.EP_RAGGED_SC
            ),
            ep_gather_reduce_version=version,
        )
    return MoeCombinePlan(combine=Combine.TENSORCORE)


def _sparse_core_kernel_runs(
    version: GatherReduceVersion, out: GmmOutSpec, target: "pltpu.TpuInfo | None"
) -> bool:
    if target is None or target.sparse_core is None:
        return False
    if version is not GatherReduceVersion.V3:
        return True
    return v3.v3_capable(rows=out.rows, hidden=out.hidden, dtype=out.dtype, chip=target)


def combine_capable(
    plan: MoeCombinePlan,
    op: MoeOpIn,
    out: GmmOutSpec,
    target: "pltpu.TpuInfo | None",
) -> bool:
    """Whether the combine `plan` names can run on this call at all.

    Capability only: the preferences `plan_combine` adds on top (the one-hot
    threshold, the VMEM cutoff of the SparseCore kernels, the top-8 and
    padding rules of local v3) are left out, so a plan they reject but this
    accepts must still produce the same output. A plan this rejects must fail
    loudly when forced.

    Args:
        plan: The combine plan to run.
        op: This call's shapes.
        out: The combine's input facts.
        target: The chip, or None without a TPU description.

    Returns:
        True when the combine can run.
    """
    match plan.combine:
        case Combine.TENSORCORE | Combine.EP_ONEHOT | Combine.EP_RAGGED_TC_FALLBACK:
            return True
        case Combine.EP_RAGGED_SC:
            assert plan.ep_gather_reduce_version is not None
            return _sparse_core_kernel_runs(plan.ep_gather_reduce_version, out, target)
        case Combine.LOCAL_V3:
            # Route-pair metadata needs at least two routes.
            return op.topk >= 2 and _sparse_core_kernel_runs(
                GatherReduceVersion.V3, out, target
            )
        case Combine.EP_OWNER_ONEHOT:
            return owner_onehot_admits(op, out, target)


def plan_bucket(
    process: MoeProcessPlan,
    op: MoeOpIn,
    out: GmmOutSpec,
    target: "pltpu.TpuInfo | None",
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
    if entry.path is MoePath.DENSE_EXPERT:
        return MoeBucketPlan(entry=entry, routing=None, combine=None)
    return MoeBucketPlan(
        entry=entry,
        routing=plan_routing(process, op),
        combine=plan_combine(process, op, out, target, owner_output=owner_output),
    )
