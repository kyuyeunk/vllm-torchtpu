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
"""The MoE combine's variants, and how the layer picks one.

Every variant sums the second GMM's routed rows back into one row per token,
and every variant answers the same two calls with the arguments the layer
passes it: `run` computes, and `supports` names a hard limit the call breaks
(or None). `pick_combine` runs the variant the process setting names, else
the TensorCore combine; a named variant that cannot run a call raises.
Nothing here reads the runtime: the chip is passed in as a `pltpu.TpuInfo`.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
    blockwise_onehot_unpermute,
    blockwise_onehot_unpermute_supports,
)
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_skip_empty as skip
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v3 as v3
from vllm_torchtpu.kernels.sparse_core import ragged_gather_reduce_v4 as v4
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import (
    ragged_gather_reduce as ragged_gather_reduce_v1,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
    ragged_gather_reduce_v2,
)
from vllm_torchtpu.kernels.sparse_core.support import get_sparse_core_info


class Combine(enum.Enum):
    """How the top-k expert outputs are summed back per token.

    Each value is the `TPU_MOE_COMBINE` spelling.

    - `tensorcore`: a plain XLA gather, weight, mask and sum.
    - `onehot`: a dense one-hot matmul.
    - `owner_onehot`: the blockwise one-hot unpermute kernel.
    - `gather_reduce_v1`, `_v2`, `_v3`: the SparseCore gather-reduce kernels.
    - `gather_reduce_v4`: a SparseCore gather-reduce for top-k up to 4.
    - `segment_sum`: an XLA scatter-add of the weighted rows, in expert order.
    - `skip_empty`: a SparseCore gather-reduce that skips tokens with no route
      on this shard; it reads routing's per-token count of local routes.
    """

    TENSORCORE = "tensorcore"
    ONEHOT = "onehot"
    OWNER_ONEHOT = "owner_onehot"
    GATHER_REDUCE_V1 = "gather_reduce_v1"
    GATHER_REDUCE_V2 = "gather_reduce_v2"
    GATHER_REDUCE_V3 = "gather_reduce_v3"
    GATHER_REDUCE_V4 = "gather_reduce_v4"
    SEGMENT_SUM = "segment_sum"
    SKIP_EMPTY = "skip_empty"


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeCombinePlan:
    """Which combine runs."""

    combine: Combine


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeProcess:
    """Settings fixed for the process, every one resolved to a concrete value.

    Attributes:
        use_sparse_core: Token movement may use the SparseCore; False on
            chips without one the SparseCore kernels are validated on.
        override: The combine every call must run, or None for the
            TensorCore combine.
    """

    use_sparse_core: bool
    override: MoeCombinePlan | None = None


def resolve_moe_process(
    env: Any, chip: pltpu.TpuInfo | None, *, use_sparse_core: bool
) -> MoeProcess:
    """Resolves the MoE settings once, from the environment and the chip.

    The only place that reads the environment; the planner and the kernels
    read the `MoeProcess` it returns.

    Args:
        env: `vllm_torchtpu.envs`, or anything with its attributes.
        chip: The chip, or None on a host without a TPU.
        use_sparse_core: The layer allows SparseCore token movement.

    Returns:
        The resolved settings.

    Raises:
        ValueError: If `TPU_MOE_COMBINE` names no combine variant.
    """
    name = env.TPU_MOE_COMBINE
    try:
        combine = None if name is None else Combine(name)
    except ValueError:
        msg = f"TPU_MOE_COMBINE={name} is not one of {[c.value for c in VARIANTS]}"
        raise ValueError(msg) from None
    return MoeProcess(
        use_sparse_core=use_sparse_core
        and chip is not None
        and get_sparse_core_info(chip) is not None,
        override=None if combine is None else MoeCombinePlan(combine=combine),
    )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeOpIn:
    """The shapes of one MoE call."""

    num_tokens: int
    topk: int
    local_num_experts: int

    @property
    def routes(self) -> int:
        """Token-expert pairs, one per top-k slot of every token."""
        return self.num_tokens * self.topk


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class GmmOut:
    """The second GMM's output and routing weights, which the combine reads.

    The output has one row per route, so its row count is `MoeOpIn.routes`
    and is not restated here.

    Attributes:
        hidden: Width of the GMM output.
        dtype: GMM output dtype.
        weights_dtype: Top-k weights dtype.
        token_local_routes_shape: Shape of routing's per-token count of
            routes on this shard, or None where routing did not pass it.
    """

    hidden: int
    dtype: jnp.dtype
    weights_dtype: jnp.dtype
    token_local_routes_shape: tuple[int, ...] | None = None


def combine_args(op: MoeOpIn, out: GmmOut) -> tuple[jax.ShapeDtypeStruct, ...]:
    """The arrays the layer passes a combine, as shapes and dtypes."""
    sds = jax.ShapeDtypeStruct
    return (
        sds((op.routes, out.hidden), out.dtype),  # gmm2_res
        sds((op.local_num_experts,), jnp.int32),  # group_sizes
        sds((op.routes,), jnp.int32),  # argsort_revert_indices
        sds((op.routes,), out.weights_dtype),  # topk_weights_flat
        sds((op.routes,), jnp.bool_),  # valid_mask_flat
        sds((op.routes,), jnp.int32),  # token_indices_sorted
    )


# The variants. Each `run` takes (gmm2_res, group_sizes,
# argsort_revert_indices, topk_weights_flat, valid_mask_flat,
# token_indices_sorted, *, num_tokens, topk); `supports` takes the same
# arguments as shapes and dtypes, plus the chip.


def _tensorcore(
    g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk
):
    del group_sizes, token_indices
    token_topk_hidden = g[revert].reshape((num_tokens, topk, g.shape[-1]))
    token_topk_hidden = token_topk_hidden * weights.reshape(num_tokens, topk)[..., None]
    token_topk_hidden = jnp.where(
        mask.reshape(num_tokens, topk)[..., None], token_topk_hidden, 0
    )
    return token_topk_hidden.sum(axis=1)


def _onehot(g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk):
    del token_indices
    # Rows past the computed prefix (non-local experts under EP, skipped
    # padded tokens) are never written by the GMM; zero them so stale NaNs
    # cannot spread through the matmul (0 * NaN == NaN).
    computed = jnp.arange(g.shape[0], dtype=jnp.int32) < group_sizes.sum(
        dtype=jnp.int32
    )
    g = jnp.where(computed[:, None], g, 0)
    onehot = jax.nn.one_hot(
        revert.reshape(num_tokens, topk), revert.size, dtype=g.dtype
    )
    weights = (weights * mask).reshape(num_tokens, topk)
    return (onehot * weights[..., None]).sum(axis=1) @ g


def _owner_onehot(
    g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk
):
    del mask, topk  # The kernel reads the valid count and sorted weights.
    # Scattering through the inverse of the sort is the gather by the
    # (unmaterialised) forward one.
    weights_sorted = jnp.zeros_like(weights).at[revert].set(weights)
    valid_count = group_sizes.sum(dtype=jnp.int32)[None]
    return blockwise_onehot_unpermute(
        g, token_indices, weights_sorted, valid_count, num_tokens=num_tokens
    )


def _segment_sum(
    g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk
):
    del topk
    # Weight each routed row in place, through the inverse of the sort, and
    # zero the rows past the computed prefix, which the GMM never writes.
    weights_sorted = jnp.zeros_like(weights).at[revert].set(weights * mask)
    computed = jnp.arange(g.shape[0], dtype=jnp.int32) < group_sizes.sum(
        dtype=jnp.int32
    )
    rows = jnp.where(computed[:, None], g * weights_sorted[:, None], 0)
    return jax.ops.segment_sum(rows, token_indices, num_segments=num_tokens)


def _segment_sum_supports(
    g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk, chip
):
    del group_sizes, weights, mask, token_indices, num_tokens, topk, chip
    if g.shape[0] != revert.shape[0]:
        return "needs one routed row per route"
    return None


def _gather_reduce(kernel: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    def run(g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk):
        del group_sizes, token_indices, num_tokens
        return kernel(g, revert, weights, mask, reduce_group_size=topk)

    return run


def _owner_onehot_supports(
    g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk, chip
):
    del group_sizes, revert, mask, topk
    if chip is None:
        return "needs a TPU"
    valid_count = jax.ShapeDtypeStruct((1,), jnp.int32)
    return blockwise_onehot_unpermute_supports(
        g, token_indices, weights, valid_count, num_tokens=num_tokens, chip=chip
    )


def _gather_reduce_supports(
    kernel_supports: Callable[..., str | None],
) -> Callable[..., str | None]:
    def supports(
        g, group_sizes, revert, weights, mask, token_indices, *, num_tokens, topk, chip
    ):
        del group_sizes, token_indices, num_tokens
        return kernel_supports(g, revert, weights, mask, topk, chip=chip)

    return supports


def _no_limits(*args, chip, **statics) -> str | None:
    del args, chip, statics
    return None


def _skip_empty(
    g,
    group_sizes,
    revert,
    weights,
    mask,
    token_indices,
    *,
    num_tokens,
    topk,
    token_local_routes,
):
    del group_sizes, token_indices, num_tokens
    return skip.ragged_gather_reduce(g, revert, weights, mask, token_local_routes, topk)


def _skip_empty_supports(
    g,
    group_sizes,
    revert,
    weights,
    mask,
    token_indices,
    *,
    num_tokens,
    topk,
    chip,
    token_local_routes,
):
    del group_sizes, token_indices, num_tokens
    return skip.ragged_gather_reduce_supports(
        g, revert, weights, mask, token_local_routes, topk, chip=chip
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Variant:
    """One way to run the combine.

    Attributes:
        run: Computes the combine.
        supports: A hard limit the call breaks, or None.
        sparse_core: Runs on the SparseCore.
        reads_token_local_routes: Also takes routing's per-token count of
            local routes, as the `token_local_routes` keyword.
    """

    run: Callable[..., jax.Array]
    supports: Callable[..., str | None]
    sparse_core: bool = False
    reads_token_local_routes: bool = False


VARIANTS: dict[Combine, Variant] = {
    Combine.TENSORCORE: Variant(run=_tensorcore, supports=_no_limits),
    Combine.ONEHOT: Variant(run=_onehot, supports=_no_limits),
    Combine.OWNER_ONEHOT: Variant(run=_owner_onehot, supports=_owner_onehot_supports),
    Combine.GATHER_REDUCE_V1: Variant(
        run=_gather_reduce(ragged_gather_reduce_v1),
        supports=_no_limits,
        sparse_core=True,
    ),
    Combine.GATHER_REDUCE_V2: Variant(
        run=_gather_reduce(ragged_gather_reduce_v2),
        supports=_no_limits,
        sparse_core=True,
    ),
    Combine.GATHER_REDUCE_V3: Variant(
        run=_gather_reduce(v3.ragged_gather_reduce),
        supports=_gather_reduce_supports(v3.ragged_gather_reduce_supports),
        sparse_core=True,
    ),
    Combine.GATHER_REDUCE_V4: Variant(
        run=_gather_reduce(v4.ragged_gather_reduce),
        supports=_gather_reduce_supports(v4.ragged_gather_reduce_supports),
        sparse_core=True,
    ),
    Combine.SEGMENT_SUM: Variant(run=_segment_sum, supports=_segment_sum_supports),
    Combine.SKIP_EMPTY: Variant(
        run=_skip_empty,
        supports=_skip_empty_supports,
        sparse_core=True,
        reads_token_local_routes=True,
    ),
}
DEFAULT = Combine.TENSORCORE


def combine_capable(
    process: MoeProcess, op: MoeOpIn, out: GmmOut, chip: pltpu.TpuInfo | None
) -> dict[Combine, str | None]:
    """Why each variant cannot run this call; None for those that can."""
    args = combine_args(op, out)
    return {
        name: "SparseCore is off"
        if v.sparse_core and not process.use_sparse_core
        else _optional_inputs(v, out)(
            *args, num_tokens=op.num_tokens, topk=op.topk, chip=chip
        )
        for name, v in VARIANTS.items()
    }


def _optional_inputs(variant: Variant, out: GmmOut) -> Callable[..., str | None]:
    """`variant.supports`, with the optional inputs it reads bound as shapes."""
    if not variant.reads_token_local_routes:
        return variant.supports
    shape = out.token_local_routes_shape
    routes = None if shape is None else jax.ShapeDtypeStruct(shape, jnp.int32)
    return functools.partial(variant.supports, token_local_routes=routes)


class Pick(enum.Enum):
    """Which rule picked the combine; the value is its name in the output."""

    OVERRIDE = "override"
    DEFAULT = "default"


class OverrideCannotRunError(ValueError):
    """The override names a variant that cannot run the call.

    Attributes:
        combine: The variant the override names.
        reason: The hard limit the call breaks, from the variant's `supports`.
    """

    def __init__(self, combine: Combine, reason: str) -> None:
        """Names the variant and the limit in the message."""
        super().__init__(f"override {combine.value} cannot run this call: {reason}")
        self.combine = combine
        self.reason = reason


def pick_combine(
    process: MoeProcess, op: MoeOpIn, out: GmmOut, chip: pltpu.TpuInfo | None
) -> tuple[MoeCombinePlan, Pick]:
    """Picks the combine for one call, and says why.

    The pick is the variant the process setting names, else the TensorCore
    combine. Production would need per-bucket choices, from measurement.

    Args:
        process: Process settings.
        op: The call's shapes.
        out: The second GMM's output.
        chip: The chip, or None on a host without a TPU.

    Returns:
        The plan, and the rule that picked it.

    Raises:
        OverrideCannotRunError: If the override names a variant that cannot
            run the call.
    """
    if process.override is None:
        return MoeCombinePlan(combine=DEFAULT), Pick.DEFAULT
    name = process.override.combine
    reason = combine_capable(process, op, out, chip)[name]
    if reason is not None:
        raise OverrideCannotRunError(name, reason)
    return process.override, Pick.OVERRIDE


def plan_combine(
    process: MoeProcess, op: MoeOpIn, out: GmmOut, chip: pltpu.TpuInfo | None
) -> MoeCombinePlan:
    """The combine `pick_combine` picks for one call."""
    return pick_combine(process, op, out, chip)[0]
