# Copyright 2025 Google LLC
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
"""The MoE combine's decision: which combine runs, decided in one place.

`moe_gmm` asks `plan_combine` and runs the `MoeCombinePlan` it gets back;
`moe_combine` runs that plan and decides nothing. The plan is resolved: it
names the EP kernel version instead of leaving it to the import-time setting.
"""

import dataclasses
import enum

import jax
from jax import numpy as jnp

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
    can_use_blockwise_onehot_unpermute,
)
from vllm_torchtpu.kernels.sparse_core import support
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
    _ROUTE_INDEX_BITS,
    below_sparse_core_cutoff,
    token_block_alignment,
)
from vllm_torchtpu.layers.core.resolution import require_no_default

# Comment in either line to plant a violation. The first reaches up from the
# family scope into the block scope's adapters; the second from the decision
# into the code that runs its plan.
# from vllm_torchtpu.layers.adapter import fused_moe
# from vllm_torchtpu.layers.core import fused_moe_gmm


class Combine(enum.Enum):
    """The combine that sums the second GMM's rows back per token."""

    LOCAL_V3 = enum.auto()
    EP_OWNER_ONEHOT = enum.auto()
    EP_ONEHOT = enum.auto()
    EP_RAGGED = enum.auto()
    TENSORCORE = enum.auto()


class GatherReduceVersion(enum.Enum):
    """The EP SparseCore kernel; values are `RAGGED_GATHER_REDUCE_VERSION`'s."""

    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class OwnerOutputMode(enum.Enum):
    """`TPU_MOE_OWNER_OUTPUT_MODE`, parsed."""

    OFF = "off"
    ON = "on"


@require_no_default
@dataclasses.dataclass(frozen=True, kw_only=True)
class MoeCombinePlan:
    """How the second GMM's rows are summed back per token.

    Attributes:
        combine: The combine that runs.
        ep_gather_reduce_version: The EP SparseCore kernel, for `ep_ragged`
            only; None for every other combine.
    """

    combine: Combine
    ep_gather_reduce_version: GatherReduceVersion | None

    def __post_init__(self) -> None:
        is_ep_ragged = self.combine is Combine.EP_RAGGED
        if (self.ep_gather_reduce_version is None) == is_ep_ragged:
            raise ValueError(f"the kernel version is for ep_ragged only: {self}")


def plan_combine(
    gmm2_res: jax.Array,
    argsort_revert_indices: jax.Array,
    topk_weights_flat: jax.Array,
    token_indices_sorted: jax.Array,
    *,
    num_tokens: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int,
) -> MoeCombinePlan:
    """Decides the combine from the shapes of its inputs and the chip."""
    # The destination-major kernel packs source indices into
    # `_ROUTE_INDEX_BITS` bits and processes 16-lane SparseCore tiles.
    use_local_ragged_combine = (
        # Local SparseCore path; route-pair metadata needs at least two routes.
        use_sparse_core
        and not use_ep
        and topk >= 2
        # TensorCore combine handles top-8 batches.
        and topk != 8
        # Honor the configured SparseCore dispatch cutoff.
        and argsort_revert_indices.size > onehot_moe_permute_threshold
        # BF16 kernel input.
        and gmm2_res.dtype == jnp.bfloat16
        # Packed source-index capacity.
        and gmm2_res.shape[0] <= 1 << _ROUTE_INDEX_BITS
    )
    if use_local_ragged_combine:
        tpu_info = support.current_tpu_info()
        sc_info = support.get_sparse_core_info(tpu_info)
        # Lane-aligned width, 16-lane SparseCore, and at or above v3's own
        # size cutoff.
        if (
            tpu_info is not None
            and sc_info is not None
            and sc_info.num_lanes == 16
            and gmm2_res.shape[-1] % tpu_info.num_lanes == 0
            and not below_sparse_core_cutoff(gmm2_res, tpu_info)
        ):
            # The kernel pads destination tokens to one 64-token block per
            # SparseCore row partition: require at least one full block and
            # padding of at most half the batch.
            alignment = token_block_alignment(gmm2_res.shape[-1], tpu_info)
            padded_tokens = -(-num_tokens // alignment) * alignment
            if num_tokens >= alignment and padded_tokens * 2 <= num_tokens * 3:
                return MoeCombinePlan(
                    combine=Combine.LOCAL_V3, ep_gather_reduce_version=None
                )
    if use_ep and use_sparse_core:
        if argsort_revert_indices.size <= onehot_moe_permute_threshold:
            # The admission reads only shapes and dtypes; the sorted weights
            # and the valid count have these.
            owner_output = OwnerOutputMode(envs.TPU_MOE_OWNER_OUTPUT_MODE.lower())
            if owner_output is OwnerOutputMode.ON and (
                can_use_blockwise_onehot_unpermute(
                    gmm2_res,
                    token_indices_sorted,
                    jax.ShapeDtypeStruct(
                        topk_weights_flat.shape, topk_weights_flat.dtype
                    ),
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                    num_tokens=num_tokens,
                )
            ):
                return MoeCombinePlan(
                    combine=Combine.EP_OWNER_ONEHOT, ep_gather_reduce_version=None
                )
            return MoeCombinePlan(
                combine=Combine.EP_ONEHOT, ep_gather_reduce_version=None
            )
        return MoeCombinePlan(
            combine=Combine.EP_RAGGED,
            # envs validates the choice; this parses it.
            ep_gather_reduce_version=GatherReduceVersion(
                envs.RAGGED_GATHER_REDUCE_VERSION
            ),
        )
    return MoeCombinePlan(combine=Combine.TENSORCORE, ep_gather_reduce_version=None)
