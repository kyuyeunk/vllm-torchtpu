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
"""Who owns the MoE data-parallel collectives.

vLLM brackets a MoE method's `apply_monolithic` with an all-gather and a
reduce-scatter unless the method's `supports_internal_mk` claims them. The
plugin's dispatch then has to issue them itself, or run a kernel that does
the exchange inside its own program. Today the answer to vLLM and the
dispatch's own choice are separate formulas per quantization path; this
module holds both, side by side, so they are written once and their
agreement (`stages_collectives` implies `claims_collectives`) is checked.

Every path's formula is preserved exactly as a per-path rule, including the
quirks (the unquantized-style paths never zero padded routing weights; the
W4AN-MXFP4 path always does and always skips them in the kernel). Changing a
rule is a behavior change, scheduled separately.

Imports no torch, JAX or TPU runtime. The `*_from_envs` builders are the only
env reads; they read on every call, as today's call sites do, so a test that
patches an env attribute still steers them.
"""

import dataclasses
import enum
import math
from collections.abc import Sequence

from vllm_torchtpu import envs


class CollectiveRule(enum.Enum):
    """The per-family collective formulas; each MoE method class names its own.

    Every rule stages the collectives when chunk pipelining is on.

    - `FP8_ALL_GATHER` (fp8): an fp8 all-gather also claims and stages them;
      padded routing weights are zeroed when skip-padded is on.
    - `FUSED_KERNEL` (nvfp4): a fused kernel (plain or adaptive) also claims.
    - `FUSED_EP_KERNEL` (DeepSeek-V4 MXFP4): only the plain fused EP kernel
      also claims.
    - `PIPELINING_ONLY` (unquantized, mxfp4, w4a16): nothing else claims; the
      kernel still reads skip-padded, and padded weights are never zeroed.
    - `ALWAYS_SKIP_PADDED` (w4an_mxfp4): as `PIPELINING_ONLY`, but always
      zeroes padded routing weights and skips them in the kernel.
    """

    FP8_ALL_GATHER = enum.auto()
    FUSED_KERNEL = enum.auto()
    FUSED_EP_KERNEL = enum.auto()
    PIPELINING_ONLY = enum.auto()
    ALWAYS_SKIP_PADDED = enum.auto()


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CollectiveOverrides:
    """Operator knobs, read once per decision by `collective_overrides_from_envs`.

    Attributes:
        chunk_pipelining: Chunked collective/compute pipelining is on
            (TPU_MOE_COLLECTION_CHUNK_SIZE > 0, via
            `enable_pipelined_collective_and_compute`).
        fp8_all_gather: TPU_MOE_FP8_ALL_GATHER.
        skip_padded_tokens: TPU_MOE_SKIP_PADDED_TOKENS.
    """

    chunk_pipelining: bool
    fp8_all_gather: bool
    skip_padded_tokens: bool


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MoeLayerInfo:
    """Per-layer facts the collective decision reads.

    Attributes:
        rule: The method class's `collective_rule`.
        ep_fused_supported: `fused_moe_ep_supported(method)`: prebuild armed
            the fused EP kernel for this layer.
        adaptive_supported: `adaptive_fused_moe_supported(method)`.
    """

    rule: CollectiveRule
    ep_fused_supported: bool
    adaptive_supported: bool


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CollectivePlan:
    """The plugin's MoE collective decisions for one layer.

    Attributes:
        claims_collectives: What `supports_internal_mk` tells vLLM: vLLM
            skips its own all-gather / reduce-scatter pair.
        stages_collectives: The dispatch runs `pipelined_fused_moe_gmm`,
            which issues the pair itself; routing tensors stay rank-local.
        quantize_all_gather: The staged all-gather sends fp8 (it still
            applies only when the collective group has more than one rank).
        zero_padded_routing_weights: The dispatch zeroes padded tokens'
            routing weights before the kernel.
        kernel_skip_padded_tokens: The GMM kernel drops zero-weight routes.
    """

    claims_collectives: bool
    stages_collectives: bool
    quantize_all_gather: bool
    zero_padded_routing_weights: bool
    kernel_skip_padded_tokens: bool

    def __post_init__(self) -> None:
        if self.stages_collectives and not self.claims_collectives:
            raise ValueError(
                f"the plugin would issue collectives vLLM also issues: {self}"
            )


def fused_kernel_armed(*, ep_fused_supported: bool, adaptive_supported: bool) -> bool:
    """Whether a fused EP kernel (plain or adaptive) does this layer's exchange.

    Prebuild arms the kernel per layer, so this is asked per quant method.

    Args:
        ep_fused_supported: `fused_moe_ep_supported(method)`.
        adaptive_supported: `adaptive_fused_moe_supported(method)`.

    Returns:
        True when an armed kernel owns dispatch and combine.
    """
    return ep_fused_supported or adaptive_supported


def collective_plan(
    layer: MoeLayerInfo, overrides: CollectiveOverrides
) -> CollectivePlan:
    """Today's per-path collective formulas, in one place.

    Every path stages the collectives itself when chunk pipelining is on, and
    an armed fused kernel returns before the GMM dispatch. Per rule:

    - `FP8_ALL_GATHER` (fp8): claims = pipelining or a fused kernel armed or
      fp8 AG; stages = pipelining or fp8 AG; zeroes padded routing weights on
      skip-padded.
    - `FUSED_KERNEL` (nvfp4): claims = pipelining or a fused kernel armed;
      stages = pipelining.
    - `FUSED_EP_KERNEL` (DeepSeek-V4 MXFP4): claims = pipelining or fused EP
      armed (no adaptive kernel); stages = pipelining.
    - `PIPELINING_ONLY` (unquantized, mxfp4, w4a16): claims = stages =
      pipelining (fp8 AG alone does not claim); never zero padded routing
      weights, while the kernel still reads skip-padded.
    - `ALWAYS_SKIP_PADDED` (w4an_mxfp4): claims = stages = pipelining; always
      zeroes padded routing weights and always skips them in the kernel,
      whatever the env says.

    Args:
        layer: Per-layer facts.
        overrides: Operator knobs.

    Returns:
        The plan.
    """
    pipelining = overrides.chunk_pipelining
    stages = pipelining
    zero_padded = False
    kernel_skip = overrides.skip_padded_tokens
    match layer.rule:
        case CollectiveRule.FP8_ALL_GATHER:
            claims = (
                pipelining
                or fused_kernel_armed(
                    ep_fused_supported=layer.ep_fused_supported,
                    adaptive_supported=layer.adaptive_supported,
                )
                or overrides.fp8_all_gather
            )
            stages = pipelining or overrides.fp8_all_gather
            zero_padded = overrides.skip_padded_tokens
        case CollectiveRule.FUSED_KERNEL:
            claims = pipelining or fused_kernel_armed(
                ep_fused_supported=layer.ep_fused_supported,
                adaptive_supported=layer.adaptive_supported,
            )
        case CollectiveRule.FUSED_EP_KERNEL:
            claims = pipelining or layer.ep_fused_supported
        case CollectiveRule.PIPELINING_ONLY:
            claims = pipelining
        case CollectiveRule.ALWAYS_SKIP_PADDED:
            claims = pipelining
            zero_padded = True
            kernel_skip = True
    return CollectivePlan(
        claims_collectives=claims,
        stages_collectives=stages,
        quantize_all_gather=stages and overrides.fp8_all_gather,
        zero_padded_routing_weights=zero_padded,
        kernel_skip_padded_tokens=kernel_skip,
    )


def chunk_size_from_envs() -> int:
    """TPU_MOE_COLLECTION_CHUNK_SIZE: max post-gather tokens per chunk."""
    return envs.TPU_MOE_COLLECTION_CHUNK_SIZE


def chunk_pipelining_from_envs() -> bool:
    """Chunked collective/compute pipelining is on (chunk size > 0)."""
    return chunk_size_from_envs() > 0


def disables_all2all_for_explicit_pcp(
    *, use_ep: bool, pcp_size: int, dp_size: int, is_sequence_parallel: bool
) -> bool:
    """Whether vLLM's all2all kernels give way to explicit PCP collectives.

    The `FusedMoEParallelConfig.use_all2all_kernels` patch in `__init__.py`:
    EP over PCP with no DP, no sequence parallelism and no chunk pipelining
    keeps the plugin's explicit PCP all-gather / reduce-scatter. The chunk
    size is read last, only when the other conditions hold, as before.

    Args:
        use_ep: Expert parallelism is on.
        pcp_size: Prefill context-parallel size.
        dp_size: Data-parallel size.
        is_sequence_parallel: MoE sequence parallelism is on.

    Returns:
        True to answer False for `use_all2all_kernels`.
    """
    return (
        use_ep
        and pcp_size > 1
        and dp_size == 1
        and not is_sequence_parallel
        and chunk_size_from_envs() <= 0
    )


def fp8_all_gather_from_envs() -> bool:
    """TPU_MOE_FP8_ALL_GATHER."""
    return envs.TPU_MOE_FP8_ALL_GATHER


def skip_padded_tokens_from_envs() -> bool:
    """TPU_MOE_SKIP_PADDED_TOKENS."""
    return envs.TPU_MOE_SKIP_PADDED_TOKENS


def collective_overrides_from_envs(*, chunk_pipelining: bool) -> CollectiveOverrides:
    """Builds the overrides from the environment.

    Args:
        chunk_pipelining: The caller's `enable_pipelined_collective_and_compute()`
            answer, passed in so tests that patch that name per module still
            steer the plan.

    Returns:
        The overrides.
    """
    return CollectiveOverrides(
        chunk_pipelining=chunk_pipelining,
        fp8_all_gather=fp8_all_gather_from_envs(),
        skip_padded_tokens=skip_padded_tokens_from_envs(),
    )


def needs_token_padding_state(
    *, skip_padded_tokens: bool, uses_mxfp4_moe: bool
) -> bool:
    """Whether the runner must create the token-padding mask.

    Args:
        skip_padded_tokens: TPU_MOE_SKIP_PADDED_TOKENS.
        uses_mxfp4_moe: Some layer runs the W4AN MXFP4 MoE method, which
            always reads the mask.

    Returns:
        True when the mask must exist.
    """
    return skip_padded_tokens or uses_mxfp4_moe


def fixed_chunk_count(
    buckets: Sequence[int], parallel_size: int, chunk_size: int
) -> int:
    """Chunk count that keeps every gather at or under ``chunk_size``.

    Every bucket must split into equal chunks, so the count is the smallest
    common divisor of the buckets that is at least what the largest bucket
    needs. When no common divisor is that large, the greatest one is used
    and the largest bucket's gather exceeds ``chunk_size``.
    """
    if chunk_size <= 0 or parallel_size <= 1 or not buckets:
        return 1
    needed = -(-max(buckets) * parallel_size // chunk_size)
    common = math.gcd(*buckets)
    count = needed
    while count < common and common % count:
        count += 1
    return min(count, common)


def calculate_moe_chunks(
    seq_len: int, parallel_size: int, chunk_size: int
) -> tuple[int, int]:
    """Calculate the number of chunks and local chunk size for MoE chunk pipelining.

    Uses ceiling division so `chunk_size` acts as a maximum global chunk threshold:
        num_chunks = ceil((seq_len * parallel_size) / chunk_size)
        chunk_size_local = seq_len // num_chunks

    Args:
        seq_len: Number of local tokens on this rank (S).
        parallel_size: World size of the active DP or PCP collective group.
        chunk_size: Maximum post-gather chunk size threshold in tokens (C).

    Returns:
        tuple[int, int]: (num_chunks N_chunk, chunk_size_local S_chunk).

    Raises:
        ValueError: If local sequence length is not divisible by chunk count.
    """
    if chunk_size <= 0 or parallel_size <= 1 or seq_len <= 0:
        return 1, seq_len

    total_tokens = seq_len * parallel_size
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    if seq_len % num_chunks != 0:
        raise ValueError(
            f"Local sequence length ({seq_len}) is not divisible by chunk count "
            f"({num_chunks})."
        )

    chunk_size_local = seq_len // num_chunks
    return num_chunks, chunk_size_local


def collection_chunks(
    seq_len: int,
    parallel_size: int,
    *,
    chunk_size: int,
    fixed_buckets: tuple[int, ...] | None,
) -> tuple[int, int]:
    """Chunk count and local chunk size for this rank's ``seq_len`` tokens.

    Args:
        seq_len: Local tokens on this rank.
        parallel_size: World size of the collective group.
        chunk_size: TPU_MOE_COLLECTION_CHUNK_SIZE.
        fixed_buckets: Token buckets a symbolically traced graph (the MTP
            draft) is padded to, or None to size from ``seq_len``.

    Returns:
        (num_chunks, chunk_size_local).
    """
    if fixed_buckets is not None and parallel_size > 1:
        num_chunks = fixed_chunk_count(fixed_buckets, parallel_size, chunk_size)
        return num_chunks, seq_len // num_chunks
    return calculate_moe_chunks(seq_len, parallel_size, chunk_size)
