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
"""DCP (Decode Context Parallelism) attention: kernel building and forward
orchestration, on top of batched_rpa's schedule_cp.CPMetadataComputer.
Process-group access lives in distributed/dcp.py.

Each DCP rank stores a shard of the KV cache (interleaved by page_size).
DCP ranks own *different* Q heads but share one GQA-replicated KV head.
"""

from __future__ import annotations

import functools

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as _batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as _batched_rpa_wrapper,
)
from vllm_torchtpu.utils import synchronize_tensors

# ---------------------------------------------------------------------------
# LSE-weighted combination
# ---------------------------------------------------------------------------


def merge_attn_results(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically stable LSE-weighted combination of two partial attentions.

    Args:
        out_a / out_b: [T, H, D]
        lse_a / lse_b: [T, H]  (log-sum-exp in natural-log base)
    Returns:
        (combined_out [T, H, D], combined_lse [T, H])
    """
    m = torch.maximum(lse_a, lse_b)
    m_finite = torch.isfinite(m)
    m_safe = torch.where(m_finite, m, torch.zeros_like(m))
    ea = torch.exp(lse_a - m_safe)
    eb = torch.exp(lse_b - m_safe)
    norm = ea + eb
    out_a = torch.nan_to_num(out_a, nan=0.0, posinf=0.0, neginf=0.0)
    out_b = torch.nan_to_num(out_b, nan=0.0, posinf=0.0, neginf=0.0)
    norm_safe = torch.where(norm > 0, norm, torch.ones_like(norm))
    out = (ea.unsqueeze(-1) * out_a + eb.unsqueeze(-1) * out_b) / norm_safe.unsqueeze(
        -1
    )
    lse_out = torch.where(m_finite, m + torch.log(norm_safe), m)
    return out, lse_out


# ---------------------------------------------------------------------------
# DCP kernel entry functions (for pallas.jax_op)
# ---------------------------------------------------------------------------


def _pallas_rpa_kernel_dcp(
    kv_cache: jax.Array,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    cp_rank: jax.Array,
    *,
    sliding_window: int | None,
    sm_scale: float | None = None,
    soft_cap: float | None = None,
    cp_group_size: int,
    attention_scope: _batched_rpa_configs.AttentionScope,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """One DCP pass (CACHE_ONLY or NEW_TOKENS_ONLY). Returns (new_kv_cache,
    output, lse). decode_query_size widens the DECODE bucket so speculative
    verify windows (K + 1 tokens) run there instead of in MIXED.
    cp_group_size/cp_rank select schedule_cp.CPMetadataComputer
    inside ragged_paged_attention, needed by both scopes: CACHE_ONLY to
    compute this rank's local cache length, NEW_TOKENS_ONLY to land the
    write on the right physical page of this rank's DCP-interleaved
    kv_cache (its own write offset is otherwise global, not rank-local).
    cp_rank is a [1] int32 operand that the schedule reads through scalar
    prefetch.
    """
    output, new_kv_cache, lse = _batched_rpa_wrapper.ragged_paged_attention(
        queries=query,
        keys=key,
        values=value,
        kv_cache=kv_cache,
        kv_lens=seq_lens,
        page_indices=block_tables,
        cu_q_lens=query_start_loc,
        distribution=request_distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank,
        attention_scope=attention_scope,
        decode_query_size=decode_query_size,
        return_lse=True,
        kv_layout=kv_layout,
    )
    return new_kv_cache, output, lse


# ---------------------------------------------------------------------------
# Kernel op builder
# ---------------------------------------------------------------------------


def _fake_dcp(kv_cache: torch.Tensor, query: torch.Tensor, *args, **kwargs):
    lse = torch.empty(
        query.shape[0], query.shape[1], dtype=query.dtype, device=query.device
    )
    return torch.empty_like(kv_cache), torch.empty_like(query), lse


# kv_cache (arg 0) is donated so XLA aliases the returned cache onto the input
# pool instead of materializing a full-sized copy. CACHE_ONLY returns it
# unchanged; NEW_TOKENS_ONLY writes the new tokens into it. The op is built at
# import, so no registration happens inside a traced forward.
_DCP_KERNEL_OP = pallas.jax_op(
    "pallas::rpa_dcp",
    _pallas_rpa_kernel_dcp,
    donate_argnums=(0,),
    mesh=None,
    input_partition_specs=None,
)
_DCP_KERNEL_OP.register_fake(_fake_dcp)


def _dcp_kernel_impl(cp_rank, kv_cache, *args, **static_kwargs):
    """Runs one DCP pass and writes the returned cache back into `kv_cache`.

    The copy keeps the caller's tensor bound to the possibly aliased result and
    compiles to a no-op for CACHE_ONLY.
    """
    new_kv_cache, output, lse = _DCP_KERNEL_OP(
        kv_cache, *args, cp_rank, **static_kwargs
    )
    kv_cache.copy_(new_kv_cache)
    return output, lse


def make_cp_rank_tensor(cp_rank: int, device: torch.device | str) -> torch.Tensor:
    """Returns `cp_rank` as the [1] int32 operand `build_dcp_kernels` expects.

    Create it once, before compilation: a tensor created inside a compiled
    forward becomes a per-rank constant in the graph.
    """
    return torch.tensor([cp_rank], dtype=torch.int32, device=device)


def build_dcp_kernels(
    sliding_window: int | None,
    sm_scale: float,
    logits_soft_cap: float | None,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    cp_group_size: int,
    cp_rank: torch.Tensor,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> tuple:
    """Binds the CACHE_ONLY and NEW_TOKENS_ONLY passes of the DCP op.

    `q_scale`, `k_scale` and `v_scale` are ignored; the returned kernels take
    them per call. `cp_rank` is a [1] int32 device tensor (see
    `make_cp_rank_tensor`) passed to the op as data rather than as a static
    argument: ranks that compile different programs under one op name can
    desynchronize the in-graph collectives and multiply compile time.

    Returns:
        (cache_kernel, new_kernel) where each is callable as
        ``kernel(kv_cache, query, key, value, seq_lens, block_tables,
                 query_start_loc, request_distribution, q_scale, k_scale,
                 v_scale) -> (output [T,H,D], lse [T,H])``
    """
    del q_scale, k_scale, v_scale
    # The adapter uses batched_rpa's equivalent enum. Normalize here so both
    # entry points pass LONGCTX's enum and hit the same compiled program.
    kv_layout = _batched_rpa_configs.KVLayout(kv_layout)
    static_kwargs = dict(
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=logits_soft_cap,
        cp_group_size=cp_group_size,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    cache_kernel = functools.partial(
        _dcp_kernel_impl,
        cp_rank,
        attention_scope=_batched_rpa_configs.AttentionScope.CACHE_ONLY,
        **static_kwargs,
    )
    new_kernel = functools.partial(
        _dcp_kernel_impl,
        cp_rank,
        attention_scope=_batched_rpa_configs.AttentionScope.NEW_TOKENS_ONLY,
        **static_kwargs,
    )
    return cache_kernel, new_kernel


# ---------------------------------------------------------------------------
# Top-level DCP forward
# ---------------------------------------------------------------------------


def forward_with_dcp(
    *,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    sliding_window: int | None,
    sm_scale: float,
    logits_soft_cap: float | None,
    kv_cache_quantized_dtype,
    dcp_world_size: int,
    dcp_rank: torch.Tensor,
    kv_layout: _batched_rpa_configs.KVLayout = (
        _batched_rpa_configs.KVLayout.HEAD_ALONG_SUBLANE
    ),
    decode_query_size: int = 1,
) -> torch.Tensor:
    """Two-pass DCP attention forward.
    1. CACHE_ONLY: query is all-gathered by heads to attend every rank's KV
        shard, then partials are merged by `cp_lse_ag_out_rs` back to own heads.
    2. NEW_TOKENS_ONLY: new tokens aren't DCP-sharded (every rank has the
        full local K/V for this step), so this rank's own heads attend
            directly; no gather/combine needed.

    """
    dcp_group = get_dcp_group()
    if dcp_group is None:
        raise RuntimeError("DCP group not initialized; cannot run DCP attention.")

    q_scale = None
    k_scale = layer._k_scale_float if kv_cache_quantized_dtype else None
    v_scale = layer._v_scale_float if kv_cache_quantized_dtype else None

    cache_kernel, new_kernel = build_dcp_kernels(
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        cp_group_size=dcp_world_size,
        cp_rank=dcp_rank,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )

    query_across_dcp = dcp_group.all_gather(query.contiguous(), dim=1)

    shared_args = (
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
        q_scale,
        k_scale,
        v_scale,
    )

    ctx_output, ctx_lse = cache_kernel(kv_cache, query_across_dcp, *shared_args)

    ctx_output, ctx_lse = cp_lse_ag_out_rs(
        ctx_output, ctx_lse, dcp_group, return_lse=True, cp_rank=dcp_rank
    )

    new_output, new_lse = new_kernel(kv_cache, query, *shared_args)

    output, _ = merge_attn_results(ctx_output, ctx_lse, new_output, new_lse)
    output = output.to(new_output.dtype)  # f32 ctx_lse promotes the merge

    if not torch.compiler.is_compiling():
        synchronize_tensors(kv_cache, wait=False)
    return output
