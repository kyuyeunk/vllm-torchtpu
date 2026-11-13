"""The staged pool layout: reads before the page is final raise."""

import ast
import collections
import pathlib

import pytest
from vllm.config import CacheConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu import pool_layout as pl

SRC = pathlib.Path(pl.__file__).parent


def _cache_config() -> CacheConfig:
    return CacheConfig(block_size=16)


def test_block_size_read_before_the_page_is_final_raises() -> None:
    """A block size read before the page is final raises; no copy of 16."""
    cc = _cache_config()
    # cache_config.block_size says 16, but the page is not final: no copy.
    with pytest.raises(pl.PoolLayoutNotFinalError, match="block_size.*stage 2"):
        _ = pl.pool_layout(cc).block_size
    cc.block_size = 2048
    pl.mark_page_final(cc)
    assert pl.pool_layout(cc).block_size == 2048


def test_the_record_holds_no_copy() -> None:
    """The record reads cache_config.block_size; it keeps no value."""
    cc = _cache_config()
    pl.mark_page_final(cc)
    pl.mark_page_final(cc)  # idempotent
    cc.block_size = 4096
    assert pl.pool_layout(cc).block_size == 4096


def test_runner_block_size_reads_the_record() -> None:
    """The runner's block size raises until the page is final."""
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    runner = object.__new__(TPUModelRunner)
    runner.cache_config = _cache_config()
    with pytest.raises(pl.PoolLayoutNotFinalError):
        _ = runner.block_size
    runner.cache_config.block_size = 2048
    pl.mark_page_final(runner.cache_config)
    assert runner.block_size == 2048


def test_kv_transfer_raises_before_vllm_resolves_the_layout() -> None:
    """KV transfer asks vLLM's resolved layout; unresolved, it raises."""
    from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc

    with pytest.raises(ValueError, match="not been resolved"):
        _ = tc._raiden_seq_on_lane_layout(_cache_config())  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - the helper under test


@pytest.mark.parametrize(
    "layout", list(KVCacheLayout), ids=[layout.name for layout in KVCacheLayout]
)
def test_kv_transfer_reads_the_layout_vllm_resolved(layout: KVCacheLayout) -> None:
    """KV transfer reads vLLM's resolved layout, not the env var."""
    from vllm_torchtpu.distributed.kv_transfer import (  # noqa: PLC0415 - only the KV transfer tests load the connector
        tpu_connector as tc,
    )

    cc = _cache_config()
    cc.kv_cache_layout = layout.name
    seq_on_lane = tc._raiden_seq_on_lane_layout(cc)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - the helper under test
    assert seq_on_lane is (layout is KVCacheLayout.LBHNC)


# Ratchet: direct reads of pool-layout facts outside the record. Each entry is
# (file, function) -> count. A new direct read fails; moving one onto
# `pool_layout()` fails until its entry is removed here, so the list shrinks.
_PRODUCERS = {
    "pool_layout.py",
    "platforms/tpu_block_size_utils.py",
    "platforms/tpu_platform.py",
}
_BASELINE = {
    ("__init__.py", "_dcp_fine_prefix_match_unit"): 1,
    ("__init__.py", "_patch_vllm_dcp_fine_prefix_hits.resolve_kv_cache_block_sizes"): 1,
    (
        "__init__.py",
        "_patch_vllm_mamba_split_scheduler_block_size.split_with_scheduler_block_size",
    ): 1,
    ("core/experimental/srpt_prefill_scheduler.py", "SrptPrefillScheduler.schedule"): 1,
    ("distributed/kv_transfer/tpu_connector.py", "TPUConnectorScheduler.__init__"): 1,
    (
        "distributed/kv_transfer/tpu_connector.py",
        "TPURaidenConnectorWorker._admit_raiden_glm_kv_cache",
    ): 1,
    (
        "distributed/kv_transfer/tpu_connector.py",
        "TPURaidenConnectorWorker._max_request_blocks",
    ): 1,
    (
        "distributed/kv_transfer/tpu_connector.py",
        "TPURaidenConnectorWorker._register_stage3_request_blocks",
    ): 1,
    (
        "distributed/kv_transfer/tpu_connector.py",
        "TPURaidenConnectorWorker._register_stage3_request_row_spans",
    ): 1,
    (
        "distributed/kv_transfer/tpu_connector.py",
        "TPURaidenConnectorWorker._stage3_fa_skip_bytes",
    ): 2,
    (
        "distributed/kv_transfer/tpu_connector.py",
        "TPURaidenConnectorWorker._validate_glm_dcp_destination",
    ): 1,
    (
        "distributed/kv_transfer/zmq_shm_base.py",
        "ZmqShmKvConnectorBase._build_pool_spec",
    ): 1,
    (
        "distributed/kv_transfer/zmq_shm_base.py",
        "ZmqShmKvConnectorBase._warmup_block_sizes",
    ): 1,
    (
        "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py",
        "VllmDeepseekV4SWACache.__init__",
    ): 1,
    (
        "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py",
        "VllmDeepseekV4SWACache.get_kv_cache_spec",
    ): 1,
    (
        "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_compressor.py",
        "VllmCompressorStateCache.get_kv_cache_spec",
    ): 1,
    (
        "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_indexer.py",
        "VllmDeepseekV4IndexerCache.get_kv_cache_spec",
    ): 1,
    (
        "layers/adapter/custom_ops/kda_attention_op.py",
        "build_kimi_pooled_kda_op.pooled_core",
    ): 1,
    ("layers/core/gdn_attention.py", "_is_seq_along_lane_pool"): 1,
    ("megakernel/adapter/plugin.py", "RunnerView.cache_block_size"): 1,
    (
        "models/vllm/deepseek_v4/attention.py",
        "VllmDeepseekV4MLAAttention._build_attn_op",
    ): 1,
    (
        "models/vllm/deepseek_v4/attention.py",
        "VllmDeepseekV4MLAAttention.get_kv_cache_spec",
    ): 1,
    ("runner/kv_cache_manager.py", "KVCacheManager._update_mamba_page_size_padded"): 1,
    ("runner/kv_cache_manager.py", "KVCacheManager.get_kv_cache_spec"): 1,
    # The placeholder that sizes the runner's placeholder block table.
    ("runner/tpu_runner.py", "TPUModelRunner.__init__"): 1,
    ("runner/tpu_runner.py", "TPUModelRunner._add_pipeline_chunk_buckets"): 2,
    ("runner/tpu_runner.py", "TPUModelRunner._attention_kernel_shapes"): 1,
    ("runner/tpu_runner.py", "TPUModelRunner.pp_handoff_hbm_reserve_bytes"): 1,
}
_FACTS = {"block_size", "mamba_block_size", "mamba_page_size_padded"}


def _chain(node: ast.expr) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _direct_reads() -> collections.Counter:
    found: collections.Counter = collections.Counter()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in _PRODUCERS:
            continue
        stack: list[str] = []

        def visit(node: ast.AST, rel: str = rel, stack: list[str] = stack) -> None:
            named = isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            if named:
                stack.append(node.name)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                chain = _chain(node)
                on_cache_config = (
                    node.attr in _FACTS
                    and len(chain) >= 2
                    and chain[-2].endswith("cache_config")
                )
                on_env = (
                    node.attr == "VLLM_KV_CACHE_LAYOUT"  # str-ok: scans source names
                )
                if on_cache_config or on_env:
                    found[rel, ".".join(stack) or "<module>"] += 1
            for child in ast.iter_child_nodes(node):
                visit(child)
            if named:
                stack.pop()

        visit(ast.parse(path.read_text()))
    return found


def test_direct_pool_layout_reads_only_shrink():
    found = _direct_reads()
    new = {k: v for k, v in found.items() if v > _BASELINE.get(k, 0)}
    gone = {k: v for k, v in _BASELINE.items() if found.get(k, 0) < v}
    assert not new, f"read these through pool_layout(), not cache_config: {new}"
    assert not gone, f"moved onto the record; drop from _BASELINE: {gone}"


def test_only_pool_layout_writes_the_record():
    writers = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if pl._ATTR in path.read_text()  # noqa: SLF001 (the test owns the check)
    ]
    assert writers == ["pool_layout.py"]
