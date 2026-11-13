"""KV transfer reads vLLM's resolved KV layout, not the env var."""

import pytest
from vllm.config import CacheConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.distributed.kv_transfer import tpu_connector as tc


def test_kv_transfer_raises_before_vllm_resolves_the_layout() -> None:
    """KV transfer asks vLLM's resolved layout; unresolved, it raises."""
    with pytest.raises(ValueError, match="not been resolved"):
        _ = tc._raiden_seq_on_lane_layout(CacheConfig(block_size=16))  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - the helper under test


@pytest.mark.parametrize(
    "layout", list(KVCacheLayout), ids=[layout.name for layout in KVCacheLayout]
)
def test_kv_transfer_reads_the_layout_vllm_resolved(layout: KVCacheLayout) -> None:
    """KV transfer reads vLLM's resolved layout, not the env var."""
    cc = CacheConfig(block_size=16)
    cc.kv_cache_layout = layout.name
    seq_on_lane = tc._raiden_seq_on_lane_layout(cc)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - the helper under test
    assert seq_on_lane is (layout is KVCacheLayout.LBHNC)
