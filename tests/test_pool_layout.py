"""The staged pool layout: reads before the page is final raise."""

import pytest
from vllm.config import CacheConfig

from vllm_torchtpu import pool_layout as pl


def test_block_size_read_before_the_page_is_final_raises() -> None:
    """A block size read before the page is final raises; no copy of 16."""
    cc = CacheConfig(block_size=16)
    # cache_config.block_size says 16, but the page is not final: no copy.
    with pytest.raises(pl.PoolLayoutNotFinalError, match=r"block_size.*stage 2"):
        _ = pl.pool_layout(cc).block_size
    cc.block_size = 2048
    _ = pl.mark_page_final(cc)
    assert pl.pool_layout(cc).block_size == 2048  # noqa: PLR2004 - the #211 size


def test_the_record_holds_no_copy() -> None:
    """The record reads cache_config.block_size; it keeps no value."""
    cc = CacheConfig(block_size=16)
    _ = pl.mark_page_final(cc)
    _ = pl.mark_page_final(cc)  # idempotent
    cc.block_size = 4096
    assert pl.pool_layout(cc).block_size == 4096  # noqa: PLR2004 - the size just set
