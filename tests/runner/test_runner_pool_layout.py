"""The runner's block size reads the staged pool layout."""

import pytest
from vllm.config import CacheConfig

from vllm_torchtpu import pool_layout as pl
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_runner_block_size_reads_the_record() -> None:
    """The runner's block size raises until the page is final."""
    runner = object.__new__(TPUModelRunner)
    runner.cache_config = CacheConfig(block_size=16)
    with pytest.raises(pl.PoolLayoutNotFinalError):
        _ = runner.block_size
    runner.cache_config.block_size = 2048
    _ = pl.mark_page_final(runner.cache_config)
    assert runner.block_size == 2048  # noqa: PLR2004 - the #211 size
