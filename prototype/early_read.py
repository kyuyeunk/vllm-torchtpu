r"""Reproduces #211: the runner reads the block size before it is final.

Run from the repository root:

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.early_read

In #211 the runner copied `cache_config.block_size` (16) in `__init__`; the
worker raised it to 2048 after `load_model`, and the mamba seed-copy
collector strode its block table with the stale 16 and indexed past the
table's end. With the record, the same early read raises at the read.

Planted toggle: comment in the second `STALE_COPY` line to restore the old
silent copy and see the #211 crash instead.
"""

from vllm.config import CacheConfig
from vllm.utils.math_utils import cdiv

from vllm_torchtpu.pool_layout import (
    PoolLayoutNotFinalError,
    mark_page_final,
    pool_layout,
)

STALE_COPY = False
# STALE_COPY = True  # noqa: ERA001 (the planted toggle)

MAX_MODEL_LEN = 32768
NUM_COMPUTED_TOKENS = 4096  # one request's progress at the seed copy


def runner_init(cache_config: CacheConfig) -> int:
    """Return the block size the runner keeps from construction (#211)."""
    if STALE_COPY:
        return cache_config.block_size  # main before this branch: a copy
    return pool_layout(cache_config).block_size  # this branch: raises


def seed_copy_column(block_size: int, table: list[int]) -> int:
    """Return the mamba block-table entry the seed-copy collector reads."""
    return table[NUM_COMPUTED_TOKENS // block_size]


def main() -> None:
    """Run the #211 order: config check, runner init, block-size update."""
    print(f"STALE_COPY={STALE_COPY}")
    cache_config = CacheConfig(block_size=16)
    runner_block_size: int | None = None
    try:
        runner_block_size = runner_init(cache_config)
        print(f"runner __init__ copies block_size={runner_block_size}")
    except PoolLayoutNotFinalError as error:
        print("runner __init__ reads the block size before the page is final:")
        print(f"  raises {type(error).__name__}: {error}")

    # update_block_size_for_backend, after load_model: 16 -> 2048.
    cache_config.block_size = 2048
    mark_page_final(cache_config)
    table = list(range(cdiv(MAX_MODEL_LEN, 2048)))  # sized with the final value
    print(f"worker finalizes block_size=2048; mamba table width {len(table)}")
    if runner_block_size is None:
        runner_block_size = pool_layout(cache_config).block_size
        print(f"runner reads it after stage 2: block_size={runner_block_size}")
    try:
        entry = seed_copy_column(runner_block_size, table)
    except IndexError as error:
        column = NUM_COMPUTED_TOKENS // runner_block_size
        print(
            f"seed copy strides by the stale {runner_block_size}: column "
            f"{column} of {len(table)} -> IndexError: {error} (the #211 crash)"
        )
        return
    column = NUM_COMPUTED_TOKENS // runner_block_size
    print(f"seed copy reads column {column} of {len(table)}: entry {entry}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
