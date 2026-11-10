r"""Walks one worker's startup and prints the pool layout as each stage is final.

Run from the repository root:

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.pool_layout

The producers are called in the order vLLM v0.33.0 runs them in one worker
(DATA_CONTRACTS.md Figure 25): the platform's config check, which resolves
the KV order onto vLLM's own record (`CacheConfig.kv_cache_layout`), the
runner's construction, then `update_block_size_for_backend` after
`load_model`, which marks the page final. Each reader line names a moved
reader's `file:line` and evaluates that line's expression against
`cache_config`. Importing `vllm.config` loads the TPU
platform plugin and libtpu, so take the TPU lock on a shared host.
"""

import pathlib
from types import SimpleNamespace

from vllm.config import CacheConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.pool_layout import (
    PoolLayoutNotFinalError,
    mark_page_final,
    pool_layout,
)

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm_torchtpu"

# (file, text on the line) of each producer and reader this script follows.
_PRODUCER_1 = ("platforms/tpu_platform.py", "layout = resolve_kv_cache_layout(")
_PRODUCER_2 = ("platforms/tpu_platform.py", "# The page is final here")
_READERS = (
    (
        "runner.block_size",
        "runner/tpu_runner.py",
        "return pool_layout(self.cache_config).block_size",
    ),
    (
        "GDN pool_block_tokens",
        "layers/adapter/custom_ops/gdn_attention_op.py",
        "pool_block_tokens=pool_layout",
    ),
    (
        "KV transfer page_tokens",
        "distributed/kv_transfer/tpu_connector.py",
        "page_tokens = pool.block_size",
    ),
    (
        "KV transfer HND check",
        "distributed/kv_transfer/tpu_connector.py",
        "return cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.LBHNC",
    ),
)


def _where(rel: str, text: str) -> str:
    """`file:line` of the first line in `rel` that contains `text`."""
    lines = (_SRC / rel).read_text().splitlines()
    line = next(i for i, s in enumerate(lines, 1) if text in s)
    return f"{rel}:{line}"


def _show(cache_config: CacheConfig) -> str:
    try:
        kv_order = cache_config.get_resolved_kv_cache_layout().name
    except ValueError:
        kv_order = "<not resolved>"
    layout = pool_layout(cache_config)
    try:
        block_size = str(layout.block_size)
    except PoolLayoutNotFinalError:
        block_size = "<not final>"
    return (
        f"kv_cache_layout={kv_order}  stage {int(layout.stage)} "
        f"{layout.stage.name}  block_size={block_size}"
    )


def main() -> None:
    """Print each stage, the runner's early read, and every reader."""
    # vLLM's default before the TPU platform picks a size.
    cache_config = CacheConfig(block_size=16)
    print(f"start      cache_config.block_size=16  {_show(cache_config)}")

    # check_and_update_config: vLLM's resolve_kv_cache_layout records the KV
    # order on cache_config (the plugin keeps no copy).
    cache_config.kv_cache_layout = KVCacheLayout.LBNHC.name
    print(f"stage 1    {_where(*_PRODUCER_1)} (vLLM's record)")
    print(f"           {_show(cache_config)}")

    runner = SimpleNamespace(cache_config=cache_config)
    try:
        _ = pool_layout(runner.cache_config).block_size
    except PoolLayoutNotFinalError as error:
        print("runner     __init__ reads the block size before load_model:")
        print(f"           raises {type(error).__name__}: {error}")

    # update_block_size_for_backend raises 16 to 2048 for this hybrid model
    # (the #211 numbers) and marks the page final.
    cache_config.block_size = 2048
    mark_page_final(cache_config)
    print(f"producer 2 {_where(*_PRODUCER_2)}")
    print(f"           {_show(cache_config)}")

    layout = pool_layout(cache_config)
    values = (
        layout.block_size,
        layout.block_size,
        layout.block_size,
        cache_config.get_resolved_kv_cache_layout() is KVCacheLayout.LBHNC,
    )
    print("readers, each reading cache_config:")
    for (name, rel, text), value in zip(_READERS, values, strict=True):
        print(f"  {name:24} {_where(rel, text):50} -> {value}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
