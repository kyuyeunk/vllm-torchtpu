**Records group** (`prototype/records/`). Backs DATA_CONTRACTS.md section 5.4 (Figure 23) and Appendix A.4's #211. Siblings: `prototype/records/step-plan`.

# Prototype: the pool's layout as a staged record

Today the pool's block size is written into `cache_config` by the
platform and copied or re-read by each reader; nothing says when the value
is final. In #211 the runner copied `block_size` 16 in `__init__`, the worker
raised it to 2048 after `load_model`, and the mamba seed-copy collector
strode its table with the stale 16 and walked off it.

## Where to look

| Claim | Code |
|---|---|
| An early read raises; the guard holds no value | [`PoolStage, PoolLayout`](../src/vllm_torchtpu/pool_layout.py#L33-L90) in `pool_layout.py` (L33-L90) |
| The page's one producer marks it final | [`mark_page_final`](../src/vllm_torchtpu/pool_layout.py#L98-L105) in `pool_layout.py` (L98-L105) |
| Stage 2 is final here | [`mark_page_final call`](../src/vllm_torchtpu/platforms/tpu_platform.py#L1370-L1372) in `tpu_platform.py` (L1370-L1372) |
| The runner keeps no copy | [`block_size, _refresh_smem_request_caps`](../src/vllm_torchtpu/runner/tpu_runner.py#L8593-L8614) in `tpu_runner.py` (L8593-L8614) |
| KV transfer reads the guard | [`_measure_stage3_layout`](../src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py#L3492-L3535) in `tpu_connector.py` (L3492-L3535) |
| One source for "is it HND?": vLLM's resolved layout | [`_raiden_seq_on_lane_layout`](../src/vllm_torchtpu/distributed/kv_transfer/tpu_connector.py#L632-L640) in `tpu_connector.py` (L632-L640) |
| The ratchet (30) and the single writer | [`check_reads, check_writers`](../tools/check_pool_layout_reads.py#L264-L285) in `check_pool_layout_reads.py` (L264-L285) |
| #211, planted | [`STALE_COPY`](../prototype/early_read.py#L26-L27) in `early_read.py` (L26-L27) |

This branch keeps each fact in the record that already holds it and guards
reads that come before that record is final:

| Stage | Fact | Record | Final at |
|---|---|---|---|
| 1 | KV order | vLLM's `CacheConfig.kv_cache_layout`, read with `get_resolved_kv_cache_layout()` (raises until resolved) | `resolve_kv_cache_layout`, called by `TpuPlatform.check_and_update_config` |
| 2 `PAGE` | block size | `CacheConfig.block_size`, read with `pool_layout(cache_config).block_size` (raises `PoolLayoutNotFinalError` until the page is final) | `TpuPlatform.update_block_size_for_backend`, after `load_model`, which marks the page final (`mark_page_final`) |

`PoolLayout` (`src/vllm_torchtpu/pool_layout.py`) holds no value of its own:
the only thing written on `cache_config` is the stage marker, and a field
reads the `cache_config` field behind it. Stages 3 to 6 of Figure 25 are not
built. Three readers move onto the guard, and KV transfer onto vLLM's
layout record:

- **Runner:** `TPUModelRunner.block_size` is a property over the guard;
  `__init__` keeps no copy and passes its placeholder to the one refresh that
  runs before stage 2. The re-sync in `_refresh_smem_request_caps` is gone.
- **GDN:** both pooled ops read `pool_block_tokens` through the guard.
- **KV transfer:** `_measure_stage3_layout` reads `page_tokens` through the
  guard. All seven "is the pool seq-on-lane (HND)?" checks go through one
  helper, `_raiden_seq_on_lane_layout(cache_config)`, which asks
  `get_resolved_kv_cache_layout() is KVCacheLayout.LBHNC` instead of
  reading `VLLM_KV_CACHE_LAYOUT`.

The tests sit next to the code each one covers:
`tests/test_pool_layout.py` checks the guard,
`tests/runner/test_runner_pool_layout.py` the runner's read, and
`tests/distributed/test_tpu_connector_pool_layout.py` the KV transfer helper.
`tools/check_pool_layout_reads.py` is the static lint: only `pool_layout.py`
writes the marker, and a ratchet of the direct reads that remain (30 reads in
28 functions; a new one fails, a moved one must leave the list).
`tests/test_pool_layout_reads.py` runs it.

Assumes vLLM v0.33.0. Run from the repository root with the project's
virtualenv and vLLM v0.33.0 on `PYTHONPATH`. Importing `vllm.config` loads
the TPU platform plugin and libtpu, so on a shared TPU host take the TPU
lock. torch-tpu's start-up lines on stderr are left out below.

This branch is for reading, not merging. Existing tests that build a runner
or config without marking the page final or resolving the KV layout (in `tests/runner`,
`tests/platforms/test_tpu_platform.py`, `tests/layers/adapter/test_gdn_attention_op.py`,
`tests/distributed/test_tpu_connector.py`) fail rather than being rewritten.

## The stages as they become final, and each reader

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.pool_layout

```
start      cache_config.block_size=16  kv_cache_layout=<not resolved>  stage 0 UNSET  block_size=<not final>
stage 1    platforms/tpu_platform.py:910 (vLLM's record)
           kv_cache_layout=LBNHC  stage 0 UNSET  block_size=<not final>
runner     __init__ reads the block size before load_model:
           raises PoolLayoutNotFinalError: PoolLayout.block_size is read before it is final: it becomes final at stage 2 (PAGE, in TpuPlatform.update_block_size_for_backend (after load_model)), and the pool is at stage 0 (UNSET). Read it after that stage, or take the value as an argument; do not copy cache_config.block_size earlier.
producer 2 platforms/tpu_platform.py:1370
           kv_cache_layout=LBNHC  stage 2 PAGE  block_size=2048
readers, each reading cache_config:
  runner.block_size        runner/tpu_runner.py:8596                          -> 2048
  GDN pool_block_tokens    layers/adapter/custom_ops/gdn_attention_op.py:825  -> 2048
  KV transfer page_tokens  distributed/kv_transfer/tpu_connector.py:3500      -> 2048
  KV transfer HND check    distributed/kv_transfer/tpu_connector.py:640       -> False
```

The reader lines point at the moved code; each evaluates that line's
expression against `cache_config`. Stage 1 is vLLM's record, so the
guard's stage stays 0 until the page is final. The HND check is `False`
because vLLM resolved `LBNHC` (the legacy NHD).

## A #211-style early read now raises

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.early_read

```
STALE_COPY=False
runner __init__ reads the block size before the page is final:
  raises PoolLayoutNotFinalError: PoolLayout.block_size is read before it is final: it becomes final at stage 2 (PAGE, in TpuPlatform.update_block_size_for_backend (after load_model)), and the pool is at stage 0 (UNSET). Read it after that stage, or take the value as an argument; do not copy cache_config.block_size earlier.
worker finalizes block_size=2048; mamba table width 16
runner reads it after stage 2: block_size=2048
seed copy reads column 2 of 16: entry 2
```

Planted: comment in `STALE_COPY = True` in `prototype/early_read.py` to
restore main's silent copy. The early read succeeds with 16 and the crash
moves to the seed copy, after the value went stale:

```
STALE_COPY=True
runner __init__ copies block_size=16
worker finalizes block_size=2048; mamba table width 16
seed copy strides by the stale 16: column 256 of 16 -> IndexError: list index out of range (the #211 crash)
```

## Tests

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
        tests/test_pool_layout.py tests/runner/test_runner_pool_layout.py \
        tests/distributed/test_tpu_connector_pool_layout.py tests/test_pool_layout_reads.py

```
12 passed
```

The lint alone, which imports nothing (so needs no TPU lock):

    python tools/check_pool_layout_reads.py

```
...
reads: 30 in 28 functions
writers: ['pool_layout.py']
```
