"""The KV pool's layout, read from `cache_config` only once it is final.

The pool's layout becomes final in stages across startup. Stage 1, the
physical KV order, is vLLM's own record: `CacheConfig.kv_cache_layout`,
resolved once in the engine core and read with
`cache_config.get_resolved_kv_cache_layout()`, which raises before then.
Stage 2, the page, is final in `TpuPlatform.update_block_size_for_backend`,
which the worker runs after `load_model`, so after the runner is constructed.

`cache_config` stays the only holder of the page's values. Its producer
marks the page final on `cache_config`, and readers call
`pool_layout(cache_config)` and read a field, which returns the
`cache_config` field once the page is final and raises
`PoolLayoutNotFinalError` before then instead of returning the value of an
earlier stage (#211: the runner copied `block_size` 16 at construction, the
worker raised it to 2048 afterwards).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import CacheConfig

# Attribute on `vllm.config.CacheConfig` that holds the stage marker. Only this
# module writes it; `tools/check_pool_layout_reads.py` checks that.
_ATTR = "_tpu_pool_stage"


class PoolStage(enum.IntEnum):
    """The point at which each part of the pool layout becomes final.

    Stage 1, the KV order, is recorded by vLLM (`CacheConfig.kv_cache_layout`)
    and is not tracked here.
    """

    UNSET = 0
    PAGE = 2  # TpuPlatform.update_block_size_for_backend, after load_model


_FINAL_AT = {
    PoolStage.PAGE: "TpuPlatform.update_block_size_for_backend (after load_model)",
}


class PoolLayoutNotFinalError(RuntimeError):
    """A pool layout field was read before the stage that decides it."""


@dataclasses.dataclass(frozen=True)
class PoolLayout:
    """The pool's layout on `cache_config`; each field raises until it is final.

    Holds no value of its own: each field reads `cache_config` behind the
    stage guard.
    """

    _cache_config: CacheConfig

    @property
    def stage(self) -> PoolStage:
        """The latest stage whose fields are final."""
        # `CacheConfig` has no field for the marker, so it rides as an extra
        # attribute; before stage 2 there is none. A mock config answers any
        # attribute; only a marker this module wrote counts.
        marker = getattr(  # str-ok: our marker on CacheConfig (no typed slot)
            self._cache_config, _ATTR, None
        )
        return marker if isinstance(marker, PoolStage) else PoolStage.UNSET

    @property
    def block_size(self) -> int:
        """Tokens per manager block (`cache_config.block_size`)."""
        self._require(PoolStage.PAGE, "block_size")
        return self._cache_config.block_size

    def _require(self, stage: PoolStage, field: str) -> None:
        if self.stage < stage:
            message = (
                f"PoolLayout.{field} is read before it is final: it becomes "
                f"final at stage {int(stage)} ({stage.name}, in "
                f"{_FINAL_AT[stage]}), and the pool is at stage "
                f"{int(self.stage)} ({self.stage.name}). Read it after that "
                "stage, or take the value as an argument; do not copy "
                "cache_config.block_size earlier."
            )
            raise PoolLayoutNotFinalError(message)


def pool_layout(cache_config: CacheConfig) -> PoolLayout:
    """The pool's layout on `cache_config`, guarded by its stage marker."""
    return PoolLayout(cache_config)


def mark_page_final(cache_config: CacheConfig) -> PoolLayout:
    """Stage 2 producer: mark the page final; called by the platform only.

    The page's values stay in `cache_config` (`block_size`,
    `mamba_block_size`); this records only that they are final.
    """
    object.__setattr__(cache_config, _ATTR, PoolStage.PAGE)
    return PoolLayout(cache_config)
