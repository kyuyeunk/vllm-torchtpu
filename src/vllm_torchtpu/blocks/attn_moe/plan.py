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
"""The attention-then-MoE block's plan: which collectives join the two halves.

Two plan values compute the same residual stream:

- ``all_reduce``: all-reduce attention's partial sum, run the norm and the
  router on every token, run the experts, all-reduce their output (upstream).
- ``reduce_scatter_all_gather`` (PR #1492): reduce-scatter
  ``attn_partial + residual / tp``, run the norm and the router on this rank's
  token shard, all-gather the normed tokens and the logits, and fold the
  attention partial sum into the MoE all-reduce.

The second needs its children to hand over unreduced outputs: attention's
output projection must not all-reduce, the MoE runner must not all-reduce and
must take the router's logits instead of recomputing them. Those are the child
contracts this plan resolves (see ``contract_adapter``).

The plan depends on the token count (collectives on a small batch cost more
than they hide), so it is resolved per compiled bucket into a frozen table.
Precedence follows the tree: the user's override (``TPU_BLOCK_PLAN``), then
the model plan (the model config's ``tpu_block_plan``), then the rule, then
the default. A pinned value that cannot run raises at startup; a value the
rule picks that cannot run falls back to the default with the reason.
"""

import dataclasses
from typing import Literal, get_args

from vllm_torchtpu.blocks.attn_moe.pattern import BlockKey
from vllm_torchtpu.layers.core.plan_marker import plan_of
from vllm_torchtpu.layers.core.resolution import require_no_default

Collectives = Literal["all_reduce", "reduce_scatter_all_gather"]
Source = Literal["user", "model plan", "rule", "default"]

# PR #1492's floor: below 8 MiB of activations per chip the extra collective
# launches cost more than the overlap hides.
MIN_PER_CHIP_BYTES = 8 * 1024 * 1024
_ELEMENT_SIZE = {"bfloat16": 2, "float16": 2, "float32": 4}


@plan_of("blocks.attn_then_moe")
@require_no_default
@dataclasses.dataclass(frozen=True)
class AttnMoEBlockPlan:
    """How one attention-then-MoE block joins its halves for one bucket."""

    collectives: Collectives


@dataclasses.dataclass(frozen=True)
class BlockPlanTable:
    """The plan per compiled bucket, with the source that decided each.

    Entries are in ascending bucket order. A warmup trace holds the token
    count as a dynamic size and splits on range guards only, so the lookup
    compares against the first bucket of each run of equal plans
    (``num_tokens >= 8192``), never against single buckets (``== 16`` would
    specialize the size). With one run it compares nothing.
    """

    entries: tuple[tuple[int, AttnMoEBlockPlan, Source], ...]

    def plan_for(self, num_tokens: int) -> AttnMoEBlockPlan:
        runs: list[tuple[int, AttnMoEBlockPlan]] = []
        for bucket, plan, _ in self.entries:
            if not runs or runs[-1][1] != plan:
                runs.append((bucket, plan))
        for start, plan in reversed(runs[1:]):
            if num_tokens >= start:
                return plan
        return runs[0][1] if runs else DEFAULT

    def uses(self, collectives: Collectives) -> bool:
        return any(p.collectives == collectives for _, p, _ in self.entries)


DEFAULT = AttnMoEBlockPlan("all_reduce")
OVERLAP = AttnMoEBlockPlan("reduce_scatter_all_gather")


def min_tokens(key: BlockKey) -> int:
    """Smallest bucket at which the rule picks the overlap."""
    return MIN_PER_CHIP_BYTES * key.tp // (key.hidden * _ELEMENT_SIZE[key.dtype])


def declines(plan: AttnMoEBlockPlan, key: BlockKey) -> list[str]:
    """Why ``plan`` cannot run for this block; empty when it can."""
    if plan.collectives == "all_reduce":
        return []
    reasons = []
    if key.tp <= 1:
        reasons.append("needs TP > 1")
    if key.sp_moe:
        reasons.append("sequence-parallel MoE reduces its own tokens")
    if key.fused_output_reduced:
        reasons.append("the fused EP MoE kernel returns a reduced output")
    if key.moe_path != "late_all_reduce":
        reasons.append(f"MoE path {key.moe_path} has no final all-reduce to fold into")
    return reasons


def _parse(text: str, source: str) -> Collectives | None:
    pins = dict(item.split("=", 1) for item in text.split(",") if item)
    if unknown := sorted(set(pins) - {"collectives"}):
        raise ValueError(f"{source}: unknown keys {unknown}")
    value = pins.get("collectives")
    if value is not None and value not in get_args(Collectives):
        raise ValueError(f"{source}: collectives={value} names no plan value")
    return value  # type: ignore[return-value]


def resolve_table(
    key: BlockKey,
    buckets: tuple[int, ...],
    override: str = "",
    model_plan: str = "",
) -> tuple[BlockPlanTable, tuple[str, ...]]:
    """Resolves the plan for every bucket, then cross-checks it.

    Args:
        key: The matched block.
        buckets: Compiled token counts.
        override: The user's ``TPU_BLOCK_PLAN``, "collectives=<value>" or "".
        model_plan: The model config's ``tpu_block_plan``, same format.

    Returns:
        The table, and the reasons the rule's choice fell back, if it did.

    Raises:
        ValueError: A pinned plan value cannot run for this block.
    """
    buckets = tuple(sorted(buckets))
    user = _parse(override, "TPU_BLOCK_PLAN")
    model = _parse(model_plan, "model plan")
    pinned, source = (user, "user") if user else (model, "model plan")
    if pinned is not None:
        plan = AttnMoEBlockPlan(pinned)
        if reasons := declines(plan, key):
            raise ValueError(
                f"block plan collectives={pinned} ({source}) cannot run on this "
                f"{key.kind} block: {'; '.join(reasons)}"
            )
        return BlockPlanTable(tuple((b, plan, source) for b in buckets)), ()
    floor = min_tokens(key)
    reasons = declines(OVERLAP, key)
    entries = []
    for bucket in buckets:
        if bucket >= floor and not reasons:
            entries.append((bucket, OVERLAP, "rule"))
        elif bucket >= floor:
            entries.append((bucket, DEFAULT, "default"))
        else:
            entries.append((bucket, DEFAULT, "rule"))
    fell_back = tuple(reasons) if any(b >= floor for b in buckets) else ()
    return BlockPlanTable(tuple(entries)), fell_back  # type: ignore[arg-type]
