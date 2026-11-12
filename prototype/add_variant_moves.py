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
"""Reports which calls a new combine variant runs, and which it fails on, on CPU.

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.add_variant_moves

Picks the combine for a grid of calls on v6e and TPU7x, once without a
setting and once with `TPU_MOE_COMBINE` naming the new variant, and counts
the calls it moves and the calls it raises on, with the kernel's reason.
"""

import collections
import itertools

from prototype.combine_plan import Chip, Raises, Runs, describe, outcome
from vllm_torchtpu.layers.core import moe_plan

NEW = moe_plan.Combine.GATHER_REDUCE_V4
TOKENS = (16, 64, 256, 512, 2048, 8192)
TOPK = (1, 2, 4, 8)


def _shown(item: tuple[tuple[Chip, Runs | Raises], int]) -> tuple[str, bool]:
    """Report order: by chip name, a chip's raises before its moves."""
    (chip, what), _ = item
    return chip.label, isinstance(what, Runs)


def main() -> None:
    """Prints how many calls the setting moves or fails, per chip."""
    default = moe_plan.MoeProcess(use_sparse_core=True)
    setting = moe_plan.MoeProcess(
        use_sparse_core=True, override=moe_plan.MoeCombinePlan(combine=NEW)
    )
    grid = list(itertools.product(Chip, TOKENS, TOPK))
    outcomes: collections.Counter[tuple[Chip, Runs | Raises]] = collections.Counter()
    unset = 0
    for chip, tokens, topk in grid:
        call = (*describe(tokens, topk), chip.info)
        unset += moe_plan.plan_combine(default, *call).combine is moe_plan.DEFAULT
        outcomes[chip, outcome(setting, *call)] += 1
    moved = sum(n for (_, what), n in outcomes.items() if isinstance(what, Runs))
    print(f"no setting: {moe_plan.DEFAULT.value} on {unset} of {len(grid)} calls")
    print(f"TPU_MOE_COMBINE={NEW.value}: applies to {moved} of {len(grid)} calls")
    for (chip, what), count in sorted(outcomes.items(), key=_shown):
        moves = isinstance(what, Runs)
        shown = f"{moe_plan.DEFAULT.value} -> {what}" if moves else what
        print(f"  {chip.label:6} {shown}: {count}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
