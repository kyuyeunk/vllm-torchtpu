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
"""Reports which calls a new combine input lets a variant run, on CPU.

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.boundary_input_moves

Picks the combine for a grid of calls on v6e and TPU7x, once as they are
and once with routing's per-token route counts passed to the combine, both
without a setting and with `TPU_MOE_COMBINE` naming the variant that reads
the counts, and counts the calls it moves and the calls it raises on.
"""

import collections
import dataclasses
import itertools

from prototype.combine_plan import Chip, Raises, Runs, describe, outcome
from vllm_torchtpu.layers.core import moe_plan

NEW = moe_plan.Combine.SKIP_EMPTY
TOKENS = (16, 64, 256, 512, 2048, 8192)
TOPK = (1, 2, 4, 8)


@dataclasses.dataclass(frozen=True)
class Case:
    """One line of the report: the setting, and whether the counts are passed."""

    setting: moe_plan.Combine | None
    counted: bool

    def __str__(self) -> str:
        """The case, as the report prints it."""
        setting = (
            "no setting"
            if self.setting is None
            else f"TPU_MOE_COMBINE={self.setting.value}"
        )
        return f"{setting}, {'with' if self.counted else 'without'} the counts"


def main() -> None:
    """Prints, with and without the counts, what each call runs or raises."""
    default = moe_plan.MoeProcess(use_sparse_core=True)
    setting = moe_plan.MoeProcess(
        use_sparse_core=True, override=moe_plan.MoeCombinePlan(combine=NEW)
    )
    grid = list(itertools.product(Chip, TOKENS, TOPK))
    outcomes: collections.Counter[tuple[Case, Runs | Raises]] = collections.Counter()
    for chip, tokens, topk in grid:
        op, out = describe(tokens, topk)
        counted = dataclasses.replace(out, token_local_routes_shape=(tokens,))
        for has_counts, spec in ((False, out), (True, counted)):
            for process, named in ((default, None), (setting, NEW)):
                what = outcome(process, op, spec, chip.info)
                outcomes[Case(named, has_counts), what] += 1
    print(f"{len(grid)} calls on v6e and TPU7x")
    for (case, what), count in outcomes.items():
        runs_named = case.setting is not None and isinstance(what, Runs)
        print(f"  {case}: {f'runs {what}' if runs_named else what}: {count}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
