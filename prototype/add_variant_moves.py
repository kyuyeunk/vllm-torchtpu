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
The counting is `variant_moves` in `tests/layers/core/moe_combine_moves.py`,
which `test_moe_combine_moves.py` checks.
"""

from tests.layers.core.moe_combine_moves import Chip, Outcome, Runs, variant_moves
from vllm_torchtpu.layers.core import moe_plan

NEW = moe_plan.Combine.GATHER_REDUCE_V4


def _shown(item: tuple[tuple[Chip, Outcome], int]) -> tuple[str, bool]:
    """Report order: by chip name, a chip's raises before its moves."""
    (chip, what), _ = item
    return chip.label, isinstance(what, Runs)


def main() -> None:
    """Prints how many calls the setting moves or fails, per chip."""
    report = variant_moves(NEW)
    calls, unset = report.calls, report.unset
    print(f"no setting: {moe_plan.DEFAULT.value} on {unset} of {calls} calls")
    print(f"TPU_MOE_COMBINE={NEW.value}: applies to {report.moved} of {calls} calls")
    for (chip, what), count in sorted(report.outcomes.items(), key=_shown):
        moves = isinstance(what, Runs)
        shown = f"{moe_plan.DEFAULT.value} -> {what}" if moves else what
        print(f"  {chip.label:6} {shown}: {count}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
