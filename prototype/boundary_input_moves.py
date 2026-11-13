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
The counting is `boundary_input_moves` in
`tests/layers/core/moe_combine_moves.py`, which `test_moe_combine_moves.py`
checks.
"""

from tests.layers.core.moe_combine_moves import Runs, boundary_input_moves, grid
from vllm_torchtpu.layers.core import moe_plan

NEW = moe_plan.Combine.SKIP_EMPTY


def main() -> None:
    """Prints, with and without the counts, what each call runs or raises."""
    outcomes = boundary_input_moves(NEW)
    print(f"{len(grid())} calls on v6e and TPU7x")
    for (case, what), count in outcomes.items():
        runs_named = case.setting is not None and isinstance(what, Runs)
        print(f"  {case}: {f'runs {what}' if runs_named else what}: {count}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
