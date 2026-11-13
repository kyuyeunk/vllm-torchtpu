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
"""Shows which MoE combine runs for a few calls, without a TPU.

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.combine_plan

The chips are JAX's own descriptions of v6e and TPU7x, which need no device.
For each call it prints the combine that runs, with its reason, and every
variant whose `supports` declines the call. Then it sets the override to a
variant that can run a call, and to one that cannot. The call descriptions
are `tests/layers/core/moe_combine_moves.py`'s.
"""

from tests.layers.core.moe_combine_moves import Chip, describe
from vllm_torchtpu.layers.core import moe_plan

# (tokens, top-k): the EP decode buckets of section 6.1 and a prefill bucket,
# with a 4096-wide BF16 GMM output and 64 local experts.
CALLS = [(16, 8), (512, 8), (2048, 4)]


def show(process: moe_plan.MoeProcess, chip: Chip, tokens: int, topk: int) -> None:
    """Prints the combine that runs one call, and the variants that cannot."""
    op, out = describe(tokens, topk)
    head = f"{chip.label} {tokens} tokens top-{topk}:"
    try:
        plan, why = moe_plan.pick_combine(process, op, out, chip.info)
    except ValueError as e:
        print(f"{head} raises ValueError: {e}")
        return
    print(f"{head} {plan.combine.value} ({why.value})")
    for name, reason in moe_plan.combine_capable(process, op, out, chip.info).items():
        if reason is not None:
            print(f"    {name.value:17} cannot: {reason}")


def main() -> None:
    """Prints every call on every chip, then one call under two overrides."""
    process = moe_plan.MoeProcess(use_sparse_core=True)
    for chip in Chip:
        for tokens, topk in CALLS:
            show(process, chip, tokens, topk)
    for name, chip in (
        (moe_plan.Combine.OWNER_ONEHOT, Chip.TPU7X),
        (moe_plan.Combine.GATHER_REDUCE_V3, Chip.V6E),
    ):
        override = moe_plan.MoeCombinePlan(combine=name)
        print(f"with TPU_MOE_COMBINE={name.value}:")
        show(
            moe_plan.MoeProcess(use_sparse_core=True, override=override),
            chip,
            16,
            8,
        )


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
