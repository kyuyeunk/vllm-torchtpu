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
variant that can run a call, and to one that cannot.
"""

import dataclasses
import enum

from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.layers.core import moe_plan


class Chip(enum.Enum):
    """The chips the prototype plans for."""

    V6E = pltpu.ChipVersion.TPU_V6E
    TPU7X = pltpu.ChipVersion.TPU_7X

    @property
    def label(self) -> str:
        """The chip's name in the output."""
        return self.name.lower()

    @property
    def info(self) -> pltpu.TpuInfo:
        """JAX's description of one chip, which needs no device."""
        return pltpu.get_tpu_info_for_chip(self.value, 1)


# (tokens, top-k): the EP decode buckets of section 6.1 and a prefill bucket,
# with a 4096-wide BF16 GMM output and 64 local experts.
CALLS = [(16, 8), (512, 8), (2048, 4)]
BF16 = jnp.dtype(jnp.bfloat16)


def describe(tokens: int, topk: int) -> tuple[moe_plan.MoeOpIn, moe_plan.GmmOut]:
    """The planner's descriptions of one call."""
    op = moe_plan.MoeOpIn(num_tokens=tokens, topk=topk, local_num_experts=64)
    out = moe_plan.GmmOut(hidden=4096, dtype=BF16, weights_dtype=BF16)
    return op, out


@dataclasses.dataclass(frozen=True)
class Runs:
    """A call runs `combine`."""

    combine: moe_plan.Combine

    def __str__(self) -> str:
        """The variant's name."""
        return self.combine.value


@dataclasses.dataclass(frozen=True)
class Raises:
    """A call raises: the override names a variant that cannot run it."""

    reason: str

    def __str__(self) -> str:
        """The reason, as the reports print it."""
        return f"raises: {self.reason}"


def outcome(
    process: moe_plan.MoeProcess,
    op: moe_plan.MoeOpIn,
    out: moe_plan.GmmOut,
    chip: pltpu.TpuInfo | None,
) -> Runs | Raises:
    """What one call runs under `process`, or why it raises."""
    try:
        return Runs(moe_plan.plan_combine(process, op, out, chip).combine)
    except moe_plan.OverrideCannotRunError as e:
        return Raises(e.reason)


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
