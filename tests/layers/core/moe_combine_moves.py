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
"""What each call of a grid of MoE calls runs under a combine setting.

The chips are JAX's own descriptions of v6e and TPU7x, which need no device.
Shared by `test_moe_combine_moves.py` and the demos in `prototype/`, which
print what these functions return. Nothing here names a variant by its enum
member, so `moe_parallel_changes.probe` can run it against older commits.
"""

import collections
import dataclasses
import enum
import itertools
from typing import override

from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.layers.core import moe_plan

TOKENS = (16, 64, 256, 512, 2048, 8192)
TOPK = (1, 2, 4, 8)
BF16 = jnp.dtype(jnp.bfloat16)  # pyright: ignore[reportAny]  # untyped in jax


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


type Call = tuple[Chip, int, int]


def grid() -> list[Call]:
    """Every chip, token count and top-k the reports plan: 48 calls."""
    return list(itertools.product(Chip, TOKENS, TOPK))


def describe(tokens: int, topk: int) -> tuple[moe_plan.MoeOpIn, moe_plan.GmmOut]:
    """The planner's descriptions of one call.

    Args:
        tokens: Tokens in the call.
        topk: Experts each token is routed to.

    Returns:
        The call's shapes, with 64 local experts, and a 4096-wide BF16 GMM
        output with BF16 weights.
    """
    op = moe_plan.MoeOpIn(num_tokens=tokens, topk=topk, local_num_experts=64)
    out = moe_plan.GmmOut(hidden=4096, dtype=BF16, weights_dtype=BF16)
    return op, out


@dataclasses.dataclass(frozen=True)
class Runs:
    """A call runs `combine`."""

    combine: moe_plan.Combine

    @override
    def __str__(self) -> str:
        """The variant's name."""
        return self.combine.value


@dataclasses.dataclass(frozen=True)
class Raises:
    """A call raises: the override names a variant that cannot run it."""

    reason: str

    @override
    def __str__(self) -> str:
        """The reason, as the reports print it."""
        return f"raises: {self.reason}"


type Outcome = Runs | Raises


def outcome(
    process: moe_plan.MoeProcess,
    op: moe_plan.MoeOpIn,
    out: moe_plan.GmmOut,
    chip: pltpu.TpuInfo | None,
) -> Outcome:
    """What one call runs under `process`, or why it raises.

    Args:
        process: The process settings, with or without an override.
        op: The call's shapes.
        out: The GMM output the combine reads.
        chip: The chip, or None on a host without a TPU.

    Returns:
        The variant that runs, or the reason the override cannot run.
    """
    try:
        return Runs(moe_plan.plan_combine(process, op, out, chip).combine)
    except moe_plan.OverrideCannotRunError as e:
        return Raises(e.reason)


def _with_override(combine: moe_plan.Combine) -> moe_plan.MoeProcess:
    return moe_plan.MoeProcess(
        use_sparse_core=True, override=moe_plan.MoeCombinePlan(combine=combine)
    )


@dataclasses.dataclass(frozen=True)
class VariantMoves:
    """What setting `TPU_MOE_COMBINE` to a new variant does to the grid.

    Attributes:
        calls: Calls in the grid.
        unset: Calls that run `moe_plan.DEFAULT` with no setting.
        outcomes: Per chip, the calls the setting runs or raises on.
    """

    calls: int
    unset: int
    outcomes: collections.Counter[tuple[Chip, Outcome]]

    @property
    def moved(self) -> int:
        """Calls the setting moves to the new variant."""
        return sum(
            n for (_, what), n in self.outcomes.items() if isinstance(what, Runs)
        )


def variant_moves(new: moe_plan.Combine) -> VariantMoves:
    """Plans the grid without a setting, and with the setting naming `new`.

    Args:
        new: The variant the setting names.

    Returns:
        The calls the default runs, and what each call does under the setting.
    """
    default = moe_plan.MoeProcess(use_sparse_core=True)
    setting = _with_override(new)
    calls = grid()
    outcomes: collections.Counter[tuple[Chip, Outcome]] = collections.Counter()
    unset = 0
    for chip, tokens, topk in calls:
        call = (*describe(tokens, topk), chip.info)
        unset += moe_plan.plan_combine(default, *call).combine is moe_plan.DEFAULT
        outcomes[chip, outcome(setting, *call)] += 1
    return VariantMoves(calls=len(calls), unset=unset, outcomes=outcomes)


@dataclasses.dataclass(frozen=True)
class Case:
    """One line of the report: the setting, and whether the counts are passed."""

    setting: moe_plan.Combine | None
    counted: bool

    @override
    def __str__(self) -> str:
        """The case, as the report prints it."""
        setting = (
            "no setting"
            if self.setting is None
            else f"TPU_MOE_COMBINE={self.setting.value}"
        )
        return f"{setting}, {'with' if self.counted else 'without'} the counts"


def boundary_input_moves(
    new: moe_plan.Combine,
) -> collections.Counter[tuple[Case, Outcome]]:
    """Plans the grid with and without routing's per-token route counts.

    Each call is planned without and with the counts, each both without a
    setting and with the setting naming `new`.

    Args:
        new: The variant that reads the counts.

    Returns:
        Per case, in the order first seen, the calls that run or raise.
    """
    default = moe_plan.MoeProcess(use_sparse_core=True)
    setting = _with_override(new)
    outcomes: collections.Counter[tuple[Case, Outcome]] = collections.Counter()
    for chip, tokens, topk in grid():
        op, out = describe(tokens, topk)
        counted = dataclasses.replace(out, token_local_routes_shape=(tokens,))
        for has_counts, spec in ((False, out), (True, counted)):
            for process, named in ((default, None), (setting, new)):
                what = outcome(process, op, spec, chip.info)
                outcomes[Case(named, has_counts), what] += 1
    return outcomes
