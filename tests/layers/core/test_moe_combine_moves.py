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
"""What each kind of change to the combine moves, as prototype/README.md records.

The recorded values are the tables `python -m prototype.add_variant_moves`,
`prototype.boundary_input_moves` and `prototype.parallel_changes` print.
The parallel-change checks read git history: the last merge in HEAD's
history must be the merge of the two variant changes.
"""

import collections
import dataclasses

import pytest

from tests.layers.core.moe_combine_moves import (
    TOKENS,
    Case,
    Chip,
    Outcome,
    Raises,
    Runs,
    VariantMoves,
    boundary_input_moves,
    variant_moves,
)
from tests.layers.core.moe_parallel_changes import (
    Moves,
    ParallelChanges,
    Side,
    moves,
    parallel_changes,
)
from vllm_torchtpu.layers.core.moe_plan import Combine

_TOPK_ABOVE_4 = Raises("top-k above 4")
_NO_COUNTS = Raises("needs routing's per-token count of local routes")
_MOE_PLAN = "src/vllm_torchtpu/layers/core/moe_plan.py"
_TEST_MOE_PLAN = "tests/layers/core/test_moe_plan.py"
# The base's variants, in `VARIANTS` order.
_EXISTING = [
    Combine.TENSORCORE,
    Combine.ONEHOT,
    Combine.OWNER_ONEHOT,
    Combine.GATHER_REDUCE_V1,
    Combine.GATHER_REDUCE_V2,
    Combine.GATHER_REDUCE_V3,
]


def test_add_variant_moves() -> None:
    """Naming the new variant moves the 36 calls its kernel supports."""
    assert variant_moves(Combine.GATHER_REDUCE_V4) == VariantMoves(
        calls=48,
        unset=48,
        outcomes=collections.Counter(
            {
                (Chip.V6E, Runs(Combine.GATHER_REDUCE_V4)): 18,
                (Chip.V6E, _TOPK_ABOVE_4): 6,
                (Chip.TPU7X, Runs(Combine.GATHER_REDUCE_V4)): 18,
                (Chip.TPU7X, _TOPK_ABOVE_4): 6,
            }
        ),
    )


def test_boundary_input_moves() -> None:
    """The variant that reads the counts runs only where they are passed."""
    assert boundary_input_moves(Combine.SKIP_EMPTY) == collections.Counter(
        {
            (Case(None, counted=False), Runs(Combine.TENSORCORE)): 48,
            (Case(Combine.SKIP_EMPTY, counted=False), _NO_COUNTS): 48,
            (Case(None, counted=True), Runs(Combine.TENSORCORE)): 48,
            (Case(Combine.SKIP_EMPTY, counted=True), Runs(Combine.SKIP_EMPTY)): 48,
        }
    )


@pytest.fixture(scope="module")
def changes() -> ParallelChanges:
    """The two parallel changes and their merge, planned once per module."""
    return parallel_changes()


def test_parallel_changes_touched(changes: ParallelChanges) -> None:
    """A and B both touch only the table and its test."""
    assert changes.touched == {
        Side.A: {
            "prototype/add_variant_moves.py",
            "src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_v4.py",
            _MOE_PLAN,
            _TEST_MOE_PLAN,
        },
        Side.B: {_MOE_PLAN, _TEST_MOE_PLAN},
    }


def test_parallel_changes_conflict_in_one_table(changes: ParallelChanges) -> None:
    """The merge conflicts in the three places both appended an entry."""
    assert changes.conflicts == {
        _MOE_PLAN: [
            "- `gather_reduce_v4`: a SparseCore gather-reduce for top-k up to 4."
            "  |  "
            "- `segment_sum`: an XLA scatter-add of the weighted rows, in expert"
            " order.",
            'GATHER_REDUCE_V4 = "gather_reduce_v4"  |  SEGMENT_SUM = "segment_sum"',
            "Combine.GATHER_REDUCE_V4: Variant(  |  Combine.SEGMENT_SUM: "
            "Variant(run=_segment_sum, supports=_segment_sum_supports),",
        ]
    }


def test_parallel_changes_leave_existing_settings(changes: ParallelChanges) -> None:
    """No existing setting runs anything differently under A, B or A+B."""
    assert changes.default == [Runs(Combine.TENSORCORE)] * 48
    assert changes.existing == [None, *_EXISTING]
    assert changes.changed == []


@dataclasses.dataclass(frozen=True)
class _NewSetting:
    """One side's new variant: what it does, and that the merge keeps it."""

    side: Side
    setting: Combine
    moves: Moves


def _new_setting_id(case: _NewSetting) -> str:
    return f"{case.side.name}-{case.setting.value}"


@pytest.mark.parametrize(
    "case",
    [
        _NewSetting(Side.A, Combine.GATHER_REDUCE_V4, Moves(36, 12, ["top-k above 4"])),
        _NewSetting(Side.B, Combine.SEGMENT_SUM, Moves(48, 0, [])),
    ],
    ids=_new_setting_id,
)
def test_parallel_changes_new_setting(
    changes: ParallelChanges, case: _NewSetting
) -> None:
    """Each new variant moves what it did alone, and the merge keeps it."""
    assert case.setting in changes.new[case.side]
    row: list[Outcome] = changes.runs[case.side][case.setting]
    assert moves(row, changes.default) == case.moves
    assert row == changes.runs[Side.MERGE][case.setting]


def test_parallel_changes_overlap(changes: ParallelChanges) -> None:
    """Both new variants run every call up to top-k 4, on both chips."""
    assert changes.overlap == [
        (chip, tokens, topk) for chip in Chip for tokens in TOKENS for topk in (1, 2, 4)
    ]
