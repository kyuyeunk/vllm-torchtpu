# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Tests for the step plan (``runner/step_plan.py``)."""

from __future__ import annotations

import ast
import collections
import dataclasses
import enum
import pathlib

import pytest

from vllm_torchtpu.runner import step_plan
from vllm_torchtpu.runner.step_plan import DpStep, StepPlan

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu"


def _main_chunk_counter(
    tokens: list[int], most_model_len: int | None, max_cap: int, most_cap: int | None
) -> list[int]:
    """Requests per chunk, as `_count_input_chunks` computed them on main."""
    sizes = []
    start = 0
    while start < len(tokens):
        use_max = most_model_len is None
        for n in tokens[start:]:
            if not use_max and n > most_model_len:
                use_max = True
        cap = max_cap if use_max else most_cap
        sizes.append(min(cap, len(tokens) - start))
        start += sizes[-1]
    return sizes


@pytest.mark.parametrize(
    ("tokens", "most_model_len", "max_cap", "most_cap"),
    [
        pytest.param([1, 1, 3, 3, 16], None, 2, None, id="no_most_model_len"),
        pytest.param([1, 1, 1, 1, 100, 2], 32, 2, 3, id="long_request_in_tail"),
        pytest.param([1, 1, 1, 1, 2, 2, 2], 32, 2, 3, id="all_within_most_model_len"),
        pytest.param([5], 4, 1, 4, id="one_long_request"),
    ],
)
def test_split_chunks_matches_main(
    tokens: list[int], most_model_len: int | None, max_cap: int, most_cap: int | None
) -> None:
    """The one chunk split agrees with main's three copies of it."""
    chunks = step_plan.split_chunks(
        tokens,
        most_model_len=most_model_len,
        num_reqs_max_model_len=max_cap,
        num_reqs_most_model_len=most_cap,
    )
    assert [c.num_reqs for c in chunks] == _main_chunk_counter(
        tokens, most_model_len, max_cap, most_cap
    )
    assert [t for c in chunks for t in c.tokens_per_req] == tokens


def test_plan_is_frozen_and_checks_the_dp_count() -> None:
    """The plan is a frozen value; DP may not allow fewer forwards."""
    kwargs = dict(
        req_order=["a", "b"],
        tokens_per_req=[1, 1],
        num_decode=2,
        num_windowed=2,
        most_model_len=None,
        num_reqs_max_model_len=1,
        num_reqs_most_model_len=None,
    )
    plan = step_plan.plan_step(dp=DpStep(bucket=16, num_forwards=3), **kwargs)
    assert (len(plan.chunks), plan.num_forwards) == (2, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.chunks[0].num_reqs = 0  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="DP agreed"):
        step_plan.plan_step(dp=DpStep(num_forwards=1), **kwargs)


def test_idle_plan_runs_only_dummy_forwards() -> None:
    """An idle rank's plan comes from the same agreement, with no chunks."""
    idle = StepPlan.idle(DpStep(bucket=64, num_forwards=2))
    assert (idle.chunks, idle.num_forwards, idle.dp.bucket) == ((), 2, 64)


class _Use(enum.Enum):
    """How a function touches an attribute."""

    READ = enum.auto()
    WRITE = enum.auto()


def _writes_and_reads(attr: str) -> collections.Counter[tuple[str, str, _Use]]:
    uses: collections.Counter[tuple[str, str, _Use]] = collections.Counter()
    for path in sorted(_SRC.rglob("*.py")):
        rel = str(path.relative_to(_SRC))
        for func in ast.walk(ast.parse(path.read_text())):
            if not isinstance(func, ast.FunctionDef):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Attribute) and node.attr == attr:
                    kind = _Use.WRITE if isinstance(node.ctx, ast.Store) else _Use.READ
                    uses[rel, func.name, kind] += 1
    return uses


def test_dp_step_has_one_writer() -> None:
    """Only the DP agreement writes the runner's `DpStep`, and only whole."""
    writers = {
        (f, fn) for (f, fn, kind) in _writes_and_reads("_dp_step") if kind is _Use.WRITE
    }
    assert writers == {
        ("runner/tpu_runner.py", "__init__"),
        ("runner/tpu_runner.py", "_dp_coordinated_step"),
    }


def test_builder_context_stash_ratchet() -> None:
    """What is left on the stash: the chunk loop's reads after the forward.

    Every metadata builder receives its context as an argument; the runner
    still stages it for the chunk loop's mamba rollback and draft capture.
    A new reader fails here; moving one must shrink this list.
    """
    assert dict(_writes_and_reads("_attn_metadata_builder_ctx")) == {
        ("runner/tpu_runner.py", "_prepare_inputs", _Use.WRITE): 1,
        ("runner/tpu_runner.py", "_dummy_run", _Use.WRITE): 1,
        ("runner/tpu_runner.py", "execute_model", _Use.READ): 2,
        ("runner/tpu_runner.py", "_mamba_state_index_groups", _Use.READ): 1,
    }
