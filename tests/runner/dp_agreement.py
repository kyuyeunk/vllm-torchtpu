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
"""The runner's real DP agreement on simulated ranks, and a two-rank batch.

`agree` runs `TPUModelRunner._dp_coordinated_step` once per rank against one
simulated all-reduce: the element-wise MAX of what every rank sends. Shared
by `tests/runner/test_step_plan.py` and `python -m prototype.step_plan`.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.runner.step_plan import split_chunks
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from vllm_torchtpu.runner.step_plan import DpStep


@dataclasses.dataclass(frozen=True)
class RankSends:
    """What one rank passes to `_dp_coordinated_step`.

    Attributes:
        local_num_tokens: The rank's scheduled tokens.
        local_num_chunks: The rank's chunk count.
        local_num_reqs: The rank's largest per-chunk request count.
        local_any_decode: Whether the rank decodes this step.
    """

    local_num_tokens: int
    local_num_chunks: int
    local_num_reqs: int = 0
    local_any_decode: bool = False


@dataclasses.dataclass(frozen=True)
class Batch:
    """Rank 0's batch after the reorder, and the runner's limits.

    Attributes:
        req_order: Request ids: decodes, verify windows, then prefills.
        tokens: Scheduled tokens per request, in `req_order`.
        num_decode: Leading requests that decode.
        num_windowed: Leading requests that decode or verify.
        most_model_len: Chunks with no longer request take more requests.
        num_reqs_max_model_len: Requests per chunk with a long request.
        num_reqs_most_model_len: Requests per chunk without one.
        paddings: The runner's token buckets.
    """

    req_order: tuple[str, ...]
    tokens: tuple[int, ...]
    num_decode: int
    num_windowed: int
    most_model_len: int
    num_reqs_max_model_len: int
    num_reqs_most_model_len: int
    paddings: tuple[int, ...]


def example_batch() -> Batch:
    """Two decodes, one verify window, three prefills; one longer than 1024."""
    return Batch(
        req_order=("d0", "d1", "v0", "p0", "p1", "p2"),
        tokens=(1, 1, 4, 2048, 40, 300),
        num_decode=2,
        num_windowed=3,
        most_model_len=1024,
        num_reqs_max_model_len=2,
        num_reqs_most_model_len=4,
        paddings=(16, 64, 256, 1024, 4096),
    )


def busy_sends(batch: Batch) -> RankSends:
    """What the rank holding `batch` sends: its tokens, chunks and decodes."""
    local = split_chunks(
        batch.tokens,
        most_model_len=batch.most_model_len,
        num_reqs_max_model_len=batch.num_reqs_max_model_len,
        num_reqs_most_model_len=batch.num_reqs_most_model_len,
    )
    return RankSends(
        local_num_tokens=sum(batch.tokens),
        local_num_chunks=len(local),
        local_num_reqs=max(c.num_reqs for c in local),
        local_any_decode=True,
    )


def idle_sends() -> RankSends:
    """What a rank with no batch sends."""
    return RankSends(local_num_tokens=0, local_num_chunks=0)


def _new_runner(paddings: Sequence[int]) -> TPUModelRunner:
    """A runner with only what `_dp_coordinated_step` reads, in DP lockstep."""
    runner = object.__new__(TPUModelRunner)
    runner._dp_lockstep_enabled = lambda: True  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - stubs the runner's lockstep switch
    runner.num_tokens_paddings = list(paddings)
    return runner


def agree(sends: Sequence[RankSends], paddings: Sequence[int]) -> list[DpStep]:
    """Run `_dp_coordinated_step` on every rank against one all-reduce.

    Args:
        sends: What each rank sends, by rank.
        paddings: The runner's token buckets.

    Returns:
        Each rank's `DpStep`, by rank.
    """
    runners = [_new_runner(paddings) for _ in sends]
    sent: list[torch.Tensor] = []
    saved_group = parallel_state.get_dp_group
    saved_all_reduce = torch.distributed.all_reduce  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] - torch leaves its signature partly untyped

    def call(rank: int, all_reduce: Callable[..., object]) -> DpStep:
        group = SimpleNamespace(
            rank_in_group=rank, world_size=len(sends), cpu_group=None
        )
        parallel_state.get_dp_group = lambda: group
        torch.distributed.all_reduce = all_reduce
        mine = sends[rank]
        return runners[rank]._dp_coordinated_step(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - the agreement under test
            local_num_tokens=mine.local_num_tokens,
            local_num_chunks=mine.local_num_chunks,
            local_num_reqs=mine.local_num_reqs,
            local_any_decode=mine.local_any_decode,
        )

    def record(t: torch.Tensor, **_kwargs: object) -> None:
        sent.append(t.clone())

    try:
        for rank in range(len(sends)):
            _ = call(rank, record)
        reduced = torch.stack(sent).amax(dim=0)

        def deliver(t: torch.Tensor, **_kwargs: object) -> None:
            _ = t.copy_(reduced)

        return [call(rank, deliver) for rank in range(len(sends))]
    finally:
        parallel_state.get_dp_group = saved_group
        torch.distributed.all_reduce = saved_all_reduce
