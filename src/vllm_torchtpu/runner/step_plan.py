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
"""The runner's decisions for one step, as two frozen values.

- ``DpStep``: what the data-parallel ranks agreed on (token bucket, forward
  count, largest chunk, encoder and decode flags). One all-reduce produces
  it, on busy and idle ranks alike.
- ``StepPlan``: this rank's request order, decode/verify/prefill counts and
  chunk split, plus its ``DpStep``. ``plan_step`` builds it once per step;
  an idle rank builds ``StepPlan.idle(dp)`` from the same agreement.

``split_chunks`` is the one chunk split. Every stage receives the plan, or
one ``ChunkPlan`` of it, as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class DpStep:
    """The data-parallel agreement for one step, final after the all-reduce.

    Attributes:
        bucket: DP-wide padded token bucket, or None without DP lockstep.
        num_forwards: Forwards every DP rank runs this step.
        max_reqs: Largest per-chunk request count on any rank.
        has_encoder_inputs: Whether any rank schedules encoder inputs.
        other_decode: Whether another rank decodes this step.
    """

    bucket: int | None = None
    num_forwards: int = 0
    max_reqs: int = 0
    has_encoder_inputs: bool = False
    other_decode: bool = False


@dataclass(frozen=True)
class ChunkPlan:
    """One forward's slice of the reordered batch.

    Attributes:
        start_index: First request of the chunk in the reordered batch.
        num_reqs: Live requests in the chunk.
        use_max_model_len: Whether any remaining request exceeds
            ``most_model_len``, which selects the max-model-len capacity.
        target_num_reqs: Request capacity of the compiled graph this chunk
            runs in.
        tokens_per_req: Scheduled tokens of each request in the chunk.
    """

    start_index: int
    num_reqs: int
    use_max_model_len: bool
    target_num_reqs: int
    tokens_per_req: tuple[int, ...]

    @property
    def end_index(self) -> int:
        """One past the chunk's last request."""
        return self.start_index + self.num_reqs


@dataclass(frozen=True)
class StepPlan:
    """The runner's decisions for one step, final after the DP agreement.

    Attributes:
        req_order: Request ids in the order every stage indexes them.
        num_decode: Leading one-token decode requests.
        num_windowed: Leading decodes plus speculative verify windows.
        chunks: This rank's real forwards, in order.
        dp: The DP agreement; forwards past ``chunks`` are dummy forwards at
            ``dp.bucket``.
    """

    req_order: tuple[str, ...]
    num_decode: int
    num_windowed: int
    chunks: tuple[ChunkPlan, ...]
    dp: DpStep

    @classmethod
    def idle(cls, dp: DpStep) -> StepPlan:
        """The plan of a rank with no requests: only dummy forwards."""
        return cls(req_order=(), num_decode=0, num_windowed=0, chunks=(), dp=dp)

    @property
    def num_forwards(self) -> int:
        """Forwards every DP rank runs this step."""
        return self.dp.num_forwards


def split_chunks(
    tokens_per_req: Sequence[int],
    *,
    most_model_len: int | None,
    num_reqs_max_model_len: int,
    num_reqs_most_model_len: int | None,
) -> tuple[ChunkPlan, ...]:
    """Splits the reordered batch into forwards.

    A chunk takes the max-model-len capacity when any remaining request
    schedules more than ``most_model_len`` tokens (or when no
    ``most_model_len`` is configured), and the most-model-len capacity
    otherwise.

    Args:
        tokens_per_req: Scheduled tokens per request, in batch order.
        most_model_len: The short-sequence threshold, or None.
        num_reqs_max_model_len: Request capacity of a max-model-len chunk.
        num_reqs_most_model_len: Request capacity of a most-model-len chunk.

    Returns:
        The chunks, covering every request once, in order.
    """
    chunks: list[ChunkPlan] = []
    start = 0
    while start < len(tokens_per_req):
        remaining = tokens_per_req[start:]
        use_max_model_len = most_model_len is None or max(remaining) > most_model_len
        if use_max_model_len:
            capacity = num_reqs_max_model_len
        else:
            assert num_reqs_most_model_len is not None
            capacity = num_reqs_most_model_len
        counts = tuple(remaining[:capacity])
        chunks.append(
            ChunkPlan(
                start_index=start,
                num_reqs=len(counts),
                use_max_model_len=use_max_model_len,
                target_num_reqs=capacity,
                tokens_per_req=counts,
            )
        )
        start += len(counts)
    return tuple(chunks)


def plan_step(
    *,
    req_order: Sequence[str],
    tokens_per_req: Sequence[int],
    num_decode: int,
    num_windowed: int,
    dp: DpStep,
    most_model_len: int | None,
    num_reqs_max_model_len: int,
    num_reqs_most_model_len: int | None,
) -> StepPlan:
    """Builds the step plan; the only producer in the runner.

    Args:
        req_order: Request ids after the batch reorder.
        tokens_per_req: Scheduled tokens per request, in ``req_order``.
        num_decode: Leading one-token decodes, from the reorder.
        num_windowed: Leading decodes plus verify windows, from the reorder.
        dp: The DP agreement for this step.
        most_model_len: The short-sequence threshold, or None.
        num_reqs_max_model_len: Request capacity of a max-model-len chunk.
        num_reqs_most_model_len: Request capacity of a most-model-len chunk.

    Returns:
        The frozen plan.

    Raises:
        ValueError: The DP agreement allows fewer forwards than this rank
            needs.
    """
    chunks = split_chunks(
        tokens_per_req,
        most_model_len=most_model_len,
        num_reqs_max_model_len=num_reqs_max_model_len,
        num_reqs_most_model_len=num_reqs_most_model_len,
    )
    if dp.num_forwards < len(chunks):
        msg = (
            f"DP agreed on {dp.num_forwards} forwards but this rank needs {len(chunks)}"
        )
        raise ValueError(msg)
    return StepPlan(
        req_order=tuple(req_order),
        num_decode=num_decode,
        num_windowed=num_windowed,
        chunks=chunks,
        dp=dp,
    )
