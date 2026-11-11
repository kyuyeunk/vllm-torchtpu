r"""One step on two data-parallel ranks, as the runner now decides it.

Run from the repository root:

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.step_plan

Rank 0 has a batch; rank 1 has none. Both call the runner's real
`_dp_coordinated_step`, with `all_reduce` simulated as the element-wise MAX
of what each rank sends, and get one frozen `DpStep`. Rank 0 builds its
`StepPlan` with `plan_step`; rank 1 builds `StepPlan.idle(dp)` from the same
agreement. Then every stage that used to re-derive a step decision is shown
with the plan value it now receives.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import torch
from vllm.distributed import parallel_state

from vllm_torchtpu.runner.step_plan import StepPlan, plan_step, split_chunks
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# Rank 0's batch after the reorder: decodes, verify windows, then prefills.
REQ_ORDER = ("d0", "d1", "v0", "p0", "p1", "p2")
TOKENS = (1, 1, 4, 2048, 40, 300)
NUM_DECODE, NUM_WINDOWED = 2, 3
MOST_MODEL_LEN = 1024  # chunks with no longer request take more requests
NUM_REQS_MAX_MODEL_LEN, NUM_REQS_MOST_MODEL_LEN = 2, 4
PADDINGS = [16, 64, 256, 1024, 4096]


def agree(sends: list[dict[str, int | bool]]) -> list[object]:
    """Runs `_dp_coordinated_step` on every rank against one all-reduce."""
    runners = []
    for _ in sends:
        runner = object.__new__(TPUModelRunner)
        runner._dp_lockstep_enabled = lambda: True  # noqa: SLF001
        runner.num_tokens_paddings = PADDINGS
        runners.append(runner)
    sent: list[torch.Tensor] = []

    def call(rank: int, all_reduce: object) -> object:
        group = SimpleNamespace(
            rank_in_group=rank, world_size=len(sends), cpu_group=None
        )
        parallel_state.get_dp_group = lambda: group
        torch.distributed.all_reduce = all_reduce
        return runners[rank]._dp_coordinated_step(**sends[rank])  # noqa: SLF001

    for rank in range(len(sends)):
        call(rank, lambda t, op=None, group=None: sent.append(t.clone()))
    reduced = torch.stack(sent).amax(dim=0)
    return [
        call(rank, lambda t, op=None, group=None: t.copy_(reduced))
        for rank in range(len(sends))
    ]


def main() -> None:
    local = split_chunks(
        TOKENS,
        most_model_len=MOST_MODEL_LEN,
        num_reqs_max_model_len=NUM_REQS_MAX_MODEL_LEN,
        num_reqs_most_model_len=NUM_REQS_MOST_MODEL_LEN,
    )
    busy = {
        "local_num_tokens": sum(TOKENS),
        "local_num_chunks": len(local),
        "local_num_reqs": max(c.num_reqs for c in local),
        "local_any_decode": True,
    }
    idle = {"local_num_tokens": 0, "local_num_chunks": 0}
    dp0, dp1 = agree([busy, idle])
    # Same agreement; only `other_decode` differs: rank 0 is the decoding
    # rank, so it alone hears that no *other* rank decodes.
    assert dataclasses.replace(dp0, other_decode=False) == dataclasses.replace(
        dp1, other_decode=False
    )
    print("DP agreement, one frozen record per rank:")
    print(f"  rank 0 {dp0}")
    print(f"  rank 1 {dp1}")

    step = plan_step(
        req_order=REQ_ORDER,
        tokens_per_req=TOKENS,
        num_decode=NUM_DECODE,
        num_windowed=NUM_WINDOWED,
        dp=dp0,
        most_model_len=MOST_MODEL_LEN,
        num_reqs_max_model_len=NUM_REQS_MAX_MODEL_LEN,
        num_reqs_most_model_len=NUM_REQS_MOST_MODEL_LEN,
    )
    print("\nrank 0 StepPlan (plan_step):")
    print(
        f"  req_order={step.req_order} num_decode={step.num_decode} "
        f"num_windowed={step.num_windowed}"
    )
    for i, chunk in enumerate(step.chunks):
        print(f"  chunk {i}: {chunk}")
    print("\nrank 1 StepPlan (StepPlan.idle):")
    print(f"  {StepPlan.idle(dp1)}")

    print("\nwho receives what (rank 0):")
    print(
        f"  execute_model loop       runs {step.num_forwards} forwards: "
        f"{len(step.chunks)} chunks + "
        f"{step.num_forwards - len(step.chunks)} dummy at dp.bucket={step.dp.bucket}"
    )
    for i, chunk in enumerate(step.chunks):
        print(
            f"  _prepare_inputs chunk {i}  requests [{chunk.start_index}, "
            f"{chunk.end_index}) capacity {chunk.target_num_reqs}"
        )
    print(f"  layout planner           pads every chunk to dp.bucket={step.dp.bucket}")
    print(
        f"  dflash / eagle3          pad to dp.num_forwards={step.dp.num_forwards}, "
        f"size drafts by dp.max_reqs={step.dp.max_reqs}"
    )
    print("  metadata builders        receive the chunk's context as `ctx`")
    print(
        f"\nrank 1 runs {StepPlan.idle(dp1).num_forwards} dummy forwards at "
        f"dp.bucket={dp1.bucket}, from the same agreement"
    )
    try:
        step.chunks[0].num_reqs = 0  # pyright: ignore[reportAttributeAccessIssue]
    except dataclasses.FrozenInstanceError as exc:
        print(f"\na stage writing a decision after the plan is built: {exc!r}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
