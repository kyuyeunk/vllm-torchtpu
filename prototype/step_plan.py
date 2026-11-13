r"""One step on two data-parallel ranks, as the runner now decides it.

Run from the repository root:

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.step_plan

Rank 0 has a batch; rank 1 has none. Both call the runner's real
`_dp_coordinated_step`, with `all_reduce` simulated as the element-wise MAX
of what each rank sends (`tests/runner/dp_agreement.py`), and get one
frozen `DpStep`; `tests/runner/test_step_plan.py` checks the two records
differ only in `other_decode`. Rank 0 builds its
`StepPlan` with `plan_step`; rank 1 builds `StepPlan.idle(dp)` from the same
agreement. Then every stage that used to re-derive a step decision is shown
with the plan value it now receives.
"""

from __future__ import annotations

import dataclasses

from tests.runner.dp_agreement import agree, busy_sends, example_batch, idle_sends
from vllm_torchtpu.runner.step_plan import StepPlan, plan_step


def _out(line: str) -> None:
    """Print one line of the demo's output."""
    print(line)  # noqa: T201 - a demo's output is its tables


def main() -> None:
    """Print the agreement, both ranks' plans, and what each stage receives."""
    batch = example_batch()
    dp0, dp1 = agree([busy_sends(batch), idle_sends()], batch.paddings)
    _out("DP agreement, one frozen record per rank:")
    _out(f"  rank 0 {dp0}")
    _out(f"  rank 1 {dp1}")

    step = plan_step(
        req_order=batch.req_order,
        tokens_per_req=batch.tokens,
        num_decode=batch.num_decode,
        num_windowed=batch.num_windowed,
        dp=dp0,
        most_model_len=batch.most_model_len,
        num_reqs_max_model_len=batch.num_reqs_max_model_len,
        num_reqs_most_model_len=batch.num_reqs_most_model_len,
    )
    _out("\nrank 0 StepPlan (plan_step):")
    _out(
        f"  req_order={step.req_order} num_decode={step.num_decode} "
        f"num_windowed={step.num_windowed}"
    )
    for i, chunk in enumerate(step.chunks):
        _out(f"  chunk {i}: {chunk}")
    _out("\nrank 1 StepPlan (StepPlan.idle):")
    _out(f"  {StepPlan.idle(dp1)}")

    _out("\nwho receives what (rank 0):")
    _out(
        f"  execute_model loop       runs {step.num_forwards} forwards: "
        f"{len(step.chunks)} chunks + "
        f"{step.num_forwards - len(step.chunks)} dummy at dp.bucket={step.dp.bucket}"
    )
    for i, chunk in enumerate(step.chunks):
        _out(
            f"  _prepare_inputs chunk {i}  requests [{chunk.start_index}, "
            f"{chunk.end_index}) capacity {chunk.target_num_reqs}"
        )
    _out(f"  layout planner           pads every chunk to dp.bucket={step.dp.bucket}")
    _out(
        f"  dflash / eagle3          pad to dp.num_forwards={step.dp.num_forwards}, "
        f"size drafts by dp.max_reqs={step.dp.max_reqs}"
    )
    _out("  metadata builders        receive the chunk's context as `ctx`")
    _out(
        f"\nrank 1 runs {StepPlan.idle(dp1).num_forwards} dummy forwards at "
        f"dp.bucket={dp1.bucket}, from the same agreement"
    )
    try:
        step.chunks[0].num_reqs = 0  # pyright: ignore[reportAttributeAccessIssue]
    except dataclasses.FrozenInstanceError as exc:
        _out(f"\na stage writing a decision after the plan is built: {exc!r}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
