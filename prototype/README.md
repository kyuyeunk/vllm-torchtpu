# Prototype: the step plan

Backs DATA_CONTRACTS.md Figure 24 (section 5.4) and Appendix J. Today the runner makes
each step's decisions in several places and leaves them on itself:

- the chunk split is computed three times (`_count_input_chunks`,
  `_pp_forward_rows`, the loop at the top of `_prepare_inputs`);
- the data-parallel agreement lands in five attributes
  (`_dp_target_bucket`, `_dp_step_num_chunks`, `_dp_step_max_reqs`,
  `_dp_step_has_encoder_inputs`, `_dp_step_other_decode`), written from four
  functions; the idle pairing re-writes the bucket "before the decoder
  dummies" because the dummies read it off the runner;
- the attention metadata builders read their context from
  `runner._attn_metadata_builder_ctx`, which dflash and eagle3 save,
  overwrite and restore around their own builds.

## Where to look

| Claim | Code |
|---|---|
| The data-parallel agreement is one frozen record | [`DpStep`](../src/vllm_torchtpu/runner/step_plan.py#L36-L52) in `step_plan.py` (L36-L52) |
| The plan record and the idle plan | [`StepPlan`](../src/vllm_torchtpu/runner/step_plan.py#L81-L108) in `step_plan.py` (L81-L108) |
| One chunk split | [`split_chunks`](../src/vllm_torchtpu/runner/step_plan.py#L111-L156) in `step_plan.py` (L111-L156) |
| The only producer | [`plan_step`](../src/vllm_torchtpu/runner/step_plan.py#L158-L205) in `step_plan.py` (L158-L205) |
| The single DpStep writer | [`_dp_coordinated_step`](../src/vllm_torchtpu/runner/tpu_runner.py#L3921-L3976) in `tpu_runner.py` (L3921-L3976) |
| The plan is built once per step | [`plan_step call in execute_model`](../src/vllm_torchtpu/runner/tpu_runner.py#L4143-L4157) in `tpu_runner.py` (L4143-L4157) |
| A stage receives its chunk | [`_prepare_inputs`](../src/vllm_torchtpu/runner/tpu_runner.py#L3282-L3286) in `tpu_runner.py` (L3282-L3286) |
| The builder takes its context as an argument | [`AttentionMetadataBuilder.build`](../src/vllm_torchtpu/layers/core/attention_metadata.py#L523-L529) in `attention_metadata.py` (L523-L529) |
| Single-writer and stash ratchet tests | [`test_dp_step_has_one_writer, test_builder_context_stash_ratchet`](../tests/runner/test_step_plan.py#L120-L143) in `test_step_plan.py` (L120-L143) |

This branch holds the same decisions in two frozen values
(`src/vllm_torchtpu/runner/step_plan.py`):

| Value | Fields | Producer | Readers |
|---|---|---|---|
| `DpStep` | bucket, forward count, largest chunk, encoder and decode flags | `_dp_coordinated_step`, the one all-reduce, on busy and idle ranks | layout planners, dflash, eagle3, the runner |
| `StepPlan` | request order, decode/verify counts, `ChunkPlan`s, its `DpStep` | `plan_step`, once per step in `execute_model`; `StepPlan.idle(dp)` on an idle rank | the chunk loop, `_prepare_inputs` (the plan and one `ChunkPlan`), `_pp_forward_rows`, the idle pairing |

What changes:

- **One chunk split.** `split_chunks` replaces the three copies;
  `_prepare_inputs` receives its `ChunkPlan` instead of re-deriving it, and
  reads the request order and decode/verify counts from the plan rather than
  from loose arguments.
- **One writer for the DP agreement.** `_dp_coordinated_step` replaces the
  runner's `DpStep` whole; the five attributes are gone. The idle rank's
  pairing takes `StepPlan.idle(dp)`.
- **The builder context is an argument.** `AttentionMetadataBuilder.build`,
  `_build_attention_metadata` and `build_attention_metadata_for_layers`
  take `ctx`; the main model, dummy runs and both drafters pass it. The
  drafters no longer touch the runner's copy.

What does not change: the runner still stages the context for the chunk
loop's reads after the forward (mamba rollback indices, draft capture);
`tests/runner/test_step_plan.py` pins those 5 accesses in 4 functions (main
has 14 in 9). The context also carries device tensors and a write-back list
(`unified_mamba_state_indices`), which a production version would split
from the decisions.

Assumes vLLM v0.33.0. Run from the repository root with the project's
virtualenv and vLLM v0.33.0 on `PYTHONPATH`. Importing the runner loads the
TPU platform plugin and libtpu, so on a shared TPU host take the TPU lock.
torch-tpu's start-up lines on stderr are left out below.

This branch is for reading, not merging. Existing tests that stub the five
DP attributes, call `_prepare_inputs` with a start index, or stage the
builder context on the runner (in `tests/runner`, `tests/spec_decode`,
`tests/layers/core`) fail rather than being rewritten; the production-style
build that keeps them is separate.

## One step on two DP ranks

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.step_plan

Both ranks call the runner's real `_dp_coordinated_step`, with `all_reduce`
simulated as the element-wise MAX of what each rank sends.

```
DP agreement, one frozen record per rank:
  rank 0 DpStep(bucket=4096, num_forwards=3, max_reqs=2, has_encoder_inputs=False, other_decode=False)
  rank 1 DpStep(bucket=4096, num_forwards=3, max_reqs=2, has_encoder_inputs=False, other_decode=True)

rank 0 StepPlan (plan_step):
  req_order=('d0', 'd1', 'v0', 'p0', 'p1', 'p2') num_decode=2 num_windowed=3
  chunk 0: ChunkPlan(start_index=0, num_reqs=2, use_max_model_len=True, target_num_reqs=2, tokens_per_req=(1, 1))
  chunk 1: ChunkPlan(start_index=2, num_reqs=2, use_max_model_len=True, target_num_reqs=2, tokens_per_req=(4, 2048))
  chunk 2: ChunkPlan(start_index=4, num_reqs=2, use_max_model_len=False, target_num_reqs=4, tokens_per_req=(40, 300))

rank 1 StepPlan (StepPlan.idle):
  StepPlan(req_order=(), num_decode=0, num_windowed=0, chunks=(), dp=DpStep(bucket=4096, num_forwards=3, max_reqs=2, has_encoder_inputs=False, other_decode=True))

who receives what (rank 0):
  execute_model loop       runs 3 forwards: 3 chunks + 0 dummy at dp.bucket=4096
  _prepare_inputs chunk 0  requests [0, 2) capacity 2
  _prepare_inputs chunk 1  requests [2, 4) capacity 2
  _prepare_inputs chunk 2  requests [4, 6) capacity 4
  layout planner           pads every chunk to dp.bucket=4096
  dflash / eagle3          pad to dp.num_forwards=3, size drafts by dp.max_reqs=2
  metadata builders        receive the chunk's context as `ctx`

rank 1 runs 3 dummy forwards at dp.bucket=4096, from the same agreement

a stage writing a decision after the plan is built: FrozenInstanceError("cannot assign to field 'num_reqs'")
```

The two records differ only in `other_decode`: rank 0 is the only decoding
rank, so only rank 1 hears that another rank decodes. Chunk 2 takes the
most-model-len capacity because no remaining request exceeds 1024 tokens.

## The checks

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q tests/runner/test_step_plan.py

```
8 passed
```

- `split_chunks` agrees with main's chunk counter on four batches.
- The plan is frozen; a DP forward count below the rank's chunk count raises.
- `StepPlan.idle` has no chunks and the agreement's forward count.
- Only `__init__` and `_dp_coordinated_step` assign the runner's `DpStep`.
- The stash ratchet: the 5 remaining accesses, in 4 functions.

## On a TPU

The project's golden workload (a harness kept outside this branch, a
TPU v6e, cold caches) on this branch against `main` at `c350823af`:

```
ok   'The capital of France is': tokens equal, max logprob drift 0.000000
ok   'def fibonacci(n):': tokens equal, max logprob drift 0.000000
ok   'Count to 5:': tokens equal, max logprob drift 0.000000
ok   'The quick brown fox jumps over the lazy ': tokens equal, max logprob drift 0.000000
GOLDEN: PASS
```

The golden workload runs one rank without a drafter, so it exercises the
chunk split and the builder argument, not the DP or drafter paths; those
are covered on CPU by the production-style build's tests.
