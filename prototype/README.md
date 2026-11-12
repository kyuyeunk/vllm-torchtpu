**Decisions group** (`prototype/decisions/`). Backs DATA_CONTRACTS.md section 6 (the MoE decision module built with today's rules, and the six places that still decide the combine), section 7's "one table per family" row, and Appendix A.7. Siblings: `prototype/decisions/moe-combine`.

# Prototype: the MoE decision module

Today `moe_gmm` and its helpers choose MoE kernel paths inline: which sort, which permute, which
combine, and which kernel fallback, each from environment reads and chip queries made in the middle
of the computation. This branch moves every one of those choices, unchanged, into one pure module,
`moe_plan.py`, plus four helpers: three kernel rules and collective ownership. The chip is
JAX's own `pltpu.TpuInfo` record, passed in as a value. The layer describes its inputs, asks the module for a plan and dispatches on it. Because
the rules move unchanged, every compiled program stays the same; what the move buys is that each
choice now has one place, runs on a CPU, and can be checked against today's code case by case.

It does not yet own the choice: six places still decide the combine (listed below), and the rules
are today's, not a structure-keyed table. Unifying them changes programs, so it is later work.

## Commits, one idea each

| Commit | Idea |
|---|---|
| Describe the chip as a value | the chip is `pltpu.TpuInfo`, passed as an argument; one function reads the runtime |
| Kernel rules as pure modules | three kernel limits move into plain modules over shapes, dtypes and `TpuInfo` that the kernel and the plan both import |
| One MoE decision module | `moe_plan.py`; `moe_gmm` describes its inputs and dispatches with `match` |
| Collective ownership | `moe_collective_plan.py`: who reduces, gathers and pads, for every MoE path |
| CPU oracles | each choice checked against today's code, transcribed, on boundary grids |
| This README and `moe_decision_module.py` | the command below |

## Where to look

| Claim | File | Symbol | Lines |
|---|---|---|---|
| One module holds every MoE path choice, as pure functions | `src/vllm_torchtpu/layers/core/moe_plan.py` | module docstring | 14-40 |
| Its inputs are described, not read from the runtime | same | `MoeProcessPlan`, `MoeOpIn`, `GmmOutSpec` | 160-187, 190-213, 216-248 |
| Routing choice (sort and permute) | same | `plan_routing` | 486-541 |
| Combine choice | same | `plan_combine` | 702-743 |
| Every stage at once | same | `plan_bucket` | 746-775 |
| The layer asks the module and dispatches on the plan | `src/vllm_torchtpu/layers/core/fused_moe_gmm.py` | `moe_gmm` (`plan_routing` at 417, `plan_combine` at 613, `match plan.combine` at 615) | 524-685 |
| The chip as a value (`pltpu.TpuInfo`), read from the runtime in one place | `src/vllm_torchtpu/layers/core/fused_moe_gmm.py` | `_runtime_tpu_info`, `_tpu_info_or_none`, `resolve_onehot_permute_threshold` | 107-109, 112-118, 121-139 |
| Kernel rule: 0.6 x VMEM TensorCore fallback | `src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_fallback.py` | `takes_tensorcore_path` | 50-79 |
| Kernel rule: v3 capability and partition | `src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_v3_partition.py` | `input_supported`, `v3_capable`, `num_column_partitions`, `token_block_alignment` | 37-48, 56-79, 82-108, 128-138 |
| Kernel rule: owner-output admission | `src/vllm_torchtpu/kernels/megablox/moe_onehot_unpermute_admission.py` | `shape_admits`, `chip_admits` | 40-76, 79-111 |
| Collective ownership in one place | `src/vllm_torchtpu/layers/core/moe_collective_plan.py` | module docstring, `CollectiveRule`, `collective_plan` | 14-32, 42-62, 142-206 |
| An adapter reads it | `src/vllm_torchtpu/layers/adapter/pipelined_fused_moe.py` | `collective_plan_for` | 76-108 |
| Oracle: today's combine chain, transcribed, against the plan `plan_combine` returns | `tests/proto_contracts/test_moe_plan_oracle.py` | `_reference_combine`, `MoeCombineOracleTest` | 337-415, 721-877 |
| The run below | `prototype/moe_decision_module.py` | `main` | 159-188 |

## Run it

CPU only: JAX on its CPU backend, chips from `pltpu.get_tpu_info_for_chip`, and no TPU runtime (the
two package `__init__`s that import `torch_tpu` are stubbed).

```
$ .venv/bin/python -I prototype/moe_decision_module.py
Plans from moe_plan.plan_bucket (bf16, hidden 2048):
  v6e                  TP     8 tokens x top-8:
    sort=single_sort permute=gather combine=tensorcore
  v6e                  TP  4096 tokens x top-8:
    sort=argsort_packed permute=gather combine=tensorcore
  7x                   EP    16 tokens x top-8:
    sort=counting_scatter permute=onehot_from_positions combine=ep_onehot
  v6e                  EP    32 tokens x top-8:
    sort=counting_packed permute=ragged_sc combine=ep_ragged_tc_fallback (EP gather-reduce v2)
  7x                   EP  8192 tokens x top-8:
    sort=counting_packed permute=ragged_sc combine=ep_ragged_sc (EP gather-reduce v2)
  v5e (no SparseCore)  EP  8192 tokens x top-8:
    sort=argsort_packed permute=gather combine=tensorcore

The six places that still decide the combine:
  layers/core/fused_moe_gmm.py:54-69
    EP gather-reduce version, fixed at import from the environment
  layers/core/fused_moe_gmm.py:121-139
    environment override of the one-hot threshold
  layers/core/moe_plan.py:373-390
    automatic one-hot threshold from the chip
  layers/core/moe_plan.py:702-743
    which combine runs, including the owner-output variant
  kernels/sparse_core/ragged_gather_reduce_fallback.py:50-79
    the 0.6 x VMEM fallback, one rule for the kernels and the plan
  kernels/megablox/moe_onehot_unpermute_admission.py:79-111
    owner-output capability, shared by the kernel and the plan
```

The line ranges are found in the source at run time, so they stay current.

## The module equals today's rules, on a CPU

Each oracle runs today's code (at `d3991eb`, this branch's base; transcribed verbatim, arrays
replaced by events) and the new module on the same inputs, and compares today's choice with the
plan the module returns: the sort and permute, the entry path, the combine and what each reads,
the kernel rules, the collective formulas and the one-hot threshold. The `fused_moe_gmm` glue
that turns a plan into array ops and runtime reads is not restated: the identity sweep below
checks it against today's code.

Each grid holds the decision-relevant values only: both sides of every threshold and capability
boundary, the 16 chips (the eleven `pltpu.get_tpu_info_for_chip` records JAX describes plus five
`dataclasses.replace` variants for branches no real chip reaches) and no chip, and each dtype and
setting that changes a decision. The tests run on JAX's CPU backend and check that no TPU runtime
(`torch_tpu`, `libtpu.so`) loaded:

```
$ .venv/bin/python -I -m unittest discover -s tests/proto_contracts -t tests/proto_contracts
[gather-reduce fallback] 13728 cases
[owner admission] 15360 cases
[owner VMEM boundary] 60 cases
[v3 partition] 150 cases
[all2all patch] 96 cases
[chunking] 6912 cases
[ownership] 224 cases
[combine] 21080 cases, 18 distinct outcomes; by path: {'ep_onehot': 5853, 'ep_owner_onehot': 81, 'ep_ragged_sc': 1629, 'ep_ragged_tc_fallback': 3363, 'local_v3': 116, 'raise': 362, 'tensorcore': 9676}
[entry] 2160 cases, 4 distinct (path, scrub)
[routing caps] 416 cases, 6 distinct (sort, permute)
[routing] 5632 cases, 12 distinct (sort, permute)
----------------------------------------------------------------------
Ran 19 tests in 0.523s

OK
```

Twelve one-token edits to the decision code (comparisons, the 0.6 VMEM fraction, the 16-lane and
one-below-the-block constants) were run against these tests: 11 fail them. The twelfth, `<=` to
`<` in local v3's padding rule, cannot change any outcome: equality needs a token alignment
divisible by 3, and every alignment is 64 times a power of two.

## Not shown here

- **Identical compiled programs on a TPU.** This branch's module and kernel-rule files follow the
  production-style build's (`scratch/data-contracts`, `e9489e004`), with its string plan values
  replaced by enums and one parse per environment value, and its hand-made chip record replaced
  by JAX's `pltpu.TpuInfo` (the rules take the record and real dtypes). That build is where
  program identity and the TPU golden were measured. This branch was checked on a CPU only:
  `fused_moe_func` traces to the base's jaxpr, or fails identically, on 34,848 configurations.
  It was not run on a TPU.
- **The fused MoE custom op with static configuration**, the enforcement checks and their
  baselines: on the production-style build, not here.
- **`GmmOutSpec.rows` still restates the routed-row count** here, as on the production-style build;
  the combine prototypes (`prototype/decisions/moe-combine`) replace it with `GmmOut`, which
  derives the count.
- Production would also need the existing test suite updated for the new call shapes.
