# Prototype: the plan check on the full MoE decision module

Backs DATA_CONTRACTS.md section 5.8 (the plan-invariance check, its two tiers) and Table 13
(what the check found on the MoE combine).

This branch is [`prototype/decisions/centralized-moe-plan`](https://github.com/kyuyeunk/vllm-torchtpu/tree/prototype/decisions/centralized-moe-plan)
(`a64038a6a`: today's MoE rules gathered into pure modules, with the full three-field
`MoeCombinePlan`) plus five commits, one idea each. `git diff a64038a6a..HEAD` is the check
and nothing else:

| Commit | Idea |
|---|---|
| `2a4fce430` | Mark the MoE plan types and enumerate their values |
| `3bf5a123c` | State what each combine can run, apart from the planner's preferences |
| `6ca341c71` | Structural tier: declared capability against the compiler, on fake chips |
| `8e6408200` | The EP gather-reduce kernel follows the plan's version (a finding of the structural tier, fixed) |
| `be0ecd34e` | Numerics tier: every capable plan value against the picked one, on a TPU |

## Where to look

| Claim | File | Symbol | Lines |
|---|---|---|---|
| A plan type is marked with its owner; its values come from its field types | [`plan_marker.py`](../src/vllm_torchtpu/layers/core/plan_marker.py#L36-L80) | `plan_of`, `plan_values` | 36-80 |
| The combine plan, marked; it refuses a field its combine would ignore | [`moe_plan.py`](../src/vllm_torchtpu/layers/core/moe_plan.py#L303-L331) | `MoeCombinePlan` | 303-331 |
| Capability apart from preference | [`moe_plan.py`](../src/vllm_torchtpu/layers/core/moe_plan.py#L764-L809) | `combine_capable` | 764-809 |
| The layer's combine, split so a checker can force any plan | [`fused_moe_gmm.py`](../src/vllm_torchtpu/layers/core/fused_moe_gmm.py#L524-L742) | `combine_facts`, `resolve_combine_plan`, `moe_combine` | 524-742 |
| The fix: the EP kernel follows the plan's version | [`fused_moe_gmm.py`](../src/vllm_torchtpu/layers/core/fused_moe_gmm.py#L622-L636) | `_ep_gather_reduce` | 622-636 |
| Structural tier: compile every value for fake chips; capable must compile, declined should not, outputs must agree, unread fields reported | [`prototype/plan_check.py`](plan_check.py#L140-L204) | `check_structure` | 140-204 |
| Structural driver: shapes (one padded), the findings ratchet, the replay switch | [`prototype/check_plans.py`](check_plans.py#L67-L132) | `SHAPES`, `REPLAY_IMPORT_TIME_KERNEL`, `KNOWN`, `REPLAYED`, `main` | 67-132, 152-154, 157-184, 192-195, 265-288 |
| Numerics tier: every capable value against the picked one, within a tolerance | [`prototype/plan_check_values.py`](plan_check_values.py#L55-L92) | `check_values` | 55-92 |
| Numerics driver: the tolerance, two input families, the findings ratchet | [`prototype/check_plan_values.py`](check_plan_values.py#L54-L57) | `TOLERANCE`, `KNOWN`, `main` | 54-57, 63-77, 210-229 |

## The structural tier (CPU, fake chips)

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.check_plans

It loads libtpu to compile, so do not run it beside a TPU job; the lock serializes it with
them. It compiles `moe_combine` under all 12 `MoeCombinePlan` values on seven shapes, ahead of
time, for v6e and TPU7x topologies, which JAX describes without a device. Output at the head
(each line is one finding; all are kept in `KNOWN`, and the run fails if any appears or goes):

```text
compiles tpu7x tp_t64_k10_h4096_f32 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles tpu7x tp_t64_k10_h4096_f32 combine='local_v3', ep_gather_reduce_version=None, local_v3_token_alignment=128
compiles tpu7x tp_t64_k10_h4096_f32 combine='local_v3', ep_gather_reduce_version=None, local_v3_token_alignment=256
compiles tpu7x tp_t64_k10_h4096_f32 combine='local_v3', ep_gather_reduce_version=None, local_v3_token_alignment=64
fails tpu7x ep_t16_k8_h2048_pad combine='ep_onehot', ep_gather_reduce_version=None, local_v3_token_alignment=None
fails tpu7x ep_t16_k8_h2048_pad combine='ep_ragged_sc', ep_gather_reduce_version='v1', local_v3_token_alignment=None
fails tpu7x tp_t2048_k1_h4096 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
fails v6e ep_t16_k8_h2048_pad combine='ep_onehot', ep_gather_reduce_version=None, local_v3_token_alignment=None
fails v6e ep_t16_k8_h2048_pad combine='ep_ragged_sc', ep_gather_reduce_version='v1', local_v3_token_alignment=None
unread tpu7x local_v3_token_alignment
findings: 10 in 122 s
```

At commit `6ca341c71`, before the fix, the same command printed 16 findings: the four
`compiles tpu7x` and two `fails ... ep_onehot` lines above, `unread tpu7x
local_v3_token_alignment`, and these nine:

```text
compiles v6e ep_t16_k8_h2048 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles v6e ep_t16_k8_h2048_pad combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles v6e ep_t512_k8_h4096 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles v6e tp_t2048_k1_h4096 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles v6e tp_t300_k10_h5120 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles v6e tp_t512_k10_h4096 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
compiles v6e tp_t64_k10_h4096_f32 combine='ep_ragged_sc', ep_gather_reduce_version='v3', local_v3_token_alignment=None
unread tpu7x ep_gather_reduce_version
unread v6e ep_gather_reduce_version
```

`REPLAY_IMPORT_TIME_KERNEL = True` (comment it in) brings the import-time kernel back at the
head and reproduces exactly that set (`findings: 16`).

## The numerics tier (TPU)

    flock /tmp/vt-tpu.lock env PYTHONPATH=src python -m prototype.check_plan_values

Runs every capable value on the host's chip, on inputs laid out as routing lays them (garbage
and NaN in the unwritten rows) and on inputs masked at random, and compares each output with
the picked plan's within about two BF16 ulps (FP32: 1e-3). Output on a v6e:

```text
chip: TPU v6 lite
value tp_t2048_k1_h4096_mask combine='ep_owner_onehot', ep_gather_reduce_version=None, local_v3_token_alignment=None 3.9e+00
value tp_t512_k10_h4096_mask combine='ep_owner_onehot', ep_gather_reduce_version=None, local_v3_token_alignment=None 5.7e-01
value tp_t64_k10_h4096_f32 combine='ep_onehot', ep_gather_reduce_version=None, local_v3_token_alignment=None 2.8e-03
value tp_t64_k10_h4096_f32 combine='ep_ragged_tc_fallback', ep_gather_reduce_version='v1', local_v3_token_alignment=None 2.0e-03
value tp_t64_k10_h4096_f32 combine='ep_ragged_tc_fallback', ep_gather_reduce_version='v2', local_v3_token_alignment=None 2.0e-03
```

That run is from `9832f0495`, before the plan values became enums; it was not repeated on a TPU.

## Table 13's findings on this branch

| # | Finding | Here |
|---|---|---|
| 1 | EP v3 at top-1 is declared capable but fails to trace (TPU7x) | Live: `fails tpu7x tp_t2048_k1_h4096 ... v3` |
| 2 | `ep_gather_reduce_version` never reaches the kernel | Live at `6ca341c71` (`unread ... ep_gather_reduce_version`, and EP v3 compiling on v6e as the import-time kernel); fixed in `8e6408200`; replayed by `REPLAY_IMPORT_TIME_KERNEL` |
| 3 | `local_v3_token_alignment` never changes the program (TPU7x) | Live: `unread tpu7x local_v3_token_alignment` |
| 4 | v3 declared BF16-only yet compiles with FP32 input | Live: the four `compiles tpu7x tp_t64_k10_h4096_f32` lines |
| 5 | Precision filed as plan | Live: one-hot and the v1/v2 TensorCore fallbacks differ by 2.0e-3 to 2.8e-3 at FP32 (v6e run at `9832f0495`) |
| 6 | A padded second-GMM output, which the input facts allow, breaks a combine | Live: `fails ... ep_t16_k8_h2048_pad` for the one-hot combine, and, once the version is read, for the v1 kernel too |
| 7 | The owner-output kernel sums masked slots | Live on the random-mask inputs: 5.7e-01 and 3.9e+00 (v6e run at `9832f0495`) |
| 8 | No split between capability and preference | Built here as `combine_capable` (`3bf5a123c`); every finding above is stated against it |

Not shown here: a TPU7x numerics run (this host is a v6e), so whether FP32 v3 fails or rounds
there stays open; and the numerics tier's tolerance is the test author's, not one a contract
states.
