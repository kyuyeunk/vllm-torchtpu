**Checks group** (`prototype/checks/`). Backs DATA_CONTRACTS.md section 5.7 (the hierarchy checks), section 3 ("Decisions flow down the scopes") and Table 12, rule 1 (d) and (e). Siblings: `prototype/checks/reach-in-lints`, `prototype/checks/moe-plan-invariance`.

# Prototype: each level owns its decisions

Decisions flow from the model scope through
block and family scopes to kernels, and each level's plan becomes the
contract of the level below. Each level owns its decisions; three checks stop
a lower level from reaching up or re-deciding what it was handed:

| What a lower level must not do | Check |
|---|---|
| Reach up: import a higher scope | import-linter `layers` contracts in `.importlinter` |
| Re-decide: accept a decision nobody made | resolved types (`resolution.require_no_default`) and `tests/layers/core/test_handoff.py` |
| Ignore: drop a decision it was handed | `tests/layers/core/test_dead_fields.py` (compile tier) |

## Where to look

| Claim | Code |
|---|---|
| Scopes import downward, with a ratchet of 9 edges | [`scopes-top-down`](../.importlinter#L14-L35) in `.importlinter` (L14-L35) |
| The decider never imports the executor | [`family-decides-below-executor`](../.importlinter#L39-L46) in `.importlinter` (L39-L46) |
| Resolved types carry no defaults | [`require_no_default`](../src/vllm_torchtpu/layers/core/resolution.py#L28-L40) in `resolution.py` (L28-L40) |
| The resolved plan | [`MoeCombinePlan`](../src/vllm_torchtpu/layers/core/moe_plan.py#L71-L88) in `moe_plan.py` (L71-L88) |
| It decides once | [`plan_combine`](../src/vllm_torchtpu/layers/core/moe_plan.py#L91-L170) in `moe_plan.py` (L91-L170) |
| The executor runs the plan only | [`moe_combine`](../src/vllm_torchtpu/layers/core/fused_moe_gmm.py#L573-L665) in `fused_moe_gmm.py` (L573-L665) |
| No X | None reaches a child | [`_module_receivers, receivers`](../tests/layers/core/handoff_check.py#L75-L97) in `handoff_check.py` (L75-L97) |
| Dead-field report | [`traced` to `fields_never_read`](../tests/layers/core/dead_fields_check.py#L78-L139) in `dead_fields_check.py` (L78-L139) |

The branch applies them to the MoE combine. It moves the combine decision out
of `moe_gmm` into `layers/core/moe_plan.py`: `plan_combine` decides once and
returns a `MoeCombinePlan`, and the new `moe_combine` in `fused_moe_gmm.py`
runs that plan and decides nothing. The plan names the expert-parallel (EP)
kernel version instead of leaving it to the import-time setting.

This branch is for reading, not merging. Tests in
`tests/layers/core/test_fused_moe_gmm.py` that patch the internals it moves
are left as they are; tests that stand in a chip patch `support.current_tpu_info`,
the one runtime read of the chip. Each check is a pytest test under
`tests/layers/core/`, with its shared code beside it (`handoff_check.py`,
`dead_fields_check.py`); `prototype/check_*.py` only print the tables below.
Run from the repository root with the project's
virtualenv and `pip install import-linter==2.16`. torch-tpu prints start-up
lines on stderr, left out below. Each planted violation is one line to
comment in; comment it out again afterwards.

## 1. No level reaches up

    pip install import-linter==2.16
    PYTHONPATH=src lint-imports

Two `layers` contracts. `scopes-top-down` orders today's packages as stand-ins
for the scopes: `runner` (model), `models`, `layers.adapter` (block),
`layers.core` (family), `kernels`. It lists the 9 upward edges of today as a
baseline: 5 adapters read the model's forward context, and 4 edges run from
kernel modules into layer code. A listed edge that disappears fails the run, so the list
only shrinks. `family-decides-below-executor` orders the inside of
`layers.core`: `fused_moe_gmm` imports `moe_plan`, never the reverse.

```
Scopes import only downward KEPT (9 ignored imports)
The decision module does not import the code that runs its plan KEPT

Contracts: 2 kept, 0 broken.
```

Comment in `from vllm_torchtpu.layers.adapter import fused_moe` (line 42 of
`moe_plan.py`), so the family's decision reaches up into the block scope:

    PYTHONPATH=src lint-imports --contract scopes-top-down

```
Scopes import only downward BROKEN (9 ignored imports)

Contracts: 0 kept, 1 broken.
...
vllm_torchtpu.layers.core is not allowed to import vllm_torchtpu.layers.adapter:

- vllm_torchtpu.layers.core.moe_plan -> vllm_torchtpu.layers.adapter.fused_moe
(l.42)
```

Line 43, `from vllm_torchtpu.layers.core import fused_moe_gmm`, breaks the
second contract the same way:
`vllm_torchtpu.layers.core.moe_plan is not allowed to import vllm_torchtpu.layers.core.fused_moe_gmm`.

## 2. A child accepts only resolved values

`MoeCombinePlan` is marked `@require_no_default`: no field has a default, so
`plan_combine` states every decision, including "no EP kernel version" as an
explicit `None`. After tokamax's `routing_moe` (`common/dataclass_util.py`).
Giving a field a default fails at import:

```
ValueError: MoeCombinePlan.ep_gather_reduce_version must not have a default, so whoever builds it states every decision
```

The check covers the receiving side: every parameter in `moe_plan` and
`fused_moe_gmm` annotated with a resolved type takes exactly that type, with
no default.

    JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/layers/core/test_handoff.py
    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.check_handoff

```
ok  moe_combine(plan: MoeCombinePlan)
violations: 0
```

Comment in `plan: moe_plan.MoeCombinePlan | None = None` (line 584 of
`fused_moe_gmm.py`) and comment out line 581, so the child may be handed
"not decided yet" and decide it again:

```
BAD moe_combine(plan: vllm_torchtpu.layers.core.moe_plan.MoeCombinePlan | None)
violations: 1
```

It reads annotations, not what each caller passes.

## 3. A child does not ignore what it was handed

    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        pytest --run-compile-tier tests/layers/core/test_dead_fields.py
    flock /tmp/vt-tpu.lock env JAX_PLATFORMS=cpu PYTHONPATH=src \
        python -m prototype.check_dead_fields

It loads libtpu, so do not run it beside a TPU job; the lock serializes it
with them. The test carries the `compile_tier` marker, so a default CPU run
skips it. It traces `moe_combine` under every `MoeCombinePlan` value for a v6e
chip that JAX describes without a device, and reports each field whose flips,
with the other field held, never change the traced program. About 9 seconds:

```
plans traced on v6e: 7
fields never read: []
```

Comment in `kernel = ragged_gather_reduce` (line 647 of `fused_moe_gmm.py`),
so the EP combine ignores the version it was handed and uses the import-time
pick:

```
plans traced on v6e: 7
fields never read: ['ep_gather_reduce_version']
```

It compares traced programs on one shape and one chip, so it finds a field the
Python code never reads. A field read but then discarded by the compiler, such
as an alignment a kernel derives again itself, needs the optimized program on
every chip, as the compile check of DATA_CONTRACTS section 5.8 does.
