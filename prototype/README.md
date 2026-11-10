# Prototype: one owner for the MoE combine

Backs DATA_CONTRACTS.md section 6.1. Today the combine at the end of
`moe_gmm` is chosen inline by thirteen conditions, so everyone who adds a
combine, changes a cutoff or brings up a chip edits the same function, and
no one can tell whose calls their edit moves. This branch gives the choice
one owner, `moe_plan.pick_combine`. Each kernel's author states the
kernel's hard limits on their side, the layer describes the call on its
side, and the table of variants between them is the contract; because it
is data, a script lists what each change moves. Run from the repository
root with the project's virtualenv; no TPU is needed. torch-tpu prints
start-up lines on stderr, left out below.

This branch is for reading, not merging. Tests in
`tests/layers/core/test_fused_moe_gmm.py` that patch the internals it removes
(`_select_ragged_gather_reduce`, the inline combine conditions, the
owner-output mode) or expect the old picks are left failing rather than
rewritten; `tests/layers/core/test_moe_plan.py` states the new rules.

## The layer stops deciding

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.combine_plan

Every combine variant in `moe_plan.VARIANTS` answers the same two calls
with the arguments the layer passes it: `run`, and `supports`, the
kernel's own hard limits as a reason or None. `pick_combine` has no
heuristic: it runs the variant that `TPU_MOE_COMBINE` names (read once,
by `resolve_moe_process`), else the TensorCore combine. A named variant
that cannot run a call raises with its kernel's reason; nothing falls back
silently.

```
v6e 16 tokens top-8: tensorcore (default)
    gather_reduce_v3  cannot: the SparseCore path needs 16-lane SparseCores
v6e 512 tokens top-8: tensorcore (default)
    gather_reduce_v3  cannot: the SparseCore path needs 16-lane SparseCores
v6e 2048 tokens top-4: tensorcore (default)
    gather_reduce_v3  cannot: the SparseCore path needs 16-lane SparseCores
tpu7x 16 tokens top-8: tensorcore (default)
tpu7x 512 tokens top-8: tensorcore (default)
tpu7x 2048 tokens top-4: tensorcore (default)
with TPU_MOE_COMBINE=owner_onehot:
tpu7x 16 tokens top-8: owner_onehot (override)
with TPU_MOE_COMBINE=gather_reduce_v3:
v6e 16 tokens top-8: raises ValueError: override gather_reduce_v3 cannot run this call: the SparseCore path needs 16-lane SparseCores
```

The thresholds and cutoffs that chose among kernels are removed, not
ported, so every call runs the TensorCore combine unless the setting names
another. Production would need per-bucket choices, from measurement.

The rules, 10 cases:

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/layers/core/test_moe_plan.py
