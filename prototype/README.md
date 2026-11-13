**Decisions group** (`prototype/decisions/`). Backs DATA_CONTRACTS.md sections 6.1, 6.2, 6.3 and 9.6. Sibling: `prototype/decisions/centralized-moe-plan`.

# Prototype: the MoE combine with one owner, and what each kind of change costs

Today the combine at the end of `moe_gmm` is chosen inline by thirteen
conditions, so everyone who adds a combine, changes a cutoff or brings up a
chip edits the same function, and no one can tell whose calls their edit
moves. This branch gives the choice one owner, `moe_plan.pick_combine`, and
then walks through the three kinds of change the doc measures, one commit
each: one person adds a variant, two people add variants at once, and one
person adds an input to the boundary. Each commit is self-contained: at
each one the package imports, that step's command runs and
`tests/layers/core/test_moe_plan.py` passes.

Run from the repository root with the project's virtualenv; no TPU is
needed. torch-tpu prints start-up lines on stderr, left out below. To see a
step as it was, check out its commit (`git checkout <commit>`); every
command below also runs at the branch head.

## Layout

At the branch head (earlier step commits predate this layout):

| Path | What it holds |
|---|---|
| `src/vllm_torchtpu/layers/core/moe_plan.py` | The planner: `Combine`, `VARIANTS`, `pick_combine` |
| `tests/layers/core/test_moe_plan.py` | Which variants can run a call and which one runs |
| `tests/layers/core/moe_combine_moves.py` | The 48-call grid, and what each call runs under a setting |
| `tests/layers/core/moe_parallel_changes.py` | The two parallel changes: touched files, conflict hunks, the plan at each commit |
| `tests/layers/core/test_moe_combine_moves.py` | Asserts the tables of steps 2, 3 and 4 below |
| `prototype/*.py` | Demos that print those tables from the `tests/` helpers above |

This branch is for reading, not merging. Tests in
`tests/layers/core/test_fused_moe_gmm.py` that patch the internals the
first step removes (`_select_ragged_gather_reduce`, the inline combine
conditions, the owner-output mode) are left failing rather than rewritten.

## Steps

| Step | Commit | Doc | What to look at |
|---|---|---|---|
| 1. The layer stops choosing | [`90c51f172`](https://github.com/kyuyeunk/vllm-torchtpu/commit/90c51f172f7fb3b544a79dee33c4e7187c333c44), then [`7e9a3751f`](https://github.com/kyuyeunk/vllm-torchtpu/commit/7e9a3751f28ed8bebc261034b89a684752a99e2a) (the GMM output's row count is derived, not restated) | 6.1 | `git show 90c51f172` |
| 2. One person adds a variant | [`28cf85a1b`](https://github.com/kyuyeunk/vllm-torchtpu/commit/28cf85a1b183efca687994b731b245934b1d2b08) | 6.2 | `git show 28cf85a1b`: one enum member, one entry, the kernel |
| 3. Two people add variants at once | person A is step 2; person B is [`b59cd432d`](https://github.com/kyuyeunk/vllm-torchtpu/commit/b59cd432d84fda9278155d8fa79444d4db7cffbe), from the same base; their merge is [`6e81518d8`](https://github.com/kyuyeunk/vllm-torchtpu/commit/6e81518d82fd946729ddd8f1118b9160a1693404); the script is [`732d902c2`](https://github.com/kyuyeunk/vllm-torchtpu/commit/732d902c24b9948738feda7afb13ddcb2eb746db) | 9.6 | the merge's conflict, in one table |
| 4. One person adds a boundary input | [`66b2f99b0`](https://github.com/kyuyeunk/vllm-torchtpu/commit/66b2f99b02e013f9cb8c08a8cd30fa041094ead6) | 6.3 | `git show 66b2f99b0`: a defaulted field, read only by the variant that needs it |

The history, from origin/main 101260f1f:

    * [Prototype] Add a boundary input as a defaulted field            (step 4)
    * [Prototype] Show what two parallel changes to the combine table move
    *   Merge the segment-sum variant into the gather_reduce_v4 change (step 3)
    |\
    | * [Prototype] Add a segment-sum combine variant                  (person B)
    * | [Prototype] Add a combine variant as one entry                 (step 2, person A)
    |/
    * [Prototype] Stop restating the GMM output's row count
    * [Prototype] Give the MoE combine one owner                       (step 1)

## Where to look

| Step | Claim | Code |
|---|---|---|
| 1 | The layer describes the call and does not decide | [`moe_combine`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/fused_moe_gmm.py#L544-L596) |
| 1 | The variant table is the contract | [`Combine`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/moe_plan.py#L47-L63), [`Variant`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/moe_plan.py#L250-L262), [`VARIANTS`, `DEFAULT`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/moe_plan.py#L265-L285) |
| 1 | Override, else the default; an override that cannot run raises | [`pick_combine`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/moe_plan.py#L323-L350) |
| 1 | Why each variant declines a call | [`combine_capable`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/moe_plan.py#L288-L298) |
| 1 | The environment is read once | [`resolve_moe_process`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/layers/core/moe_plan.py#L88-L118) |
| 1 | The kernel's author owns its limits | [`ragged_gather_reduce_supports`](https://github.com/kyuyeunk/vllm-torchtpu/blob/7e9a3751f28ed8bebc261034b89a684752a99e2a/src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_v3.py#L114-L144) (v3) |
| 2 | One enum member | [`Combine`](https://github.com/kyuyeunk/vllm-torchtpu/blob/28cf85a1b183efca687994b731b245934b1d2b08/src/vllm_torchtpu/layers/core/moe_plan.py#L48-L66) |
| 2 | One entry | [`VARIANTS[Combine.GATHER_REDUCE_V4]`](https://github.com/kyuyeunk/vllm-torchtpu/blob/28cf85a1b183efca687994b731b245934b1d2b08/src/vllm_torchtpu/layers/core/moe_plan.py#L287-L291) in [`VARIANTS`](https://github.com/kyuyeunk/vllm-torchtpu/blob/28cf85a1b183efca687994b731b245934b1d2b08/src/vllm_torchtpu/layers/core/moe_plan.py#L268-L292) |
| 2 | The kernel declares its limits | [`ragged_gather_reduce_supports`](https://github.com/kyuyeunk/vllm-torchtpu/blob/28cf85a1b183efca687994b731b245934b1d2b08/src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_v4.py#L42-L57) (v4) |
| 3 | Both changes meet in one table | [`Combine`](https://github.com/kyuyeunk/vllm-torchtpu/blob/6e81518d82fd946729ddd8f1118b9160a1693404/src/vllm_torchtpu/layers/core/moe_plan.py#L48-L68), [`VARIANTS`](https://github.com/kyuyeunk/vllm-torchtpu/blob/6e81518d82fd946729ddd8f1118b9160a1693404/src/vllm_torchtpu/layers/core/moe_plan.py#L293-L318) |
| 3 | B's variant merges cleanly | [`_segment_sum`, `_segment_sum_supports`](https://github.com/kyuyeunk/vllm-torchtpu/blob/6e81518d82fd946729ddd8f1118b9160a1693404/src/vllm_torchtpu/layers/core/moe_plan.py#L218-L238) |
| 3 | The conflict hunks; the plan at each commit; the overlap report | At the script's commit: [`_conflicts`](https://github.com/kyuyeunk/vllm-torchtpu/blob/732d902c24b9948738feda7afb13ddcb2eb746db/prototype/parallel_changes.py#L132-L146), [`_plan_at`](https://github.com/kyuyeunk/vllm-torchtpu/blob/732d902c24b9948738feda7afb13ddcb2eb746db/prototype/parallel_changes.py#L111-L129), [`main`](https://github.com/kyuyeunk/vllm-torchtpu/blob/732d902c24b9948738feda7afb13ddcb2eb746db/prototype/parallel_changes.py#L165-L222). At the branch head: `_conflicts` and `_plan_at` in `tests/layers/core/moe_parallel_changes.py` (L153-L167, L124-L150), the report in `prototype/parallel_changes.py` (`main`, L56-L94) |
| 4 | The new input is a defaulted field | [`GmmOut`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/layers/core/moe_plan.py#L145-L163) |
| 4 | Only the variant that reads it receives it | [`Variant`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/layers/core/moe_plan.py#L321-L336), [`VARIANTS[Combine.SKIP_EMPTY]`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/layers/core/moe_plan.py#L364-L369), [`_optional_inputs`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/layers/core/moe_plan.py#L389-L395) |
| 4 | The variant's adapters | [`_skip_empty`, `_skip_empty_supports`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/layers/core/moe_plan.py#L286-L318) |
| 4 | One call site fills it | [`moe_combine`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/layers/core/fused_moe_gmm.py#L544-L606) |
| 4 | The kernel declines without the counts | [`ragged_gather_reduce_supports`](https://github.com/kyuyeunk/vllm-torchtpu/blob/66b2f99b02e013f9cb8c08a8cd30fa041094ead6/src/vllm_torchtpu/kernels/sparse_core/ragged_gather_reduce_skip_empty.py#L44-L61) (skip-empty) |

## Step 1: the layer stops choosing (section 6.1)

    git checkout 7e9a3751f
    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.combine_plan

Every combine variant in `moe_plan.VARIANTS` answers the same two calls
with the arguments the layer passes it: `run`, and `supports`, the
kernel's own hard limits as a reason or None. `pick_combine` has no
heuristic: it runs the variant that `TPU_MOE_COMBINE` names (read once, by
`resolve_moe_process`), else the TensorCore combine. A named variant that
cannot run a call raises with its kernel's reason; nothing falls back
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

At later commits the same command also lists the variants those steps add.
The thresholds and cutoffs that chose among kernels are removed, not
ported, so every call runs the TensorCore combine unless the setting names
another. Production would need per-bucket choices, from measurement.

## Step 2: one person adds a variant (section 6.2)

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.add_variant_moves

The hypothetical `gather_reduce_v4` is one member of the `Combine` enum
and one entry in `moe_plan.VARIANTS`, built from its kernel's own
`supports` (top-k up to 4, a SparseCore). The layer does not change, and no
setting is added. Without a setting nothing moves; with
`TPU_MOE_COMBINE=gather_reduce_v4`, the variant runs every call its kernel
supports, and every other call raises with the kernel's reason instead of
falling back. The kernel is a stub that delegates to v2. The command picks
the combine for 48 calls on v6e and TPU7x:

```
no setting: tensorcore on 48 of 48 calls
TPU_MOE_COMBINE=gather_reduce_v4: applies to 36 of 48 calls
  tpu7x  raises: top-k above 4: 6
  tpu7x  tensorcore -> gather_reduce_v4: 18
  v6e    raises: top-k above 4: 6
  v6e    tensorcore -> gather_reduce_v4: 18
```

## Step 3: two people add variants at once (section 9.6)

Person A is step 2. Person B, from the same base, adds `segment_sum`, an
XLA scatter-add. The merge's conflict reproduces from the history:

    git checkout --detach 28cf85a1b   # person A
    git merge b59cd432d               # person B: the conflict
    git merge --abort

```
Auto-merging src/vllm_torchtpu/layers/core/moe_plan.py
CONFLICT (content): Merge conflict in src/vllm_torchtpu/layers/core/moe_plan.py
Auto-merging tests/layers/core/test_moe_plan.py
Automatic merge failed; fix conflicts and then commit the result.
```

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.parallel_changes

For the base, A, B and the merge, the script extracts that commit's `src/`,
plans 48 calls under every `TPU_MOE_COMBINE` setting, and compares:

```
base 7e9a3751f, A 28cf85a1b, B b59cd432d, merge 6e81518d8
A touched: prototype/add_variant_moves.py, kernels/sparse_core/ragged_gather_reduce_v4.py, layers/core/moe_plan.py, tests/layers/core/test_moe_plan.py
B touched: layers/core/moe_plan.py, tests/layers/core/test_moe_plan.py
both touched: layers/core/moe_plan.py, tests/layers/core/test_moe_plan.py
git merge: 1 conflicted file(s)
  layers/core/moe_plan.py, 3 hunks, ours  |  theirs:
    - `gather_reduce_v4`: a SparseCore gather-reduce for top-k up to 4.  |  - `segment_sum`: an XLA scatter-add of the weighted rows, in expert order.
    GATHER_REDUCE_V4 = "gather_reduce_v4"  |  SEGMENT_SUM = "segment_sum"
    Combine.GATHER_REDUCE_V4: Variant(  |  Combine.SEGMENT_SUM: Variant(run=_segment_sum, supports=_segment_sum_supports),
48 calls: v6e and TPU7x, 6 token counts, top-k (1, 2, 4, 8)
same under base, A, B and A+B: no setting (tensorcore on 48) and 6 of 6 existing settings; changed: none
A's TPU_MOE_COMBINE=gather_reduce_v4: moves 36, raises on 12 (top-k above 4); same under A+B
B's TPU_MOE_COMBINE=segment_sum: moves 48, raises on 0; same under A+B
calls both new variants run, the overlap the merger reviews: 36
  v6e    top-k [1, 2, 4] x tokens [16, 64, 256, 512, 2048, 8192]
  tpu7x  top-k [1, 2, 4] x tokens [16, 64, 256, 512, 2048, 8192]
```

- **The conflict is in one table.** git stops in `moe_plan.py` only, in the
  three places both people appended an entry: the `Combine` enum's
  docstring, its members and `VARIANTS`. Resolving it means keeping both
  entries.
- **Outside it, nothing collides.** A's kernel file and B's variant
  function merge cleanly, and so do both new cases in `test_moe_plan.py`.
  No existing setting changes under A, B or the merge.
- **The overlap list is what the merger reviews.** The 36 calls both new
  variants can run are where a later per-bucket choice must pick between
  them; the other 12 (top-k 8) only `segment_sum` can run.

## Step 4: one person adds a boundary input (section 6.3)

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.boundary_input_moves

A combine that needs an input the boundary does not carry adds a defaulted
field to the boundary's description, `GmmOut.token_local_routes_shape`, and
a defaulted argument to `moe_combine`, passed only to variants that read
it. Callers that do not pass it and the other variants' authors change
nothing and keep what they run. The command picks the combine for 48 calls
on v6e and TPU7x, without and with the field filled the way routing would,
and without and with the setting naming the variant:

```
48 calls on v6e and TPU7x
  no setting, without the counts: tensorcore: 48
  TPU_MOE_COMBINE=skip_empty, without the counts: raises: needs routing's per-token count of local routes: 48
  no setting, with the counts: tensorcore: 48
  TPU_MOE_COMBINE=skip_empty, with the counts: runs skip_empty: 48
```

Production would also need routing to emit the counts; the kernel is a
stub.

## The rules

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/layers/core/test_moe_plan.py

10 cases at step 1, 11 at step 2, 12 after the merge, 14 at step 4; all
pass at each step.

    JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/layers/core/test_moe_combine_moves.py

At the branch head only: 8 cases asserting the tables of steps 2, 3 and 4
above, so a change that moves a call fails a test, not just a demo. The
step 3 cases read git history and need the merge as the last merge in
HEAD's history. All of these run on CPU.
