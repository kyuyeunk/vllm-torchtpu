**Scopes group** (`prototype/scopes/`). Backs DATA_CONTRACTS.md section 5.3. Siblings: `prototype/scopes/attention-layer-plan`, `prototype/scopes/declared-blocks`.

# Prototype: one attention fusion for every model that declares the sub-block

PR #3096 (open) fuses a decoder layer's input RMSNorm, the residual add and
attention's QKV projection into one Pallas kernel (`fused_qkv`). It reaches
each model through its own patch: a rewritten forward per model family, plus
eligibility checks written against attribute names. This branch shows the
same fusion through a declared scope instead:

1. **The framework publishes a pattern** for the pre-norm attention
   sub-block, `norm_then_attention`: `input_norm(x, residual)`, then
   attention on the normed tokens, with the rest of the layer (second norm,
   MLP or MoE) opaque.
2. **Each model owner declares it once**, one `declare_block` call in the
   model's own `registration.py`.
3. **The framework verifies the declaration** on the meta device: the roles
   are the right kinds, the layer's forward computes the pattern's graph with
   the roles opaque, and attention hands its input straight to its QKV
   projection. A mismatch is declined at install with the edge and line, and
   fails the check.
4. **One fused value then serves every declared model.** `fused_qkv` is
   written once, against the pattern's roles: one function adds, normalizes
   and projects; attention then runs its own forward on the projected tokens.
   No model is named.

The fused value here is a plain-torch reference standing in for #3096's
kernel. Its point is the route, not the speed.

## Where to look

| Claim | Code |
|---|---|
| The pattern (framework-owned) | [`norm_then_attention`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L40-L53) |
| The role kinds | [`check_roles`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L101-L119) |
| The conformance check | [`conform`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L78-L201), with the handoff probe at [L166-L186](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L166-L186) |
| The plan (`unfused` or `fused_qkv`) | [`NormAttnPlan`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L37-L43), [`resolve`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L46-L66) |
| The one fused value | [`fused_norm_add_qkv`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L37-L51) |
| The forward that runs it, through roles | [`NormThenAttentionBlock.forward`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L82-L107) |
| Install, declining a mismatch | [`NormThenAttention.install`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L213-L237) |
| Scopes nest: several patterns per class | [`declarations_for`](../src/vllm_torchtpu/blocks/__init__.py#L84-L87) |
| The three declarations | [Llama](../src/vllm_torchtpu/models/vllm/llama/registration.py#L21-L30), [Qwen3-MoE](../src/vllm_torchtpu/models/vllm/qwen3_moe/registration.py#L21-L30), [Qwen3.5](../src/vllm_torchtpu/models/vllm/qwen3_5/registration.py#L43-L54) |
| Per family: conforms, fused value matches upstream | [`family`](check_subblock.py#L129-L158) |
| The planted mismatch | [`ScaledAttention`](check_subblock.py#L64-L72), [`planted`](check_subblock.py#L161-L188) |
| The count against #3096 | [`PR_3096`](check_subblock.py#L192-L215) |

## Running it

On a CPU against vLLM v0.33.0 (`<vllm>`); nothing loads libtpu.

```bash
export PYTHONPATH=src:.:<vllm>
python -m prototype.check_subblock
TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu python -m pytest tests/blocks
```

`prototype.check_subblock` (vLLM's import-time notices omitted):

```text
1 each family: declared -> conforms -> fused_qkv, against upstream
    Llama (dense)                  declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen3-MoE                      declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen3.5-MoE (full attention)   declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
  -> PASS
2 a layer that does not conform is declined
    ScaledAttention: declined: declared norm_then_attention, but extra op aten.mul between attn and post_norm at check_subblock.py:70
  -> PASS
3 the same fusion as #3096, by count
    #3096 per-model forwards: 4
    #3096 structural eligibility checks (here: conformance): 5
    #3096 kernel limits (here: the primitive's supports(), not this check): 7
    here: 1 plan value (fused_norm_add_qkv), 3 declarations, 1 conformance check (norm_attn.conform)
  -> PASS
```

- **The comparison.** Each family's layer is built through its own
  constructor on a CPU in float32 with random weights, and run upstream, then
  with `fused_qkv` installed. "fused calls 1" shows the fused path ran.
  The attention backend and the MLP (dense or MoE) need TPU or CUDA kernels,
  so both are replaced by the same stand-ins in every run (causal attention
  on q, k, v; `tanh`); everything else is the model's own code: the norms,
  the residual, the projections, q/k norms, RoPE, Qwen3.5's gate and the
  output projection. The difference is exactly 0 because, in float32, the
  fused value does the same arithmetic in one function.
- **CI.** Comment in `PLANT_MISMATCH = True` and the command exits 1 with
  `FAIL declared ScaledAttention (planted) does not conform`.
- **pytest.** `10 passed` (5 for this sub-block, 5 for `attn_then_moe`).

## The count against #3096

Counted from #3096's diff, file by file:

| #3096 | Here |
|---|---|
| 4 per-model forwards: `fused_attn_proj_patch.py`'s `_patched_qwen3_moe_forward`, `_patched_qwen3_next_forward` (Qwen3-Next and Qwen3.5) and `_patched_gemma4_attention_forward`, plus an edit to #1492's `_overlap_forward` in `qwen3_5_patch.py` | 1 forward, `NormThenAttentionBlock.forward`, through roles |
| 5 structural eligibility checks in `layers/adapter/fused_attn_proj.py`: unsupported norm, sequence-parallel input, `layer_scale`, GDN's separate in-projections, first layer without a residual | 1 conformance check, the same for every model |
| 7 kernel limits (quant method, weight layout, FP8 per channel, batched RPA, head_dim 64 or DCP, KV sharing or sinks, GDN in-projection not FP8) | The kernel's own `supports(...)` in the doc's design; not this check |
| Finding layers by attribute names (`input_layernorm`, `self_attn`, `linear_attn`) and `isinstance` on model classes | 3 declarations, one per family, in the model's own directory |

#3096 is careful work; the count is about where the knowledge lives, not
about the PR. It covers Gemma 4 and Gated DeltaNet layers too, which this
branch does not (see below).

## How it relates to #1492's block

`norm_then_attention` is the front half of the skeleton `attn_then_moe`
checks (norm, attention, norm, MoE), so the two plans nest by scope: the
sub-block's plan (`input_proj`) inside the block's plan (`collectives`).
Qwen3.5 declares both. Where both plans are non-default, the block's forward
would call the sub-block's front half; that composition is not built here,
and install reports "outer block owns the forward" instead of installing.

## Not shown here

- **Speed.** The fused value is a plain-torch reference; #3096's Pallas
  kernel is the implementation it stands in for.
- **What the kernel also fuses.** #3096 fuses q/k norm and RoPE and writes
  the attention kernel's layout directly; here attention runs those itself.
- **Gemma 4 and Gated DeltaNet.** Gemma 4 adds the residual after its second
  norm and its input norm takes no residual; GDN's in-projections differ.
  Each would be its own pattern or a decline with a reason.
- **TP > 1, quantized weights, the residual tiling pin** (a contract on the
  residual stream between blocks, at model scope).
- **Production** would also need the composed forward with `attn_then_moe`,
  the plan in the startup override path, and a numerics tier on TPU.
