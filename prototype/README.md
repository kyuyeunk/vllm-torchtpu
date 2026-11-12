**Scopes group** (`prototype/scopes/`). Backs DATA_CONTRACTS.md section 5.3. Siblings: `prototype/scopes/attention-layer-plan`, `prototype/scopes/declared-blocks`.

# Prototype: one declaration line gives a model #3096's attention fusion

PR #3096 (open) fuses a decoder layer's input RMSNorm, the residual add and
attention's QKV projection into one Pallas kernel (`fused_qkv`). It reaches
each model through its own patch: a rewritten forward per model family, plus
eligibility checks written against attribute names. This branch shows the
same fusion through a declared scope:

1. **The framework publishes a pattern** for the pre-norm attention
   sub-block, `norm_then_attention`: `input_norm(x, residual)`, then
   attention on the normed tokens, with the rest of the layer (second norm,
   MLP or MoE) opaque.
2. **A model owner declares it in one call**, in the model's own
   `registration.py`:
   `declare_block(MistralDecoderLayer, pattern=norm_then_attention, roles=PRE_NORM_ROLES)`.
3. **The framework verifies the declaration** on the meta device: the roles
   are the right kinds; with the roles opaque, the layer's forward computes
   the pattern's graph (so the normed tokens reach nothing but attention);
   and attention's input reaches nothing but its QKV projection. A mismatch
   is declined at install with the edge and line, and fails the check.
4. **One fused value then serves every declared model.** `fused_qkv`
   rewires two roles: the input norm becomes `AddOnly` (add the residual) and
   the QKV projection becomes `FusedNormProjection` (normalize and project in
   one call, through the projection's own matmul). The layer's forward is
   untouched, so nothing names a model.

The fused value here is a plain-torch reference standing in for #3096's
kernel; section B below routes #3096's kernel itself behind it.

## Where to look

| Claim | Code |
|---|---|
| The pattern (framework-owned) | [`norm_then_attention`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L48-L61), [`PRE_NORM_ROLES`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L39-L45) |
| The role kinds | [`check_roles`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L109-L127) |
| The conformance check | [`conform`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L79-L233); attention's input reaches only its projection at [L167-L219](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L167-L219) |
| The plan (`unfused` or `fused_qkv`) | [`NormAttnPlan`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L37-L43), [`resolve`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L46-L66) |
| The one fused value: two rewired roles | [`AddOnly`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L37-L50), [`FusedNormProjection`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L53-L74), [`install_fused_qkv`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L77-L86) |
| Install, declining a mismatch | [`NormThenAttention.install`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L248-L264) |
| Scopes nest: several patterns per class | [`declarations_for`](../src/vllm_torchtpu/blocks/__init__.py#L84-L86) |
| One call per model | [Llama](../src/vllm_torchtpu/models/vllm/llama/registration.py#L21), [Mistral](../src/vllm_torchtpu/models/vllm/mistral/registration.py#L21), [Qwen3](../src/vllm_torchtpu/models/vllm/qwen3/registration.py#L21), [gpt-oss](../src/vllm_torchtpu/models/vllm/gpt_oss/registration.py#L21-L25), [Qwen3-MoE](../src/vllm_torchtpu/models/vllm/qwen3_moe/registration.py#L21), [Qwen3.5](../src/vllm_torchtpu/models/vllm/qwen3_5/registration.py#L43-L48) |
| Per model: conforms, fused value matches upstream | [`family`](check_subblock.py#L152-L180), [`new_models`](check_subblock.py#L182-L190) |
| The planted mismatch | [`ScaledAttention`](check_subblock.py#L84-L92), [`planted`](check_subblock.py#L193-L220) |
| The count against #3096 | [`PR_3096`](check_subblock.py#L224-L247) |
| Reach over current models, from real configs | [`CANDIDATES`](subblock_reach.py#L61-L191), [`check`](subblock_reach.py#L266-L294) |
| The startup override | `TPU_SUBBLOCK_PLAN` in `envs.py`, passed by the runner's `_install_blocks` |

## Running it

On a CPU against vLLM v0.33.0 (`<vllm>`); nothing loads libtpu.

```bash
export PYTHONPATH=src:.:<vllm>
python -m prototype.check_subblock
python -m prototype.subblock_reach      # fetches config.json files only
TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu python -m pytest tests/blocks
```

### `prototype.check_subblock` (vLLM and transformers notices omitted)

```text
1 each family: declared -> conforms -> fused_qkv, against upstream
    Llama (dense)                  declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen3-MoE                      declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen3.5-MoE (full attention)   declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
  -> PASS
2 current models #3096 does not target, real configs
    Mistral (mistralai/Ministral-8B-Instruct-2410), not a #3096 target: declaration 1 call; per-model code 0
    mistralai/Ministral-8B-Instruct-2410 declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
    OpenAI (openai/gpt-oss-20b), not a #3096 target: declaration 1 call; per-model code 0
    openai/gpt-oss-20b             declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen (Qwen/Qwen3-32B), not a #3096 target: declaration 1 call; per-model code 0
    Qwen/Qwen3-32B                 declared -> conforms -> installed; input_proj=fused_qkv (rule); fused calls 1; max |fused - upstream| 0.0e+00
  -> PASS
3 a layer that does not conform is declined
    ScaledAttention: declined: declared norm_then_attention, but extra op aten.mul between attn and post_norm at check_subblock.py:90
  -> PASS
4 the same fusion as #3096, by count
    #3096 per-model forwards: 4
    #3096 structural eligibility checks (here: conformance): 5
    #3096 kernel limits (here: the primitive's supports(), not this check): 7
    here: 1 plan value (FusedNormProjection), 6 declarations, 1 conformance check (norm_attn.conform)
  -> PASS
```

- **The comparison.** Each layer is built through its own constructor on a
  CPU in float32 with random weights: the first three from tiny configs, the
  next three from their real `config.json` with sizes shrunk. It runs
  upstream, then with `fused_qkv` installed; "fused calls 1" shows the fused
  path ran. The attention backend and the MLP (dense or MoE) need TPU or
  CUDA kernels, so both are replaced by the same stand-ins in every run
  (causal attention on q, k, v; `tanh`); everything else is the model's own
  code: norms, residual, projections, q/k norms, RoPE, Qwen3.5's gate,
  gpt-oss's attention, the output projection. The difference is exactly 0:
  in float32 the fused value does the same arithmetic.
- **CI.** Comment in `PLANT_MISMATCH = True` and the command exits 1 with
  `FAIL declared ScaledAttention (planted) does not conform`.
- **pytest.** `10 passed` (5 for this sub-block, 5 for `attn_then_moe`).

### `prototype.subblock_reach`

Only each model's `config.json` is fetched; structural fields are kept
(q/k norms, partial RoPE, biases, sliding windows, layer types, MLA ranks),
sizes shrunk, and the layer built with vLLM v0.33.0's own class:

```text
  MiniMax                MiniMaxAI/MiniMax-M2.5                         MiniMaxM2DecoderLayer                  conforms
  GLM (Zhipu)            zai-org/GLM-4.6                                Glm4MoeDecoderLayer                    conforms
  GLM (Zhipu)            zai-org/GLM-4.5-Air                            Glm4MoeDecoderLayer                    conforms
  Mistral                mistralai/Ministral-8B-Instruct-2410           MistralDecoderLayer                    conforms
  Mistral                mistralai/Mistral-Small-3.2-24B-Instruct-2506  MistralDecoderLayer                    conforms
  OpenAI                 openai/gpt-oss-20b                             TransformerBlock                       conforms
  Meta                   meta-llama/Llama-3.3-70B-Instruct              LlamaDecoderLayer                      conforms
  Meta                   meta-llama/Llama-4-Scout-17B-16E-Instruct      Llama4DecoderLayer                     conforms
  Microsoft              microsoft/phi-4                                LlamaDecoderLayer                      conforms
  IBM                    ibm-granite/granite-4.0-h-small                GraniteMoeHybridAttentionDecoderLayer  not buildable        StrictDataclassClassValidationError: Class validation error for granitemoehybrid (layer types)
  Baidu                  baidu/ERNIE-4.5-21B-A3B-PT                     Ernie4_5_MoeDecoderLayer               conforms
  LG                     LGAI-EXAONE/EXAONE-4.0-32B                     Exaone4DecoderLayer                    declines             in_norm: the layer has none of ('input_layernorm',)
  DeepSeek               deepseek-ai/DeepSeek-V3.1                      DeepseekV2DecoderLayer                 declines             attn=self_attn has 0 QKVParallelLinear from the hidden size 64, not one
  Moonshot               moonshotai/Kimi-K2-Instruct                    DeepseekV2DecoderLayer                 declines             attn=self_attn has 0 QKVParallelLinear from the hidden size 64, not one
  Qwen (#3096 target)    Qwen/Qwen3-30B-A3B                             Qwen3MoeDecoderLayer                   conforms      #3096
  Qwen                   Qwen/Qwen3-32B                                 Qwen3DecoderLayer                      conforms
  Qwen (#3096 target)    Qwen/Qwen3.5-35B-A3B                           Qwen3_5DecoderLayer                    conforms      #3096
  Google                 google/gemma-3-27b-it                          Gemma3DecoderLayer                     declines             role post_norm runs after attn at gemma3.py:269, where the pattern does not call it
  Google (#3096 target)  google/gemma-4-31b-it                          Gemma4DecoderLayer                     declines      #3096  the forward does not export: 'GateLinear' runs outside the bound roles
  Cohere                 CohereForAI/c4ai-command-r-v01                 CohereDecoderLayer                     not buildable        GatedRepoError: 403 (gated repo)
  conform: 13 of 20; not a #3096 target: 11 (Baidu, GLM (Zhipu), Meta, Microsoft, MiniMax, Mistral, OpenAI, Qwen)
```

- **11 conforming layer types belong to models #3096 does not target**,
  across MiniMax, GLM, Mistral, OpenAI gpt-oss, Meta Llama 3 and 4,
  Microsoft Phi-4, Baidu ERNIE 4.5 and dense Qwen3. Each needs one
  declaration call.
- **The declines are the pattern's real limits**, named at the edge: MLA
  (DeepSeek, Kimi K2) has no QKV projection; Gemma 3's norm after attention
  and EXAONE 4's post-norm layout are not pre-norm; Gemma 4 runs a router
  outside the bound roles (and #3096 itself reaches Gemma 4 through its
  attention, not the decoder layer).
- **A static scan of vLLM main** (`1ecc9cc4ce`, 2026-11-07; script and JSON
  under `static-op-args/subblock/`) finds 89 model files whose attention
  runs `qkv_proj -> attn`, 29 with a plain pre-norm decoder forward, and one
  attention forward body copied verbatim into 17 classes (Llama's, in
  DeepSeek, ERNIE, Gemma, GLM-4, Mixtral, Qwen2-MoE and others): the same
  sub-block implemented again and again.

## On a TPU

TPU_RESULTS_PLACEHOLDER

## The count against #3096

Counted from #3096's diff, file by file:

| #3096 | Here |
|---|---|
| 4 per-model forwards: `fused_attn_proj_patch.py`'s `_patched_qwen3_moe_forward`, `_patched_qwen3_next_forward` (Qwen3-Next and Qwen3.5) and `_patched_gemma4_attention_forward`, plus an edit to #1492's `_overlap_forward` in `qwen3_5_patch.py` | No forward: two rewired roles |
| 5 structural eligibility checks in `layers/adapter/fused_attn_proj.py`: unsupported norm, sequence-parallel input, `layer_scale`, GDN's separate in-projections, first layer without a residual | 1 conformance check, the same for every model |
| 7 kernel limits (quant method, weight layout, FP8 per channel, batched RPA, head_dim 64 or DCP, KV sharing or sinks, GDN in-projection not FP8) | The kernel's own `supports(...)` in the doc's design; not this check |
| Finding layers by attribute names (`input_layernorm`, `self_attn`, `linear_attn`) and `isinstance` on model classes | One declaration call per model, in the model's own directory |

#3096 is careful work; the count is about where the knowledge lives, not
about the PR. It also covers Gemma 4 and Gated DeltaNet layers, which this
pattern does not.

## How it relates to #1492's block

`norm_then_attention` is the front half of the skeleton `attn_then_moe`
checks (norm, attention, norm, MoE), so the two plans nest by scope: the
sub-block's plan (`input_proj`) inside the block's plan (`collectives`).
Qwen3.5 declares both. Because `fused_qkv` rewires roles instead of
replacing the forward, the block's forward picks it up through the same
roles.

## Not shown here

- **What the kernel also fuses.** #3096 fuses q/k norm, RoPE, attention and
  the output projection too; the reference fuses norm and projection only.
- **Gemma 4 and Gated DeltaNet**, each its own pattern or a decline.
- **TP > 1 numerics on the fused path, quantized projections, and the
  residual tiling pin** (a contract on the residual stream between blocks,
  at model scope).
- **Production** would also need the plan in the per-bucket table and a
  numerics tier against the residual stream's tolerance.
