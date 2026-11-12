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
   `declare_block(Glm4MoeDecoderLayer, pattern=norm_then_attention, roles=PRE_NORM_ROLES)`.
3. **The framework verifies the declaration**: the roles are the right
   kinds; with the roles opaque, the layer's forward computes the pattern's
   graph (so the normed tokens reach nothing but attention); and attention's
   input reaches nothing but its QKV projection. **A declaration that does
   not conform stops startup**, naming the edge and line: it is the model
   owner's explicit claim.
4. **One fused value then serves every declared model.** `fused_qkv`
   rewires two roles: the input norm becomes `AddOnly` (add the residual) and
   the QKV projection becomes `FusedNormProjection` (the norm's own op, then
   the projection's own matmul, in one module). The layer's forward is
   untouched, so nothing names a model.
5. **Capability flows up.** `fused_qkv` stands for #3096's kernel, so it
   declares that kernel's capability (`fused_qkv_supports`, below) and asks
   the bound attention role. Where the capability does not cover a
   conforming layer, the rule's pick falls back to `unfused` quietly; a user
   pin to `fused_qkv` that cannot run stops startup.

The fused value here is a plain-torch reference standing in for #3096's
kernel; "On a TPU" below also routes #3096's kernel itself behind it (a
local scratch build, not on this branch).

## Where to look

| Claim | Code |
|---|---|
| The pattern (framework-owned) | [`norm_then_attention`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L48-L61), [`PRE_NORM_ROLES`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L39-L45) |
| The role kinds | [`check_roles`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L109-L127) |
| The conformance check | [`conform`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L80-L243); attention's input reaches only its projection at [L175-L229](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L175-L229) |
| A non-conforming declaration stops startup | [`install`](../src/vllm_torchtpu/blocks/__init__.py#L162-L199) |
| The plan and the kernel's capability | [`NormAttnPlan`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L47-L54), [`rule` and fallbacks](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L62-L70), [`fused_qkv_supports`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L73-L142), [`resolve`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L145-L164); the chip fact [`has_fp8_matmul`](../src/vllm_torchtpu/tpu_info.py#L90-L99) |
| The plan consults the capability; automatic falls back, a pin that cannot run raises | [`NormThenAttention.install`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L263-L300) |
| The one fused value: two rewired roles | [`AddOnly`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L35-L49), [`FusedNormProjection`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L52-L70), [`install_fused_qkv`](../src/vllm_torchtpu/blocks/norm_attn/forward.py#L73-L82) |
| Scopes nest: several patterns per class | [`declarations_for`](../src/vllm_torchtpu/blocks/__init__.py#L84-L86) |
| One call per model | [GLM-4.5/4.6](../src/vllm_torchtpu/models/vllm/glm4_moe/registration.py#L21), [Mistral](../src/vllm_torchtpu/models/vllm/mistral/registration.py#L21), [Llama](../src/vllm_torchtpu/models/vllm/llama/registration.py#L21), [Qwen3](../src/vllm_torchtpu/models/vllm/qwen3/registration.py#L21), [gpt-oss](../src/vllm_torchtpu/models/vllm/gpt_oss/registration.py#L21-L25), [Qwen3-MoE](../src/vllm_torchtpu/models/vllm/qwen3_moe/registration.py#L21), [Qwen3.5](../src/vllm_torchtpu/models/vllm/qwen3_5/registration.py#L43-L48) |
| Per model: conforms, fused value matches upstream | [`family`](check_subblock.py#L171-L203), [`new_models`](check_subblock.py#L206-L214) |
| The planted non-conforming declaration | [`ScaledAttention`](check_subblock.py#L103-L111), [`planted`](check_subblock.py#L217-L243) |
| The count against #3096 | [`PR_3096`](check_subblock.py#L248-L271) |
| Reach over current models, from real configs | [`CANDIDATES`](subblock_reach.py#L62-L192), [`MAIN_CANDIDATES`](subblock_reach.py#L198-L254), [`check`](subblock_reach.py#L332-L358) |
| The startup override | `TPU_SUBBLOCK_PLAN` in `envs.py`, passed by the runner's `_install_blocks` |

## The kernel's capability, as its author would declare it

`fused_qkv_supports(attention, matmul=, chip=)` returns why #3096's fused_qkv cannot serve an
attention role, or None. The chip is the third description, after the model's and the kernel's.
Each condition is read from #3096's code (paths in
its diff, `d2d5ded6e`; "adapter" is `layers/adapter/fused_attn_proj.py`):

| Condition | #3096 | Checked where |
|---|---|---|
| No attention sinks (RPA v3 serves them; fused_qkv writes the batched-RPA layout) | adapter:246 | attention role |
| No KV sharing | adapter:246 | attention role |
| head_dim not 64 | adapter:244 | attention role |
| head_dim a multiple of 32 (8 sublanes x 4 FP8 values) | gqa/config.py:147 | attention role |
| An even number of query heads per KV head (bf16 Q packs 2) | gqa/config.py:146 | attention role |
| An output gate only with head_dim a multiple of 128 | gqa/config.py:149 | attention role |
| q/k norms per head: RMSNorm, (1 + w) or weightless | adapter:74-82 | attention role |
| NeoX-style RoPE (partial RoPE fine) | common/model_config.py:59-71, common/rope.py:17; #3096's adapter does not check this | attention role |
| QKV weights unquantized or FP8 per channel, after the platform's requantization (block FP8 is requantized on load unless `REQUANTIZE_BLOCK_SIZE` is set) | adapter:88-103 | attention role; the platform fact when the kernel is built |
| Batched-RPA attention backend, no DCP | adapter:242-244 | platform, when the kernel is built |
| FP8 KV cache, bf16 queries | gqa/config.py:142-143 | platform, when the kernel is built |
| Weights stored (K, N) | adapter:101 | platform, when the kernel is built |
| **fp8 variant: a chip with FP8 matmul** (v7x; v6e runs an fp8 x fp8 dot only at the bf16 rate). Without it the bf16 variant applies: automatic falls back, a pin raises | none: `common/quantized_matmul.py:111-112, 164` run the fp8 x fp8 dot on any chip | the chip, from JAX's chip table (`tpu_info.has_fp8_matmul`) |
| No `layer_scale`, no sequence-parallel input | adapter:457-459 | this pattern's conformance check |

With FP8 weights the kernel also quantizes activations (dynamic sub-channel
FP8 after the add and norm), so a golden through it is compared against a
stated tolerance, not zero.

`fused_qkv` has two variants, `matmul=fp8 | bf16`. The rule takes the
checkpoint's precision; on a chip without FP8 matmul the fp8 variant
declines and automatic picks `bf16` (`TPU_SUBBLOCK_PLAN=matmul=fp8` there is a
startup error). fused_gdn (Gated DeltaNet) is FP8-only, so on v6e GDN layers
stay unfused; it is out of this pattern either way.

## Which models get #3096's kernel from one declaration today

Conformance and capability together, from each model's real `config.json`
(capability on the model's real head sizes). vLLM v0.33.0
(`python -m prototype.subblock_reach`):

| Family | Model | Class | Conforms | #3096's fused_qkv can serve it |
|---|---|---|---|---|
| MiniMax | `MiniMaxAI/MiniMax-M2.5` | `MiniMaxM2DecoderLayer` | yes | no: q_norm is MiniMaxText01RMSNormTP (a norm over the whole projection); fused_qkv fuses a per-head RMSNorm |
| GLM (Zhipu) | `zai-org/GLM-4.6` | `Glm4MoeDecoderLayer` | yes | yes |
| GLM (Zhipu) | `zai-org/GLM-4.5-Air` | `Glm4MoeDecoderLayer` | yes | yes |
| Mistral | `mistralai/Ministral-8B-Instruct-2410` | `MistralDecoderLayer` | yes | yes |
| Mistral | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | `MistralDecoderLayer` | yes | yes |
| OpenAI | `openai/gpt-oss-20b` | `TransformerBlock` | yes | no: attention sinks need RPA v3; fused_qkv writes the batched-RPA layout |
| Meta | `meta-llama/Llama-3.3-70B-Instruct` | `LlamaDecoderLayer` | yes | yes |
| Meta | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | `Llama4DecoderLayer` | yes | no: 5 query heads per KV head (bf16 Q packs 2) |
| Microsoft | `microsoft/phi-4` | `LlamaDecoderLayer` | yes | yes |
| IBM | `ibm-granite/granite-4.0-h-small` | `GraniteMoeHybridAttentionDecoderLayer` | not built: StrictDataclassClassValidationError (config) | — |
| Baidu | `baidu/ERNIE-4.5-21B-A3B-PT` | `Ernie4_5_MoeDecoderLayer` | yes | no: 5 query heads per KV head (bf16 Q packs 2) |
| LG | `LGAI-EXAONE/EXAONE-4.0-32B` | `Exaone4DecoderLayer` | no: in_norm: the layer has none of ('input_layernorm',) | — |
| DeepSeek | `deepseek-ai/DeepSeek-V3.1` | `DeepseekV2DecoderLayer` | no: attn=self_attn has 0 QKVParallelLinear from the hidden size 128, not one | — |
| Moonshot | `moonshotai/Kimi-K2-Instruct` | `DeepseekV2DecoderLayer` | no: attn=self_attn has 0 QKVParallelLinear from the hidden size 128, not one | — |
| Qwen (#3096 target) | `Qwen/Qwen3-30B-A3B` | `Qwen3MoeDecoderLayer` | yes | yes |
| Qwen | `Qwen/Qwen3-32B` | `Qwen3DecoderLayer` | yes | yes |
| Qwen (#3096 target) | `Qwen/Qwen3.5-35B-A3B` | `Qwen3_5DecoderLayer` | yes | yes |
| Google | `google/gemma-3-27b-it` | `Gemma3DecoderLayer` | no: role post_norm runs after attn at gemma3.py:269, where the pattern does not call it | — |
| Google (#3096 target) | `google/gemma-4-31b-it` | `Gemma4DecoderLayer` | no: the check could not export it (a tool limit) | — |
| Cohere | `CohereForAI/c4ai-command-r-v01` | `CohereDecoderLayer` | not built: GatedRepoError: 403 (gated) | — |

vLLM main `1ecc9cc4ce` (`python -m prototype.subblock_reach --main` against a
vLLM main checkout), for classes the static scan listed as plain pre-norm:

| Family | Model | Class | Conforms | #3096's fused_qkv can serve it |
|---|---|---|---|---|
| Mistral | `mistralai/Mixtral-8x22B-Instruct-v0.1` | `MixtralDecoderLayer` | yes | yes |
| GLM (Zhipu) | `zai-org/GLM-4-32B-0414` | `Glm4DecoderLayer` | no: the check could not export it (a tool limit) | — |
| Tencent | `tencent/Hunyuan-A13B-Instruct` | `HunYuanDecoderLayer` | not built: ModuleNotFoundError: No module named 'vllm.model_executor.models.hunyuan_v1' | — |
| inclusionAI | `inclusionAI/Ling-lite-1.5` | `BailingMoeBlock` | not built: KeyError: "no config class for model_type 'bailing_moe'" | — |
| NVIDIA | `nvidia/Minitron-8B-Base` | `NemotronDecoderLayer` | no: in_norm=input_layernorm is not an RMSNorm called with a residual; post_norm=post_attention_layernorm is not an | — |
| Upstage | `upstage/solar-pro-preview-instruct` | `SolarDecoderLayer` | not built: KeyError: "no config class for model_type 'solar'" | — |
| Arcee | `arcee-ai/Trinity-Mini` | `AfmoeDecoderLayer` | no: the check could not export it (a tool limit) | — |
| Qwen | `Qwen/Qwen2.5-72B-Instruct` | `Qwen2DecoderLayer` | yes | yes |
| Qwen | `Qwen/Qwen1.5-MoE-A2.7B` | `Qwen2MoeDecoderLayer` | yes | no: 1 query heads per KV head (bf16 Q packs 2) |
| Google | `google/gemma-7b` | `GemmaDecoderLayer` | yes | no: 1 query heads per KV head (bf16 Q packs 2) |
| AllenAI | `allenai/Molmo-7B-D-0924` | `MolmoDecoderLayer` | not built: KeyError: "no config class for model_type 'molmo'" | — |
| StepFun | `stepfun-ai/step3` | `Step3TextDecoderLayer` | not built: KeyError: "no config class for model_type 'step3_text'" | — |
| LG | `LGAI-EXAONE/K-EXAONE-236B-A23B` | `ExaoneMoeDecoderLayer` | yes | yes |
| Shanghai AI Lab | `internlm/Intern-S1-Pro` | `InternS1ProMoeDecoderLayer` | not built: KeyError: "no config class for model_type 'interns1_pro_text'" | — |

- **In the intersection and not #3096 targets:** GLM-4.5-Air and GLM-4.6,
  Ministral-8B and Mistral-Small-3.2, Mixtral-8x22B, Llama-3.3-70B, Phi-4,
  dense Qwen3 and Qwen2.5, K-EXAONE. Each is one declaration call away from
  the kernel.
- **Waiting on a capability:** MiniMax-M2.5 (q/k norm over the whole
  projection), gpt-oss (sinks), Llama 4 Scout and ERNIE 4.5 (5 query heads
  per KV head), Qwen1.5-MoE and Gemma 7B (one query head per KV head).
- **Not this pattern:** MLA (DeepSeek, Kimi K2), post-norm (EXAONE 4),
  sandwich norms (Gemma 3), LayerNorm1P (Nemotron).
- **Not checked:** configs this build cannot construct (Granite 4 hybrid,
  gated Cohere, remote-code configs), and two layers the check could not
  export (a tool limit, not a verdict).
- **A static scan of vLLM main** (script and JSON under
  `static-op-args/subblock/`) finds 89 model files whose attention runs
  `qkv_proj -> attn`, 29 with a plain pre-norm decoder forward, and one
  attention forward copied verbatim into 17 classes (Llama's, in DeepSeek,
  ERNIE, Gemma, GLM-4, Mixtral, Qwen2-MoE and others).

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
    Llama (dense)                  declared -> conforms -> installed; input_proj=fused_qkv matmul=bf16 (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen3-MoE                      declared -> conforms -> installed; input_proj=fused_qkv matmul=bf16 (rule); fused calls 1; max |fused - upstream| 0.0e+00
    Qwen3.5-MoE (full attention)   declared -> conforms -> installed; input_proj=fused_qkv matmul=bf16 (rule); fused calls 1; max |fused - upstream| 0.0e+00
  -> PASS
2 current models #3096 does not target, real configs
    Mistral (mistralai/Ministral-8B-Instruct-2410), not a #3096 target: declaration 1 call; per-model code 0
    mistralai/Ministral-8B-Instruct-2410 declared -> conforms -> installed; input_proj=fused_qkv matmul=bf16 (rule); fused calls 1; max |fused - upstream| 0.0e+00
    OpenAI (openai/gpt-oss-20b), not a #3096 target: declaration 1 call; per-model code 0
    openai/gpt-oss-20b             declared -> conforms -> conforms; input_proj=unfused (fused_qkv/bf16 declined: attention sinks need RPA v3; fused_qkv writes the batched-RPA layout); fused calls 0; max |fused - upstream| 0.0e+00
    Qwen (Qwen/Qwen3-32B), not a #3096 target: declaration 1 call; per-model code 0
    Qwen/Qwen3-32B                 declared -> conforms -> installed; input_proj=fused_qkv matmul=bf16 (rule); fused calls 1; max |fused - upstream| 0.0e+00
  -> PASS
3 a declaration that does not conform stops startup
    startup error: l (ScaledAttention) is declared norm_then_attention: declared norm_then_attention, but extra op aten.mul between attn and post_norm at check_subblock.py:109
  -> PASS
4 the same fusion as #3096, by count
    #3096 per-model forwards: 4
    #3096 structural eligibility checks (here: conformance): 5
    #3096 kernel limits (here: the primitive's supports(), not this check): 7
    here: 1 plan value (FusedNormProjection), 7 declarations, 1 conformance check (norm_attn.conform)
  -> PASS

```

- **The comparison.** Each layer is built through its own constructor on a
  CPU in float32 with random weights: the first three from tiny configs
  (head_dim 128), the next three from their real `config.json`, sizes
  shrunk. It runs upstream, then with the plan installed. The attention
  backend and the MLP (dense or MoE) need TPU or CUDA kernels, so both are
  replaced by the same stand-ins in every run; everything else is the model's
  own code. The difference is exactly 0.
- **gpt-oss** conforms, and `fused_qkv` declines it (sinks): unfused runs.
- **CI.** Comment in `PLANT_MISMATCH = True` and the command exits 1.
- **pytest.** `12 passed`, including a non-conforming declaration stopping
  startup, gpt-oss falling back automatically, and a pin to `fused_qkv` on
  gpt-oss stopping startup.

## On a TPU (v6e-8)

Golden runs: four prompts, greedy, 32 tokens, logprobs compared per token
(`static-op-args/subblock/run_sb.sh`, `golden_run_sb.py`; logs beside each).

| Run | Model | Setup | Result |
|---|---|---|---|
| A | Qwen3-0.6B (bf16), declared `Qwen3DecoderLayer` | TP 1, batched RPA, bf16 KV | `block install: 28 layer(s): installed; input_proj=fused_qkv (rule)`; against `input_proj=unfused`: **tokens equal, max logprob drift 0.000000 on all four prompts** |
| A | GLM-4.5-Air (bf16, 46 layers), one declaration line | TP 8, EP, batched RPA | unfused golden written (startup 382 s); the fused run reached `block install: 46 layer(s): installed; input_proj=fused_qkv (rule)` but did not finish (see below) |
| B | Qwen3-0.6B through #3096's kernel | TP 1, batched RPA, bf16 KV | `block install: 28 layer(s): installed; input_proj=fused_qkv (rule) through #3096's kernel`; compile failed with `DMA source and target element type mismatch`: the kernel requires an FP8 KV cache (gqa/config.py:143). The FP8-KV rerun did not run (see below) |
| — | gpt-oss-20b | — | not run on a TPU; on a CPU `fused_qkv` declines it (sinks) |

- **Not finished:** the GLM fused golden, the FP8-KV runs (Qwen3-0.6B
  unfused, through #3096's kernel; GLM unfused, through #3096's kernel) and
  gpt-oss. Worker processes left by a stopped run still hold four TPU
  devices, so these runs could not start.
- **A local adaptation to run #3096's kernel on v6e; on v7x the fp8 path
  is the author's.** Scratch commit `8df28ab73` dequantizes FP8 QKV weights
  once at load in #3096's `_linear` when the chip has no FP8 matmul, so every
  kernel takes its existing bf16 path. Queued (`queueH.sh`) behind the held
  devices: Qwen3-0.6B bf16 and -FP8 through the kernel with an FP8 KV cache,
  each against its unfused golden.
- **A probe bug, caught by a TPU run:** the first attempt declined every
  layer, because the conformance probe built its tensors on the meta device
  while attention's weights were on the TPU. That silent decline is why a
  non-conforming declaration now stops startup.

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
