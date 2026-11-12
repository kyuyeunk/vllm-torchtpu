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
3. **The framework verifies the declaration**, twice: with the roles opaque,
   the layer's forward computes the pattern's graph (the normed tokens and
   the positions reach nothing but attention); and with attention's parts
   opaque, attention's own forward is `AttentionRunner`'s composition (split,
   per-head q/k norm, RoPE, output gate, output projection, nothing else).
   **A declaration that does not conform stops startup**, naming the edge and
   line: it is the model owner's explicit claim.
4. **One framework forward then runs every declared model:
   `AttentionRunner`**, shaped like vLLM's `MoERunner` (below). It receives
   the model's own parts, keeps references to them, and owns only their
   composition; the plan picks how that composition runs. `fused_qkv` is one
   branch of that one forward, written against the parts, never against a
   model class.
5. **Capability flows up.** `fused_qkv` stands for #3096's kernel, so it
   declares that kernel's capability (`fused_qkv_supports`, below) and asks
   the bound attention role. Where the capability does not cover a
   conforming layer, the rule's pick falls back to `unfused` quietly; a user
   pin to `fused_qkv` that cannot run stops startup.

The fused value here is a plain-torch reference standing in for #3096's
kernel; "On a TPU" below also routes #3096's kernel itself behind it (a
local scratch build, not on this branch).

## The runner, side by side with `MoERunner`

| vLLM `MoERunner` (`layers/fused_moe/runner/moe_runner.py`) | `AttentionRunner` (`blocks/norm_attn/runner.py`) |
|---|---|
| `__init__` (L251-L299) receives already-built `router`, `routed_experts`, `gate`, `shared_experts`, transforms and stores them | `__init__` receives the model's `input_norm`, `qkv_proj`, `attn` (core), `o_proj`, `rotary`, `q_norm` / `k_norm` and keeps references (no new parameters; weight loading and the `state_dict` are unchanged) |
| `forward` (L693) calls them in order: `router.select_experts`, `routed_experts.forward_*`, shared experts, transforms | `forward` calls them in order: input norm (with the residual), QKV projection, split, per-head q/k norm, RoPE, attention, output gate, output projection |
| `_forward_entry = self._select_forward()` (L295) picks the implementation once | `_select_forward` resolves the plan once: `unfused`, or `fused_qkv` (the reference here, #3096's kernel in the scratch build) |
| Models build their MoE through `FusedMoEFactory`, so 39 model files share one MoE forward | Models keep building their own parts; one declaration line routes them through the runner, so the attention forward copied into 17 classes (the reach scan) runs once |

The runner owns the composition, not the parts: each part keeps its own
weights and contract, and the plan is the runner's.

## Where to look

| Claim | Code |
|---|---|
| The pattern (framework-owned) | [`norm_then_attention`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L50-L63), [`PRE_NORM_ROLES`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L41-L47) |
| The runner: parts in, composition owned, plan resolved once | [`AttentionRunner`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L74-L185) ([`__init__`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L81-L112), [`_select_forward`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L114-L120), [`_unfused`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L132-L145), [`_fused_qkv`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L147-L159), [`attention_from_qkv`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L163-L185)), [`AttnConfig`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L48-L71) |
| The fused value's reference (stands in for #3096's kernel) | [`reference_fused_qkv`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L188-L206) |
| Conformance: the layer's wiring | [`conform`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L96-L213) |
| Conformance: attention's forward is the runner's composition; a variant the runner does not implement is a stated missing property | [`attention_parts`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L223-L270), [`conform_attention`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L273-L346) |
| A non-conforming declaration stops startup | [`install`](../src/vllm_torchtpu/blocks/__init__.py#L164-L195) |
| The plan and the kernel's capability | [`NormAttnPlan`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L47-L54), [`rule` and fallbacks](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L62-L70), [`fused_qkv_supports`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L73-L142), [`resolve`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L145-L164); the chip fact [`has_fp8_matmul`](../src/vllm_torchtpu/tpu_info.py#L90-L99) |
| Install: capability consulted (automatic falls back, a pin that cannot run raises), runner built from the layer's parts, the layer routed through it | [`NormThenAttention`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L423-L480), [`build_runner`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L394-L416), [`NormAttnDecoder`](../src/vllm_torchtpu/blocks/norm_attn/__init__.py#L360-L380) |
| Scopes nest: inside `attn_then_moe`'s block, the block runs its front half through the runner | [`AttentionMoEBlock.forward`](../src/vllm_torchtpu/blocks/attn_moe/forward.py#L82-L94), [`test_scopes_nest_on_qwen3_5`](../tests/blocks/test_norm_attn_block.py#L109-L126) |
| Scopes nest: several patterns per class | [`declarations_for`](../src/vllm_torchtpu/blocks/__init__.py#L84-L86) |
| One call per model | [GLM-4.5/4.6](../src/vllm_torchtpu/models/vllm/glm4_moe/registration.py#L21), [Mistral](../src/vllm_torchtpu/models/vllm/mistral/registration.py#L21), [Llama](../src/vllm_torchtpu/models/vllm/llama/registration.py#L21), [Qwen3](../src/vllm_torchtpu/models/vllm/qwen3/registration.py#L21), [gpt-oss](../src/vllm_torchtpu/models/vllm/gpt_oss/registration.py#L21-L25), [Qwen3-MoE](../src/vllm_torchtpu/models/vllm/qwen3_moe/registration.py#L21), [Qwen3.5](../src/vllm_torchtpu/models/vllm/qwen3_5/registration.py#L43-L48) |
| Per model: A (runner unfused vs upstream) and B (the rule's value through the runner) | [`family`](check_subblock.py#L211-L244), [`_install_and_run`](check_subblock.py#L190-L208), [`new_models`](check_subblock.py#L256-L264) |
| Planted non-conforming declarations (the layer; attention) | [`ScaledAttention`](check_subblock.py#L105-L113), [`_scale_query`](check_subblock.py#L120-L131), [`_planted`](check_subblock.py#L277-L305) |
| Every code object the installed forward runs has source (torch.compile's AOT cache) | [`_source_less`](check_subblock.py#L308-L352) |
| The count against #3096 | [`counts`](check_subblock.py#L392-L399) |
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
1 each family: declared -> conforms -> AttentionRunner, against upstream
    Llama (dense)                  declared -> conforms (4q/2kv x 128)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=fused_qkv matmul=bf16 (rule); fused runs 1; max diff 0.0e+00
    Qwen3-MoE                      declared -> conforms (4q/2kv x 128, per-head q/k norm)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=fused_qkv matmul=bf16 (rule); fused runs 1; max diff 0.0e+00
    Qwen3.5-MoE (full attention)   declared -> conforms (4q/2kv x 128, per-head q/k norm, output gate)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=fused_qkv matmul=bf16 (rule); fused runs 1; max diff 0.0e+00
  -> PASS
2 current models #3096 does not target, real configs
    GLM (Zhipu) (zai-org/GLM-4.5-Air), not a #3096 target: declaration 1 call; per-model code 0
    zai-org/GLM-4.5-Air            declared -> conforms (4q/2kv x 128)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=fused_qkv matmul=bf16 (rule); fused runs 1; max diff 0.0e+00
    Mistral (mistralai/Ministral-8B-Instruct-2410), not a #3096 target: declaration 1 call; per-model code 0
    mistralai/Ministral-8B-Instruct-2410 declared -> conforms (4q/2kv x 128)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=fused_qkv matmul=bf16 (rule); fused runs 1; max diff 0.0e+00
    OpenAI (openai/gpt-oss-20b), not a #3096 target: declaration 1 call; per-model code 0
    openai/gpt-oss-20b             declared -> conforms (4q/2kv x 128)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=unfused (rule); fused runs 0; max diff 0.0e+00
    Qwen (Qwen/Qwen3-32B), not a #3096 target: declaration 1 call; per-model code 0
    Qwen/Qwen3-32B                 declared -> conforms (4q/2kv x 128, per-head q/k norm)
                                     A runner unfused vs upstream: max diff 0.0e+00
                                     B runner installed; input_proj=fused_qkv matmul=bf16 (rule); fused runs 1; max diff 0.0e+00
  -> PASS
3 a declaration that does not conform stops startup
    planted in the layer: startup error: l (ScaledAttention) is declared norm_then_attention: declared norm_then_attention, but extra op aten.mul between attn and post_norm at check_subblock.py:111
    planted in attention: startup error: l (ScaledQueryAttention) is declared norm_then_attention: declared norm_then_attention (attention on ScaledQueryAttention), but extra op aten.mul between rope and attn at check_subblock.py:127
  -> PASS
4 the same fusion as #3096, by count
    #3096 per-model forwards: 4
    #3096 structural eligibility checks (here: conformance): 5
    #3096 kernel limits (here: the primitive's supports(), not this check): 7
    here: 1 forward (AttentionRunner) with 1 fused branch, 7 declarations, 1 conformance check (norm_attn.conform)
  -> PASS
5 every code object the installed forward runs has source (for torch.compile)
    Llama (dense)                  0 code object(s) without source
    Qwen3-MoE                      0 code object(s) without source
    Qwen3.5-MoE (full attention)   0 code object(s) without source
    zai-org/GLM-4.5-Air            0 code object(s) without source
    mistralai/Ministral-8B-Instruct-2410 0 code object(s) without source
    openai/gpt-oss-20b             0 code object(s) without source
    Qwen/Qwen3-32B                 0 code object(s) without source
  -> PASS
```

- **The comparison.** Each layer is built through its own constructor on a
  CPU in float32 with random weights: the first three from tiny configs
  (head_dim 128), the next four from their real `config.json`, sizes
  shrunk. It runs upstream, then with the plan installed. The attention
  backend and the MLP (dense or MoE) need TPU or CUDA kernels, so both are
  replaced by the same stand-ins in every run; everything else is the model's
  own code. **A** (the runner's unfused composition against the model's own
  forward) and **B** (the rule's value through the runner) both differ by
  exactly 0: A shows the framework's forward is a faithful replacement; B
  shows the fused branch runs, here as a reference equal by construction.
- **gpt-oss** conforms, and `fused_qkv` declines it (sinks): the runner runs
  unfused.
- **CI.** Comment in `PLANT_MISMATCH = True` and the command exits 1.
- **pytest.** `17 passed` (with the declared-blocks tests), including a
  non-conforming declaration stopping startup, gpt-oss falling back
  automatically, a pin to `fused_qkv` on gpt-oss stopping startup, the
  source lookup, and the two scopes nesting on Qwen3.5.

## On a TPU (v6e-8)

Golden runs: four prompts, greedy, 32 tokens, logprobs compared per token
(`static-op-args/subblock/run_sb.sh`, `golden_run_sb.py`; logs beside each).
Runs before this commit used the earlier rewiring of two roles; the
AttentionRunner rerun is queued (`queueR.sh`, below).

| Run | Model | Setup | Result |
|---|---|---|---|
| A, rewiring | Qwen3-0.6B (bf16), declared `Qwen3DecoderLayer` | TP 1, batched RPA, bf16 KV | 28 layers; against `input_proj=unfused`: **tokens equal, max logprob drift 0.000000 on all four prompts** |
| B | Qwen3-0.6B (bf16) through #3096's kernel | TP 1, batched RPA, **FP8 KV in both arms** | 28 layers `through #3096's kernel`; **tokens equal on all four prompts, max logprob drift 0.069 to 0.313** |
| B | Qwen3-0.6B-FP8 through #3096's kernel, FP8 weights dequantized at load (v6e has no FP8 matmul) | TP 1, FP8 KV both arms | tokens differ on one prompt; elsewhere drift 0.110 to 0.290 |
| B | **GLM-4.5-Air (bf16, 46 layers), a family #3096 does not target, through #3096's kernel** | TP 8, EP, FP8 KV both arms; baseline: the AttentionRunner's unfused composition | `block install: 46 layer(s): installed; input_proj=fused_qkv … through #3096's kernel` on all 8 ranks; **tokens equal on all four prompts, max logprob drift 0.042 to 0.088** |
| — | gpt-oss-20b | — | on a CPU `fused_qkv` declines it (sinks); TPU run queued |

- **Drift is the fusion's, not the cache's:** both arms of every B run use
  the same FP8 KV cache. Whether 0.04 to 0.31 is within the residual
  stream's tolerance is the block owner's call (DATA_CONTRACTS.md 5.8).
- **A failed compile that was not the model:** the GLM fused run first
  failed in torch.compile's AOT cache (`inspect.getsourcelines`: "could not
  get source code") because this branch's source files were being edited
  while the run compiled. The CPU check 5 now looks up the source of every
  code object the installed forward runs; the runner path has none without
  source, and runs must import from a tree no one edits.
- **Queued (`queueR.sh`, after the other TPU queue):** R1 Qwen3-0.6B, the
  runner's unfused composition against upstream (A); R2 the runner's
  `fused_qkv` through #3096's kernel (B); R3/R4 GLM-4.5-Air upstream against
  the runner's unfused composition (A); R5 GLM through the runner and #3096's
  kernel (B). All with FP8 KV.
- **A local adaptation to run #3096's kernel on v6e; on v7x the fp8 path
  is the author's.** The scratch build dequantizes FP8 QKV weights once at
  load in #3096's `_linear` when the chip has no FP8 matmul, so every kernel
  takes its bf16 path. The kernel is bound as the runner's fused op from the
  runner's own parts (`blocks/norm_attn/pr3096.py`, scratch only).
- **A probe bug, caught by a TPU run:** an earlier version's conformance
  probe built tensors on the meta device while attention's weights were on
  the TPU and declined every layer silently. That is why a non-conforming
  declaration now stops startup; the attention check here stubs every part,
  so it runs on any device.

## The count against #3096

Counted from #3096's diff, file by file:

| #3096 | Here |
|---|---|
| 4 per-model forwards: `fused_attn_proj_patch.py`'s `_patched_qwen3_moe_forward`, `_patched_qwen3_next_forward` (Qwen3-Next and Qwen3.5) and `_patched_gemma4_attention_forward`, plus an edit to #1492's `_overlap_forward` in `qwen3_5_patch.py` | One framework forward, `AttentionRunner`, for every declared model |
| 5 structural eligibility checks in `layers/adapter/fused_attn_proj.py`: unsupported norm, sequence-parallel input, `layer_scale`, GDN's separate in-projections, first layer without a residual | 1 conformance check, the same for every model |
| 7 kernel limits (quant method, weight layout, FP8 per channel, batched RPA, head_dim 64 or DCP, KV sharing or sinks, GDN in-projection not FP8) | The kernel's own `supports(...)`, here `fused_qkv_supports`, consulted by the runner's plan |
| Finding layers by attribute names (`input_layernorm`, `self_attn`, `linear_attn`) and `isinstance` on model classes | One declaration call per model, in the model's own directory |

#3096 is careful work; the count is about where the knowledge lives, not
about the PR. It also covers Gemma 4 and Gated DeltaNet layers, which this
pattern does not.

## How it relates to #1492's block

`norm_then_attention` is the front half of the skeleton `attn_then_moe`
checks (norm, attention, norm, MoE), so the two plans nest by scope: the
sub-block's plan (`input_proj`) inside the block's plan (`collectives`).
Qwen3.5 declares both. Where the block's plan owns the layer's forward
(the overlap), the block runs its front half through the sub-block's
runner, so both plans apply; elsewhere the runner owns the forward.

## Not shown here

- **What the kernel also fuses.** #3096 fuses q/k norm, RoPE, attention and
  the output projection too; the reference fuses norm and projection only.
- **Gemma 4 and Gated DeltaNet**, each its own pattern or a decline.
- **TP > 1 numerics on the fused path, quantized projections, and the
  residual tiling pin** (a contract on the residual stream between blocks,
  at model scope).
- **Production** would also need the plan in the per-bucket table and a
  numerics tier against the residual stream's tolerance.
