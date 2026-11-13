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
2. **A model owner declares it with one typed binding function**, in the
   model's own `registration.py`:

   ```python
   @declare_block(Glm4MoeDecoderLayer)
   def glm4_moe_attention(layer: Glm4MoeDecoderLayer) -> NormAttnBinding:
       attention = layer.self_attn
       qk_norm = (attention.q_norm, attention.k_norm) if attention.use_qk_norm else None
       return pre_norm_binding(
           layer,
           attention,
           qk_norm=qk_norm,
           output_gate=False,
           stream=positions_first(layer),
       )
   ```

   The return type names the pattern; the binding dataclass is what
   `AttentionRunner.__init__` takes; every part is typed attribute access
   (see "Declarations are typed binding functions" below).
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
weights and contract, and the plan is the runner's. The runners are plain
objects, not modules: the model owns the state; the runner is behaviour
(ownership would move to the runner only if upstream constructed it, as
vLLM's MoERunner factory does). A runner unpacks the role binding (the
layer's own modules bound to the pattern's roles) into typed attributes, so
installing one leaves the model's `state_dict` and named parameters as built.

## Where to look

| Claim | Code |
|---|---|
| The pattern (framework-owned) | [`norm_then_attention`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L46-L59), [`NormAttnBinding`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L73-L147), [`pre_norm_binding`](../src/vllm_torchtpu/blocks/norm_attn/pattern.py#L199-L226) |
| The runner: parts in, composition owned, plan resolved once | [`AttentionBinding`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L102-L125), [`AttentionRunner`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L128-L269) ([`__init__`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L146-L169), [`_select_forward`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L171-L183), [`_unfused`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L196-L209), [`_fused_qkv`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L211-L225), [`attention_from_qkv`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L229-L254)), [`AttnConfig`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L74-L99) |
| The fused value's reference (stands in for #3096's kernel) | [`reference_fused_qkv`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L272-L290) |
| Conformance: the layer's wiring | [`conform`](../src/vllm_torchtpu/blocks/norm_attn/install.py#L88-L145) |
| Conformance: attention's forward is the runner's composition; a variant the runner does not implement is a stated missing property | [`attention_config`](../src/vllm_torchtpu/blocks/norm_attn/install.py#L155-L181), [`conform_attention`](../src/vllm_torchtpu/blocks/norm_attn/install.py#L184-L266) |
| A non-conforming declaration stops startup | [`install`](../src/vllm_torchtpu/blocks/installer.py#L39-L76) |
| The plan and the kernel's capability | [`NormAttnPlan`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L127-L134), [`rule` and fallbacks](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L212-L215), [`AttentionFacts`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L154-L173), [`facts_of`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L185-L209), [`fused_qkv_supports`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L224-L295), [`resolve`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L320-L327), [`choose`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L330-L372); the chip fact [`has_fp8_matmul`](../src/vllm_torchtpu/tpu_info.py#L90-L98) |
| Install: capability consulted (automatic falls back, a pin that cannot run raises), runner built from the layer's parts, the layer routed through it | [`NormThenAttention`](../src/vllm_torchtpu/blocks/norm_attn/install.py#L384-L403), [`build_runner`](../src/vllm_torchtpu/blocks/norm_attn/install.py#L293-L310), [`NormAttnLayer`](../src/vllm_torchtpu/blocks/norm_attn/install.py#L269-L290) |
| Scopes nest: the block runner is built from the declared parts and the nested `AttentionRunner`, constructed with `output_reduced=False` | [`AttnMoERunner`](../src/vllm_torchtpu/blocks/attn_moe/runner.py#L121-L237), [`MoEBinding`](../src/vllm_torchtpu/blocks/attn_moe/runner.py#L101-L118), [`AttnMoEBinding`](../src/vllm_torchtpu/blocks/attn_moe/pattern.py#L113-L179), [`AttnThenMoE`](../src/vllm_torchtpu/blocks/attn_moe/install.py#L258-L317), [`unreduced_output_supports`](../src/vllm_torchtpu/blocks/norm_attn/plan.py#L298-L317), [`project_out`](../src/vllm_torchtpu/blocks/norm_attn/runner.py#L256-L269), [`test_scopes_nest_on_qwen3_5`](../tests/blocks/test_norm_attn_integration.py#L241-L261) |
| Scopes nest: several patterns per class | [`declarations_for`](../src/vllm_torchtpu/blocks/__init__.py#L137-L139), [`install`](../src/vllm_torchtpu/blocks/installer.py#L39-L76) |
| Typed declarations: the binding function, its parts, kinds checked at construction | [`declare_block`](../src/vllm_torchtpu/blocks/__init__.py#L142-L164), [`BlockDeclaration`](../src/vllm_torchtpu/blocks/__init__.py#L65-L131), [`BlockBinding`](../src/vllm_torchtpu/blocks/binding.py#L44-L64), [`BindingKindError`](../src/vllm_torchtpu/blocks/binding.py#L67-L87), [`as_kind`](../src/vllm_torchtpu/blocks/binding.py#L115-L129) |
| One binding function per model | [GLM-4.5/4.6](../src/vllm_torchtpu/models/vllm/glm4_moe/registration.py#L24-L51), [Mistral](../src/vllm_torchtpu/models/vllm/mistral/registration.py#L24-L37), [Llama](../src/vllm_torchtpu/models/vllm/llama/registration.py#L24-L37), [Qwen3](../src/vllm_torchtpu/models/vllm/qwen3/registration.py#L23-L32), [gpt-oss](../src/vllm_torchtpu/models/vllm/gpt_oss/registration.py#L24-L37), [Qwen3-MoE](../src/vllm_torchtpu/models/vllm/qwen3_moe/registration.py#L23-L32), [Qwen3.5](../src/vllm_torchtpu/models/vllm/qwen3_5/registration.py#L36-L75) |
| Per model: A (runner unfused vs upstream) and B (the rule's value through the runner) | [`family`](../tests/blocks/subblock_check.py#L225-L261), [`_install_and_run`](../tests/blocks/subblock_check.py#L199-L216), [`new_models`](../tests/blocks/subblock_check.py#L284-L293) |
| Planted non-conforming declarations (the layer; attention) | [`ScaledAttention`](../tests/blocks/subblock_fixtures.py#L34-L49), [`scale_query`](../tests/blocks/subblock_fixtures.py#L56-L67), [`_planted`](../tests/blocks/subblock_check.py#L307-L329) |
| Every code object the installed forward runs has source (torch.compile's AOT cache) | [`source_less`](../tests/blocks/subblock_check.py#L332-L371) |
| The count against #3096 | [`counts`](../tests/blocks/subblock_check.py#L430-L438) |
| Reach over current models, from real configs | [`CANDIDATES`](../tests/blocks/subblock_reach.py#L76-L214), [`MAIN_CANDIDATES`](../tests/blocks/subblock_reach.py#L220-L276), [`check`](../tests/blocks/subblock_reach.py#L432-L452) |
| The startup override | `TPU_SUBBLOCK_PLAN` in `envs.py`, passed by the runner's `_install_blocks` |

## The kernel's capability, as its author would declare it

`fused_qkv_supports(attention, matmul=, chip=, kv_cache_dtype=)` returns why #3096's fused_qkv cannot serve an
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
| q/k norms per head (over head_dim, not TP-wide over the projection): RMSNorm, (1 + w) or weightless | adapter:74-82 | attention role |
| NeoX-style RoPE (partial RoPE fine) | common/model_config.py:59-71, common/rope.py:17; #3096's adapter does not check this | attention role |
| QKV weights unquantized or FP8 per channel, after the platform's requantization (block FP8 is requantized on load unless `REQUANTIZE_BLOCK_SIZE` is set) | adapter:88-103 | attention role; the platform fact when the kernel is built |
| Batched-RPA attention backend, no DCP | adapter:242-244 | platform, when the kernel is built |
| FP8 (e4m3) KV cache | gqa/config.py:143 | the platform's `kv_cache_dtype` (empty, unchecked, while the fused value is the plain-torch reference, which writes no KV cache) |
| bf16 queries | gqa/config.py:142 | platform, when the kernel is built |
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

## Two kinds of test

The runner splits the tests in two. With per-model patches the fusion and the
model are one function, so every model needs its own numerics test of every
fusion; here they are separate objects with a contract between them.

- **Execution, model-free** ([`test_attention_runner.py`](../tests/blocks/test_attention_runner.py)):
  `AttentionRunner` built from synthetic parts written in the test (an
  RMSNorm with the `(x, residual)` contract, linears returning `(out, bias)`,
  a NeoX rotary, a causal GQA core). Over every role configuration (per-head
  q/k norm, output gate, GQA 1/4/8, head_dim 64/128, QKV bias, full or
  partial RoPE, first layer or not): `_unfused` equals a straight-line
  reference, and `_fused_qkv` equals `_unfused` (bit for bit for the
  reference op). Also: the plan's choice (rule, fallback, pin, a pin that
  cannot run raises) and `fused_qkv_supports` over synthetic attentions,
  chips (v6e, 7x) and KV cache dtypes. It imports nothing from
  `vllm.model_executor.models`, and `test_imports_no_model` checks that.
- **Integration, per model** ([`test_norm_attn_integration.py`](../tests/blocks/test_norm_attn_integration.py)):
  layers built on the meta device from tiny configs, for each declared
  family. The declaration conforms (a layer that does not is named with
  its edge and line, or the role property it lacks), the runner holds the
  model's own part objects (`runner.qkv_proj is attention.qkv_proj`, no
  copies, an unchanged `state_dict`), and the decoder routes through the
  runner. Numerics: one smoke check, Llama against upstream.

The argument is compositional: the runner correct for every role
configuration, plus the model conforming, gives the model correct under
every plan value, so test cost grows with fusions + models, not
fusions x models.

```text
$ python -m pytest tests/blocks/test_attention_runner.py -q
419 passed, 16 warnings in 8.41s
$ python -m pytest tests/blocks/test_norm_attn_integration.py -q
34 passed, 62 warnings in 17.03s
```

## Running it

On a CPU against vLLM v0.33.0 (`<vllm>`); nothing loads libtpu.

```bash
export PYTHONPATH=src:.:<vllm>
python -m prototype.check_subblock
python -m prototype.subblock_reach      # fetches config.json files only
TORCH_DEVICE_BACKEND_AUTOLOAD=0 JAX_PLATFORMS=cpu python -m pytest tests/blocks
```

The scripts in `prototype/` only print. The checks they print live in
`tests/blocks/` (`subblock_check.py`, `subblock_reach.py`, the planted layers
in `subblock_fixtures.py`), and pytest asserts the same checks
(`test_subblock_check.py`, `test_subblock_reach.py`).

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
    planted in the layer: startup error: l (ScaledAttention) is declared norm_then_attention: declared norm_then_attention, but extra op aten.mul between attn and post_norm at subblock_fixtures.py:47
    planted in attention: startup error: l (ScaledQueryAttention) is declared norm_then_attention: declared norm_then_attention (attention on ScaledQueryAttention), but extra op aten.mul between rope and attn at subblock_fixtures.py:63
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
- **CI.** Comment in `PLANT_MISMATCH = True` (`tests/blocks/subblock_check.py`)
  and the command exits 1; `test_subblock_check.py` fails with it.
- **pytest.** `590 passed, 3 xfailed` in `tests/blocks` (with the
  declared-blocks tests and the demos' checks; the xfails are the three
  layers `suggest_blocks` raises on); the two kinds of test above,
  including a non-conforming declaration stopping startup, gpt-oss falling
  back automatically, a pin to `fused_qkv` on gpt-oss stopping startup, the
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
| Finding layers by attribute names (`input_layernorm`, `self_attn`, `linear_attn`) and `isinstance` on model classes | One typed binding function per model, in the model's own directory |

#3096 is careful work; the count is about where the knowledge lives, not
about the PR. It also covers Gemma 4 and Gated DeltaNet layers, which this
pattern does not.

## How it relates to #1492's block: `AttnMoERunner` nests `AttentionRunner`

`norm_then_attention` is the front of the skeleton `attn_then_moe` checks
(norm, attention, norm, MoE), so the two plans nest by scope: the
sub-block's plan (`input_proj`) inside the block's plan (`collectives`).
Qwen3.5 and GLM-4.5 declare both.

The block is a runner too, shaped like `MoERunner` one scope up
(`blocks/attn_moe/runner.py`):

- **The binding in.** `AttnMoERunner.__init__` takes `MoEBinding` (post norm,
  router, experts; `AttnMoEBinding` extends it), the attention part and the per-bucket
  plan table; `_select_forward` picks the value once; the runner becomes the
  layer's `forward` on the instance (no mixin, no `__class__` rebinding).
- **The attention part.** Input norm, attention, output projection,
  returning the unreduced attention output and the residual stream. Where
  the layer declares `norm_then_attention`, the block builds that
  `AttentionRunner` itself, **with its contract stated at construction**:
  `output_reduced=False`, because the block reduces attention itself under
  both values (an all-reduce, or #1492's reduce-scatter folded into the MoE
  all-reduce). `o_proj.reduce_results` is never written.
- **The capability guards the contract.** Each attention plan value states
  whether it can return an unreduced output (`unreduced_output_supports`):
  `unfused` runs `o_proj` without its all-reduce (the flag passed per call)
  when `o_proj` has no bias (TPU adds it after the all-reduce); `fused_qkv`
  also needs its op to honour the flag. A block pin that cannot get
  unreduced attention raises; the rule falls back to upstream's
  collectives.
- **What is still written after construction.** The MoE child's two
  decisions (`experts.gate`, `moe_config.skip_final_all_reduce`): the model
  builds its MoE, so `contract_adapter` writes them once, logs them and
  freezes them. For a layer with no declared sub-block (Qwen3.5's Gated
  DeltaNet layers) the attention part is the model's own attention
  (`ModelAttention`), and its `o_proj.reduce_results` is written too.

Tests: [`test_attn_moe_runner.py`](../tests/blocks/test_attn_moe_runner.py)
(model-free: block {all_reduce, overlap} x attention {unfused, fused_qkv}
against upstream's composition at TP=1, the collective order, the warmup
gate, the capability and the contract) and
[`test_attn_moe_block.py`](../tests/blocks/test_attn_moe_block.py) (per
model: the nested runner built unreduced from the model's own parts, nothing
written to `o_proj`, the interim MoE writes, the fallbacks).

## Declarations are typed binding functions

A declaration is a function from the layer to the pattern's parts:

```python
@declare_block(Glm4MoeDecoderLayer)
def glm4_moe_block(layer: Glm4MoeDecoderLayer) -> AttnMoEBinding | None:
    if not isinstance(layer.mlp, Glm4MoE):  # The first_k_dense_replace layers.
        return None
    attention = layer.self_attn
    return pre_norm_moe_binding(
        layer,
        attention,
        attention.o_proj,
        layer.mlp,
        stream=positions_first(layer),
        attend=attend_with_positions(attention),
    )
```

- **The pattern is the return type**; `None` means the declaration does not
  cover this layer. There is no `pattern=`, role dict or `when=`.
- **Nothing is found by name or kind.** The binding states every part
  (`out_proj` included, so the block never searches attention for it), the
  optional ones (`qk_norm`, `output_gate`), and how the
  layer's forward and its attention are called (`stream`, `attend`, typed
  Protocols checked against the model's signature). Conformance tracks
  arguments by identity and reads no parameter name.
- **The binding is the runner's argument**: `NormAttnBinding` extends
  `AttentionBinding` and `AttnMoEBinding` extends `MoEBinding`, the dataclasses the
  runners take; model-free tests build the base dataclass from synthetic
  parts.
- **Kinds are checked when the parts are built**, and errors name the
  attribute, recovered by identity: `q_norm=self_attn.q_norm and k_norm come
  together or not at all`.
- **Shared names are typed helpers, not tables**: `pre_norm_binding` and
  `pre_norm_moe_binding` take Protocol-typed arguments; `as_kind` narrows a
  child a model builds from a class parameter (Llama's `attn_layer_type`,
  gpt-oss's `attention_cls`).
- **A misspelled attribute is a type error.** `nn.Module.__getattr__`
  returns `Tensor | Module`, which no part field accepts. basedpyright on
  `post_norm=layer.post_attention_layernrom`:

  ```text
  error: Argument of type "Tensor | Module" cannot be assigned to parameter "post_norm" of type "Module"
  ```

  and on `pre_norm_binding(layer, layer.self_atn)`: `"Module" is incompatible
  with protocol "PreNormAttention"`.

## Not shown here

- **What the kernel also fuses.** #3096 fuses q/k norm, RoPE, attention and
  the output projection too; the reference fuses norm and projection only.
- **Gemma 4 and Gated DeltaNet**, each its own pattern or a decline.
- **TP > 1 numerics on the fused path, quantized projections, and the
  residual tiling pin** (a contract on the residual stream between blocks,
  at model scope).
- **Production** would also need the plan in the per-bucket table and a
  numerics tier against the residual stream's tolerance.
