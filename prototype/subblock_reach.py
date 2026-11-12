# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Which current models conform to norm_then_attention, from their real configs.

For each candidate, only its Hugging Face ``config.json`` is fetched (no
weights). The config's structural fields are kept (q/k norms, partial RoPE,
biases, sliding windows, layer types, MLA ranks) and its sizes shrunk except
the attention heads, which capability reads from the built layer; the
decoder layer is built on the meta device with vLLM v0.33.0's own class
and checked against the pattern with vLLM's usual child names. #3096's
targets are marked.

    python -m prototype.subblock_reach
"""

import dataclasses
import importlib
import inspect
import json
import sys

import transformers
from huggingface_hub import hf_hub_download
from torch import nn
from vllm.config import set_current_vllm_config

from prototype import _build
from vllm_torchtpu.blocks import BlockDeclaration
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.norm_attn import install as norm_attn
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnBinding
from vllm_torchtpu.blocks.norm_attn.plan import facts_of, fused_qkv_supports
from vllm_torchtpu.blocks.stream import (
    Stream,
    attend_with_positions,
    hidden_positions_residual,
    hidden_residual_positions,
    positions_first,
)

# #3096 patches these decoder classes (fused_attn_proj_patch.py).
PR_3096_TARGETS = {
    "Qwen3MoeDecoderLayer",
    "Qwen3NextDecoderLayer",
    "Qwen3_5DecoderLayer",
    "Gemma4DecoderLayer",
}


@dataclasses.dataclass(frozen=True)
class Candidate:
    family: str
    repo: str
    module: str
    layer_class: str
    extra: dict = dataclasses.field(default_factory=dict)
    text_config: bool = False
    model: dict = dataclasses.field(default_factory=dict)


CANDIDATES = (
    Candidate(
        "MiniMax",
        "MiniMaxAI/MiniMax-M2.5",
        "vllm.model_executor.models.minimax_m2",
        "MiniMaxM2DecoderLayer",
    ),
    Candidate(
        "GLM (Zhipu)",
        "zai-org/GLM-4.6",
        "vllm.model_executor.models.glm4_moe",
        "Glm4MoeDecoderLayer",
    ),
    Candidate(
        "GLM (Zhipu)",
        "zai-org/GLM-4.5-Air",
        "vllm.model_executor.models.glm4_moe",
        "Glm4MoeDecoderLayer",
    ),
    Candidate(
        "Mistral",
        "mistralai/Ministral-8B-Instruct-2410",
        "vllm.model_executor.models.mistral",
        "MistralDecoderLayer",
    ),
    Candidate(
        "Mistral",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "vllm.model_executor.models.mistral",
        "MistralDecoderLayer",
        text_config=True,
    ),
    Candidate(
        "OpenAI",
        "openai/gpt-oss-20b",
        "vllm.model_executor.models.gpt_oss",
        "TransformerBlock",
    ),
    Candidate(
        "Meta",
        "meta-llama/Llama-3.3-70B-Instruct",
        "vllm.model_executor.models.llama",
        "LlamaDecoderLayer",
    ),
    Candidate(
        "Meta",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "vllm.model_executor.models.llama4",
        "Llama4DecoderLayer",
        text_config=True,
    ),
    Candidate(
        "Microsoft",
        "microsoft/phi-4",
        "vllm.model_executor.models.llama",
        "LlamaDecoderLayer",
    ),
    Candidate(
        "IBM",
        "ibm-granite/granite-4.0-h-small",
        "vllm.model_executor.models.granitemoehybrid",
        "GraniteMoeHybridAttentionDecoderLayer",
    ),
    Candidate(
        "Baidu",
        "baidu/ERNIE-4.5-21B-A3B-PT",
        "vllm.model_executor.models.ernie45_moe",
        "Ernie4_5_MoeDecoderLayer",
    ),
    Candidate(
        "LG",
        "LGAI-EXAONE/EXAONE-4.0-32B",
        "vllm.model_executor.models.exaone4",
        "Exaone4DecoderLayer",
    ),
    Candidate(
        "DeepSeek",
        "deepseek-ai/DeepSeek-V3.1",
        "vllm.model_executor.models.deepseek_v2",
        "DeepseekV2DecoderLayer",
        model={"use_mla": True},
    ),
    Candidate(
        "Moonshot",
        "moonshotai/Kimi-K2-Instruct",
        "vllm.model_executor.models.deepseek_v2",
        "DeepseekV2DecoderLayer",
        model={"use_mla": True},
    ),
    Candidate(
        "Qwen (#3096 target)",
        "Qwen/Qwen3-30B-A3B",
        "vllm.model_executor.models.qwen3_moe",
        "Qwen3MoeDecoderLayer",
    ),
    Candidate(
        "Qwen",
        "Qwen/Qwen3-32B",
        "vllm.model_executor.models.qwen3",
        "Qwen3DecoderLayer",
    ),
    Candidate(
        "Qwen (#3096 target)",
        "Qwen/Qwen3.5-35B-A3B",
        "vllm.model_executor.models.qwen3_5",
        "Qwen3_5DecoderLayer",
        {"layer_type": "full_attention"},
        text_config=True,
    ),
    Candidate(
        "Google",
        "google/gemma-3-27b-it",
        "vllm.model_executor.models.gemma3",
        "Gemma3DecoderLayer",
        text_config=True,
    ),
    Candidate(
        "Google (#3096 target)",
        "google/gemma-4-31b-it",
        "vllm.model_executor.models.gemma4",
        "Gemma4DecoderLayer",
        {"layer_idx": 0},
        text_config=True,
    ),
    Candidate(
        "Cohere",
        "CohereForAI/c4ai-command-r-v01",
        "vllm.model_executor.models.commandr",
        "CohereDecoderLayer",
    ),
)

# Pattern-level candidates from vLLM main's static scan (files whose decoder
# forward is plain pre-norm), checked with ``--main`` against a vLLM main
# checkout; the families v0.33.0 lacks are only checkable there.
_M = "vllm.model_executor.models."
MAIN_CANDIDATES = (
    Candidate(
        "Mistral",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        _M + "mixtral",
        "MixtralDecoderLayer",
    ),
    Candidate("GLM (Zhipu)", "zai-org/GLM-4-32B-0414", _M + "glm4", "Glm4DecoderLayer"),
    Candidate(
        "Tencent",
        "tencent/Hunyuan-A13B-Instruct",
        _M + "hunyuan_v1",
        "HunYuanDecoderLayer",
    ),
    Candidate(
        "inclusionAI",
        "inclusionAI/Ling-lite-1.5",
        _M + "bailing_moe",
        "BailingMoeBlock",
    ),
    Candidate(
        "NVIDIA", "nvidia/Minitron-8B-Base", _M + "nemotron", "NemotronDecoderLayer"
    ),
    Candidate(
        "Upstage",
        "upstage/solar-pro-preview-instruct",
        _M + "solar",
        "SolarDecoderLayer",
    ),
    Candidate("Arcee", "arcee-ai/Trinity-Mini", _M + "afmoe", "AfmoeDecoderLayer"),
    Candidate("Qwen", "Qwen/Qwen2.5-72B-Instruct", _M + "qwen2", "Qwen2DecoderLayer"),
    Candidate(
        "Qwen", "Qwen/Qwen1.5-MoE-A2.7B", _M + "qwen2_moe", "Qwen2MoeDecoderLayer"
    ),
    Candidate("Google", "google/gemma-7b", _M + "gemma", "GemmaDecoderLayer"),
    Candidate("AllenAI", "allenai/Molmo-7B-D-0924", _M + "molmo", "MolmoDecoderLayer"),
    Candidate(
        "StepFun",
        "stepfun-ai/step3",
        _M + "step3_text",
        "Step3TextDecoderLayer",
        text_config=True,
    ),
    Candidate(
        "LG",
        "LGAI-EXAONE/K-EXAONE-236B-A23B",
        _M + "exaone_moe",
        "ExaoneMoeDecoderLayer",
    ),
    Candidate(
        "Shanghai AI Lab",
        "internlm/Intern-S1-Pro",
        _M + "interns1_pro",
        "InternS1ProMoeDecoderLayer",
        text_config=True,
    ),
)

_SHRINK_HEADS = dict(
    hidden_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=128,
)
_SHRINK = dict(
    intermediate_size=256,
    moe_intermediate_size=32,
    num_hidden_layers=2,
    vocab_size=128,
    max_position_embeddings=128,
)
_EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def _config(c: Candidate, *, shrink_heads: bool = True):  # noqa: ANN202
    """The real config, shrunk; ``shrink_heads=False`` keeps the head geometry."""
    raw = json.load(open(hf_hub_download(c.repo, "config.json")))  # noqa: SIM115
    if c.text_config:
        raw = raw.get("text_config", raw)
    kind = {"kimi_k2": "deepseek_v3"}.get(raw.get("model_type"), raw.get("model_type"))
    drop = ("architectures", "auto_map", "model_type", "quantization_config")
    data = {k: v for k, v in raw.items() if k not in drop}
    data.update(_SHRINK)
    if shrink_heads:
        data.update(_SHRINK_HEADS)
    for key in _EXPERTS:
        if key in data:
            data[key] = 8
    if "num_experts_per_tok" in data:
        data["num_experts_per_tok"] = 2
    for key, small in (
        ("kv_lora_rank", 32),
        ("q_lora_rank", 32),
        ("qk_nope_head_dim", 16),
        ("qk_rope_head_dim", 8),
        ("v_head_dim", 16),
        ("rotary_dim", 8),
        ("shared_expert_intermediate_size", 32),
        ("moe_shared_expert_intermediate_size", 32),
        ("n_shared_experts", 1),
        ("first_k_dense_replace", 0),
        ("n_group", 1),
        ("topk_group", 1),
        ("sliding_window", 16),
    ):
        if data.get(key) is not None:
            data[key] = small
    # Multimodal RoPE sections are sized for the real head; the rotary is
    # opaque to the check, so a shrunk layer uses plain RoPE.
    for key in ("rope_parameters", "rope_scaling"):
        if isinstance(data.get(key), dict):
            data[key] = {
                k: v for k, v in data[key].items() if not k.startswith("mrope")
            }
    for key in ("layer_types", "mlp_layer_types"):
        if isinstance(data.get(key), list):
            data[key] = data[key][:2]
    if kind in transformers.CONFIG_MAPPING:
        return transformers.AutoConfig.for_model(kind, **data)
    # A type transformers does not know: vLLM's own config class, if any.
    configs = importlib.import_module("vllm.transformers_utils.configs")
    cls = next(
        (
            v
            for v in vars(configs).values()
            if isinstance(v, type) and getattr(v, "model_type", None) == kind
        ),
        None,
    )
    if cls is None:
        raise KeyError(f"no config class for model_type {kind!r}")
    return cls(**data)


def _child(layer: nn.Module, *names: str) -> nn.Module:
    for name in names:
        if isinstance(child := getattr(layer, name, None), nn.Module):
            return child
    raise AttributeError(f"the layer has none of {names}")


def guess_binding(layer: nn.Module) -> NormAttnBinding:
    """The binding under vLLM's usual names, for the survey (not a declaration).

    A model owner's binding function reads its own class's attributes; the
    survey runs over many classes, so it tries the common names.
    """
    attention = _child(layer, "self_attn", "attn")
    norms = [getattr(attention, n, None) for n in ("q_norm", "k_norm")]
    return NormAttnBinding(
        input_norm=_child(layer, "input_layernorm"),
        qkv_proj=_child(attention, "qkv_proj"),
        attn=_child(attention, "attn"),
        o_proj=_child(attention, "o_proj"),
        rotary=_child(attention, "rotary_emb"),
        q_norm=norms[0],
        k_norm=norms[1],
        attention=attention,
        post_norm=_child(layer, "post_attention_layernorm"),
        mlp=_child(layer, "mlp", "block_sparse_moe", "feed_forward"),
        output_gate=bool(getattr(attention, "attn_output_gate", False)),
        stream=guess_stream(layer),
        attend=attend_with_positions(attention),  # pyright: ignore[reportArgumentType]
    )


def guess_stream(layer: nn.Module) -> Stream:
    """The survey reads the forward's parameter order; a declaration states it."""
    order = tuple(inspect.signature(type(layer).forward).parameters)[1:4]
    if order == ("hidden_states", "positions", "residual"):
        return hidden_positions_residual(layer)  # pyright: ignore[reportArgumentType]
    if order == ("hidden_states", "residual", "positions"):
        return hidden_residual_positions(layer)  # pyright: ignore[reportArgumentType]
    return positions_first(layer)  # pyright: ignore[reportArgumentType]


def check(c: Candidate) -> tuple[str, str]:
    try:
        cfg = _config(c, shrink_heads=False)
        spec = _build.Spec(
            c.repo, lambda: cfg, c.module, c.layer_class, c.extra, c.model
        )
        layer = _build.build(spec)
    except Exception as e:  # noqa: BLE001 - report why it does not build
        first = (str(e).splitlines() or [""])[0][:90]
        return "not buildable", f"{type(e).__name__}: {first}"
    declaration = BlockDeclaration.of(type(layer), guess_binding)
    try:
        with set_current_vllm_config(_build.current_config()):
            config = norm_attn.conform(layer, declaration).config
    except (BlockConformanceError, AttributeError) as e:
        return "declines", str(e).split(", but ", 1)[-1][:110]
    # Capability on the model's real head sizes, which the layer was built with.
    why = fused_qkv_supports(facts_of(guess_binding(layer), config))
    return "conforms", f"fused_qkv: {'declines: ' + why if why else 'can serve it'}"


def main() -> None:
    rows = []
    for c in MAIN_CANDIDATES if "--main" in sys.argv else CANDIDATES:
        verdict, why = check(c)
        target = "#3096" if c.layer_class in PR_3096_TARGETS else ""
        rows.append((c, verdict, why, target))
        print(
            f"  {c.family:22} {c.repo:46} {c.layer_class:38} {verdict:13} "
            f"{target:6} {why}"
        )
    conforming = [r for r in rows if r[1] == "conforms"]
    new = [r for r in conforming if not r[3]]
    families = sorted({r[0].family for r in new})
    print(
        f"  conform: {len(conforming)} of {len(rows)}; not a #3096 target: "
        f"{len(new)} ({', '.join(families)})"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
