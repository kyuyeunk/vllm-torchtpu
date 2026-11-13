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
targets are marked. ``python -m prototype.subblock_reach`` prints the table
(``--main`` for ``MAIN_CANDIDATES`` against a vLLM main checkout);
``test_subblock_reach.py`` asserts each candidate's verdict.
"""

import dataclasses
import enum
import importlib
import inspect
import json
import pathlib
from collections.abc import Sequence
from typing import Any

import transformers
from huggingface_hub import hf_hub_download
from torch import nn
from vllm.config import set_current_vllm_config

from tests.blocks import _build
from tests.blocks._report import Emit
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


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One current model: its family, repo and decoder class.

    ``pr_3096``: #3096 patches the decoder class (fused_attn_proj_patch.py
    patches Qwen3MoeDecoderLayer, Qwen3NextDecoderLayer, Qwen3_5DecoderLayer
    and Gemma4DecoderLayer).
    """

    family: str
    repo: str
    module: str
    layer_class: str
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    text_config: bool = False
    model: dict[str, Any] = dataclasses.field(default_factory=dict)
    pr_3096: bool = False


GLM_4_5_AIR = Candidate(
    "GLM (Zhipu)",
    "zai-org/GLM-4.5-Air",
    "vllm.model_executor.models.glm4_moe",
    "Glm4MoeDecoderLayer",
)
MINISTRAL_8B = Candidate(
    "Mistral",
    "mistralai/Ministral-8B-Instruct-2410",
    "vllm.model_executor.models.mistral",
    "MistralDecoderLayer",
)
GPT_OSS_20B = Candidate(
    "OpenAI",
    "openai/gpt-oss-20b",
    "vllm.model_executor.models.gpt_oss",
    "TransformerBlock",
)
QWEN3_32B = Candidate(
    "Qwen",
    "Qwen/Qwen3-32B",
    "vllm.model_executor.models.qwen3",
    "Qwen3DecoderLayer",
)

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
    GLM_4_5_AIR,
    MINISTRAL_8B,
    Candidate(
        "Mistral",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "vllm.model_executor.models.mistral",
        "MistralDecoderLayer",
        text_config=True,
    ),
    GPT_OSS_20B,
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
        pr_3096=True,
    ),
    QWEN3_32B,
    Candidate(
        "Qwen (#3096 target)",
        "Qwen/Qwen3.5-35B-A3B",
        "vllm.model_executor.models.qwen3_5",
        "Qwen3_5DecoderLayer",
        {"layer_type": "full_attention"},
        text_config=True,
        pr_3096=True,
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
        pr_3096=True,
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

# config.json fields, overwritten in the parsed file (the parse boundary).
_SHRINK_HEADS = {
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
}
_SHRINK = {
    "intermediate_size": 256,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 128,
}
_EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def config_of(c: Candidate, *, shrink_heads: bool = True) -> Any:  # noqa: ANN401 - a transformers or vLLM config class
    """The real config, shrunk; ``shrink_heads=False`` keeps the head geometry.

    config.json is keyed by field name: this function is where it is parsed,
    so its string keys are the file's (``# str-ok`` on each).
    """
    path = pathlib.Path(hf_hub_download(c.repo, "config.json"))
    raw = json.loads(path.read_text())
    if c.text_config:
        raw = raw.get("text_config", raw)  # str-ok: config.json field
    model_type = raw.get("model_type")  # str-ok: config.json field
    aliases = {"kimi_k2": "deepseek_v3"}
    kind = aliases.get(model_type, model_type)
    drop = ("architectures", "auto_map", "model_type", "quantization_config")
    data = {k: v for k, v in raw.items() if k not in drop}
    _shrink(data, shrink_heads=shrink_heads)
    if kind in transformers.CONFIG_MAPPING:
        return transformers.AutoConfig.for_model(kind, **data)
    return _vllm_config_class(kind)(**data)


def _shrink(data: dict[str, Any], *, shrink_heads: bool) -> None:
    """Shrinks the parsed config.json's sizes in place (structure kept)."""
    data.update(_SHRINK)
    if shrink_heads:
        data.update(_SHRINK_HEADS)
    for key in _EXPERTS:
        if key in data:
            data[key] = 8
    if "num_experts_per_tok" in data:  # str-ok: config.json field
        data["num_experts_per_tok"] = 2  # str-ok: config.json field
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
    _trim(data)


def _trim(data: dict[str, Any]) -> None:
    """Drops what the shrunk layer cannot use: mRoPE sections, extra layers."""
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


def _vllm_config_class(kind: str) -> type:
    """VLLM's own config class for a model_type transformers does not know."""
    configs = importlib.import_module("vllm.transformers_utils.configs")
    for v in vars(configs).values():
        if not isinstance(v, type):
            continue
        model_type = getattr(v, "model_type", None)  # str-ok: config classes
        if model_type == kind:
            return v
    msg = f"no config class for model_type {kind!r}"
    raise KeyError(msg)


def _child(layer: nn.Module, *names: str) -> nn.Module:
    """The first of ``names`` that is a child of ``layer`` (the survey's guess)."""
    for name in names:
        # The survey tries vLLM's usual names on classes no model declares.
        child = getattr(layer, name, None)  # str-ok: the survey's guess
        if isinstance(child, nn.Module):
            return child
    msg = f"the layer has none of {names}"
    raise AttributeError(msg)


def guess_binding(layer: nn.Module) -> NormAttnBinding:
    """The binding under vLLM's usual names, for the survey (not a declaration).

    A model owner's binding function reads its own class's attributes; the
    survey runs over many classes, so it tries the common names.
    """
    attention = _child(layer, "self_attn", "attn")
    q_norm = getattr(attention, "q_norm", None)  # str-ok: the survey's guess
    k_norm = getattr(attention, "k_norm", None)  # str-ok: the survey's guess
    gate = getattr(attention, "attn_output_gate", False)  # str-ok: ditto
    return NormAttnBinding(
        input_norm=_child(layer, "input_layernorm"),
        qkv_proj=_child(attention, "qkv_proj"),
        attn=_child(attention, "attn"),
        o_proj=_child(attention, "o_proj"),
        rotary=_child(attention, "rotary_emb"),
        q_norm=q_norm,
        k_norm=k_norm,
        attention=attention,
        post_norm=_child(layer, "post_attention_layernorm"),
        mlp=_child(layer, "mlp", "block_sparse_moe", "feed_forward"),
        output_gate=bool(gate),
        stream=guess_stream(layer),
        attend=attend_with_positions(attention),  # pyright: ignore[reportArgumentType]
    )


def guess_stream(layer: nn.Module) -> Stream:
    """The survey reads the forward's parameter order; a declaration states it."""
    order = tuple(inspect.signature(type(layer).forward).parameters)[1:4]
    # str-ok below: the survey reads the forward's parameter names.
    if order == ("hidden_states", "positions", "residual"):  # str-ok: names
        return hidden_positions_residual(layer)  # pyright: ignore[reportArgumentType]
    if order == ("hidden_states", "residual", "positions"):  # str-ok: names
        return hidden_residual_positions(layer)  # pyright: ignore[reportArgumentType]
    return positions_first(layer)  # pyright: ignore[reportArgumentType]


class Verdict(enum.Enum):
    """What the survey found for one candidate."""

    _value_: str

    CONFORMS = "conforms"
    DECLINES = "declines"
    NOT_BUILDABLE = "not buildable"


def check(c: Candidate) -> tuple[Verdict, str]:
    """The candidate's verdict, and why (or what fused_qkv says)."""
    try:
        cfg = config_of(c, shrink_heads=False)
        spec = _build.Spec(
            c.repo, lambda: cfg, c.module, c.layer_class, c.extra, c.model
        )
        layer = _build.build(spec)
    except Exception as e:  # noqa: BLE001 - report why it does not build
        first = (str(e).splitlines() or [""])[0][:90]
        return Verdict.NOT_BUILDABLE, f"{type(e).__name__}: {first}"
    declaration = BlockDeclaration.of(type(layer), guess_binding)
    try:
        with set_current_vllm_config(_build.current_config()):
            config = norm_attn.conform(layer, declaration).config
    except (BlockConformanceError, AttributeError) as e:
        return Verdict.DECLINES, str(e).split(", but ", 1)[-1][:110]
    # Capability on the model's real head sizes, which the layer was built with.
    why = fused_qkv_supports(facts_of(guess_binding(layer), config))
    serves = f"declines: {why}" if why else "can serve it"
    return Verdict.CONFORMS, f"fused_qkv: {serves}"


def report(candidates: Sequence[Candidate], emit: Emit) -> None:
    """Emits one row per candidate, then the count of those that conform."""
    rows: list[tuple[Candidate, Verdict]] = []
    for c in candidates:
        verdict, why = check(c)
        target = "#3096" if c.pr_3096 else ""
        rows.append((c, verdict))
        emit(
            f"  {c.family:22} {c.repo:46} {c.layer_class:38} {verdict.value:13} "
            f"{target:6} {why}"
        )
    conforming = [c for c, verdict in rows if verdict is Verdict.CONFORMS]
    new = [c for c in conforming if not c.pr_3096]
    families = sorted({c.family for c in new})
    emit(
        f"  conform: {len(conforming)} of {len(rows)}; not a #3096 target: "
        f"{len(new)} ({', '.join(families)})"
    )
