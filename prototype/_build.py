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
"""Builds one decoder layer of each surveyed vLLM MoE model on the meta device.

Every layer is built from a tiny config through the model's own constructor in
a single-rank gloo world, with vLLM's platform pinned to CPU by the package
``__init__``. Nothing loads libtpu. Constructor arguments are supplied by
name, since each model takes a different subset of (vllm_config, config,
cache_config, quant_config, ...).
"""

import contextlib
import dataclasses
import importlib
import inspect
import types
from collections.abc import Callable, Iterator
from typing import Any

import portpicker
import torch
import transformers
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)

# Shared tiny sizes: hidden 64, 4 heads of 16, 8 experts of width 32, top-2.
_TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    num_hidden_layers=2,
    max_position_embeddings=128,
    rms_norm_eps=1e-6,
    vocab_size=128,
)


@dataclasses.dataclass(frozen=True)
class Spec:
    """One surveyed model: its config, decoder class and layer arguments."""

    name: str
    config: Callable[[], Any]
    module: str
    layer_class: str
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    model: dict[str, Any] = dataclasses.field(default_factory=dict)


def _cfg(cls_name: str, **overrides: Any) -> Callable[[], Any]:  # noqa: ANN401
    def make() -> Any:  # noqa: ANN401
        cls = getattr(transformers, cls_name)
        return cls(**(_TINY | overrides))

    return make


_MOE = dict(num_experts=8, num_experts_per_tok=2)
SPECS: tuple[Spec, ...] = (
    Spec(
        "Qwen3.5-MoE (full attention)",
        _cfg("Qwen3_5MoeTextConfig", **_MOE, shared_expert_intermediate_size=32),
        "vllm.model_executor.models.qwen3_5",
        "Qwen3_5DecoderLayer",
        {"layer_type": "full_attention"},
    ),
    Spec(
        "Qwen3.5-MoE (linear attention)",
        _cfg(
            "Qwen3_5MoeTextConfig",
            **_MOE,
            shared_expert_intermediate_size=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        ),
        "vllm.model_executor.models.qwen3_5",
        "Qwen3_5DecoderLayer",
        {"layer_type": "linear_attention"},
    ),
    Spec(
        "Qwen3-Next",
        _cfg(
            "Qwen3NextConfig",
            **_MOE,
            shared_expert_intermediate_size=32,
            decoder_sparse_step=1,
        ),
        "vllm.model_executor.models.qwen3_next",
        "Qwen3NextDecoderLayer",
        {"layer_type": "full_attention"},
    ),
    Spec(
        "Qwen3-MoE",
        _cfg("Qwen3MoeConfig", **_MOE, decoder_sparse_step=1),
        "vllm.model_executor.models.qwen3_moe",
        "Qwen3MoeDecoderLayer",
    ),
    Spec(
        "Qwen2-MoE",
        _cfg(
            "Qwen2MoeConfig",
            **_MOE,
            shared_expert_intermediate_size=32,
            decoder_sparse_step=1,
        ),
        "vllm.model_executor.models.qwen2_moe",
        "Qwen2MoeDecoderLayer",
    ),
    Spec(
        "GLM-4 MoE",
        _cfg(
            "Glm4MoeConfig",
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
        ),
        "vllm.model_executor.models.glm4_moe",
        "Glm4MoeDecoderLayer",
    ),
    Spec(
        "Mixtral",
        _cfg("MixtralConfig", num_local_experts=8, num_experts_per_tok=2),
        "vllm.model_executor.models.mixtral",
        "MixtralDecoderLayer",
    ),
    Spec(
        "MiniMax-M2",
        _cfg(
            "MiniMaxM2Config",
            num_local_experts=8,
            num_experts_per_tok=2,
            rotary_dim=8,
            scoring_func="sigmoid",
            use_routing_bias=True,
        ),
        "vllm.model_executor.models.minimax_m2",
        "MiniMaxM2DecoderLayer",
    ),
    Spec(
        "Llama4 (MoE layer)",
        _cfg(
            "Llama4TextConfig",
            num_local_experts=8,
            num_experts_per_tok=1,
            interleave_moe_layer_step=1,
            intermediate_size_mlp=128,
            no_rope_layers=[1, 1],
        ),
        "vllm.model_executor.models.llama4",
        "Llama4DecoderLayer",
    ),
    Spec(
        "GPT-OSS",
        _cfg(
            "GptOssConfig",
            num_local_experts=8,
            num_experts_per_tok=2,
            intermediate_size=32,
            sliding_window=16,
        ),
        "vllm.model_executor.models.gpt_oss",
        "TransformerBlock",
    ),
    Spec(
        "DeepSeek-V3",
        _cfg(
            "DeepseekV3Config",
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
        ),
        "vllm.model_executor.models.deepseek_v2",
        "DeepseekV2DecoderLayer",
        model={"use_mla": True},
    ),
    Spec(
        "Bailing-MoE",
        _cfg(
            "PretrainedConfig",
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_shared_expert_intermediate_size=32,
            norm_topk_prob=True,
            use_bias=False,
            use_qkv_bias=False,
            first_k_dense_replace=0,
            hidden_act="silu",
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        ),
        "vllm.model_executor.models.bailing_moe",
        "BailingMoeBlock",
    ),
    Spec(
        "Kimi-Linear (nvidia)",
        lambda: importlib.import_module(
            "vllm.transformers_utils.configs.kimi_linear"
        ).KimiLinearConfig(
            **_TINY,
            num_experts=8,
            num_experts_per_token=2,
            num_shared_experts=1,
            first_k_dense_replace=0,
            moe_layer_freq=1,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            kv_lora_rank=32,
            mla_use_nope=True,
        ),
        "vllm.models.kimi_k3.nvidia.model",
        "KimiDecoderLayer",
        model={"use_mla": True},
    ),
    Spec(
        "Jamba (attention layer)",
        _cfg(
            "JambaConfig",
            num_experts=8,
            num_experts_per_tok=2,
            expert_layer_period=1,
            expert_layer_offset=0,
            attn_layer_period=1,
            attn_layer_offset=0,
        ),
        "vllm.model_executor.models.jamba",
        "JambaAttentionDecoderLayer",
        {"layer_idx": 0},
    ),
    Spec(
        "Jamba (Mamba layer)",
        _cfg(
            "JambaConfig",
            num_experts=8,
            num_experts_per_tok=2,
            expert_layer_period=1,
            expert_layer_offset=0,
            mamba_d_state=8,
            mamba_dt_rank=8,
        ),
        "vllm.model_executor.models.jamba",
        "JambaMambaDecoderLayer",
        {"layer_idx": 0},
    ),
    Spec(
        "Nemotron-H (MoE layer)",
        _cfg(
            "NemotronHConfig",
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_shared_expert_intermediate_size=32,
            n_group=1,
            topk_group=1,
        ),
        "vllm.model_executor.models.nemotron_h",
        "NemotronHMoEDecoderLayer",
        {"layer_idx": 0},
    ),
)


class _AnyModelConfig(types.SimpleNamespace):
    """A model config stand-in: unknown attributes read as None."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - any attribute
        del name
        return None


_WORLD = False


def _world() -> None:
    global _WORLD  # noqa: PLW0603 - one world per process
    if not _WORLD:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{portpicker.pick_unused_port()}",
            backend="gloo",
        )
        with set_current_vllm_config(VllmConfig()):
            ensure_model_parallel_initialized(1, 1)
        torch.set_default_dtype(torch.bfloat16)
        _WORLD = True


@contextlib.contextmanager
def _config_for(hf: Any, model: dict[str, Any]) -> Iterator[VllmConfig]:  # noqa: ANN401
    config = VllmConfig()
    fields = {
        "hf_config": hf,
        "hf_text_config": hf,
        "dtype": torch.bfloat16,
        "is_attention_free": False,
        "use_mla": False,
        "max_model_len": 128,
        "get_head_size": lambda: 16,
    }
    config.model_config = _AnyModelConfig(**(fields | model))  # pyright: ignore[reportAttributeAccessIssue]
    with set_current_vllm_config(config):
        yield config


_LAST: list[VllmConfig] = []


def current_config() -> VllmConfig:
    """The config of the last build, for running what was built."""
    return _LAST[-1]


def build(
    spec: Spec, prefix: str = "model.layers.0", device: str = "meta"
) -> torch.nn.Module:
    """Builds the spec's decoder layer through its own constructor."""
    _world()
    hf = spec.config()
    cls = getattr(importlib.import_module(spec.module), spec.layer_class)
    with _config_for(hf, spec.model) as config, torch.device(device):
        _LAST[:] = [config]
        supply = {
            "vllm_config": config,
            "config": hf,
            "model_config": config.model_config,
            "cache_config": config.cache_config,
            "quant_config": None,
            "parallel_config": config.parallel_config,
            "prefix": prefix,
        } | spec.extra
        params = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in supply.items() if k in params}
        return cls(**kwargs)
