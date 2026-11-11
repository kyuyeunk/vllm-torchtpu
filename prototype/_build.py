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
``__init__``. Nothing loads libtpu. Each spec calls its class's constructor
with the arguments that class takes (``make``), since each model takes a
different subset of (vllm_config, config, cache_config, quant_config, ...).
"""

import contextlib
import dataclasses
import types
from collections.abc import Callable, Iterator
from typing import Any

import portpicker
import torch
import transformers
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.models import nemotron_h
from vllm.model_executor.models.bailing_moe import BailingMoeBlock
from vllm.model_executor.models.deepseek_v2 import DeepseekV2DecoderLayer
from vllm.model_executor.models.glm4_moe import Glm4MoeDecoderLayer
from vllm.model_executor.models.gpt_oss import TransformerBlock
from vllm.model_executor.models.jamba import (
    JambaAttentionDecoderLayer,
    JambaMambaDecoderLayer,
)
from vllm.model_executor.models.llama4 import Llama4DecoderLayer
from vllm.model_executor.models.minimax_m2 import MiniMaxM2DecoderLayer
from vllm.model_executor.models.mixtral import MixtralDecoderLayer
from vllm.model_executor.models.qwen2_moe import Qwen2MoeDecoderLayer
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextDecoderLayer
from vllm.models.kimi_k3.nvidia.model import KimiDecoderLayer
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig


def _tiny[C](
    cls: Callable[..., C], *, intermediate_size: int = 128, **overrides: Any
) -> Callable[[], C]:
    """The config ``cls`` at the shared tiny sizes, with ``overrides``.

    Hidden 64, 4 heads of 16, 8 experts of width 32, top-2.
    """

    def make() -> C:
        return cls(
            hidden_size=64,
            intermediate_size=intermediate_size,
            moe_intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128,
            **overrides,
        )

    return make


@dataclasses.dataclass(frozen=True)
class Built:
    """What a spec's constructor call reads."""

    vllm_config: VllmConfig
    hf: Any
    prefix: str


@dataclasses.dataclass(frozen=True)
class Spec:
    """One surveyed model: its config, decoder class and constructor call."""

    name: str
    config: Callable[[], Any]
    layer_class: type[nn.Module]
    make: Callable[[Built], nn.Module]
    use_mla: bool = False


def _hf_args(b: Built) -> tuple[Any, Any]:
    """``(config, cache_config)``, the arguments most older layers take first."""
    return b.hf, b.vllm_config.cache_config


SPECS: tuple[Spec, ...] = (
    Spec(
        "Qwen3.5-MoE (full attention)",
        _tiny(
            transformers.Qwen3_5MoeTextConfig,
            num_experts=8,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=32,
        ),
        Qwen3_5DecoderLayer,
        lambda b: Qwen3_5DecoderLayer(
            b.vllm_config, layer_type="full_attention", prefix=b.prefix
        ),
    ),
    Spec(
        "Qwen3.5-MoE (linear attention)",
        _tiny(
            transformers.Qwen3_5MoeTextConfig,
            num_experts=8,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=32,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        ),
        Qwen3_5DecoderLayer,
        lambda b: Qwen3_5DecoderLayer(
            b.vllm_config, layer_type="linear_attention", prefix=b.prefix
        ),
    ),
    Spec(
        "Qwen3-Next",
        _tiny(
            transformers.Qwen3NextConfig,
            num_experts=8,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=32,
            decoder_sparse_step=1,
        ),
        Qwen3NextDecoderLayer,
        lambda b: Qwen3NextDecoderLayer(
            b.vllm_config, layer_type="full_attention", prefix=b.prefix
        ),
    ),
    Spec(
        "Qwen3-MoE",
        _tiny(
            transformers.Qwen3MoeConfig,
            num_experts=8,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
        ),
        Qwen3MoeDecoderLayer,
        lambda b: Qwen3MoeDecoderLayer(b.vllm_config, prefix=b.prefix),
    ),
    Spec(
        "Qwen2-MoE",
        _tiny(
            transformers.Qwen2MoeConfig,
            num_experts=8,
            num_experts_per_tok=2,
            shared_expert_intermediate_size=32,
            decoder_sparse_step=1,
        ),
        Qwen2MoeDecoderLayer,
        lambda b: Qwen2MoeDecoderLayer(
            *_hf_args(b), quant_config=None, prefix=b.prefix
        ),
    ),
    Spec(
        "GLM-4 MoE",
        _tiny(
            transformers.Glm4MoeConfig,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            first_k_dense_replace=0,
            n_group=1,
            topk_group=1,
        ),
        Glm4MoeDecoderLayer,
        lambda b: Glm4MoeDecoderLayer(*_hf_args(b), quant_config=None, prefix=b.prefix),
    ),
    Spec(
        "Mixtral",
        _tiny(transformers.MixtralConfig, num_local_experts=8, num_experts_per_tok=2),
        MixtralDecoderLayer,
        lambda b: MixtralDecoderLayer(*_hf_args(b), quant_config=None, prefix=b.prefix),
    ),
    Spec(
        "MiniMax-M2",
        _tiny(
            transformers.MiniMaxM2Config,
            num_local_experts=8,
            num_experts_per_tok=2,
            rotary_dim=8,
            scoring_func="sigmoid",
            use_routing_bias=True,
        ),
        MiniMaxM2DecoderLayer,
        lambda b: MiniMaxM2DecoderLayer(
            b.hf,
            prefix=b.prefix,
            model_config=b.vllm_config.model_config,
            cache_config=b.vllm_config.cache_config,
            quant_config=None,
        ),
    ),
    Spec(
        "Llama4 (MoE layer)",
        _tiny(
            transformers.Llama4TextConfig,
            num_local_experts=8,
            num_experts_per_tok=1,
            interleave_moe_layer_step=1,
            intermediate_size_mlp=128,
            no_rope_layers=[1, 1],
        ),
        Llama4DecoderLayer,
        lambda b: Llama4DecoderLayer(b.vllm_config, prefix=b.prefix, config=b.hf),
    ),
    Spec(
        "GPT-OSS",
        _tiny(
            transformers.GptOssConfig,
            num_local_experts=8,
            num_experts_per_tok=2,
            intermediate_size=32,
            sliding_window=16,
        ),
        TransformerBlock,
        lambda b: TransformerBlock(b.vllm_config, quant_config=None, prefix=b.prefix),  # pyright: ignore[reportArgumentType]
    ),
    Spec(
        "DeepSeek-V3",
        _tiny(
            transformers.DeepseekV3Config,
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
        DeepseekV2DecoderLayer,
        lambda b: DeepseekV2DecoderLayer(b.vllm_config, prefix=b.prefix, config=b.hf),
        use_mla=True,
    ),
    Spec(
        "Bailing-MoE",
        _tiny(
            transformers.PretrainedConfig,
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
        BailingMoeBlock,
        lambda b: BailingMoeBlock(*_hf_args(b), quant_config=None, prefix=b.prefix),
    ),
    Spec(
        "Kimi-Linear (nvidia)",
        _tiny(
            KimiLinearConfig,
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
        KimiDecoderLayer,
        lambda b: KimiDecoderLayer(b.hf, vllm_config=b.vllm_config, prefix=b.prefix),
        use_mla=True,
    ),
    Spec(
        "Jamba (attention layer)",
        _tiny(
            transformers.JambaConfig,
            num_experts=8,
            num_experts_per_tok=2,
            expert_layer_period=1,
            expert_layer_offset=0,
            attn_layer_period=1,
            attn_layer_offset=0,
        ),
        JambaAttentionDecoderLayer,
        lambda b: JambaAttentionDecoderLayer(
            b.hf,
            layer_idx=0,
            model_config=b.vllm_config.model_config,
            cache_config=b.vllm_config.cache_config,
            quant_config=None,
            prefix=b.prefix,
        ),
    ),
    Spec(
        "Jamba (Mamba layer)",
        _tiny(
            transformers.JambaConfig,
            num_experts=8,
            num_experts_per_tok=2,
            expert_layer_period=1,
            expert_layer_offset=0,
            mamba_d_state=8,
            mamba_dt_rank=8,
        ),
        JambaMambaDecoderLayer,
        lambda b: JambaMambaDecoderLayer(
            b.hf,
            layer_idx=0,
            model_config=b.vllm_config.model_config,
            cache_config=b.vllm_config.cache_config,
            quant_config=None,
            prefix=b.prefix,
        ),
    ),
    Spec(
        "Nemotron-H (MoE layer)",
        _tiny(
            transformers.NemotronHConfig,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_shared_expert_intermediate_size=32,
            n_group=1,
            topk_group=1,
        ),
        nemotron_h.NemotronHMoEDecoderLayer,
        lambda b: nemotron_h.NemotronHMoEDecoderLayer(
            b.hf,
            layer_idx=0,
            model_config=b.vllm_config.model_config,
            cache_config=b.vllm_config.cache_config,
            quant_config=None,
            parallel_config=b.vllm_config.parallel_config,
            prefix=b.prefix,
        ),
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
def _config_for(hf: Any, use_mla: bool) -> Iterator[VllmConfig]:  # noqa: ANN401
    config = VllmConfig()
    config.model_config = _AnyModelConfig(  # pyright: ignore[reportAttributeAccessIssue]
        hf_config=hf,
        hf_text_config=hf,
        dtype=torch.bfloat16,
        is_attention_free=False,
        use_mla=use_mla,
        max_model_len=128,
        get_head_size=lambda: 16,
    )
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
    with _config_for(hf, spec.use_mla) as config, torch.device(device):
        _LAST[:] = [config]
        return spec.make(Built(config, hf, prefix))
