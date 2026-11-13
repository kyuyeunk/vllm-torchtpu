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
"""Builds Qwen3NextAttention on the meta device for the attention-layer checks.

Building the layer needs vLLM's model-parallel groups and a current
VllmConfig; a single-rank gloo world provides both on a CPU. Building goes
through upstream's class, so the expansion point decides what is built.
"""

import contextlib
import os
from collections.abc import Generator

import portpicker
import torch
from transformers import Qwen3NextConfig
from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.models import qwen3_next
from vllm.platforms.interface import PlatformEnum

from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod


def tiny_config(
    *, attn_output_gate: bool = True, model_plan: str | None = None
) -> Qwen3NextConfig:
    """A tiny Qwen3-Next full-attention layer: gated output, partial NeoX RoPE.

    Args:
        attn_output_gate: Whether the layer gates its output.
        model_plan: The model plan text, set as the config's
            ``tpu_attention_layer_plan``; None leaves the key unset.

    Returns:
        The config.
    """
    config = Qwen3NextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,  # pyright: ignore[reportCallIssue]  # HF config extra
        attn_output_gate=attn_output_gate,  # pyright: ignore[reportCallIssue]  # HF config extra
    )
    if model_plan is not None:
        config.tpu_attention_layer_plan = model_plan
    return config


@contextlib.contextmanager
def runtime() -> Generator[None]:
    """Single-rank world, model-parallel groups and bf16 default dtype.

    Yields:
        Nothing; the world stays initialized for the rest of the process.
    """
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
    yield


def build(
    config: Qwen3NextConfig,
    *,
    chip: PlatformEnum = PlatformEnum.TPU,
    override: str = "",
) -> layer_mod.TpuQwen3NextAttention:
    """Builds the upstream class on meta, describing the given chip.

    Args:
        config: The model config.
        chip: The chip the layer's plan is resolved for.
        override: Set as the env var's text, the way a user pins a plan.

    Returns:
        The replacement class the expansion point built.
    """
    real_describe_chip = layer_mod.describe_chip
    layer_mod.describe_chip = lambda: alp.ChipDesc(chip)
    os.environ["TPU_ATTENTION_LAYER_PLAN"] = override  # str-ok: env var boundary
    try:
        with set_current_vllm_config(VllmConfig()), torch.device("meta"):
            layer = qwen3_next.Qwen3NextAttention(
                config, prefix="model.layers.3.self_attn"
            )
    finally:
        layer_mod.describe_chip = real_describe_chip
        _ = os.environ.pop("TPU_ATTENTION_LAYER_PLAN")  # str-ok: env var
    # The expansion point decides the class; the checks need the replacement.
    assert isinstance(layer, layer_mod.TpuQwen3NextAttention)
    return layer
