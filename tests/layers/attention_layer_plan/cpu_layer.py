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
"""Builds the attention layer with real weights on a CPU, with no TPU runtime.

Import this module first. It keeps vLLM from detecting the TPU (no libtpu, no
plugin platform), so the checks never touch the device and need no lock.
The attention kernel is the layer's child: the layer's plan hands it only its
output format, so it is replaced by `RefAttention`, a plain causal
grouped-query attention with that format. The checks then cover the layer's
own decisions, which is what its plan owns.
"""

# isort: off
from tests.layers.attention_layer_plan import no_tpu_runtime  # noqa: F401  # pyright: ignore[reportUnusedImport]  # first: no TPU runtime

# isort: on
import typing

import torch
from torch import nn
from transformers import Qwen3NextConfig
from vllm.model_executor.models import qwen3_next
from vllm.platforms.interface import PlatformEnum

from tests.layers.attention_layer_plan import layer_builder
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod


class RefAttention(nn.Module):
    """Causal grouped-query attention over (num_tokens, heads * head_dim)."""

    num_heads: int
    head_size: int
    num_kv_heads: int
    scale: float

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_heads, self.head_size = num_heads, head_size
        self.num_kv_heads, self.scale = num_kv_heads, scale

    @typing.override
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Attends causally, repeating each KV head over its query group.

        Args:
            q: Queries, (num_tokens, num_heads * head_size).
            k: Keys, (num_tokens, num_kv_heads * head_size).
            v: Values, (num_tokens, num_kv_heads * head_size).

        Returns:
            The output, (num_tokens, num_heads * head_size).
        """
        t = q.shape[0]
        q = q.view(t, self.num_heads, self.head_size).transpose(0, 1)
        k = k.view(t, self.num_kv_heads, self.head_size).transpose(0, 1)
        v = v.view(t, self.num_kv_heads, self.head_size).transpose(0, 1)
        rep = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale
        )
        return out.transpose(0, 1).reshape(t, self.num_heads * self.head_size)


qwen3_next.Attention = RefAttention  # pyright: ignore[reportPrivateLocalImportUsage]  # the reference child


def use_cpu_gemm(layer: nn.Module) -> None:
    """Sets vLLM's CPU GEMM hook, which vLLM sets at weight load.

    Args:
        layer: The layer whose linear submodules get the hook.
    """
    for module in layer.modules():
        if hasattr(module, "quant_method") and hasattr(module, "weight"):
            module.cpu_linear = nn.functional.linear  # pyright: ignore[reportArgumentType]  # a function, not a tensor


def build_real(
    config: Qwen3NextConfig,
    plan: alp.AttentionLayerPlan | None = None,
    seed: int = 0,
) -> layer_mod.TpuQwen3NextAttention:
    """The layer with seeded weights on the CPU, optionally forced to `plan`.

    Forcing writes the plan directly, past `resolve_plan`, so a check can run a
    value the module declines and see what the declaration hides.

    Args:
        config: The model config.
        plan: The plan value to force, or None for the resolved plan.
        seed: The weights' seed.

    Returns:
        The layer, in eval mode.
    """
    layer = layer_builder.build(config, chip=PlatformEnum.TPU).to_empty(device="cpu")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.05)
    rope = layer.rotary_emb
    rope.cos_sin_cache = rope._compute_cos_sin_cache().to(  # noqa: SLF001  # vLLM sets it at load
        rope.cos_sin_cache.dtype
    )
    use_cpu_gemm(layer)
    if plan is not None:
        layer.plan = plan
    return layer.eval()


def inputs(
    config: Qwen3NextConfig, num_tokens: int, seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract-satisfying (positions, hidden_states) for one case.

    Args:
        config: The model config.
        num_tokens: The number of tokens.
        seed: The hidden states' seed.

    Returns:
        The positions and the bfloat16 hidden states.
    """
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, config.hidden_size, generator=gen)
    return torch.arange(num_tokens), hidden.to(torch.bfloat16)
