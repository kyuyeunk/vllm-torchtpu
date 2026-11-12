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
plugin platform), so the plan checks never touch the device and need no lock.
The attention kernel is the layer's child: the layer's plan hands it only its
output format, so it is replaced by `RefAttention`, a plain causal
grouped-query attention with that format. The checks then cover the layer's
own decisions, which is what its plan owns.
"""

import os
import pathlib
import sys
import types

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")  # str-ok: env var
os.environ.setdefault("VLLM_PLUGINS", "")  # str-ok: env var
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # str-ok: env var
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # str-ok: env var
sys.modules["libtpu"] = None  # type: ignore[assignment]  # str-ok: module table

# The plugin's package __init__ files load the TPU runtime; the plan modules
# do not need them, so the packages are mapped without running them.
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm_torchtpu"
for _name, _sub in [
    ("vllm_torchtpu", ""),
    ("vllm_torchtpu.layers", "layers"),
    ("vllm_torchtpu.models", "models"),
    ("vllm_torchtpu.models.vllm", "models/vllm"),
]:
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_SRC / _sub)]
        sys.modules[_name] = _module

import torch  # noqa: E402
from torch import nn  # noqa: E402
from vllm.model_executor.models import qwen3_next  # noqa: E402
from vllm.platforms.interface import PlatformEnum  # noqa: E402


class RefAttention(nn.Module):
    """Causal grouped-query attention over (num_tokens, heads * head_dim)."""

    def __init__(self, num_heads, head_size, scale, num_kv_heads, **_):
        super().__init__()
        self.num_heads, self.head_size = num_heads, head_size
        self.num_kv_heads, self.scale = num_kv_heads, scale

    def forward(self, q, k, v):
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


qwen3_next.Attention = RefAttention  # type: ignore[assignment]

from prototype import _layer  # noqa: E402
from vllm_torchtpu.layers import attention_layer_plan as alp  # noqa: E402


def build_real(config, plan: alp.AttentionLayerPlan | None = None, seed: int = 0):
    """The layer with seeded weights on the CPU, optionally forced to `plan`.

    Forcing writes the plan directly, past `resolve_plan`, so a check can run a
    value the module declines and see what the declaration hides.
    """
    layer = _layer.build(config, chip=PlatformEnum.TPU).to_empty(device="cpu")
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.05)
    layer.rotary_emb.cos_sin_cache = layer.rotary_emb._compute_cos_sin_cache().to(
        layer.rotary_emb.cos_sin_cache.dtype
    )
    for module in layer.modules():  # vLLM's CPU GEMM hook, set at weight load.
        if hasattr(module, "quant_method") and hasattr(module, "weight"):
            module.cpu_linear = nn.functional.linear
    if plan is not None:
        layer.plan = plan
    return layer.eval()


def inputs(config, num_tokens: int, seed: int = 1):
    """Contract-satisfying (positions, hidden_states) for one case."""
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_tokens, config.hidden_size, generator=gen)
    return torch.arange(num_tokens), hidden.to(torch.bfloat16)
