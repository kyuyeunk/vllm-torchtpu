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
"""The sub-block's plan: whether the input norm joins attention's projection.

Two plan values compute the same output within the residual stream's
tolerance:

- ``unfused``: the input RMSNorm adds the residual and normalizes, then
  attention's QKV projection reads the normed tokens (upstream).
- ``fused_qkv``: one function adds, normalizes and projects (PR #3096's
  fused_qkv kernel; here a plain-torch reference written against the roles).

The rule picks ``fused_qkv`` for every layer that conforms. The user's
override (``TPU_SUBBLOCK_PLAN``) comes first, as for every plan.
"""

import dataclasses
from typing import Literal, get_args

from torch import nn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

from vllm_torchtpu.layers.core.plan_marker import plan_of
from vllm_torchtpu.layers.core.resolution import require_no_default

InputProj = Literal["unfused", "fused_qkv"]


@plan_of("blocks.norm_then_attention")
@require_no_default
@dataclasses.dataclass(frozen=True)
class NormAttnPlan:
    """How one conforming sub-block reaches attention's projection."""

    input_proj: InputProj


RULE = NormAttnPlan(input_proj="fused_qkv")


def fused_qkv_supports(
    attention: nn.Module,
    *,
    head_dim: int | None = None,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> str | None:
    """Why ``fused_qkv`` cannot serve this attention, or None: its capability.

    ``fused_qkv`` stands for PR #3096's fused_qkv kernel; this is the
    capability that kernel's author would declare, read from #3096's code
    (paths below are in its diff, d2d5ded6e). Each condition reads the bound
    attention role, so capability flows up from the primitive to the plan.
    Head sizes may be given (a shrunk layer's are not the model's). Facts
    only the platform knows are checked when the kernel is built on it:
    the batched-RPA attention backend (``layers/adapter/fused_attn_proj.py:242``),
    an FP8 KV cache and bf16 queries (``gqa/config.py:142-143``),
    weights stored (K, N) (``fused_attn_proj.py:101``) and no DCP (``:244``).
    A ``layer_scale`` or sequence-parallel layer (``fused_attn_proj.py:457-459``)
    fails this pattern's conformance check instead.
    """
    inner = [m for m in attention.modules() if m is not attention]
    # Sinks need RPA v3's kernel; fused_qkv writes the batched-RPA layout
    # (fused_attn_proj.py:246).
    if any(getattr(m, "sinks", None) is not None for m in (attention, *inner)):
        return "attention sinks need RPA v3; fused_qkv writes the batched-RPA layout"
    if any(getattr(m, "kv_sharing_target_layer_name", None) for m in inner):
        return "KV sharing (fused_attn_proj.py:246)"
    head_dim = head_dim or getattr(attention, "head_dim", None)
    if head_dim == 64:  # fused_attn_proj.py:244
        return "head_dim 64 runs on RPA v3; fused_qkv writes the batched-RPA layout"
    if head_dim is not None and head_dim % 32:  # gqa/config.py:147 (8 sublanes x 4 fp8)
        return f"head_dim {head_dim} is not a multiple of 32"
    q_heads = num_heads or getattr(attention, "num_heads", None)
    kv_heads = num_kv_heads or getattr(attention, "num_kv_heads", None)
    if q_heads and kv_heads and (q_heads // kv_heads) % 2:  # gqa/config.py:146
        return f"{q_heads // kv_heads} query heads per KV head (bf16 Q packs 2)"
    if getattr(attention, "attn_output_gate", False) and head_dim and head_dim % 128:
        return "an output gate needs head_dim % 128 == 0 (gqa/config.py:149)"
    for name in ("q_norm", "k_norm"):
        norm = getattr(attention, name, None)
        # fused_attn_proj.py:74-82: per-head RMSNorm, (1 + w) or weightless only.
        if norm is not None and not isinstance(norm, (RMSNorm, GemmaRMSNorm)):
            return (
                f"{name} is {type(norm).__name__} (a norm over the whole projection); "
                "fused_qkv fuses a per-head RMSNorm"
            )
    rotary = getattr(attention, "rotary_emb", None)
    # The kernel rotates NeoX-style pairs only (common/model_config.py:59-71,
    # common/rope.py:17); #3096's adapter does not check this.
    if rotary is not None and not getattr(rotary, "is_neox_style", True):
        return "GPT-J-style RoPE; fused_qkv rotates NeoX-style pairs"
    qkv = getattr(attention, "qkv_proj", None)
    method = type(getattr(qkv, "quant_method", None)).__name__
    # fused_attn_proj.py:88-103: unquantized or per-channel FP8 only.
    if qkv is not None and not ("Unquantized" in method or "Fp8" in method):
        return f"qkv_proj quantized by {method}"
    return None


def resolve(override: str) -> tuple[NormAttnPlan, str]:
    """The plan and its source: the user's override, else the rule.

    Raises:
        ValueError: The override names an unknown key or value.
    """
    if not override:
        return RULE, "rule"
    pins = dict(item.split("=", 1) for item in override.split(","))
    if unknown := sorted(set(pins) - {"input_proj"}):
        raise ValueError(f"TPU_SUBBLOCK_PLAN: unknown keys {unknown}")
    value = pins["input_proj"]
    if value not in get_args(InputProj):
        raise ValueError(
            f"TPU_SUBBLOCK_PLAN: input_proj={value} is not one of "
            f"{list(get_args(InputProj))}"
        )
    return NormAttnPlan(input_proj=value), "user"  # type: ignore[arg-type]
