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

import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

from vllm_torchtpu.layers.core.plan_marker import plan_of
from vllm_torchtpu.layers.core.resolution import require_no_default
from vllm_torchtpu.tpu_info import has_fp8_matmul

InputProj = Literal["unfused", "fused_qkv"]
# How fused_qkv multiplies: the checkpoint's FP8 weights against quantized
# activations (#3096's fp8 variant), or bf16 weights (its bf16 variant; FP8
# weights are dequantized once at load). Unfused ignores it.
Matmul = Literal["fp8", "bf16"]
# vLLM's ``cache_dtype`` values that name an e4m3 KV cache.
FP8_KV_CACHE = frozenset({"fp8", "fp8_e4m3"})


@plan_of("blocks.norm_then_attention")
@require_no_default
@dataclasses.dataclass(frozen=True)
class NormAttnPlan:
    """How one conforming sub-block reaches attention's projection."""

    input_proj: InputProj
    matmul: Matmul


def _fp8_weights(attention: nn.Module) -> bool:
    weight = getattr(getattr(attention, "qkv_proj", None), "weight", None)
    return weight is not None and weight.dtype == torch.float8_e4m3fn


def rule(attention: nn.Module) -> NormAttnPlan:
    """Fused everywhere, in the checkpoint's precision."""
    return NormAttnPlan("fused_qkv", "fp8" if _fp8_weights(attention) else "bf16")


def candidates(first: NormAttnPlan) -> list[NormAttnPlan]:
    """``first``, then the values to fall back to, fastest first."""
    order = [first, NormAttnPlan("fused_qkv", "bf16"), NormAttnPlan("unfused", "bf16")]
    return list(dict.fromkeys(order))


def fused_qkv_supports(
    attention: nn.Module,
    *,
    matmul: Matmul = "bf16",
    chip: str = "",
    kv_cache_dtype: str = "",
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
    bf16 queries (``gqa/config.py:142``), weights stored (K, N)
    (``fused_attn_proj.py:101``) and no DCP (``:244``).
    A ``layer_scale`` or sequence-parallel layer (``fused_attn_proj.py:457-459``)
    fails this pattern's conformance check instead.

    The chip is the third description, after the model's and the kernel's:
    the fp8 variant needs FP8 matmul (``chip`` is a `pltpu.ChipVersion`
    value; empty off-TPU, where only the model and the kernel are checked).
    ``kv_cache_dtype`` is the platform's KV cache setting (vLLM's
    ``cache_dtype``); empty when unknown, which skips its condition.
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
        weight = getattr(norm, "weight", None)
        if weight is not None and head_dim and weight.numel() != head_dim:
            return (
                f"{name} normalizes {weight.numel()} values (a TP-wide norm over "
                f"the whole projection); fused_qkv fuses a per-head RMSNorm over "
                f"{head_dim}"
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
    # gqa/config.py:143: the kernel writes an FP8 (e4m3) KV cache only.
    if kv_cache_dtype and kv_cache_dtype not in FP8_KV_CACHE:
        return f"KV cache {kv_cache_dtype}; fused_qkv writes an FP8 (e4m3) KV cache"
    if matmul == "fp8" and not _fp8_weights(attention):
        return "the fp8 variant needs FP8 weights; qkv_proj's are not"
    # quantized_matmul.py:111-112 and :164 run an fp8 x fp8 dot on any chip;
    # #3096 has no chip guard, so this condition is ours, read from the chip.
    if matmul == "fp8" and chip and not has_fp8_matmul(pltpu.ChipVersion(chip)):
        return f"the fp8 variant needs FP8 matmul; chip {chip} has none"
    return None


def unreduced_output_supports(
    plan: NormAttnPlan, attention: nn.Module, *, fused_unreduced: bool
) -> str | None:
    """Why ``plan`` cannot return attention's output unreduced over TP, or None.

    The capability a parent block reads before it builds the runner with
    ``output_reduced=False``. ``unfused`` runs the output projection without
    its all-reduce (``AttentionRunner.project_out``), which needs a projection
    that takes its input split over ranks and adds no bias: the platform adds
    the bias after the all-reduce (``TpuRowParallelLinear``), so an unreduced
    output has no point to add it at. ``fused_qkv`` also needs its op to
    honour the flag (``fused_unreduced``, the op author's declaration).
    """
    o_proj = getattr(attention, "o_proj", None)
    if o_proj is None or not getattr(o_proj, "input_is_parallel", False):
        return "the output projection does not take a TP-split input"
    if getattr(o_proj, "bias", None) is not None and not o_proj.skip_bias_add:
        return "the output projection adds a bias, which TPU adds after the all-reduce"
    if plan.input_proj == "fused_qkv" and not fused_unreduced:
        return "the fused op always all-reduces its output"
    return None


def resolve(override: str, attention: nn.Module) -> tuple[NormAttnPlan, frozenset[str]]:
    """The plan and the fields the user pinned; the rest come from the rule.

    Raises:
        ValueError: The override names an unknown key or value.
    """
    plan = rule(attention)
    if not override:
        return plan, frozenset()
    pins = dict(item.split("=", 1) for item in override.split(","))
    allowed = {"input_proj": InputProj, "matmul": Matmul}
    if unknown := sorted(set(pins) - set(allowed)):
        raise ValueError(f"TPU_SUBBLOCK_PLAN: unknown keys {unknown}")
    for key, value in pins.items():
        if value not in get_args(allowed[key]):
            raise ValueError(
                f"TPU_SUBBLOCK_PLAN: {key}={value} is not one of "
                f"{list(get_args(allowed[key]))}"
            )
    return dataclasses.replace(plan, **pins), frozenset(pins)


def choose(
    override: str,
    attention: nn.Module,
    *,
    chip: str = "",
    kv_cache_dtype: str = "",
    where: str = "",
    output_reduced: bool = True,
    fused_unreduced: bool = True,
) -> tuple[NormAttnPlan, frozenset[str], list[str]]:
    """The first value the pins allow that ``fused_qkv``'s capability accepts.

    With ``output_reduced=False`` (a parent block reduces attention's output
    itself) each value must also return an unreduced output
    (``unreduced_output_supports``).

    Returns:
        The plan, the fields the user pinned, and why each faster value was
        declined (the rule's value falling back is quiet).

    Raises:
        ValueError: Every value the pins allow is declined: a pin that
            cannot run stops startup.
    """
    first, pinned = resolve(override, attention)
    declined: list[str] = []
    for plan in candidates(first):
        if any(getattr(plan, f) != getattr(first, f) for f in pinned):
            continue  # A fallback never moves a field the user pinned.
        why = None
        if plan.input_proj == "fused_qkv":
            why = fused_qkv_supports(
                attention, matmul=plan.matmul, chip=chip, kv_cache_dtype=kv_cache_dtype
            )
        if why is None and not output_reduced:
            why = unreduced_output_supports(
                plan, attention, fused_unreduced=fused_unreduced
            )
        if why is None:
            return plan, pinned, declined
        declined.append(f"{plan.input_proj}/{plan.matmul} declined: {why}")
    pins = ",".join(f"{f}={getattr(first, f)}" for f in sorted(pinned)) or "(rule)"
    raise ValueError(f"TPU_SUBBLOCK_PLAN {pins} on {where}: {declined[0]}")
