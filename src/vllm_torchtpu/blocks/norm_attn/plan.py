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
- ``fused_qkv``: one function normalizes and projects (PR #3096's fused_qkv
  kernel; here a plain-torch reference written against the roles).

``fused_qkv`` has two fields besides its precision (``matmul``):
``residual_add`` says where the residual is added. ``folded``: the kernel
takes the residual and returns the new residual, written once. ``outside``:
the runner adds before the kernel, which only normalizes. Both keep the
output within the contract; on TPU, ``outside`` leaves a sum XLA recomputes
in every later fusion that reads it (Llama-3.1-70B prefill, 8192 tokens:
gate_up +150 us, down +89 us per layer).

The rule picks ``fused_qkv`` for every layer that conforms, ``folded`` where
the fused op implements it. The user's override (``TPU_SUBBLOCK_PLAN``)
comes first, as for every plan.
"""

import dataclasses
import enum
from typing import cast, override

import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

from vllm_torchtpu.blocks.binding import as_kind
from vllm_torchtpu.blocks.norm_attn.runner import (
    AttentionBinding,
    AttnConfig,
    InputProj,
    ResidualAdd,
)
from vllm_torchtpu.layers.core.plan_marker import plan_of
from vllm_torchtpu.layers.core.resolution import require_no_default
from vllm_torchtpu.tpu_info import has_fp8_matmul


class Matmul(enum.Enum):
    """How fused_qkv multiplies.

    The checkpoint's FP8 weights against quantized activations (#3096's fp8
    variant), or bf16 weights (its bf16 variant; FP8 weights are dequantized
    once at load). Unfused ignores it.
    """

    _value_: str

    FP8 = "fp8"
    BF16 = "bf16"


@dataclasses.dataclass(frozen=True)
class Pins:
    """The fields the user pinned (``TPU_SUBBLOCK_PLAN``, parsed once)."""

    input_proj: InputProj | None = None
    matmul: Matmul | None = None
    residual_add: ResidualAdd | None = None

    def allow(self, plan: "NormAttnPlan") -> bool:
        """Whether ``plan`` keeps every pinned field."""
        return (
            (self.input_proj is None or plan.input_proj is self.input_proj)
            and (self.matmul is None or plan.matmul is self.matmul)
            and (self.residual_add is None or plan.residual_add is self.residual_add)
        )

    def __bool__(self) -> bool:
        """Whether any field is pinned."""
        return any(
            v is not None for v in (self.input_proj, self.matmul, self.residual_add)
        )

    @override
    def __str__(self) -> str:
        fields: tuple[tuple[str, enum.Enum | None], ...] = (
            ("input_proj", self.input_proj),
            ("matmul", self.matmul),
            ("residual_add", self.residual_add),
        )
        return (
            ",".join(f"{k}={v.value}" for k, v in fields if v is not None) or "(rule)"
        )


def is_fp8_e4m3(cache_dtype: str) -> bool:
    """The one parse point for vLLM's ``cache_dtype`` string."""
    return cache_dtype in {"fp8", "fp8_e4m3"}  # str-ok: parses vLLM's cache_dtype


def parse_pins(text: str) -> Pins:
    """The one parse point: ``TPU_SUBBLOCK_PLAN`` into typed pins.

    Raises:
        ValueError: An unknown key or value, listing the valid ones.
    """
    input_proj: InputProj | None = None
    matmul: Matmul | None = None
    residual_add: ResidualAdd | None = None
    for item in filter(None, text.split(",")):
        key, _, value = item.partition("=")
        if key == "input_proj":  # str-ok: the TPU_SUBBLOCK_PLAN parse point
            input_proj = _parse_value(InputProj, key, value)
        elif key == "matmul":  # str-ok: the TPU_SUBBLOCK_PLAN parse point
            matmul = _parse_value(Matmul, key, value)
        elif key == "residual_add":  # str-ok: the TPU_SUBBLOCK_PLAN parse point
            residual_add = _parse_value(ResidualAdd, key, value)
        else:
            msg = (
                f"TPU_SUBBLOCK_PLAN: unknown key {key!r}; "
                "valid: input_proj, matmul, residual_add"
            )
            raise ValueError(msg)
    return Pins(input_proj, matmul, residual_add)


def _parse_value[E: enum.Enum](kind: type[E], key: str, value: str) -> E:
    try:
        return kind(value)
    except ValueError:
        valid = ", ".join(str(cast("object", v.value)) for v in kind)
        msg = f"TPU_SUBBLOCK_PLAN: {key}={value}; valid: {valid}"
        raise ValueError(msg) from None


@plan_of("blocks.norm_then_attention")
@require_no_default
@dataclasses.dataclass(frozen=True)
class NormAttnPlan:
    """How one conforming sub-block reaches attention's projection.

    ``matmul`` and ``residual_add`` are fields of the fused value; unfused
    ignores them.
    """

    input_proj: InputProj
    matmul: Matmul
    residual_add: ResidualAdd


FUSED_BF16 = NormAttnPlan(InputProj.FUSED_QKV, Matmul.BF16, ResidualAdd.FOLDED)
UNFUSED = NormAttnPlan(InputProj.UNFUSED, Matmul.BF16, ResidualAdd.FOLDED)


RPA_V3_HEAD_DIM = 64
"""The head size RPA v3 serves (fused_attn_proj.py:244); fused_qkv does not."""


@dataclasses.dataclass(frozen=True)
class NormFacts:
    """A per-head q or k norm, as the kernel sees it."""

    rms: bool  # vLLM's RMSNorm family: (1 + w) or plain w, or weightless.
    width: int  # Values it normalizes over: head_dim for a per-head norm.
    kind: str  # The class name, for messages only.


@dataclasses.dataclass(frozen=True)
class AttentionFacts:
    """What the capabilities read about one declared attention, typed.

    Built once from the declared binding (``facts_of``), the one place vLLM's
    modules are read; capability conditions read only this record, so tests
    enumerate them without building a module.
    """

    config: AttnConfig
    sinks: bool
    kv_sharing: bool
    q_norm: NormFacts | None
    k_norm: NormFacts | None
    neox_rope: bool
    qkv_quant_supported: bool  # Unquantized, or per-channel FP8.
    qkv_quant: str  # The quant method's class name, for messages only.
    fp8_weights: bool
    o_proj_split_input: bool
    o_proj_adds_bias: bool


def _norm_facts(norm: object) -> NormFacts | None:
    if norm is None:
        return None
    weights = list(as_kind(norm, nn.Module).parameters())  # Weightless: none.
    width = weights[0].numel() if weights else 0
    rms = isinstance(norm, (RMSNorm, GemmaRMSNorm))
    return NormFacts(rms, width, type(norm).__name__)


def facts_of(binding: AttentionBinding, config: AttnConfig) -> AttentionFacts:
    """The typed boundary: reads the declared binding' vLLM classes once."""
    core = as_kind(binding.attn, Attention)
    rotary = as_kind(binding.rotary, RotaryEmbeddingBase)
    qkv = as_kind(binding.qkv_proj, QKVParallelLinear)
    o_proj = as_kind(binding.o_proj, RowParallelLinear)
    method = qkv.quant_method
    return AttentionFacts(
        config=config,
        sinks=core.has_sink,
        kv_sharing=core.kv_sharing_target_layer_name is not None,
        q_norm=_norm_facts(binding.q_norm),
        k_norm=_norm_facts(binding.k_norm),
        neox_rope=rotary.is_neox_style,
        qkv_quant_supported=isinstance(
            method, (UnquantizedLinearMethod, Fp8LinearMethod)
        ),
        qkv_quant=type(method).__name__,
        fp8_weights=qkv.weight.dtype == torch.float8_e4m3fn,
        o_proj_split_input=o_proj.input_is_parallel,
        o_proj_adds_bias=(
            o_proj.bias is not None  # pyright: ignore[reportUnnecessaryComparison] - vLLM registers a None bias
            and not o_proj.skip_bias_add
        ),
    )


def rule(facts: AttentionFacts) -> NormAttnPlan:
    """Fused everywhere, in the checkpoint's precision, the residual folded in."""
    matmul = Matmul.FP8 if facts.fp8_weights else Matmul.BF16
    return NormAttnPlan(InputProj.FUSED_QKV, matmul, ResidualAdd.FOLDED)


def candidates(first: NormAttnPlan) -> list[NormAttnPlan]:
    """``first``, then the values to fall back to, fastest first.

    The residual add moves outside before the precision drops: ``outside``
    costs a recomputed sum, the bf16 variant a dequantized weight.
    """
    outside = ResidualAdd.OUTSIDE
    order = [
        first,
        dataclasses.replace(first, residual_add=outside),
        FUSED_BF16,
        dataclasses.replace(FUSED_BF16, residual_add=outside),
        UNFUSED,
    ]
    return list(dict.fromkeys(order))


def fused_qkv_supports(  # noqa: C901, PLR0911, PLR0912 - one return per capability condition, each cited to #3096
    facts: AttentionFacts,
    *,
    matmul: Matmul = Matmul.BF16,
    chip: pltpu.ChipVersion | None = None,
    fp8_kv_cache: bool | None = None,
) -> str | None:
    """Why ``fused_qkv`` cannot serve this attention, or None: its capability.

    ``fused_qkv`` stands for PR #3096's fused_qkv kernel; this is the
    capability that kernel's author would declare, read from #3096's code
    (paths below are in its diff, d2d5ded6e). Each condition reads the
    declared attention's facts, so capability flows up from the primitive to
    the plan. Facts only the platform knows are checked when the kernel is
    built on it: the batched-RPA attention backend
    (``layers/adapter/fused_attn_proj.py:242``), bf16 queries
    (``gqa/config.py:142``), weights stored (K, N) (``fused_attn_proj.py:101``)
    and no DCP (``:244``). A ``layer_scale`` or sequence-parallel layer
    (``fused_attn_proj.py:457-459``) fails this pattern's conformance check.

    The chip is the third description, after the model's and the kernel's:
    the fp8 variant needs FP8 matmul (None off-TPU, where only the model and
    the kernel are checked). ``fp8_kv_cache`` is the platform's KV cache
    setting; None when unknown, which skips its condition.
    """
    c = facts.config
    # Sinks need RPA v3's kernel; fused_qkv writes the batched-RPA layout
    # (fused_attn_proj.py:246).
    if facts.sinks:
        return "attention sinks need RPA v3; fused_qkv writes the batched-RPA layout"
    if facts.kv_sharing:
        return "KV sharing (fused_attn_proj.py:246)"
    if c.head_dim == RPA_V3_HEAD_DIM:
        return "head_dim 64 runs on RPA v3; fused_qkv writes the batched-RPA layout"
    if c.head_dim % 32:  # gqa/config.py:147 (8 sublanes x 4 fp8)
        return f"head_dim {c.head_dim} is not a multiple of 32"
    if (c.num_heads // c.num_kv_heads) % 2:  # gqa/config.py:146
        return (
            f"{c.num_heads // c.num_kv_heads} query heads per KV head (bf16 Q packs 2)"
        )
    if c.output_gate and c.head_dim % 128:
        return "an output gate needs head_dim % 128 == 0 (gqa/config.py:149)"
    for label, norm in (("q_norm", facts.q_norm), ("k_norm", facts.k_norm)):
        # fused_attn_proj.py:74-82: per-head RMSNorm, (1 + w) or weightless only.
        if norm is not None and not norm.rms:
            return (
                f"{label} is {norm.kind} (a norm over the whole projection); "
                "fused_qkv fuses a per-head RMSNorm"
            )
        if norm is not None and norm.width and norm.width != c.head_dim:
            return (
                f"{label} normalizes {norm.width} values (a TP-wide norm over "
                f"the whole projection); fused_qkv fuses a per-head RMSNorm over "
                f"{c.head_dim}"
            )
    # The kernel rotates NeoX-style pairs only (common/model_config.py:59-71,
    # common/rope.py:17); #3096's adapter does not check this.
    if not facts.neox_rope:
        return "GPT-J-style RoPE; fused_qkv rotates NeoX-style pairs"
    # fused_attn_proj.py:88-103: unquantized or per-channel FP8 only.
    if not facts.qkv_quant_supported:
        return f"qkv_proj quantized by {facts.qkv_quant}"
    # gqa/config.py:143: the kernel writes an FP8 (e4m3) KV cache only.
    if fp8_kv_cache is False:
        return "the KV cache is not FP8; fused_qkv writes an FP8 (e4m3) KV cache"
    if matmul is Matmul.FP8 and not facts.fp8_weights:
        return "the fp8 variant needs FP8 weights; qkv_proj's are not"
    # quantized_matmul.py:111-112 and :164 run an fp8 x fp8 dot on any chip;
    # #3096 has no chip guard, so this condition is ours, read from the chip.
    if matmul is Matmul.FP8 and chip is not None and not has_fp8_matmul(chip):
        return f"the fp8 variant needs FP8 matmul; chip {chip.value} has none"
    return None


BOTH_RESIDUAL_ADDS = frozenset(ResidualAdd)
"""An op that implements both residual adds (the reference's declaration)."""


def residual_add_supports(
    value: ResidualAdd, declared: frozenset[ResidualAdd], *, first_layer: bool = False
) -> str | None:
    """Why the fused op cannot add the residual where ``value`` says, or None.

    ``declared`` is the op author's declaration
    (``FusedCapability.residual_add``). The first decoder layer has no
    residual, so only ``outside`` serves it (the runner's call there adds
    nothing).
    """
    if first_layer and value is ResidualAdd.FOLDED:
        return "the first layer has no residual to fold into the fused op"
    if value not in declared:
        return f"the fused op does not implement residual_add={value.value}"
    return None


def unreduced_output_supports(
    plan: NormAttnPlan, facts: AttentionFacts, *, fused_unreduced: bool
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
    if not facts.o_proj_split_input:
        return "the output projection does not take a TP-split input"
    if facts.o_proj_adds_bias:
        return "the output projection adds a bias, which TPU adds after the all-reduce"
    if plan.input_proj is InputProj.FUSED_QKV and not fused_unreduced:
        return "the fused op always all-reduces its output"
    return None


def describe(plan: NormAttnPlan) -> str:
    """The plan's fields, for messages; unfused has none that matter."""
    if plan.input_proj is InputProj.UNFUSED:
        return plan.input_proj.value
    return f"{plan.input_proj.value}/{plan.matmul.value}/{plan.residual_add.value}"


def resolve(pins: Pins, facts: AttentionFacts) -> NormAttnPlan:
    """The rule's plan with the user's pinned fields applied."""
    plan = rule(facts)
    if pins.input_proj is not None:
        plan = dataclasses.replace(plan, input_proj=pins.input_proj)
    if pins.matmul is not None:
        plan = dataclasses.replace(plan, matmul=pins.matmul)
    if pins.residual_add is not None:
        plan = dataclasses.replace(plan, residual_add=pins.residual_add)
    return plan


def choose(  # noqa: PLR0913 - the facts and the platform's descriptions, all keyword-only
    pins: Pins,
    facts: AttentionFacts,
    *,
    chip: pltpu.ChipVersion | None = None,
    fp8_kv_cache: bool | None = None,
    where: str = "",
    output_reduced: bool = True,
    fused_unreduced: bool = True,
    fused_residual_add: frozenset[ResidualAdd] = BOTH_RESIDUAL_ADDS,
) -> tuple[NormAttnPlan, list[str]]:
    """The first value the pins allow that ``fused_qkv``'s capability accepts.

    With ``output_reduced=False`` (a parent block reduces attention's output
    itself) each value must also return an unreduced output
    (``unreduced_output_supports``). A fused value must add the residual
    where the fused op can (``fused_residual_add``, the op's declaration;
    ``residual_add_supports``).

    Returns:
        The plan, and why each faster value was declined (the rule's value
        falling back is quiet).

    Raises:
        ValueError: Every value the pins allow is declined: a pin that
            cannot run stops startup.
    """
    first = resolve(pins, facts)
    declined: list[str] = []
    for plan in candidates(first):
        if not pins.allow(plan):
            continue  # A fallback never moves a field the user pinned.
        why = None
        if plan.input_proj is InputProj.FUSED_QKV:
            why = fused_qkv_supports(
                facts, matmul=plan.matmul, chip=chip, fp8_kv_cache=fp8_kv_cache
            ) or residual_add_supports(plan.residual_add, fused_residual_add)
        if why is None and not output_reduced:
            why = unreduced_output_supports(
                plan, facts, fused_unreduced=fused_unreduced
            )
        if why is None:
            return plan, declined
        declined.append(f"{describe(plan)} declined: {why}")
    msg = f"TPU_SUBBLOCK_PLAN {pins} on {where}: {declined[0]}"
    raise ValueError(msg)
