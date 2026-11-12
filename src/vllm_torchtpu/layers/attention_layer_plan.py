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
"""Decision module for the gated attention layer (Qwen3NextAttention).

Prototype; see prototype/README.md.

The layer's contract is (positions, hidden_states) -> output, plus its weights
and config. Everything inside that boundary is the layer's plan, resolved once
from descriptions:

- the layer's description (config and the RoPE module it built);
- its children's declared capability (what o_proj accepts as input);
- the chip.

The plan picks the whole layer as one fused kernel, or the unfused path. On the
unfused path the same plan resolves the contracts it hands down to its
children: the attention kernel's output format and o_proj's input format, and
which gate variant sits between them. Each child then plans only its own
internals. On the fused path no intermediate tensor exists, so neither
child contract exists. The two paths are two plan types, dispatched by type;
every field is an enum.

Capability is declared by each variant's ``supports``, which returns a reason
when the variant cannot run. Among capable plans the family's rules pick, else
the fixed default. The user's override (``TPU_ATTENTION_LAYER_PLAN``) and the
model plan (read from the model's config as its ``tpu_attention_layer_plan``)
are partial plans, each parsed once by ``parse_pins``, that merge field by
field, the user's first. Resolution fills the unpinned fields, then raises with
the reason when one source's pins contradict each other, a pin names a contract
field, or every completion declines.
"""

import enum
import itertools
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import torch
from vllm.platforms.interface import PlatformEnum

from vllm_torchtpu.layers.core.plan_marker import plan_of


class Quant(enum.Enum):
    """An activation's quantization."""

    NONE = "none"
    FP8_STATIC_TENSOR_SYM = "fp8_static_tensor_sym"


class RopePositions(enum.Enum):
    """The shape of the RoPE module's positions."""

    ONE_D = "1d"
    MROPE_INTERLEAVED_THW = "mrope_interleaved_thw"
    MROPE_OTHER = "mrope_other"


class Layer(enum.Enum):
    """Which plan type a value is; pinned as ``layer=``."""

    UNFUSED = "unfused"
    FUSED = "fused"


class QkPath(enum.Enum):
    """How q and k are split, normed and rotated."""

    EAGER = "eager"
    FUSED_TRITON = "fused_triton"


class GatePath(enum.Enum):
    """How the output gate multiplies the attention output."""

    NONE = "none"
    EAGER_MUL = "eager_mul"
    FUSED_MUL = "fused_mul"
    FUSED_MUL_QUANT = "fused_mul_quant"


class Scale(enum.Enum):
    """Where a quantized activation's scale comes from."""

    NONE = "none"
    O_PROJ_INPUT_SCALE = "o_proj.input_scale"


_HALF = (torch.float16, torch.bfloat16)


def dtype_name(dtype: torch.dtype) -> str:
    """A dtype's torch name without the module prefix, for display only."""
    return str(dtype).removeprefix("torch.")


def platform_name(platform: PlatformEnum) -> str:
    """A platform's name in lower case, for display only."""
    return platform.name.lower()


# ---------------------------------------------------------------------------
# Descriptions: what the plan is derived from.


@dataclass(frozen=True)
class ChipDesc:
    platform: PlatformEnum


@dataclass(frozen=True)
class AttentionLayerDesc:
    """The layer as built: config facts and the RoPE module's facts."""

    hidden_size: int
    head_dim: int
    dtype: torch.dtype
    output_gate: bool
    rope_neox: bool
    rope_dtype: torch.dtype | None
    rotary_dim: int
    rope_positions: RopePositions
    text_only: bool
    reduce_results: bool


@dataclass(frozen=True)
class ChildCapability:
    """What the children declare they accept, passed up as data."""

    o_proj_inputs: frozenset[Quant]


# ---------------------------------------------------------------------------
# The contract: the only thing that crosses the layer boundary.


class Sharding(enum.Enum):
    """How a boundary tensor is laid out over TP ranks."""

    REPLICATED = "replicated"
    PARTIAL_SUM_OVER_TP = "partial_sum_over_tp"


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[str | int, ...]  # a str is a symbolic dimension's label
    dtype: torch.dtype
    sharding: Sharding


@dataclass(frozen=True)
class AttentionLayerContract:
    positions: TensorSpec
    hidden_states: TensorSpec
    output: TensorSpec
    # Stated by the consuming layer (the decoder layer); every capable plan
    # must meet it against the eager reference.
    atol: float
    rtol: float
    weights: tuple[tuple[str, tuple[int, ...]], ...]


# bf16 activations: the eager reference against itself under re-association.
_TOLERANCE = {torch.bfloat16: (2e-2, 1.6e-2), torch.float16: (1e-3, 1e-3)}


def layer_contract(
    desc: AttentionLayerDesc, weights: tuple[tuple[str, tuple[int, ...]], ...]
) -> AttentionLayerContract:
    one_d = desc.rope_positions is RopePositions.ONE_D
    positions = ("num_tokens",) if one_d else (3, "num_tokens")
    atol, rtol = _TOLERANCE.get(desc.dtype, (1e-5, 1.3e-6))
    return AttentionLayerContract(
        positions=TensorSpec(positions, torch.int64, Sharding.REPLICATED),
        hidden_states=TensorSpec(
            ("num_tokens", desc.hidden_size), desc.dtype, Sharding.REPLICATED
        ),
        output=TensorSpec(
            ("num_tokens", desc.hidden_size),
            desc.dtype,
            Sharding.REPLICATED
            if desc.reduce_results
            else Sharding.PARTIAL_SUM_OVER_TP,
        ),
        atol=atol,
        rtol=rtol,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# The plan. Resolved contracts have no defaults: the plan sets every field.


@dataclass(frozen=True)
class ActivationFormat:
    """A resolved contract the layer hands to one child."""

    dtype: torch.dtype
    quant: Quant
    scale: Scale


@plan_of("attention_layer")
@dataclass(frozen=True)
class UnfusedLayerPlan:
    qk_norm_rope_gate: QkPath
    gate: GatePath
    attn_output: ActivationFormat  # handed to self.attn
    o_proj_input: ActivationFormat  # handed to self.o_proj


@plan_of("attention_layer")
@dataclass(frozen=True)
class FusedLayerPlan:
    """The whole layer as one kernel: no intermediate, no child contracts."""


AttentionLayerPlan = UnfusedLayerPlan | FusedLayerPlan


class PlanField(enum.Enum):
    """The plan fields an override pins, by their override key."""

    LAYER = "layer"
    QK_NORM_ROPE_GATE = "qk_norm_rope_gate"
    GATE = "gate"
    O_PROJ_INPUT = "o_proj_input"


PlanValue = Layer | QkPath | GatePath | Quant


def plan_fields(plan: AttentionLayerPlan) -> dict[PlanField, PlanValue]:
    """The plan's value of every field an override can pin, in pin order."""
    if isinstance(plan, FusedLayerPlan):
        return {PlanField.LAYER: Layer.FUSED}
    return {
        PlanField.LAYER: Layer.UNFUSED,
        PlanField.QK_NORM_ROPE_GATE: plan.qk_norm_rope_gate,
        PlanField.GATE: plan.gate,
        PlanField.O_PROJ_INPUT: plan.o_proj_input.quant,
    }


def _pins_text(pins: dict[PlanField, PlanValue]) -> str:
    return ",".join(f"{k.value}={v.value}" for k, v in pins.items())


def plan_name(plan: AttentionLayerPlan) -> str:
    """The canonical override string for a plan value."""
    return _pins_text(plan_fields(plan))


# ---------------------------------------------------------------------------
# Variants: each declares capability with its kernel's own arguments.


def _fused_sigmoid_gate_mul() -> Callable[..., Any] | None:
    try:
        from vllm.kernels.triton.fused_sigmoid_gate_mul import (
            fused_sigmoid_gate_mul,
        )
    except ImportError:
        return None
    return fused_sigmoid_gate_mul


def supports_fused_qk_rmsnorm_rope_gate(
    *,
    q_gate_dtype: torch.dtype,
    cos_sin_cache_dtype: torch.dtype | None,
    head_dim: int,
    rotary_dim: int,
    mrope_section_ok: bool,
    is_neox_style: bool,
    platform: PlatformEnum,
) -> str | None:
    """vllm.model_executor.layers.fused_qk_norm_rope.fused_qk_rmsnorm_rope_gate."""
    if platform not in (PlatformEnum.CUDA, PlatformEnum.XPU):
        return f"Triton kernel for CUDA/XPU; chip is {platform_name(platform)}"
    if q_gate_dtype not in _HALF:
        return f"q_gate dtype {dtype_name(q_gate_dtype)}; kernel takes float16/bfloat16"
    if cos_sin_cache_dtype not in _HALF:
        cache = dtype_name(cos_sin_cache_dtype) if cos_sin_cache_dtype else None
        return f"cos_sin_cache dtype {cache}; needs float16/bfloat16"
    if not is_neox_style:
        return "kernel implements NeoX-style RoPE only"
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        return f"rotary_dim {rotary_dim} must be even and <= head_dim {head_dim}"
    if not mrope_section_ok:
        return "MRoPE needs 3 interleaved sections summing to rotary_dim/2"
    return None


def supports_fused_sigmoid_gate_mul(
    *, x_dtype: torch.dtype, scale: bool, platform: PlatformEnum
) -> str | None:
    """vllm.kernels.triton.fused_sigmoid_gate_mul(x, gate, scale)."""
    if _fused_sigmoid_gate_mul() is None:
        return "fused_sigmoid_gate_mul is not in this vLLM (added after v0.33.0)"
    if platform is not PlatformEnum.CUDA:
        return f"Triton kernel for CUDA tensors; chip is {platform_name(platform)}"
    if x_dtype not in _HALF:
        return f"x dtype {dtype_name(x_dtype)}; kernel takes float16/bfloat16"
    del scale  # FP8 e4m3fn output is CUDA-only, already required above.
    return None


def supports_fused_attention_layer(
    *, desc: AttentionLayerDesc, platform: PlatformEnum
) -> str | None:
    """A kernel for the whole layer: qkv_proj through o_proj in one launch."""
    del desc, platform
    return "no fused attention-layer kernel registered"


# ---------------------------------------------------------------------------
# Enumeration, capability and the decision function.


def _formats(desc: AttentionLayerDesc, quant: Quant) -> ActivationFormat:
    if quant is Quant.NONE:
        return ActivationFormat(desc.dtype, Quant.NONE, Scale.NONE)
    return ActivationFormat(torch.float8_e4m3fn, quant, Scale.O_PROJ_INPUT_SCALE)


def enumerate_plans(desc: AttentionLayerDesc) -> list[AttentionLayerPlan]:
    """Every structurally consistent plan value for this layer."""
    plans: list[AttentionLayerPlan] = [FusedLayerPlan()]
    gates = [g for g in GatePath if (g is GatePath.NONE) != desc.output_gate]
    for qk, gate in itertools.product(QkPath, gates):
        # Only the fused gate+quant kernel writes o_proj's FP8 input; the
        # attention kernel's own output stays in the activation dtype because
        # the gate multiplies after it.
        o_proj_quant = (
            Quant.FP8_STATIC_TENSOR_SYM
            if gate is GatePath.FUSED_MUL_QUANT
            else Quant.NONE
        )
        plans.append(
            UnfusedLayerPlan(
                qk_norm_rope_gate=qk,
                gate=gate,
                attn_output=_formats(desc, Quant.NONE),
                o_proj_input=_formats(desc, o_proj_quant),
            )
        )
    return plans


def fused_qk_declines(desc: AttentionLayerDesc, chip: ChipDesc) -> list[str]:
    """Why the fused QK-norm-RoPE-gate kernel cannot serve this layer on ``chip``.

    Upstream decides the same question in ``Qwen3NextAttention.__init__``
    (``use_fused_qk_norm_rope_gate``) with its own copy of these conditions;
    the layer compares the two at build time (``check_upstream_decision``).
    """
    reasons = []
    if not desc.output_gate:
        reasons.append("fused_qk_rmsnorm_rope_gate splits a gated q; no gate")
    reason = supports_fused_qk_rmsnorm_rope_gate(
        q_gate_dtype=desc.dtype,
        cos_sin_cache_dtype=desc.rope_dtype,
        head_dim=desc.head_dim,
        rotary_dim=desc.rotary_dim,
        mrope_section_ok=desc.text_only
        or desc.rope_positions is not RopePositions.MROPE_OTHER,
        is_neox_style=desc.rope_neox,
        platform=chip.platform,
    )
    if reason:
        reasons.append(f"fused_qk_rmsnorm_rope_gate: {reason}")
    return reasons


def check_upstream_decision(
    *, upstream: bool | None, desc: AttentionLayerDesc, chip: ChipDesc
) -> None:
    """Raises when upstream's recorded fused-QK decision disagrees with the plan's.

    ``fused_qk_declines`` restates upstream's ``use_fused_qk_norm_rope_gate``
    conditions (vLLM v0.33.0, inline in ``Qwen3NextAttention.__init__``, with
    no helper to call). Upstream still computes its value while ``__init__``
    runs, and the layer records it, so a drift in either copy shows up here
    at build time instead of as a silently different kernel choice.

    Args:
        upstream: The value upstream wrote, or None when it wrote none.
        desc: The layer's description.
        chip: The platform upstream decided for.

    Raises:
        LookupError: Upstream no longer writes the decision.
        ValueError: The two decisions disagree.
    """
    if upstream is None:
        msg = (
            "upstream Qwen3NextAttention.__init__ no longer writes "
            "use_fused_qk_norm_rope_gate; re-derive fused_qk_declines from it"
        )
        raise LookupError(msg)
    reasons = fused_qk_declines(desc, chip)
    if bool(upstream) == (not reasons):
        return
    plan_says = "; ".join(reasons) if reasons else "capable"
    msg = (
        f"upstream use_fused_qk_norm_rope_gate={bool(upstream)} on "
        f"{platform_name(chip.platform)}, but fused_qk_declines says: {plan_says}. "
        "The copied conditions have drifted from upstream."
    )
    raise ValueError(msg)


def plan_declines(
    plan: AttentionLayerPlan,
    desc: AttentionLayerDesc,
    caps: ChildCapability,
    chip: ChipDesc,
) -> list[str]:
    """The reasons this plan cannot run; empty when it can."""
    if isinstance(plan, FusedLayerPlan):
        reason = supports_fused_attention_layer(desc=desc, platform=chip.platform)
        return [reason] if reason else []
    reasons = []
    if plan.qk_norm_rope_gate is QkPath.FUSED_TRITON:
        reasons += fused_qk_declines(desc, chip)
    if plan.gate in (GatePath.FUSED_MUL, GatePath.FUSED_MUL_QUANT):
        reason = supports_fused_sigmoid_gate_mul(
            x_dtype=plan.attn_output.dtype,
            scale=plan.gate is GatePath.FUSED_MUL_QUANT,
            platform=chip.platform,
        )
        if reason:
            reasons.append(f"fused_sigmoid_gate_mul: {reason}")
    if plan.o_proj_input.quant not in caps.o_proj_inputs:
        quant = plan.o_proj_input.quant.value
        reasons.append(f"o_proj does not accept {quant} input")
    return reasons


def _matches(plan: AttentionLayerPlan, pins: dict[PlanField, PlanValue]) -> bool:
    values = plan_fields(plan)
    return all(values.get(k) is v for k, v in pins.items())


@dataclass(frozen=True)
class PartialLayerPlan:
    """An unresolved plan from one source: None leaves a field to the next."""

    layer: Layer | None = None
    qk_norm_rope_gate: QkPath | None = None
    gate: GatePath | None = None
    o_proj_input: Quant | None = None

    def pins(self) -> dict[PlanField, PlanValue]:
        """The pinned fields, in pin order."""
        pinned: dict[PlanField, PlanValue | None] = {
            PlanField.LAYER: self.layer,
            PlanField.QK_NORM_ROPE_GATE: self.qk_norm_rope_gate,
            PlanField.GATE: self.gate,
            PlanField.O_PROJ_INPUT: self.o_proj_input,
        }
        return {k: v for k, v in pinned.items() if v is not None}


NO_PINS = PartialLayerPlan()

_CONTRACT_KEYS = frozenset(f.name for f in fields(AttentionLayerContract))


def _parse_value[E: enum.Enum](
    kind: type[E], raw: dict[PlanField, str], field: PlanField, source: str
) -> E | None:
    text = raw.get(field)
    if text is None:
        return None
    try:
        return kind(text)
    except ValueError:
        valid = ", ".join(str(v.value) for v in kind)
        raise ValueError(f"{source}: {field.value}={text}; valid: {valid}") from None


def parse_pins(text: str, source: str) -> PartialLayerPlan:
    """The one parse point: one source's "field=value,..." into its partial plan.

    Args:
        text: The source's override text; empty pins nothing.
        source: Who wrote it, for the error text (the env var, "model plan").

    Raises:
        ValueError: A key names a contract field or no field, or a value is
            not one of its field's values.
    """
    raw: dict[PlanField, str] = {}
    contract: list[str] = []
    unknown: list[str] = []
    for item in filter(None, text.split(",")):
        key, _, value = item.partition("=")
        try:
            raw[PlanField(key)] = value
        except ValueError:
            (contract if key in _CONTRACT_KEYS else unknown).append(key)
    if contract:
        raise ValueError(
            f"{source}: {sorted(contract)} belong to the layer contract; "
            "an override pins plan fields, never the contract"
        )
    if unknown:
        raise ValueError(f"{source}: unknown keys {sorted(unknown)}")
    return PartialLayerPlan(
        layer=_parse_value(Layer, raw, PlanField.LAYER, source),
        qk_norm_rope_gate=_parse_value(
            QkPath, raw, PlanField.QK_NORM_ROPE_GATE, source
        ),
        gate=_parse_value(GatePath, raw, PlanField.GATE, source),
        o_proj_input=_parse_value(Quant, raw, PlanField.O_PROJ_INPUT, source),
    )


def _inconsistent(pins: PartialLayerPlan) -> str | None:
    """Why the pinned fields cannot hold together in one plan value."""
    gate, o_proj = pins.gate, pins.o_proj_input
    internal = (pins.qk_norm_rope_gate, gate, o_proj)
    if pins.layer is Layer.FUSED and any(v is not None for v in internal):
        return "layer=fused has no internal fields to pin"
    fp8 = Quant.FP8_STATIC_TENSOR_SYM
    if gate is GatePath.FUSED_MUL_QUANT and o_proj not in (None, fp8):
        return (
            "gate=fused_mul_quant writes an FP8 o_proj input, "
            f"but o_proj_input={o_proj.value if o_proj else None}"
        )
    if o_proj is fp8 and gate not in (None, GatePath.FUSED_MUL_QUANT):
        gate_text = gate.value if gate else None
        return f"only gate=fused_mul_quant writes an FP8 o_proj input, not {gate_text}"
    return None


def _rule(desc: AttentionLayerDesc, plan: UnfusedLayerPlan) -> bool:
    """The family's rules: which capable unfused plan wins for this key.

    One rule, carried over from upstream: the fused QK-norm-RoPE-gate kernel
    serves wherever it can run. The fused sigmoid gate (upstream's
    ``use_fused_output_gate``, set by Qwen3.5 from outside the layer) has no
    rule yet, so it serves only through the override.
    """
    del desc
    return plan.gate in (GatePath.NONE, GatePath.EAGER_MUL)


class Source(enum.Enum):
    """Who decided one field of the resolved plan."""

    USER = "user"
    USER_OVER_MODEL = "user, over the model plan"
    MODEL = "model plan"
    RULE = "rule"
    DEFAULT = "default"


def _fused_qk(plan: AttentionLayerPlan) -> bool:
    return (
        isinstance(plan, UnfusedLayerPlan)
        and plan.qk_norm_rope_gate is QkPath.FUSED_TRITON
    )


def resolve_plan(
    desc: AttentionLayerDesc,
    caps: ChildCapability,
    chip: ChipDesc,
    user: PartialLayerPlan = NO_PINS,
    model: PartialLayerPlan = NO_PINS,
) -> tuple[AttentionLayerPlan, dict[PlanField, Source]]:
    """Completes a partial plan down the tree, then cross-checks it.

    Precedence follows the tree: the user's override, then the model plan,
    then the family's rules, then the default. Returns the plan and, per field,
    the source that decided it. Raises with the reason when one source's pins
    contradict each other, the pins name no plan, or every completion declines.
    """
    # Each source's own pins must be consistent.
    for name, own in (("user", user), ("model plan", model)):
        if reason := _inconsistent(own):
            raise ValueError(
                f"attention plan {_pins_text(own.pins())} ({name}): {reason}"
            )
    # The model plan is itself a partial plan, so it merges field by field
    # under the user's pins: precedence, never a contradiction.
    pins = model.pins() | user.pins()
    sources = dict.fromkeys(model.pins(), Source.MODEL) | {
        k: Source.USER_OVER_MODEL if k in model.pins() else Source.USER
        for k in user.pins()
    }
    label = _pins_text(pins)
    matched = [p for p in enumerate_plans(desc) if _matches(p, pins)]
    if not matched:
        raise ValueError(f"attention plan {label} names no plan for this layer")
    runnable = [p for p in matched if not plan_declines(p, desc, caps, chip)]
    if not runnable:
        reasons = "; ".join(
            r for p in matched for r in plan_declines(p, desc, caps, chip)
        )
        raise ValueError(f"attention plan {label}: {reasons}")

    # Unpinned fields: a rule-picked plan first, preferring the fused QK
    # kernel (upstream behavior); else the default.
    def ruled(p: AttentionLayerPlan) -> bool:
        return isinstance(p, UnfusedLayerPlan) and _rule(desc, p)

    plan = min(runnable, key=lambda p: (not ruled(p), not _fused_qk(p)))
    fill = Source.RULE if ruled(plan) else Source.DEFAULT
    return plan, {k: sources.get(k, fill) for k in plan_fields(plan)}
