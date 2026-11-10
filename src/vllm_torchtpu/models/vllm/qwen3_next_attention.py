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
"""Qwen3NextAttention as an expansion point: one plan behind its boundary.

Prototype for DATA_CONTRACTS section 5.3; see prototype/README.md.

Upstream makes three decisions inside the layer, each from a different place:

- ``use_fused_qk_norm_rope_gate``, computed in ``__init__`` from the RoPE
  module, the dtype and the platform;
- ``use_fused_output_gate``, off by default and set to True by Qwen3.5's
  decoder layer from outside (newer vLLM; absent in v0.33.0);
- the forward's ``get_input_quant_key(self.o_proj) == kFp8StaticTensorSym``
  peek, which fuses the gate with o_proj's FP8 input quantization (newer vLLM).

Here they are one ``AttentionLayerPlan`` resolved in ``__init__``; the two
attributes become read-only views of it, and o_proj's input format is a
contract the plan hands down instead of a peek.
"""

import dataclasses
from collections.abc import Callable
from typing import overload

import torch
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase
from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
    DualChunkRotaryEmbedding,
)
from vllm.model_executor.models import qwen3_next
from vllm.platforms import current_platform

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.layers.expansion import expansion_point
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.patch_registry import ExpansionKey

logger = init_logger(__name__)


def _platform_chip() -> alp.ChipDesc:
    """The chip vLLM's platform reports: the one upstream decides for."""
    # vLLM keeps the platform's enum private; its is_cuda() and kin read it.
    return alp.ChipDesc(current_platform._enum)  # noqa: SLF001


def describe_chip() -> alp.ChipDesc:
    """The chip the plan is resolved for (the prototype substitutes one)."""
    return _platform_chip()


def _rope_positions(rope: torch.nn.Module) -> alp.RopePositions:
    # Exact type, as upstream: MRoPE subclasses compute positions differently.
    if type(rope) is not MRotaryEmbedding:
        return alp.RopePositions.ONE_D
    section = rope.mrope_section
    if (
        section
        and len(section) == 3
        and sum(section) == rope.rotary_dim // 2
        and rope.mrope_interleaved
    ):
        return alp.RopePositions.MROPE_INTERLEAVED_THW
    return alp.RopePositions.MROPE_OTHER


def describe_layer(
    layer: qwen3_next.Qwen3NextAttention, model_config, reduce_results: bool
) -> alp.AttentionLayerDesc:
    rope = layer.rotary_emb
    # The RoPE classes that carry a style and a cache dtype; others (Phi-3's
    # long RoPE) offer the fused kernel neither.
    if isinstance(rope, (RotaryEmbeddingBase, DualChunkRotaryEmbedding)):
        rope_neox, rope_dtype = bool(rope.is_neox_style), rope.dtype
    else:
        rope_neox, rope_dtype = False, None
    mm_config = model_config.multimodal_config if model_config else None
    dtype = model_config.dtype if model_config else torch.get_default_dtype()
    return alp.AttentionLayerDesc(
        hidden_size=layer.hidden_size,
        head_dim=layer.head_dim,
        dtype=dtype,
        output_gate=bool(layer.attn_output_gate),
        rope_neox=rope_neox,
        rope_dtype=rope_dtype,
        rotary_dim=rope.rotary_dim,
        rope_positions=_rope_positions(rope),
        text_only=mm_config is None or mm_config.language_model_only,
        reduce_results=reduce_results,
    )


def describe_children(layer: qwen3_next.Qwen3NextAttention) -> alp.ChildCapability:
    accepts = {alp.Quant.NONE}
    try:
        from vllm.model_executor.layers.fusion.quant_activation import (
            get_input_quant_key,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8StaticTensorSym,
        )
    except ImportError:  # v0.33.0: o_proj takes only unquantized input.
        pass
    else:
        if get_input_quant_key(layer.o_proj) == kFp8StaticTensorSym:
            accepts.add(alp.Quant.FP8_STATIC_TENSOR_SYM)
    return alp.ChildCapability(o_proj_inputs=frozenset(accepts))


def _apply_gate(plan: alp.UnfusedLayerPlan, attn_output, gate, o_proj):
    """Produces o_proj's input in exactly the format the plan resolved."""
    fused_gate_mul = alp._fused_sigmoid_gate_mul()
    match plan.gate:
        case alp.GatePath.NONE:
            return attn_output
        case alp.GatePath.EAGER_MUL:
            return attn_output * torch.sigmoid(gate)
        case alp.GatePath.FUSED_MUL:
            assert fused_gate_mul is not None  # supports() checked it.
            return fused_gate_mul(attn_output, gate)
        case alp.GatePath.FUSED_MUL_QUANT:
            from vllm.model_executor.layers.fusion.quant_activation import (
                QuantizedActivation,
            )
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                kFp8StaticTensorSym,
            )

            assert fused_gate_mul is not None  # supports() checked it.
            scale = o_proj.input_scale
            return QuantizedActivation(
                data=fused_gate_mul(attn_output, gate, scale),
                scale=scale,
                orig_dtype=attn_output.dtype,
                orig_shape=attn_output.shape,
                quant_key=kFp8StaticTensorSym,
            )


class _PlanView:
    """An upstream decision attribute, now derived from ``self.plan``.

    Writes are accepted only while upstream ``__init__`` runs (before the plan
    exists); they are recorded in ``upstream_decisions``, not obeyed. Any later
    write raises.
    """

    def __init__(self, derive: Callable[[alp.AttentionLayerPlan], bool]) -> None:
        self.derive = derive
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name  # for the error text

    @overload
    def __get__(self, layer: None, owner: type) -> "_PlanView": ...
    @overload
    def __get__(self, layer: "TpuQwen3NextAttention", owner: type) -> bool: ...
    def __get__(
        self, layer: "TpuQwen3NextAttention | None", owner: type
    ) -> "bool | _PlanView":
        if layer is None:  # read on the class, as a property returns itself
            return self
        return self.derive(layer.plan)

    def __set__(self, layer: "TpuQwen3NextAttention", value: bool) -> None:
        if layer.upstream_decisions.open:
            layer.upstream_decisions.writes[self] = value
            return
        raise AttributeError(
            f"{self.name} is derived from {type(layer).__name__}.plan; pin a plan "
            "with TPU_ATTENTION_LAYER_PLAN instead "
            f"(current: {alp.plan_name(layer.plan)})"
        )


@dataclasses.dataclass
class UpstreamDecisions:
    """Upstream ``__init__``'s writes to the plan views: recorded, not obeyed."""

    writes: dict[_PlanView, bool] = dataclasses.field(default_factory=dict)
    open: bool = True  # closed once the plan is resolved


@expansion_point
class TpuQwen3NextAttention(qwen3_next.Qwen3NextAttention):
    """Qwen3NextAttention whose internals follow one resolved plan."""

    upstream_init_signature = (
        "(self, config, model_config=None, cache_config=None, quant_config=None, "
        "reduce_results=True, prefix='', mrope_positions_factor=4)"
    )
    upstream_forward_signature = "(self, positions, hidden_states)"
    upstream_parameter_names = (
        "qkv_proj.weight",
        "o_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
    )

    expansion_key = ExpansionKey.QWEN3_NEXT_ATTENTION

    use_fused_qk_norm_rope_gate = _PlanView(
        lambda p: isinstance(p, alp.UnfusedLayerPlan)
        and p.qk_norm_rope_gate is alp.QkPath.FUSED_TRITON
    )
    use_fused_output_gate = _PlanView(
        lambda p: isinstance(p, alp.UnfusedLayerPlan)
        and p.gate in (alp.GatePath.FUSED_MUL, alp.GatePath.FUSED_MUL_QUANT)
    )

    def __init__(
        self,
        config,
        model_config=None,
        cache_config=None,
        quant_config=None,
        reduce_results: bool = True,
        prefix: str = "",
        mrope_positions_factor: int = 4,
    ) -> None:
        # Before upstream __init__, which writes the plan views.
        self.upstream_decisions = UpstreamDecisions()
        super().__init__(
            config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=prefix,
            mrope_positions_factor=mrope_positions_factor,
        )
        self.desc = describe_layer(self, model_config, reduce_results)
        self.contract = alp.layer_contract(
            self.desc,
            tuple((n, tuple(p.shape)) for n, p in self.named_parameters()),
        )
        # Each source's text is parsed once, here, into a typed partial plan.
        user = alp.parse_pins(envs.TPU_ATTENTION_LAYER_PLAN, "TPU_ATTENTION_LAYER_PLAN")
        key = "tpu_attention_layer_plan"  # The model plan: an optional HF config key.
        model_plan = getattr(config, key, "") or ""  # str-ok: optional HF config key
        model = alp.parse_pins(model_plan, "model plan")
        self.plan, self.plan_sources = alp.resolve_plan(
            self.desc, describe_children(self), describe_chip(), user, model
        )
        self.upstream_decisions.open = False
        # Upstream decided use_fused_qk_norm_rope_gate for the platform it runs
        # on, so the plan's copy of that decision is checked on the same one.
        alp.check_upstream_decision(
            upstream=self.upstream_decisions.writes.get(
                type(self).use_fused_qk_norm_rope_gate
            ),
            desc=self.desc,
            chip=_platform_chip(),
        )
        logger.debug("%s: %s %s", prefix, alp.plan_name(self.plan), self.plan_sources)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        match self.plan:
            case alp.FusedLayerPlan():
                # A whole-layer kernel would be called here with the layer's
                # inputs and weights. None is registered, so its supports()
                # declines and resolve_plan never returns this value.
                raise AssertionError("no fused attention-layer kernel")
            case alp.UnfusedLayerPlan() as plan:
                qkv, _ = self.qkv_proj(hidden_states)
                # Upstream's helper branches on use_fused_qk_norm_rope_gate,
                # which now reads plan.qk_norm_rope_gate.
                q, k, v, gate = self._project_qkv_gate(qkv, positions)
                attn_output = self.attn(q, k, v)
                output, _ = self.o_proj(
                    _apply_gate(plan, attn_output, gate, self.o_proj)
                )
                return output
