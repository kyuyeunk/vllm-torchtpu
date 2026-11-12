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
"""Conforms a declared attn_then_moe layer and installs its runner."""

import contextlib
import dataclasses
import logging

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu.blocks import BlockDeclaration, Pattern, calls
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.attn_moe import plan as plan_mod
from vllm_torchtpu.blocks.attn_moe.contract_adapter import (
    ChildContracts,
    bind_child_contracts,
)
from vllm_torchtpu.blocks.attn_moe.pattern import (
    AttnMoEBinding,
    attn_then_moe,
    block_key,
    moe_explicit_router,
    moe_runner_router,
)
from vllm_torchtpu.blocks.attn_moe.runner import (
    AttnMoERunner,
    ModelAttention,
    ModelAttentionBinding,
)
from vllm_torchtpu.blocks.binding import as_kind
from vllm_torchtpu.blocks.context import InstallContext, Installed
from vllm_torchtpu.blocks.layer_check import (
    TOKENS,
    LayerBinding,
    check_layer,
    norm_stub,
)
from vllm_torchtpu.blocks.norm_attn import install as norm_attn
from vllm_torchtpu.blocks.norm_attn import plan as subblock_plan
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention
from vllm_torchtpu.blocks.stream import route

# vllm_torchtpu.logger's proxy resolves to vLLM's logger of this name.
logger = logging.getLogger("vllm." + __name__)

Tensor = torch.Tensor
_conformed: set[tuple[object, ...]] = set()


def declared_binding(layer: nn.Module, declaration: BlockDeclaration) -> AttnMoEBinding:
    """The binding the binding function returns for ``layer``.

    Raises:
        BlockConformanceError: The function returns None or other binding, or a
            part is the wrong kind (named by its attribute).
    """
    binding = declaration.binding(layer)
    if not isinstance(binding, AttnMoEBinding):
        msg = (
            f"{declaration.binding_of.__qualname__} returns {type(binding).__name__} "
            "for this layer, not AttnMoEBinding"
        )
        raise conf.BlockConformanceError(msg)
    return binding


def conform(layer: nn.Module, declaration: BlockDeclaration) -> AttnMoEBinding:
    """Checks a declared layer against attn_then_moe; returns its binding.

    The binding' kinds are checked when the binding function builds them; this
    checks the wiring: the layer's forward (``check_layer``) and the MoE
    child's forward, which the block replaces by calling the router and the
    experts itself.

    Raises:
        BlockConformanceError: A part is the wrong kind, or the layer's
            forward or its MoE child's forward computes another graph.
    """
    binding = declared_binding(layer, declaration)
    state: dict[str, object] = vars(layer)
    flags = tuple(
        sorted((k, v) for k, v in state.items() if isinstance(v, (bool, str)))
    )
    shapes = tuple(tuple(t.shape) for t in layer.parameters())
    cache_key = (type(layer), declaration.binding_of, flags, shapes)
    if cache_key in _conformed:
        return binding
    router = as_kind(binding.router, GateLinear)
    runner = as_kind(binding.experts, MoERunner)
    width, hidden = router.output_size, router.input_size
    _ = check_layer(
        layer,
        LayerBinding(
            input_norm=binding.input_norm,
            attention=binding.attention,
            post_norm=binding.post_norm,
            rest=binding.moe,
            stream=binding.stream,
            attend=binding.attend,
        ),
        hidden,
        lambda h, r: attn_then_moe(
            h,
            r,
            in_norm=norm_stub("in_norm"),
            attn=lambda t: conf.unary(t, "attn"),
            post_norm=norm_stub("post_norm"),
            moe=lambda t: conf.unary(t, "rest"),
        ),
        "attn_then_moe",
    )
    _check_moe(layer, binding.moe, router, runner, width, hidden)
    _conformed.add(cache_key)
    return binding


def _check_moe(  # noqa: PLR0913 - the layer, its binding, plan, TP size and attention part
    layer: nn.Module,
    moe: nn.Module,
    router: GateLinear,
    runner: MoERunner,
    width: int,
    hidden: int,
) -> None:
    """The MoE child's forward is exactly one of the two router patterns."""
    holds_router = runner.gate is router
    where = f"declared attn_then_moe on {type(layer).__qualname__} (moe)"

    def router_stub(t: Tensor) -> object:
        logits = conf.project(t, width, "router")
        # A vLLM linear returns (output, bias) unless built with return_bias=False.
        return (logits, None) if router.return_bias else logits

    def experts_stub(
        hidden_states: Tensor, router_logits: Tensor | None = None, **extra: object
    ) -> Tensor:
        if any(v is not None for v in extra.values()):
            msg = f"{where}, but the runner takes inputs the pattern does not pass"
            raise conf.BlockConformanceError(msg)
        if holds_router:
            return conf.unary(hidden_states, "experts")
        if router_logits is None:
            msg = f"{where}, but no router logits reach the runner"
            raise conf.BlockConformanceError(msg)
        return conf.binary(hidden_states, router_logits, "experts")

    x = torch.empty(TOKENS, hidden, device="meta", dtype=torch.bfloat16)
    with contextlib.ExitStack() as stack:
        stack.enter_context(conf.opaque(router, router_stub))
        stack.enter_context(conf.opaque(runner, experts_stub))
        got = conf.canonicalize(conf.export(moe, (x,), moe), ("x",))

    def reference(t: Tensor) -> Tensor:
        if holds_router:
            return moe_runner_router(t, experts=lambda u: conf.unary(u, "experts"))
        return moe_explicit_router(
            t,
            router=lambda u: conf.project(u, width, "router"),
            experts=lambda u, v: conf.binary(u, v, "experts"),
        )

    want = conf.canonicalize(conf.export(reference, (x,), None), ("x",))
    if error := conf.compare("attn_then_moe", got, want, " (moe)"):
        raise conf.BlockConformanceError(error)


# What the block hands its MoE child, under either plan value: the block
# reduces the MoE's output itself and computes the router's logits.
_MOE_CONTRACTS = ChildContracts(experts_hold_router=False, experts_reduce=False)


def _summary(table: plan_mod.BlockPlanTable) -> str:
    overlap = [b for b, p, _ in table.entries if p == plan_mod.OVERLAP]
    if not overlap:
        return "all_reduce on every bucket"
    sources = sorted({s.value for _, p, s in table.entries if p == plan_mod.OVERLAP})
    return (
        f"reduce_scatter_all_gather on {len(overlap)}/{len(table.entries)} "
        f"buckets (>= {min(overlap)}, from {', '.join(sources)})"
    )


def build_runner(
    name: str,
    binding: AttnMoEBinding,
    table: plan_mod.BlockPlanTable,
    tp: int,
    attention: calls.AttentionPart | None = None,
) -> AttnMoERunner:
    """The block's runner: the declared binding, and the attention part.

    Args:
        name: The layer's module path.
        binding: The declared binding.
        table: The plan per bucket.
        tp: The tensor-parallel world size.
        attention: The attention part, the nested ``AttentionRunner`` built
            with ``output_reduced=False``; None runs the model's own norm and
            attention (``ModelAttention``).
    """
    if attention is None:
        attention = ModelAttention(
            ModelAttentionBinding(input_norm=binding.input_norm, attend=binding.attend)
        )
    return AttnMoERunner(
        name, binding, attention=attention, table=table, tp=tp, stream=binding.stream
    )


def apply_block(  # noqa: PLR0913 - the layer, its binding, plan, TP size and attention part
    name: str,
    layer: nn.Module,
    binding: AttnMoEBinding,
    table: plan_mod.BlockPlanTable,
    tp: int,
    attention: calls.AttentionPart | None = None,
) -> AttnMoERunner:
    """Builds the block's runner and makes it the layer's forward.

    The MoE child's contracts are written (the interim adapter); attention's
    is a constructor argument of the attention part when there is one.
    """
    bind_child_contracts(
        name,
        None if attention is not None else binding.out_proj,
        binding.experts,
        _MOE_CONTRACTS,
        source=_summary(table),
    )
    runner = build_runner(name, binding, table, tp, attention)
    route(layer, runner)
    return runner


def _all_default(table: plan_mod.BlockPlanTable) -> plan_mod.BlockPlanTable:
    return dataclasses.replace(
        table,
        entries=tuple(
            (b, plan_mod.DEFAULT, plan_mod.Source.DEFAULT) for b, _, _ in table.entries
        ),
    )


class AttnThenMoE:
    """Conforms, resolves and installs one declared attn_then_moe layer.

    Where the layer also declares ``norm_then_attention``, the block builds
    that sub-block's runner itself, stating its contract
    (``output_reduced=False``); a block left at upstream's collectives leaves
    the sub-block to run on its own.
    """

    nests: tuple[Pattern, ...] = (norm_then_attention,)

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
        nested: dict[Pattern, BlockDeclaration] | None = None,
    ) -> Installed:
        """Conforms the layer, resolves its table, installs the runner if needed."""
        binding = conform(layer, declaration)
        key = block_key(binding, context.tp, context.dtype)
        table, fell_back = plan_mod.resolve_table(
            key, context.buckets, context.block_pin, context.block_model_pin
        )
        inner = (nested or {}).get(norm_then_attention)
        if inner is not None and table.uses(plan_mod.OVERLAP):
            # The block needs attention unreduced: ask the floor value first.
            config = norm_attn.conform(layer, inner).config
            inner_binding = norm_attn.declared_binding(layer, inner)
            why = subblock_plan.unreduced_output_supports(
                subblock_plan.UNFUSED,
                subblock_plan.facts_of(inner_binding, config),
                fused_unreduced=False,
            )
            if why and table.pinned():
                msg = (
                    f"{name}: block plan pinned to reduce_scatter_all_gather, but "
                    f"attention cannot return an unreduced output: {why}"
                )
                raise ValueError(msg)
            if why:  # The rule's choice cannot run: upstream's collectives.
                logger.debug("%s: overlap declined: %s", name, why)
                table, fell_back = _all_default(table), (*fell_back, why)
        cls = type(layer).__qualname__
        if not table.uses(plan_mod.OVERLAP):
            why = f" (rule fell back: {'; '.join(fell_back)})" if fell_back else ""
            outcome = f"conforms; {_summary(table)}{why}"
            if inner is not None:
                sub = norm_attn.install_standalone(name, layer, inner, context)
                outcome += f"; sub-block {sub.outcome}"
            return Installed(name, cls, outcome)
        attention, what = None, ""
        if inner is not None:
            _, attention, chosen = norm_attn.resolve_runner(
                name, layer, inner, context, output_reduced=False
            )
            what = f"; nested AttentionRunner {chosen}"
        _ = apply_block(name, layer, binding, table, context.tp, attention)
        return Installed(name, cls, f"AttnMoERunner installed; {_summary(table)}{what}")
