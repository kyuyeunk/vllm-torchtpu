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
"""The attention-then-MoE block: PR #1492's collective overlap, framework-owned.

``pattern`` is the reference forward and the role kinds; ``conform`` checks a
declared layer against it; ``plan`` resolves which collectives join the two
halves per bucket; ``runner`` is the one forward every installed block runs,
built from the layer's parts (the nested ``AttentionRunner`` among them, with
the block's contract stated at construction); ``contract_adapter`` writes the
MoE child's contracts, which the model built without them.
"""

import contextlib
import inspect
from typing import Any

import torch
from torch import nn

from vllm_torchtpu.blocks import (
    BlockDeclaration,
    InstallContext,
    Installed,
    block_implementation,
    norm_attn,
)
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.attn_moe import plan as plan_mod
from vllm_torchtpu.blocks.attn_moe.contract_adapter import (
    ChildContracts,
    bind_child_contracts,
)
from vllm_torchtpu.blocks.attn_moe.pattern import (
    AttnMoERoles,
    attn_then_moe,
    block_key,
    check_roles,
    moe_explicit_router,
    moe_parts,
    moe_runner_router,
    out_proj_of,
)
from vllm_torchtpu.blocks.attn_moe.runner import AttnMoERunner, ModelAttention
from vllm_torchtpu.blocks.norm_attn import plan as subblock_plan
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention
from vllm_torchtpu.blocks.stream import StreamParams, route
from vllm_torchtpu.logger import init_logger

_STREAM = {"hidden": ("hidden_states", "x"), "residual": ("residual",)}
_POSITIONS = ("positions", "position_ids")
_TOKENS = 8
_conformed: dict[tuple[object, ...], AttnMoERoles] = {}


def _stream_params(layer: nn.Module) -> tuple[str, str, str]:
    params = inspect.signature(layer.forward).parameters
    hidden = next((p for p in _STREAM["hidden"] if p in params), "")
    residual = next((p for p in _STREAM["residual"] if p in params), "")
    positions = next((p for p in _POSITIONS if p in params), "")
    required = [
        p
        for p, v in params.items()
        if v.default is inspect.Parameter.empty
        and v.kind not in (v.VAR_KEYWORD, v.VAR_POSITIONAL)
        and p not in (hidden, residual, positions)
    ]
    if not hidden or not residual or required:
        raise conf.BlockConformanceError(
            f"declared attn_then_moe, but forward{tuple(params)} does not take "
            "(hidden_states, residual[, positions])"
        )
    return hidden, residual, positions


def _norm_stub(role: str):  # noqa: ANN202
    def stub(x: torch.Tensor, residual: torch.Tensor | None = None) -> Any:  # noqa: ANN401
        if residual is None:
            return conf.ops.unary(x, role)
        return conf.ops.pair(x, residual, role)

    return stub


def conform(layer: nn.Module, declaration: BlockDeclaration) -> AttnMoERoles:
    """Checks a declared layer against attn_then_moe; returns its roles.

    Raises:
        BlockConformanceError: The roles are the wrong kinds, or the layer's
            forward or its MoE child's forward computes another graph.
    """
    binding = declaration.bind(layer)
    # Layers of one class differ in flags their forward branches on
    # (layer_type, layer_scale); those are bool or str attributes.
    flags = tuple(
        sorted((k, v) for k, v in vars(layer).items() if isinstance(v, (bool, str)))
    )
    cache_key = (type(layer), tuple(sorted(binding.items())), flags)
    if cache_key in _conformed:
        return _conformed[cache_key]
    if problems := check_roles(layer, binding):
        raise conf.BlockConformanceError(
            f"declared attn_then_moe on {type(layer).__qualname__}, but "
            + "; ".join(problems)
        )
    hidden_p, residual_p, positions_p = _stream_params(layer)
    moe = layer.get_submodule(binding["moe"])
    runner_name, router_name = moe_parts(moe)
    runner, router = moe.get_submodule(runner_name), moe.get_submodule(router_name)
    width = int(router.output_size)
    hidden = int(router.input_size)
    meta = {"device": "meta", "dtype": torch.bfloat16}
    x = torch.empty(_TOKENS, hidden, **meta)
    attn = layer.get_submodule(binding["attn"])
    attn_params = list(inspect.signature(attn.forward).parameters)
    attn_input = next(
        (p for p in _STREAM["hidden"] if p in attn_params), attn_params[0]
    )
    seen: dict[str, Any] = {}

    def attn_stub(*args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        bound = dict(zip(attn_params, args, strict=False)) | kwargs
        seen["attn"] = bound
        return conf.ops.unary(bound[attn_input], "attn")

    def layer_fn(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        seen["inputs"] = {hidden_p: h, residual_p: r, positions_p: p}
        call = {hidden_p: h, residual_p: r} | ({positions_p: p} if positions_p else {})
        return layer(**call)

    def pattern_fn(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        del p
        return attn_then_moe(
            h,
            r,
            in_norm=_norm_stub("in_norm"),
            attn=lambda t: conf.ops.unary(t, "attn"),
            post_norm=_norm_stub("post_norm"),
            moe=lambda t: conf.ops.unary(t, "moe"),
        )

    args = (
        x,
        torch.empty_like(x),
        torch.empty(_TOKENS, device="meta", dtype=torch.long),
    )
    names = ("hidden", "residual", "positions")
    in_norm_stub = _norm_stub("in_norm")

    def recording_in_norm(*a: Any, **k: Any) -> Any:  # noqa: ANN401
        seen["normed"] = out = in_norm_stub(*a, **k)
        return out

    stubs = {
        binding["in_norm"]: recording_in_norm,
        binding["post_norm"]: _norm_stub("post_norm"),
        binding["attn"]: attn_stub,
        binding["moe"]: lambda hidden_states, *a, **k: conf.ops.unary(
            hidden_states, "moe"
        ),
    }
    with contextlib.ExitStack() as stack:
        for child, stub in stubs.items():
            stack.enter_context(conf.opaque(layer.get_submodule(child), stub))
        got = conf.canonicalize(conf.export(layer_fn, args, layer), names)
    want = conf.canonicalize(conf.export(pattern_fn, args, None), names)
    if error := conf.compare("attn_then_moe", got, want):
        raise conf.BlockConformanceError(error)

    # The block forward calls the router and the runner directly, so the MoE
    # child's own forward must be exactly one of the two router patterns.
    runner_holds_router = runner.gate is router
    with contextlib.ExitStack() as stack:
        # A vLLM linear returns (output, bias) unless built with return_bias=False.
        tupled = getattr(router, "return_bias", True)
        stack.enter_context(
            conf.opaque(
                router,
                lambda t: (
                    (conf.ops.project(t, width, "router"), None)
                    if tupled
                    else conf.ops.project(t, width, "router")
                ),
            )
        )

        def stub(
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor | None = None,
            **extra: Any,  # noqa: ANN401
        ) -> torch.Tensor:
            if inputs := sorted(k for k, v in extra.items() if v is not None):
                raise conf.BlockConformanceError(
                    f"declared attn_then_moe (moe={binding['moe']}), but the "
                    f"runner also takes {inputs}, which the pattern does not pass"
                )
            if runner_holds_router:
                return conf.ops.unary(hidden_states, "experts")
            assert router_logits is not None
            return conf.ops.binary(hidden_states, router_logits, "experts")

        stack.enter_context(conf.opaque(runner, stub))
        got = conf.canonicalize(conf.export(lambda t: moe(t), (x,), moe), ("x",))
    if runner_holds_router:
        ref = lambda t: moe_runner_router(  # noqa: E731
            t, experts=lambda u: conf.ops.unary(u, "experts")
        )
    else:
        ref = lambda t: moe_explicit_router(  # noqa: E731
            t,
            router=lambda u: conf.ops.project(u, width, "router"),
            experts=lambda u, v: conf.ops.binary(u, v, "experts"),
        )
    want = conf.canonicalize(conf.export(ref, (x,), None), ("x",))
    if error := conf.compare("attn_then_moe", got, want, f" (moe={binding['moe']})"):
        raise conf.BlockConformanceError(error)

    sources = {id(v): k for k, v in seen["inputs"].items() if k}
    sources[id(seen["normed"][0])] = "normed"
    attention_args = []
    for k, v in seen["attn"].items():
        if isinstance(v, torch.Tensor):
            attention_args.append((k, sources[id(v)]))
    roles = AttnMoERoles(
        input_norm=binding["in_norm"],
        post_norm=binding["post_norm"],
        attention=binding["attn"],
        out_proj=f"{binding['attn']}.{out_proj_of(attn, hidden)}",
        moe=binding["moe"],
        experts=f"{binding['moe']}.{runner_name}",
        router=f"{binding['moe']}.{router_name}",
        attention_args=tuple(attention_args),
        hidden_param=hidden_p,
        residual_param=residual_p,
        positions_param=positions_p,
    )
    _conformed[cache_key] = roles
    return roles


logger = init_logger(__name__)

# What the block hands its MoE child, under either plan value: the block
# reduces the MoE's output itself and computes the router's logits.
_MOE_CONTRACTS = ChildContracts(experts_hold_router=False, experts_reduce=False)


def _summary(table: plan_mod.BlockPlanTable) -> str:
    overlap = [b for b, p, _ in table.entries if p == plan_mod.OVERLAP]
    if not overlap:
        return "all_reduce on every bucket"
    sources = sorted({s for _, p, s in table.entries if p == plan_mod.OVERLAP})
    return (
        f"reduce_scatter_all_gather on {len(overlap)}/{len(table.entries)} "
        f"buckets (>= {min(overlap)}, from {', '.join(sources)})"
    )


def build_runner(
    name: str,
    layer: nn.Module,
    roles: AttnMoERoles,
    table: plan_mod.BlockPlanTable,
    tp: int,
    front: nn.Module | None = None,
) -> AttnMoERunner:
    """The block's runner, from the layer's parts and the nested front half.

    Args:
        front: The nested ``AttentionRunner``, built with
            ``output_reduced=False``; None runs the model's own norm and
            attention (``ModelAttention``).
    """
    if front is None:
        front = ModelAttention(
            input_norm=layer.get_submodule(roles.input_norm),
            attention=layer.get_submodule(roles.attention),
            attention_args=roles.attention_args,
            positions_param=roles.positions_param,
        )
    return AttnMoERunner(
        name,
        front=front,
        post_norm=layer.get_submodule(roles.post_norm),
        router=layer.get_submodule(roles.router),
        experts=layer.get_submodule(roles.experts),
        table=table,
        tp=tp,
        stream=StreamParams.of(
            layer, roles.hidden_param, roles.residual_param, roles.positions_param
        ),
    )


def apply_block(
    name: str,
    layer: nn.Module,
    roles: AttnMoERoles,
    table: plan_mod.BlockPlanTable,
    tp: int,
    front: nn.Module | None = None,
) -> AttnMoERunner:
    """Builds the block's runner and makes it the layer's forward.

    The MoE child's contracts are written (the interim adapter); attention's
    is a constructor argument of ``front`` when there is one.
    """
    bind_child_contracts(
        name,
        None if front is not None else layer.get_submodule(roles.out_proj),
        layer.get_submodule(roles.experts),
        _MOE_CONTRACTS,
        source=_summary(table),
    )
    runner = build_runner(name, layer, roles, table, tp, front)
    route(layer, runner)
    return runner


def _all_default(table: plan_mod.BlockPlanTable) -> plan_mod.BlockPlanTable:
    return plan_mod.BlockPlanTable(
        tuple((b, plan_mod.DEFAULT, "default") for b, _, _ in table.entries)
    )


@block_implementation(attn_then_moe)
class AttnThenMoE:
    """Conforms, resolves and installs one declared attn_then_moe layer.

    Where the layer also declares ``norm_then_attention``, the block builds
    that sub-block's runner itself, stating its contract
    (``output_reduced=False``); a block left at upstream's collectives leaves
    the sub-block to run on its own.
    """

    nests = (norm_then_attention,)

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
        nested: dict[Any, BlockDeclaration],
    ) -> Installed:
        roles = conform(layer, declaration)
        key = block_key(layer, declaration.bind(layer), context.tp, context.dtype)
        table, fell_back = plan_mod.resolve_table(
            key, context.buckets, context.override, context.model_plan
        )
        inner = nested.get(norm_then_attention)
        if inner is not None and table.uses("reduce_scatter_all_gather"):
            # The block needs attention unreduced: ask the floor value first.
            sub_roles = norm_attn.conform(layer, inner)
            floor = subblock_plan.NormAttnPlan("unfused", "bf16")
            why = subblock_plan.unreduced_output_supports(
                floor,
                layer.get_submodule(sub_roles.attention),
                fused_unreduced=False,
            )
            if why and any(s in ("user", "model plan") for _, _, s in table.entries):
                raise ValueError(
                    f"{name}: block plan pinned to reduce_scatter_all_gather, but "
                    f"attention cannot return an unreduced output: {why}"
                )
            if why:  # The rule's choice cannot run: upstream's collectives.
                logger.debug("%s: overlap declined: %s", name, why)
                table, fell_back = _all_default(table), (*fell_back, why)
        cls = type(layer).__qualname__
        if not table.uses("reduce_scatter_all_gather"):
            why = f" (rule fell back: {'; '.join(fell_back)})" if fell_back else ""
            outcome = f"conforms; {_summary(table)}{why}"
            if inner is not None:
                sub = norm_attn.install_standalone(name, layer, inner, context)
                outcome += f"; sub-block {sub.outcome}"
            return Installed(name, cls, outcome)
        front, what = None, ""
        if inner is not None:
            _, front, chosen = norm_attn.resolve_runner(
                name, layer, inner, context, output_reduced=False
            )
            what = f"; nested AttentionRunner {chosen}"
        apply_block(name, layer, roles, table, context.tp, front)
        return Installed(name, cls, f"AttnMoERunner installed; {_summary(table)}{what}")
