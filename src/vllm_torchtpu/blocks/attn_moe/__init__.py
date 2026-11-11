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

``pattern`` is the reference forward and the parts' kinds; ``conform`` checks
a declared layer against it; ``plan`` resolves which collectives join the two
halves per bucket; ``contract_adapter`` hands the children the contracts that
plan needs; ``forward`` runs the block through the declared parts.
"""

import contextlib
import dataclasses
import enum
import inspect
from collections.abc import Callable
from typing import overload

import torch
from torch import nn
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu.blocks import (
    BlockDeclaration,
    InstallContext,
    Installed,
    block_implementation,
)
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.attn_moe import plan as plan_mod
from vllm_torchtpu.blocks.attn_moe.contract_adapter import (
    ChildContracts,
    bind_child_contracts,
)
from vllm_torchtpu.blocks.attn_moe.forward import AttentionMoEBlock
from vllm_torchtpu.blocks.attn_moe.pattern import (
    AttnMoEParts,
    attn_then_moe,
    block_key,
    moe_explicit_router,
    moe_runner_router,
)
from vllm_torchtpu.blocks.parts import as_kind

Tensor = torch.Tensor
_TOKENS = 8
_conformed: set[tuple[object, ...]] = set()


class Arg(enum.Enum):
    """What one tensor argument of attention's call is, by identity."""

    NORMED = enum.auto()
    POSITIONS = enum.auto()
    OTHER = enum.auto()


def _bound_arguments(
    module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[object, ...]:
    """A call's arguments in the parameter order of ``module``'s class forward.

    Two calls that pass the same objects to the same parameters give the same
    tuple, however each spelled the call (positional or keyword).
    """
    signature = inspect.signature(type(module).forward)
    bound = signature.bind(module, *args, **kwargs)
    return tuple(bound.arguments.values())[1:]


def _classify(
    call: tuple[object, ...], normed: object, positions: object
) -> tuple[Arg, ...]:
    """Each tensor argument of a call as normed, positions or other."""
    return tuple(
        Arg.NORMED if a is normed else Arg.POSITIONS if a is positions else Arg.OTHER
        for a in call
        if isinstance(a, Tensor)
    )


class _NormStub:
    """An opaque RMSNorm: ``x -> normed`` or ``(x, residual) -> (normed, sum)``."""

    def __init__(self, label: str) -> None:
        self.label = label

    @overload
    def __call__(self, x: Tensor) -> Tensor: ...
    @overload
    def __call__(self, x: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]: ...
    def __call__(
        self, x: Tensor, residual: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        if residual is None:
            return conf.ops.unary(x, self.label)
        return conf.ops.pair(x, residual, self.label)


@dataclasses.dataclass
class _Seen:
    normed: Tensor | None = None
    positions: Tensor | None = None
    call: tuple[object, ...] = ()


def declared_parts(layer: nn.Module, declaration: BlockDeclaration) -> AttnMoEParts:
    """The parts the binding function returns for ``layer``.

    Raises:
        BlockConformanceError: The function returns None or other parts, or a
            part is the wrong kind (named by its attribute).
    """
    parts = declaration.parts(layer)
    if not isinstance(parts, AttnMoEParts):
        raise conf.BlockConformanceError(
            f"{declaration.parts_of.__qualname__} returns {type(parts).__name__} "
            "for this layer, not AttnMoEParts"
        )
    return parts


def conform(layer: nn.Module, declaration: BlockDeclaration) -> AttnMoEParts:
    """Checks a declared layer against attn_then_moe; returns its parts.

    The parts' kinds are checked when the binding function builds them; this
    checks the wiring: the layer's forward, called through the declared
    stream, computes the pattern and calls attention as the declared
    ``attend`` does; the MoE child's forward is one of the router patterns.

    Raises:
        BlockConformanceError: A part is the wrong kind, or the layer's
            forward or its MoE child's forward computes another graph.
    """
    parts = declared_parts(layer, declaration)
    # Layers of one class differ in flags their forward branches on
    # (layer_type, layer_scale); those are bool or str attributes.
    flags = tuple(
        sorted((k, v) for k, v in vars(layer).items() if isinstance(v, (bool, str)))
    )
    shapes = tuple(tuple(t.shape) for t in layer.parameters())
    cache_key = (type(layer), declaration.parts_of, flags, shapes)
    if cache_key in _conformed:
        return parts
    router = as_kind(parts.router, GateLinear)
    runner = as_kind(parts.experts, MoERunner)
    _check_layer(layer, parts, router.input_size)
    check_moe(layer, parts.moe, router, runner)
    _conformed.add(cache_key)
    return parts


def _check_layer(layer: nn.Module, parts: AttnMoEParts, hidden: int) -> None:
    """The layer's forward computes attn_then_moe and calls attention as declared."""
    seen = _Seen()
    in_norm = _NormStub("in_norm")
    where = f"declared attn_then_moe on {type(layer).__qualname__}, but"

    def recording_in_norm(
        x: Tensor, residual: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        if residual is None:
            seen.normed = in_norm(x)
            return seen.normed
        normed, summed = in_norm(x, residual)
        seen.normed = normed
        return normed, summed

    def attention_stub(*a: object, **k: object) -> Tensor:
        seen.call = _bound_arguments(parts.attention, a, k)
        # The stream attention reads: the normed tokens, else the first
        # floating tensor (the graph comparison then names the wrong edge).
        tensors = [t for t in seen.call if isinstance(t, Tensor)]
        normed = next((t for t in tensors if t is seen.normed), None)
        floating = (t for t in tensors if t.is_floating_point())
        stream = normed if normed is not None else next(floating, None)
        if stream is None:
            raise conf.BlockConformanceError(f"{where} attention reads no tokens")
        return conf.ops.unary(stream, "attn")

    def moe_stub(x: Tensor, *a: object, **k: object) -> Tensor:
        del a, k
        return conf.ops.unary(x, "moe")

    def layer_fn(h: Tensor, r: Tensor, p: Tensor) -> object:
        seen.positions = p
        return parts.stream.call(h, r, p)

    def pattern_fn(h: Tensor, r: Tensor, p: Tensor) -> object:
        del p
        return attn_then_moe(
            h,
            r,
            in_norm=_NormStub("in_norm"),
            attn=lambda t: conf.ops.unary(t, "attn"),
            post_norm=_NormStub("post_norm"),
            moe=lambda t: conf.ops.unary(t, "moe"),
        )

    x = torch.empty(_TOKENS, hidden, device="meta", dtype=torch.bfloat16)
    args = (
        x,
        torch.empty_like(x),
        torch.empty(_TOKENS, device="meta", dtype=torch.long),
    )
    names = ("hidden", "residual", "positions")
    stubs: tuple[tuple[nn.Module, Callable[..., object]], ...] = (
        (parts.input_norm, recording_in_norm),
        (parts.post_norm, _NormStub("post_norm")),
        (parts.attention, attention_stub),
        (parts.moe, moe_stub),
    )
    with contextlib.ExitStack() as stack:
        for module, stub in stubs:
            stack.enter_context(conf.opaque(module, stub))
        got = conf.canonicalize(conf.export(layer_fn, args, layer), names)
        layer_call = _classify(seen.call, seen.normed, seen.positions)
        # The declared attend, on fresh tensors, must make the same call.
        normed, positions = torch.empty_like(x), torch.empty_like(args[2])
        seen.normed = normed
        try:
            _ = parts.attend(normed, positions)
        except TypeError as e:
            attend_error = e
        else:
            attend_error = None
        attend_call = _classify(seen.call, normed, positions)
    want = conf.canonicalize(conf.export(pattern_fn, args, None), names)
    if error := conf.compare("attn_then_moe", got, want):
        raise conf.BlockConformanceError(error)
    if attend_error is not None:
        raise conf.BlockConformanceError(
            f"{where} the declared attend does not fit attention's forward: "
            f"{attend_error}"
        ) from attend_error
    if Arg.NORMED not in layer_call:
        raise conf.BlockConformanceError(
            f"{where} attention does not read the normed tokens"
        )
    if Arg.OTHER in layer_call:
        raise conf.BlockConformanceError(
            f"{where} attention reads more than the normed tokens and the positions"
        )
    if attend_call != layer_call:
        raise conf.BlockConformanceError(
            f"{where} the declared attend calls attention another way than the layer"
        )


def check_moe(
    layer: nn.Module, moe: nn.Module, router: GateLinear, runner: MoERunner
) -> None:
    """The MoE child's forward is exactly one of the two router patterns.

    The block forward calls the router and the runner directly, so the MoE
    child's own forward must compute nothing else.
    """
    width, hidden = router.output_size, router.input_size
    holds_router = runner.gate is router
    where = f"declared attn_then_moe on {type(layer).__qualname__} (moe)"

    def router_stub(t: Tensor) -> object:
        logits = conf.ops.project(t, width, "router")
        # A vLLM linear returns (output, bias) unless built with return_bias=False.
        return (logits, None) if router.return_bias else logits

    def experts_stub(
        hidden_states: Tensor, router_logits: Tensor | None = None, **extra: object
    ) -> Tensor:
        if inputs := sorted(k for k, v in extra.items() if v is not None):
            raise conf.BlockConformanceError(
                f"{where}, but the runner also takes {inputs}, which the pattern "
                "does not pass"
            )
        if holds_router:
            return conf.ops.unary(hidden_states, "experts")
        if router_logits is None:
            raise conf.BlockConformanceError(
                f"{where}, but no router logits reach the runner"
            )
        return conf.ops.binary(hidden_states, router_logits, "experts")

    x = torch.empty(_TOKENS, hidden, device="meta", dtype=torch.bfloat16)
    with contextlib.ExitStack() as stack:
        stack.enter_context(conf.opaque(router, router_stub))
        stack.enter_context(conf.opaque(runner, experts_stub))
        got = conf.canonicalize(conf.export(moe, (x,), moe), ("x",))

    def reference(t: Tensor) -> Tensor:
        if holds_router:
            return moe_runner_router(t, experts=lambda u: conf.ops.unary(u, "experts"))
        return moe_explicit_router(
            t,
            router=lambda u: conf.ops.project(u, width, "router"),
            experts=lambda u, v: conf.ops.binary(u, v, "experts"),
        )

    want = conf.canonicalize(conf.export(reference, (x,), None), ("x",))
    if error := conf.compare("attn_then_moe", got, want, " (moe)"):
        raise conf.BlockConformanceError(error)


_block_classes: dict[type, type] = {}

# What the overlap plan hands its children.
_OVERLAP_CONTRACTS = ChildContracts(
    out_proj_reduces=False, experts_hold_router=False, experts_reduce=False
)


def _block_class(cls: type) -> type:
    if cls not in _block_classes:
        block = type(cls.__name__, (AttentionMoEBlock, cls), {})
        block.__qualname__ = cls.__qualname__
        _block_classes[cls] = block
    return _block_classes[cls]


def _summary(table: plan_mod.BlockPlanTable) -> str:
    overlap = [b for b, p, _ in table.entries if p == plan_mod.OVERLAP]
    if not overlap:
        return "all_reduce on every bucket"
    sources = sorted({s.value for _, p, s in table.entries if p == plan_mod.OVERLAP})
    return (
        f"reduce_scatter_all_gather on {len(overlap)}/{len(table.entries)} "
        f"buckets (>= {min(overlap)}, from {', '.join(sources)})"
    )


def apply_block(
    name: str,
    layer: nn.Module,
    parts: AttnMoEParts,
    table: plan_mod.BlockPlanTable,
    tp: int,
) -> None:
    """Binds the overlap's child contracts and swaps in the block forward."""
    bind_child_contracts(
        name,
        parts.out_proj,
        as_kind(parts.experts, MoERunner),
        _OVERLAP_CONTRACTS,
        source=_summary(table),
    )
    layer.__class__ = _block_class(type(layer))
    as_kind(layer, AttentionMoEBlock).bind_block(parts, table, tp)


@block_implementation(attn_then_moe)
class AttnThenMoE:
    """Conforms, resolves and installs one declared attn_then_moe layer."""

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
    ) -> Installed:
        parts = conform(layer, declaration)
        key = block_key(parts, context.tp, context.dtype)
        table, fell_back = plan_mod.resolve_table(
            key, context.buckets, context.block_pin, context.block_model_pin
        )
        cls = type(layer).__qualname__
        if not table.uses(plan_mod.OVERLAP):
            why = f" (rule fell back: {'; '.join(fell_back)})" if fell_back else ""
            return Installed(name, cls, f"conforms; {_summary(table)}{why}")
        apply_block(name, layer, parts, table, context.tp)
        return Installed(name, cls, f"installed; {_summary(table)}")
