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
"""Conforms a declared norm_then_attention layer and installs its runner."""

import contextlib
import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, override

import torch
from torch import nn
from vllm.model_executor.layers.linear import (
    LinearBase,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_torchtpu.blocks import BlockDeclaration, Pattern
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.context import InstallContext, Installed
from vllm_torchtpu.blocks.layer_check import Arg, LayerParts, check_layer, norm_stub
from vllm_torchtpu.blocks.norm_attn import plan as plan_mod
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnParts, norm_then_attention
from vllm_torchtpu.blocks.norm_attn.runner import (
    AttentionParts,
    AttentionRunner,
    AttnConfig,
    FusedOp,
    InputProj,
    reference_fused_qkv,
)
from vllm_torchtpu.blocks.parts import as_kind
from vllm_torchtpu.blocks.stream import Stream, route

if TYPE_CHECKING:
    from vllm_torchtpu.blocks import calls

# vllm_torchtpu.logger's proxy resolves to vLLM's logger of this name.
logger = logging.getLogger("vllm." + __name__)

_TOKENS = 8
Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class Conformed:
    """What conformance found for one declared layer."""

    config: AttnConfig


_conformed: dict[tuple[object, ...], Conformed] = {}


def _fail(layer: nn.Module, why: str) -> conf.BlockConformanceError:
    return conf.BlockConformanceError(
        f"declared norm_then_attention on {type(layer).__qualname__}, but {why}"
    )


def declared_parts(layer: nn.Module, declaration: BlockDeclaration) -> NormAttnParts:
    """The parts the binding function returns for ``layer``.

    Raises:
        BlockConformanceError: The function returns None or other parts, or a
            part is the wrong kind (named by its attribute).
    """
    parts = declaration.parts(layer)
    if not isinstance(parts, NormAttnParts):
        raise _fail(
            layer,
            f"{declaration.parts_of.__qualname__} returns {type(parts).__name__}, "
            "not NormAttnParts",
        )
    return parts


def conform(layer: nn.Module, declaration: BlockDeclaration) -> Conformed:
    """Checks a declared layer against norm_then_attention.

    Three checks, all on the meta device: the parts are the right kinds (when
    the binding function builds them); with the parts opaque, the layer's
    forward computes the pattern's graph and calls attention exactly as the
    declared ``attend`` does (the normed tokens and the positions, nothing
    else); and, with attention's parts opaque, attention's own forward is the
    runner's composition (``conform_attention``), so the runner can replace
    it.

    Raises:
        BlockConformanceError: Naming the part, edge and source line.
    """
    declared = declared_parts(layer, declaration)
    # Layers of one class differ in flags their forward branches on
    # (layer_type, layer_scale) and, built at other sizes, in their shapes.
    state: dict[str, object] = vars(layer)
    flags = tuple(
        sorted((k, v) for k, v in state.items() if isinstance(v, (bool, str)))
    )
    shapes = tuple(tuple(t.shape) for t in layer.parameters())
    cache_key = (type(layer), declaration.parts_of, flags, shapes)
    if cache_key in _conformed:
        return _conformed[cache_key]
    reads = check_layer(
        layer,
        LayerParts(
            input_norm=declared.input_norm,
            attention=declared.attention,
            post_norm=declared.post_norm,
            rest=declared.mlp,
            stream=declared.stream,
            attend=declared.attend,
        ),
        as_kind(declared.qkv_proj, LinearBase).input_size,
        lambda h, r: norm_then_attention(
            h,
            r,
            in_norm=norm_stub("in_norm"),
            attn=lambda t: conf.unary(t, "attn"),
            post_norm=norm_stub("post_norm"),
            mlp=lambda t: conf.unary(t, "rest"),
        ),
        "norm_then_attention",
    )
    if Arg.POSITIONS not in reads:
        raise _fail(
            layer, "attention does not read the positions the runner's RoPE needs"
        )

    # Attention's own forward against the runner's composition. The declared
    # parts are opaque there, so a forward that calls another module than the
    # one declared computes another graph.
    config = conform_attention(declared, f" on {type(layer).__qualname__}")
    conformed = Conformed(config)
    _conformed[cache_key] = conformed
    return conformed


def _missing(where: str, what: str) -> conf.BlockConformanceError:
    return conf.BlockConformanceError(
        f"declared norm_then_attention{where}, but attention lacks a role property "
        f"the runner implements: {what}"
    )


def attention_config(parts: NormAttnParts, where: str = "") -> AttnConfig:
    """The properties the runner's composition reads, from the declared parts.

    Raises:
        BlockConformanceError: A property the runner needs is missing, or
            present in a form it does not implement.
    """
    # The projection holds this rank's head counts; its query heads include
    # the declared output gate's copies.
    qkv = as_kind(parts.qkv_proj, QKVParallelLinear)
    head_dim = qkv.head_size
    for label, norm in (("q_norm", parts.q_norm), ("k_norm", parts.k_norm)):
        weight = None if norm is None else next(norm.parameters(), None)
        if weight is not None and weight.numel() != head_dim:
            raise _missing(
                where,
                f"{label} over {weight.numel()} values; the runner implements a "
                f"per-head norm over head_dim {head_dim}",
            )
    return AttnConfig(
        num_heads=qkv.num_heads // 2 if parts.output_gate else qkv.num_heads,
        num_kv_heads=qkv.num_kv_heads,
        head_dim=head_dim,
        qk_norm=parts.q_norm is not None,
        output_gate=parts.output_gate,
    )


def conform_attention(parts: NormAttnParts, where: str = "") -> AttnConfig:
    """Checks that attention's forward is the runner's composition of its parts.

    With every declared part opaque, attention's own forward (called as the
    declared ``attend`` calls it) must export to the same graph as
    ``AttentionRunner.attention_from_qkv`` after the projection: the same
    split, q/k norms, RoPE, gate and output projection, and nothing else.

    Raises:
        BlockConformanceError: Naming the part, edge and source line.
    """
    config = attention_config(parts, where)
    qkv_proj = as_kind(parts.qkv_proj, QKVParallelLinear)
    o_proj = as_kind(parts.o_proj, RowParallelLinear)

    def projection(linear: LinearBase, width: int, label: str) -> Pattern:
        def stub(x: Tensor, *a: object, **k: object) -> object:
            del a, k
            out = conf.project(x, width, label)
            return (out, None) if linear.return_bias else out

        return stub

    def rope_stub(
        positions: Tensor, q: Tensor, k: Tensor, *a: object, **kw: object
    ) -> object:
        del positions, a, kw
        return conf.pair(q, k, "rope")

    def attn_stub(q: Tensor, k: Tensor, v: Tensor, *a: object, **kw: object) -> Tensor:
        del a, kw
        return conf.binary(conf.binary(q, k, "attn"), v, "attn_v")

    def norm_stub(label: str) -> Callable[[Tensor], Tensor]:
        return lambda x: conf.unary(x, label)

    stubs: list[tuple[nn.Module, Pattern]] = [
        (
            qkv_proj,
            projection(qkv_proj, qkv_proj.output_size_per_partition, "qkv_proj"),
        ),
        (o_proj, projection(o_proj, o_proj.output_size_per_partition, "o_proj")),
        (parts.rotary, rope_stub),
        (parts.attn, attn_stub),
    ]
    if parts.q_norm is not None and parts.k_norm is not None:
        stubs += [
            (parts.q_norm, norm_stub("q_norm")),
            (parts.k_norm, norm_stub("k_norm")),
        ]
    runner = AttentionRunner(
        "conformance",
        AttentionParts(
            input_norm=nn.Identity(),
            qkv_proj=parts.qkv_proj,
            attn=parts.attn,
            o_proj=parts.o_proj,
            rotary=parts.rotary,
            q_norm=parts.q_norm,
            k_norm=parts.k_norm,
        ),
        config=config,
    )
    args = (
        torch.empty(_TOKENS, qkv_proj.input_size, device="meta", dtype=torch.bfloat16),
        torch.empty(_TOKENS, device="meta", dtype=torch.long),
    )
    names = ("hidden", "positions")

    def runner_fn(x: Tensor, p: Tensor) -> Tensor:
        qkv: Tensor = runner.qkv_proj(x)[0]
        return runner.attention_from_qkv(qkv, p)

    with contextlib.ExitStack() as stack:
        for module, stub in stubs:
            stack.enter_context(conf.opaque(module, stub))
        got = conf.canonicalize(conf.export(parts.attend, args, parts.attention), names)
        want = conf.canonicalize(conf.export(runner_fn, args, None), names)
    if error := conf.compare("norm_then_attention", got, want, f" (attention{where})"):
        raise conf.BlockConformanceError(error)
    return config


class NormAttnLayer(nn.Module):
    """A declared layer's forward with no enclosing block: the runner, then the rest.

    Runs the sub-block through its ``AttentionRunner``, then the layer's
    post norm and MLP (opaque), which conformance has shown is all the layer
    computes. ``route`` makes it the layer's forward on the instance.
    """

    runner: AttentionRunner  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    post_norm: "calls.ResidualNorm"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    mlp: "calls.Mlp"  # pyright: ignore[reportUninitializedInstanceVariable] - set in __init__ by object.__setattr__
    stream: Stream

    def __init__(self, runner: AttentionRunner, parts: NormAttnParts) -> None:
        """Keeps references to the parts; creates no parameter."""
        super().__init__()
        for name, part in (
            ("runner", runner),
            ("post_norm", parts.post_norm),
            ("mlp", parts.mlp),
        ):
            object.__setattr__(self, name, part)  # a reference, not a child
        self.stream = parts.stream

    @override
    def forward(self, *args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
        """Takes the layer's own arguments: it stands in for its forward."""
        hidden, residual, positions = self.stream.unpack(*args, **kwargs)
        x, stream = self.runner.forward(hidden, residual, positions)
        x, stream = self.post_norm(x, stream)
        return self.mlp(x), stream


def build_runner(  # noqa: PLR0913 - the runner's arguments, as AttentionRunner takes them
    name: str,
    parts: AttentionParts,
    config: AttnConfig,
    input_proj: InputProj,
    fused_op: FusedOp | None = None,
    *,
    output_reduced: bool = True,
) -> AttentionRunner:
    """The runner for one declared layer: the declared parts are its arguments."""
    return AttentionRunner(
        name,
        parts,
        config=config,
        input_proj=input_proj,
        fused_op=fused_op if input_proj is InputProj.FUSED_QKV else None,
        output_reduced=output_reduced,
    )


FUSED_OP: FusedOp = reference_fused_qkv
"""The fused value's implementation: the plain-torch reference, or a kernel."""

FUSED_OP_UNREDUCED = True
"""``FUSED_OP``'s capability: it honours ``runner.output_reduced=False``."""


def fused_op_for(name: str, parts: NormAttnParts) -> tuple[FusedOp, bool, str]:
    """The fused value's op for this layer, its unreduced capability, and a note."""
    del name, parts
    return FUSED_OP, FUSED_OP_UNREDUCED, ""


def resolve_runner(
    name: str,
    layer: nn.Module,
    declaration: BlockDeclaration,
    context: InstallContext,
    *,
    output_reduced: bool,
) -> tuple[NormAttnParts, AttentionRunner, str]:
    """Conforms, chooses the plan and builds the runner for one declared layer.

    The parent states the runner's contract (``output_reduced``); the plan
    takes only values that can honour it.

    Returns:
        The parts, the runner, and what was chosen, for the install report.

    Raises:
        BlockConformanceError: The layer does not conform.
        ValueError: A pinned value cannot run on this layer.
    """
    config = conform(layer, declaration).config
    parts = declared_parts(layer, declaration)
    fused_op, fused_unreduced, via = fused_op_for(name, parts)
    pins = context.subblock_pins or plan_mod.Pins()
    plan, declined = plan_mod.choose(
        pins,
        plan_mod.facts_of(parts, config),
        chip=context.chip,
        fp8_kv_cache=context.fp8_kv_cache,
        where=type(layer).__qualname__,
        output_reduced=output_reduced,
        fused_unreduced=fused_unreduced,
    )
    for item in declined:  # The rule's value fell back: quiet.
        logger.debug("%s: %s", name, item)
    runner = build_runner(
        name, parts, config, plan.input_proj, fused_op, output_reduced=output_reduced
    )
    source = "user" if pins else "rule"
    fused = plan.input_proj is InputProj.FUSED_QKV
    matmul = f" matmul={plan.matmul.value}" if fused else ""
    via = via if fused else ""
    reduced = "" if output_reduced else " output_reduced=False"
    what = f"input_proj={plan.input_proj.value}{matmul}{reduced} ({source}){via}"
    return parts, runner, what


def install_standalone(
    name: str, layer: nn.Module, declaration: BlockDeclaration, context: InstallContext
) -> Installed:
    """Routes a declared layer with no enclosing block through its runner."""
    parts, runner, what = resolve_runner(
        name, layer, declaration, context, output_reduced=True
    )
    route(layer, NormAttnLayer(runner, parts))
    return Installed(name, type(layer).__qualname__, f"runner installed; {what}")


class NormThenAttention:
    """Conforms, resolves and installs one declared norm_then_attention layer.

    Inside a declared block that nests this pattern, the block's installer
    calls ``resolve_runner`` itself, with the block's contract.
    """

    nests: tuple[Pattern, ...] = ()

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
        nested: dict[Pattern, BlockDeclaration] | None = None,
    ) -> Installed:
        """Installs the runner on one layer that declares no enclosing block."""
        del nested  # This pattern nests none.
        return install_standalone(name, layer, declaration, context)
