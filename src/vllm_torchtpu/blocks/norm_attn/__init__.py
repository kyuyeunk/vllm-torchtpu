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
"""The pre-norm attention sub-block: one fusion for every model that declares it.

``pattern`` is the reference forward and the role kinds; ``conform`` checks a
declared layer against it, twice: the layer's wiring against the pattern, and
attention's own forward against ``AttentionRunner``'s composition; ``plan``
picks the value; ``runner`` is the one forward every declared layer then runs,
shaped like vLLM's ``MoERunner``.
"""

import contextlib
import inspect
from typing import Any

import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, QKVParallelLinear

from vllm_torchtpu.blocks import (
    BlockDeclaration,
    InstallContext,
    Installed,
    block_implementation,
)
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.norm_attn import plan as plan_mod
from vllm_torchtpu.blocks.norm_attn.pattern import (
    NormAttnParts,
    NormAttnRoles,
    norm_then_attention,
)
from vllm_torchtpu.blocks.norm_attn.runner import (
    AttentionParts,
    AttentionRunner,
    AttnConfig,
    FusedOp,
    reference_fused_qkv,
)
from vllm_torchtpu.blocks.parts import as_kind, attribute_of, require
from vllm_torchtpu.blocks.stream import StreamParams
from vllm_torchtpu.logger import init_logger

_PATTERN = "norm_then_attention"
_TOKENS = 8
_conformed: dict[tuple[object, ...], NormAttnRoles] = {}


def _fail(layer: nn.Module, why: str) -> conf.BlockConformanceError:
    return conf.BlockConformanceError(
        f"declared {_PATTERN} on {type(layer).__qualname__}, but {why}"
    )


def original_class(layer: nn.Module) -> type:
    """The model's own class, under the mixin a standalone install placed."""
    for cls in type(layer).__mro__:
        if cls is not NormAttnDecoder and NormAttnDecoder not in cls.__bases__:
            return cls
    raise TypeError(type(layer))


def _stream_params(layer: nn.Module) -> tuple[str, str, str]:
    params = inspect.signature(original_class(layer).forward).parameters
    hidden = next((p for p in ("hidden_states", "x") if p in params), "")
    residual = "residual" if "residual" in params else ""
    positions = next((p for p in ("positions", "position_ids") if p in params), "")
    if not hidden or not residual:
        raise _fail(layer, f"forward{tuple(params)} takes no (hidden, residual)")
    return hidden, residual, positions


def _norm_stub(role: str):  # noqa: ANN202
    def stub(x: torch.Tensor, residual: torch.Tensor | None = None) -> Any:  # noqa: ANN401
        if residual is None:
            return conf.ops.unary(x, role)
        return conf.ops.pair(x, residual, role)

    return stub


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


def conform(layer: nn.Module, declaration: BlockDeclaration) -> NormAttnRoles:
    """Checks a declared layer against norm_then_attention; returns its roles.

    Three checks, all on the meta device: the roles are the right kinds; with
    the roles opaque, the layer's forward computes the pattern's graph (the
    normed tokens and the positions reach nothing but attention); and, with
    attention's parts opaque, attention's own forward is the runner's
    composition (``conform_attention``), so the runner can replace it.

    The parts' kinds are checked when the binding function builds them; the
    attention parts it declares must be the ones attention's forward calls.

    Raises:
        BlockConformanceError: Naming the role, edge and source line.
    """
    declared = declared_parts(layer, declaration)
    binding = declaration.bind(layer)
    flags = tuple(
        sorted((k, v) for k, v in vars(layer).items() if isinstance(v, (bool, str)))
    )
    # Layers of one class built at other sizes (another config) differ in
    # their parameters' shapes, which the roles' config is read from.
    shapes = tuple((n, tuple(t.shape)) for n, t in layer.named_parameters())
    cache_key = (type(layer), tuple(sorted(binding.items())), flags, shapes)
    if cache_key in _conformed:
        return _conformed[cache_key]
    hidden_p, residual_p, positions_p = _stream_params(layer)
    hidden = as_kind(declared.qkv_proj, QKVParallelLinear).input_size
    attn = declared.attention
    attn_params = list(inspect.signature(attn.forward).parameters)
    attn_input = next(
        (p for p in ("hidden_states", "x") if p in attn_params), attn_params[0]
    )
    meta = {"device": "meta", "dtype": torch.bfloat16}
    x = torch.empty(_TOKENS, hidden, **meta)
    args = (
        x,
        torch.empty_like(x),
        torch.empty(_TOKENS, device="meta", dtype=torch.long),
    )
    names = ("hidden", "residual", "positions")
    seen: dict[str, Any] = {}

    def attn_stub(*a: Any, **k: Any) -> torch.Tensor:  # noqa: ANN401
        bound = dict(zip(attn_params, a, strict=False)) | k
        seen["attn"] = bound
        return conf.ops.unary(bound[attn_input], "attn")

    in_norm_stub = _norm_stub("in_norm")

    def recording_in_norm(*a: Any, **k: Any) -> Any:  # noqa: ANN401
        seen["normed"] = out = in_norm_stub(*a, **k)
        return out

    forward = original_class(layer).forward.__get__(layer)

    def layer_fn(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        seen["inputs"] = {hidden_p: h, residual_p: r, positions_p: p}
        call = {hidden_p: h, residual_p: r} | ({positions_p: p} if positions_p else {})
        return forward(**call)

    def pattern_fn(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        del p
        return norm_then_attention(
            h,
            r,
            in_norm=_norm_stub("in_norm"),
            attn=lambda t: conf.ops.unary(t, "attn"),
            post_norm=_norm_stub("post_norm"),
            mlp=lambda t: conf.ops.unary(t, "mlp"),
        )

    stubs = {
        binding["input_norm"]: recording_in_norm,
        binding["post_norm"]: _norm_stub("post_norm"),
        binding["attention"]: attn_stub,
        binding["mlp"]: lambda hidden_states, *a, **k: conf.ops.unary(
            hidden_states, "mlp"
        ),
    }
    with contextlib.ExitStack() as stack:
        for child, stub in stubs.items():
            stack.enter_context(conf.opaque(layer.get_submodule(child), stub))
        got = conf.canonicalize(conf.export(layer_fn, args, layer), names)
    want = conf.canonicalize(conf.export(pattern_fn, args, None), names)
    if error := conf.compare(_PATTERN, got, want):
        raise conf.BlockConformanceError(error)

    sources = {id(v): k for k, v in seen["inputs"].items() if k}
    sources[id(seen["normed"][0])] = "normed"
    attention_args = tuple(
        (k, sources[id(v)])
        for k, v in seen["attn"].items()
        if isinstance(v, torch.Tensor)
    )
    reads = dict(attention_args)
    if sorted(reads.values()) != sorted(["normed", positions_p]) or not positions_p:
        raise _fail(
            layer, f"attention reads {reads}, not the normed tokens and the positions"
        )
    attn_hidden = next(k for k, src in attention_args if src == "normed")
    attn_positions = next(k for k, src in attention_args if src == positions_p)

    # Attention's own forward against the runner's composition.
    # The declared parts are opaque there, so a forward that calls another
    # module than the one declared computes another graph.
    config = conform_attention(
        attn,
        declared,
        declared.output_gate,
        attn_hidden,
        attn_positions,
        f" on {type(layer).__qualname__}",
    )
    parts = {
        field: attribute_of(attn, part, field)
        for field, part in (
            ("qkv_proj", declared.qkv_proj),
            ("attn", declared.attn),
            ("o_proj", declared.o_proj),
            ("rotary", declared.rotary),
            ("q_norm", declared.q_norm),
            ("k_norm", declared.k_norm),
        )
        if part is not None
    }

    roles = NormAttnRoles(
        input_norm=binding["input_norm"],
        attention=binding["attention"],
        qkv_proj=f"{binding['attention']}.{parts['qkv_proj']}",
        post_norm=binding["post_norm"],
        mlp=binding["mlp"],
        attention_args=attention_args,
        hidden_param=hidden_p,
        residual_param=residual_p,
        positions_param=positions_p,
        parts=tuple(sorted(parts.items())),
        config=config,
    )
    _conformed[cache_key] = roles
    return roles


def _missing(where: str, what: str) -> conf.BlockConformanceError:
    return conf.BlockConformanceError(
        f"declared {_PATTERN}{where}, but attention lacks a role property the "
        f"runner implements: {what}"
    )


def attention_config(
    parts: AttentionParts, output_gate: bool, where: str = ""
) -> AttnConfig:
    """The properties the runner's composition reads, from the declared parts.

    The parts are declared (the binding function); this reads the sizes the
    split needs from the QKV projection and checks the per-head norms' width.

    Raises:
        BlockConformanceError: A property the runner needs is missing, or
            present in a form it does not implement.
    """
    # The projection holds this rank's head counts; its query heads include
    # the declared output gate's copies.
    qkv = as_kind(parts.qkv_proj, QKVParallelLinear)
    head_dim = qkv.head_size
    for name, norm in (("q_norm", parts.q_norm), ("k_norm", parts.k_norm)):
        weight = None if norm is None else require(norm, "weight", name)
        if isinstance(weight, torch.Tensor) and weight.numel() != head_dim:
            raise _missing(
                where,
                f"{name} over {weight.numel()} values; the runner implements a "
                f"per-head norm over head_dim {head_dim}",
            )
    return AttnConfig(
        num_heads=qkv.num_heads // 2 if output_gate else qkv.num_heads,
        num_kv_heads=qkv.num_kv_heads,
        head_dim=head_dim,
        qk_norm=parts.q_norm is not None,
        output_gate=output_gate,
    )


def conform_attention(
    attn: nn.Module,
    parts: AttentionParts,
    output_gate: bool,
    hidden_param: str,
    positions_param: str,
    where: str = "",
) -> AttnConfig:
    """Checks that attention's forward is the runner's composition of its parts.

    With every part opaque, attention's own forward must export to the same
    graph as ``AttentionRunner.attention_from_qkv`` after the projection: the
    same split, q/k norms, RoPE, gate and output projection, and nothing else.

    Raises:
        BlockConformanceError: Naming the role, edge and source line.
    """
    config = attention_config(parts, output_gate, where)
    mod = {
        field: part
        for field, part in (
            ("qkv_proj", parts.qkv_proj),
            ("attn", parts.attn),
            ("o_proj", parts.o_proj),
            ("rotary", parts.rotary),
            ("q_norm", parts.q_norm),
            ("k_norm", parts.k_norm),
        )
        if part is not None
    }
    hidden = as_kind(parts.qkv_proj, LinearBase).input_size

    def projection(m: nn.Module, role: str):  # noqa: ANN202
        width = as_kind(m, LinearBase).output_size_per_partition

        def stub(x: torch.Tensor, *a: Any, **k: Any) -> Any:  # noqa: ANN401
            out = conf.ops.project(x, width, role)
            return (out, None) if require(m, "return_bias", role) else out

        return stub

    stubs = {
        "qkv_proj": projection(mod["qkv_proj"], "qkv_proj"),
        "o_proj": projection(mod["o_proj"], "o_proj"),
        "rotary": lambda positions, q, k, *a, **kw: conf.ops.pair(q, k, "rope"),
        "attn": lambda q, k, v, *a, **kw: conf.ops.binary(
            conf.ops.binary(q, k, "attn"), v, "attn_v"
        ),
    }
    stubs |= {
        n: (lambda role: lambda x: conf.ops.unary(x, role))(n)
        for n in ("q_norm", "k_norm")
        if n in mod
    }
    runner = AttentionRunner(
        "conformance",
        AttentionParts(
            input_norm=nn.Identity(),
            qkv_proj=mod["qkv_proj"],
            attn=mod["attn"],
            o_proj=mod["o_proj"],
            rotary=mod["rotary"],
            q_norm=mod.get("q_norm"),
            k_norm=mod.get("k_norm"),
        ),
        config=config,
    )
    meta = {"device": "meta", "dtype": torch.bfloat16}
    args = (
        torch.empty(_TOKENS, hidden, **meta),
        torch.empty(_TOKENS, device="meta", dtype=torch.long),
    )
    names = ("hidden", "positions")

    def model_fn(x: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        return attn(**{hidden_param: x, positions_param: p})

    def runner_fn(x: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        qkv, _ = runner.qkv_proj(x)
        return runner.attention_from_qkv(qkv, p)

    with contextlib.ExitStack() as stack:
        for part, stub in stubs.items():
            stack.enter_context(conf.opaque(mod[part], stub))
        got = conf.canonicalize(conf.export(model_fn, args, attn), names)
        want = conf.canonicalize(conf.export(runner_fn, args, None), names)
    if error := conf.compare(_PATTERN, got, want, f" (attention{where})"):
        raise conf.BlockConformanceError(error)
    return config


logger = init_logger(__name__)


class NormAttnDecoder(nn.Module):
    """Mixin placed before a declared decoder layer's class.

    Runs the sub-block through its ``AttentionRunner``, then the rest of the
    layer (the second norm and the MLP, opaque), which conformance has shown
    is all the layer computes. The installer sets ``_norm_attn_roles``,
    ``_norm_attn_stream`` and ``_norm_attn_runner``. Inside a declared block
    the block's runner owns the layer's forward instead.
    """

    _norm_attn_roles: NormAttnRoles
    _norm_attn_stream: StreamParams
    _norm_attn_runner: AttentionRunner

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ANN401
        r = self._norm_attn_roles
        hidden, residual, positions = self._norm_attn_stream.bind(args, kwargs)
        assert positions is not None  # conformance requires a positions parameter
        x, residual = self._norm_attn_runner(hidden, residual, positions)
        x, residual = self.get_submodule(r.post_norm)(x, residual)
        return self.get_submodule(r.mlp)(x), residual


_decoder_classes: dict[type, type] = {}


def _decoder_class(cls: type) -> type:
    if cls not in _decoder_classes:
        decoder = type(cls.__name__, (NormAttnDecoder, cls), {})
        decoder.__qualname__ = cls.__qualname__
        _decoder_classes[cls] = decoder
    return _decoder_classes[cls]


def build_runner(
    name: str,
    parts: AttentionParts,
    config: AttnConfig,
    input_proj: str,
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
        fused_op=fused_op if input_proj == "fused_qkv" else None,
        output_reduced=output_reduced,
    )


FUSED_OP: FusedOp = reference_fused_qkv
"""The fused value's implementation: the plain-torch reference, or a kernel."""

FUSED_OP_UNREDUCED = True
"""``FUSED_OP``'s capability: it honours ``runner.output_reduced=False``."""


def fused_op_for(
    name: str, layer: nn.Module, roles: NormAttnRoles
) -> tuple[FusedOp, bool, str]:
    """The fused value's op for this layer, its unreduced capability, and a note."""
    del name, layer, roles
    return FUSED_OP, FUSED_OP_UNREDUCED, ""


def resolve_runner(
    name: str,
    layer: nn.Module,
    declaration: BlockDeclaration,
    context: InstallContext,
    *,
    output_reduced: bool,
) -> tuple[NormAttnRoles, AttentionRunner, str]:
    """Conforms, chooses the plan and builds the runner for one declared layer.

    The parent states the runner's contract (``output_reduced``); the plan
    takes only values that can honour it.

    Returns:
        The roles, the runner, and what was chosen, for the install report.

    Raises:
        BlockConformanceError: The layer does not conform.
        ValueError: A pinned value cannot run on this layer.
    """
    roles = conform(layer, declaration)
    attention = layer.get_submodule(roles.attention)
    fused_op, fused_unreduced, via = fused_op_for(name, layer, roles)
    where = original_class(layer).__qualname__
    plan, pinned, declined = plan_mod.choose(
        context.subblock_plan,
        attention,
        chip=context.chip,
        kv_cache_dtype=context.kv_cache_dtype,
        where=where,
        output_gate=roles.config.output_gate,
        output_reduced=output_reduced,
        fused_unreduced=fused_unreduced,
    )
    for item in declined:  # The rule's value fell back: quiet.
        logger.debug("%s: %s", name, item)
    runner = build_runner(
        name,
        declared_parts(layer, declaration),
        roles.config,
        plan.input_proj,
        fused_op,
        output_reduced=output_reduced,
    )
    source = "user" if pinned else "rule"
    matmul = f" matmul={plan.matmul}" if plan.input_proj == "fused_qkv" else ""
    via = via if plan.input_proj == "fused_qkv" else ""
    reduced = "" if output_reduced else " output_reduced=False"
    return (
        roles,
        runner,
        f"input_proj={plan.input_proj}{matmul}{reduced} ({source}){via}",
    )


def install_standalone(
    name: str, layer: nn.Module, declaration: BlockDeclaration, context: InstallContext
) -> Installed:
    """Routes a declared layer with no enclosing block through its runner."""
    cls = type(layer)
    roles, runner, what = resolve_runner(
        name, layer, declaration, context, output_reduced=True
    )
    stream = StreamParams.of(
        layer, roles.hidden_param, roles.residual_param, roles.positions_param
    )
    object.__setattr__(layer, "_norm_attn_roles", roles)
    object.__setattr__(layer, "_norm_attn_stream", stream)
    object.__setattr__(layer, "_norm_attn_runner", runner)
    layer.__class__ = _decoder_class(cls)
    return Installed(name, cls.__qualname__, f"runner installed; {what}")


@block_implementation(norm_then_attention)
class NormThenAttention:
    """Conforms, resolves and installs one declared norm_then_attention layer.

    Inside a declared block that nests this pattern, the block's installer
    calls ``resolve_runner`` itself, with the block's contract.
    """

    nests: tuple[Any, ...] = ()

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
        nested: dict[Any, BlockDeclaration] | None = None,  # noqa: ARG002 - nests none
    ) -> Installed:
        if isinstance(layer, NormAttnDecoder):
            return Installed(name, type(layer).__qualname__, "already installed")
        return install_standalone(name, layer, declaration, context)
