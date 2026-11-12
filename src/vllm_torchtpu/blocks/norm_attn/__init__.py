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
declared layer against it; ``plan`` picks whether the input norm joins
attention's projection; ``forward`` runs the fused value through the roles.
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
)
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.norm_attn import plan as plan_mod
from vllm_torchtpu.blocks.norm_attn.forward import NormThenAttentionBlock
from vllm_torchtpu.blocks.norm_attn.pattern import (
    NormAttnRoles,
    check_roles,
    norm_then_attention,
    qkv_proj_of,
)

_PATTERN = "norm_then_attention"
_TOKENS = 8
_conformed: dict[tuple[object, ...], NormAttnRoles] = {}


class _Handoff(Exception):  # noqa: N818 - control flow, not an error
    pass


def _fail(layer: nn.Module, why: str) -> conf.BlockConformanceError:
    return conf.BlockConformanceError(
        f"declared {_PATTERN} on {type(layer).__qualname__}, but {why}"
    )


def _stream_params(layer: nn.Module) -> tuple[str, str, str]:
    params = inspect.signature(layer.forward).parameters
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


def conform(layer: nn.Module, declaration: BlockDeclaration) -> NormAttnRoles:
    """Checks a declared layer against norm_then_attention; returns its roles.

    Three checks, all on the meta device: the roles are the right kinds; with
    the roles opaque, the layer's forward computes the pattern's graph; and
    attention hands its input straight to its QKV projection, so the fused
    value can project for it.

    Raises:
        BlockConformanceError: Naming the role, edge and source line.
    """
    binding = declaration.bind(layer)
    flags = tuple(
        sorted((k, v) for k, v in vars(layer).items() if isinstance(v, (bool, str)))
    )
    cache_key = (type(layer), tuple(sorted(binding.items())), flags)
    if cache_key in _conformed:
        return _conformed[cache_key]
    if problems := check_roles(layer, binding):
        raise _fail(layer, "; ".join(problems))
    hidden_p, residual_p, positions_p = _stream_params(layer)
    hidden = int(layer.get_submodule(binding["in_norm"]).weight.shape[0])
    attn = layer.get_submodule(binding["attn"])
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

    def layer_fn(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> Any:  # noqa: ANN401
        seen["inputs"] = {hidden_p: h, residual_p: r, positions_p: p}
        call = {hidden_p: h, residual_p: r} | ({positions_p: p} if positions_p else {})
        return layer(**call)

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
        binding["in_norm"]: recording_in_norm,
        binding["post_norm"]: _norm_stub("post_norm"),
        binding["attn"]: attn_stub,
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

    # The handoff: attention's first use of its input is its QKV projection.
    (qkv_name,) = qkv_proj_of(attn, hidden)
    probe = torch.empty(_TOKENS, hidden, **meta)
    by_source = {"normed": probe, hidden_p: probe, residual_p: args[1]}
    by_source[positions_p] = args[2]
    call = {k: by_source[src] for k, src in attention_args}

    def projection(t: torch.Tensor) -> Any:  # noqa: ANN401
        raise _Handoff(t is probe)

    try:
        with conf.opaque(attn.get_submodule(qkv_name), projection):
            attn(**call)
    except _Handoff as handed:
        if not handed.args[0]:
            raise _fail(
                layer, f"attention changes its input before {qkv_name}"
            ) from None
    else:
        raise _fail(layer, f"attention never calls {qkv_name}")

    roles = NormAttnRoles(
        input_norm=binding["in_norm"],
        attention=binding["attn"],
        qkv_proj=f"{binding['attn']}.{qkv_name}",
        post_norm=binding["post_norm"],
        mlp=binding["mlp"],
        attention_args=attention_args,
        hidden_param=hidden_p,
        residual_param=residual_p,
        positions_param=positions_p,
    )
    _conformed[cache_key] = roles
    return roles


_block_classes: dict[type, type] = {}


def _block_class(cls: type) -> type:
    if cls not in _block_classes:
        block = type(cls.__name__, (NormThenAttentionBlock, cls), {})
        block.__qualname__ = cls.__qualname__
        _block_classes[cls] = block
    return _block_classes[cls]


@block_implementation(norm_then_attention)
class NormThenAttention:
    """Conforms, resolves and installs one declared norm_then_attention layer."""

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
    ) -> Installed:
        roles = conform(layer, declaration)
        plan, source = plan_mod.resolve(context.subblock_plan)
        cls = type(layer).__qualname__
        if plan.input_proj == "unfused":
            return Installed(name, cls, f"conforms; input_proj=unfused ({source})")
        if type(layer) in _block_classes.values() or hasattr(layer, "_block_roles"):
            # Nesting under an installed attn_then_moe block forward is not
            # shown here: that forward would call this one's front half.
            return Installed(name, cls, "conforms; outer block owns the forward")
        order = tuple(inspect.signature(layer.forward).parameters)
        layer.__class__ = _block_class(type(layer))
        layer._subblock_roles = roles
        layer._subblock_order = order
        return Installed(name, cls, f"installed; input_proj=fused_qkv ({source})")
