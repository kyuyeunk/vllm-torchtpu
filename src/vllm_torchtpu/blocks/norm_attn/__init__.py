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
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.blocks import (
    BlockDeclaration,
    InstallContext,
    Installed,
    block_implementation,
)
from vllm_torchtpu.blocks import conformance as conf
from vllm_torchtpu.blocks.norm_attn import plan as plan_mod
from vllm_torchtpu.blocks.norm_attn.forward import (
    AddOnly,
    FusedNormProjection,
    install_fused_qkv,
)
from vllm_torchtpu.blocks.norm_attn.pattern import (
    NormAttnRoles,
    check_roles,
    norm_then_attention,
    qkv_proj_of,
)

_PATTERN = "norm_then_attention"
_TOKENS = 8
_conformed: dict[tuple[object, ...], NormAttnRoles] = {}


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
    the roles opaque, the layer's forward computes the pattern's graph (so the
    normed tokens reach nothing but attention); and attention's input reaches
    nothing but its QKV projection, so the fused value can normalize there.

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
    # The attention probe below runs attention's own children (RoPE, q/k
    # norms), so its tensors live where attention's weights do.
    param = next(attn.parameters(), None)
    live = {
        "device": param.device if param is not None else "meta",
        "dtype": torch.bfloat16,
    }
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

    # Attention's input reaches nothing but its QKV projection: run attention
    # on the meta device with the projection and the attention core opaque,
    # and record every other op that touches the input.
    (qkv_name,) = qkv_proj_of(attn, hidden)
    proj = attn.get_submodule(qkv_name)
    probe = torch.empty(_TOKENS, hidden, **live)
    by_source = {"normed": probe, hidden_p: probe, residual_p: torch.empty_like(probe)}
    by_source[positions_p] = torch.zeros(
        _TOKENS, device=live["device"], dtype=torch.long
    )
    call = {k: by_source[src] for k, src in attention_args}
    width = int(proj.output_size_per_partition)
    used: list[str] = []
    projected: list[bool] = []

    def projection(t: torch.Tensor) -> Any:  # noqa: ANN401
        projected.append(t is probe)
        out = torch.empty(_TOKENS, width, **live)
        return (out, None) if getattr(proj, "return_bias", True) else out

    def core(q: torch.Tensor, *a: Any, **k: Any) -> torch.Tensor:  # noqa: ANN401
        return torch.empty_like(q)

    class _Uses(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, a=(), k=None):  # noqa: ANN001, ANN204
            k = k or {}
            if any(x is probe for x in (*a, *k.values())):
                used.append(getattr(func, "__name__", str(func)))
            return func(*a, **k)

    cores = [m for m in attn.modules() if isinstance(m, conf_core_types())]
    linears = [m for m in attn.modules() if isinstance(m, LinearBase) and m is not proj]

    def linear_stub(m: nn.Module):  # noqa: ANN202
        def stub(x: torch.Tensor, *a: Any, **k: Any) -> Any:  # noqa: ANN401
            out = x.new_empty((*x.shape[:-1], int(m.output_size_per_partition)))
            return (out, None) if getattr(m, "return_bias", True) else out

        return stub

    with contextlib.ExitStack() as stack:
        stack.enter_context(conf.opaque(proj, projection))
        for m in cores:
            stack.enter_context(conf.opaque(m, core))
        for m in linears:
            stack.enter_context(conf.opaque(m, linear_stub(m)))
        stack.enter_context(_Uses())
        try:
            attn(**call)
        except Exception as e:  # noqa: BLE001 - a forward that will not run here
            raise _fail(layer, f"attention does not run on a probe: {e}") from e
    if projected != [True]:
        raise _fail(layer, f"attention does not pass its input to {qkv_name} once")
    if used:
        raise _fail(layer, f"attention also reads its input in {sorted(set(used))}")

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


def conf_core_types() -> tuple[type, ...]:
    from vllm.model_executor.layers.attention_layer_base import (  # noqa: PLC0415
        AttentionLayerBase,
    )

    return (AttentionLayerBase,)


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
        if plan.input_proj == "fused_qkv" and (
            why := plan_mod.fused_qkv_supports(layer.get_submodule(roles.attention))
        ):
            if source == "user":  # A pin that cannot run raises at startup.
                raise ValueError(
                    f"TPU_SUBBLOCK_PLAN input_proj=fused_qkv on {cls}: {why}"
                )
            return Installed(
                name, cls, f"conforms; input_proj=unfused (fused_qkv declined: {why})"
            )
        if plan.input_proj == "unfused":
            return Installed(name, cls, f"conforms; input_proj=unfused ({source})")
        if isinstance(layer.get_submodule(roles.input_norm), AddOnly):
            return Installed(name, cls, "already installed")
        install_fused_qkv(layer, roles.input_norm, roles.qkv_proj)
        assert isinstance(layer.get_submodule(roles.qkv_proj), FusedNormProjection)
        return Installed(name, cls, f"installed; input_proj=fused_qkv ({source})")
