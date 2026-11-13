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
"""The pre-norm attention sub-block: one fused value for three model families.

Runs on a CPU against vLLM v0.33.0; ``python -m prototype.check_subblock``
prints the checks and ``test_subblock_check.py`` asserts them.

For each family the model owner declares ``norm_then_attention`` (one
``declare_block`` call in ``models/vllm/<model>/registration.py``); the check
then confirms the declaration, installs the plan value ``fused_qkv`` (written
once, against the pattern's roles) and compares the layer's output with
upstream's on random weights. A planted layer with an extra multiply between
attention and the second norm is declined with the edge and line
(``subblock_fixtures.py``).
"""

import dataclasses
import importlib
import inspect
import sys
from collections.abc import Callable
from types import CodeType, FrameType
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 - torch's own alias
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.llama import LlamaAttention

from tests.blocks import _build, subblock_fixtures, subblock_reach
from tests.blocks._report import Check, Emit
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import installer
from vllm_torchtpu.blocks.binding import as_kind
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.context import InstallContext
from vllm_torchtpu.blocks.norm_attn import install as norm_attn
from vllm_torchtpu.blocks.norm_attn import plan as subblock_plan
from vllm_torchtpu.blocks.norm_attn import runner as runner_mod
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention
from vllm_torchtpu.blocks.stream import routed_runner

# Comment in to see CI fail: the planted layer becomes a declared model.
PLANT_MISMATCH = False
# PLANT_MISMATCH = True  # noqa: ERA001 - the comment-in switch

TOLERANCE = 1e-5  # float32 on a CPU; the fused value only reorders one add.

HIDDEN, HEAD_DIM = 128, 128  # head_dim 128: every condition of fused_qkv holds
_TOKENS = 6
_LAYER = "l"  # The layer's name in the install, which keys FUSED_CALLS.


def _sized(spec: _build.Spec) -> _build.Spec:
    """The spec with hidden 128 and head_dim 128 (4 query, 2 KV heads)."""
    make = spec.config

    def config() -> Any:  # noqa: ANN401 - the spec's config, whatever its class
        cfg = make()
        cfg.hidden_size, cfg.head_dim = HIDDEN, HEAD_DIM
        return cfg

    return dataclasses.replace(spec, config=config)


FAMILIES = tuple(
    _sized(s)
    for s in (
        _build.Spec(
            "Llama (dense)",
            _build._cfg("LlamaConfig"),  # noqa: SLF001 - the shared tiny sizes
            "vllm.model_executor.models.llama",
            "LlamaDecoderLayer",
        ),
        _build.SPECS[3],  # Qwen3-MoE
        _build.SPECS[0],  # Qwen3.5-MoE, full attention
    )
)


# Current models #3096 does not target, from their real configs (shrunk):
# each gets the fusion from one declare_block call in its registration.py.
NEW_MODELS = (
    subblock_reach.GLM_4_5_AIR,
    subblock_reach.MINISTRAL_8B,
    subblock_reach.GPT_OSS_20B,
    subblock_reach.QWEN3_32B,
)


def spec_of(c: subblock_reach.Candidate) -> _build.Spec:
    """The candidate's layer spec, from its shrunk real config."""
    cfg = subblock_reach.config_of(c)
    return _build.Spec(c.repo, lambda: cfg, c.module, c.layer_class, c.extra, c.model)


def _core(
    heads: int, kv_heads: int, head_dim: int
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Causal attention on q, k, v, standing in for the TPU attention backend."""

    def core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        t = q.shape[0]
        q = q.view(t, heads, head_dim).transpose(0, 1)
        k = (
            k.view(t, kv_heads, head_dim)
            .transpose(0, 1)
            .repeat_interleave(heads // kv_heads, 0)
        )
        v = (
            v.view(t, kv_heads, head_dim)
            .transpose(0, 1)
            .repeat_interleave(heads // kv_heads, 0)
        )
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(0, 1).reshape(t, heads * head_dim)

    return core


def _cpu_linear(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None
) -> torch.Tensor:
    return F.linear(x, w, b)


def _mlp(x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
    """The MLP stand-in, the same in every run."""
    del args, kwargs
    return torch.tanh(x)


def _runnable(spec: _build.Spec) -> nn.Module:
    """The family's layer on a CPU in float32 with random weights.

    The attention backend and the MLP (dense or MoE) need TPU or CUDA
    kernels; both are replaced by the same stand-ins in every run, so the
    comparison covers the norms, the residual, the projections, q/k norms,
    RoPE, gates and the output projection.
    """
    _ = torch.manual_seed(0)
    rotary_embedding._ROPE_DICT.clear()  # noqa: SLF001 - drop meta-device ropes
    layer = _build.build(spec, device="cpu")
    for p in layer.parameters():
        _ = nn.init.normal_(p, std=0.05)
    for m in layer.modules():
        if isinstance(m, LinearBase):
            m.cpu_linear = _cpu_linear
        if isinstance(m, AttentionLayerBase):
            m.forward = _core(4, 2, HEAD_DIM)
    layer.get_submodule("mlp").forward = _mlp
    return layer


def _context(plan: str = "") -> InstallContext:
    """The install context, with ``plan`` parsed as the runner parses the pin."""
    return InstallContext(
        tp=1,
        dtype=torch.float32,
        buckets=(16,),
        subblock_pins=subblock_plan.parse_pins(plan),
        chip=None,
    )


@dataclasses.dataclass(frozen=True)
class Run:
    """One install-and-run against upstream on a fresh CPU layer."""

    outcome: str
    input_proj: runner_mod.InputProj | None
    fused_runs: int
    max_diff: float


def _input_proj(layer: nn.Module) -> runner_mod.InputProj | None:
    """The installed runner's input projection; None if none was installed."""
    routed = routed_runner(layer)
    if isinstance(routed, norm_attn.NormAttnLayer):
        return routed.runner.input_proj
    return None


def _install_and_run(
    spec: _build.Spec, declaration: blocks.BlockDeclaration, plan: str
) -> Run:
    """Installs the runner on a fresh CPU layer and runs it against upstream."""
    torch.set_default_dtype(torch.float32)
    layer = _runnable(spec)
    positions = torch.arange(_TOKENS)
    hidden, residual = torch.randn(_TOKENS, HIDDEN), torch.randn(_TOKENS, HIDDEN)
    with set_current_vllm_config(_build.current_config()), torch.no_grad():
        want = layer(positions=positions, hidden_states=hidden, residual=residual)
        before = runner_mod.FUSED_CALLS.get(_LAYER, 0)
        impl = installer.IMPLEMENTATIONS[norm_then_attention]
        item = impl.install(_LAYER, layer, declaration, _context(plan))
        got = layer(positions=positions, hidden_states=hidden, residual=residual)
    ran = runner_mod.FUSED_CALLS.get(_LAYER, 0) - before
    diff = max((g - w).abs().max().item() for g, w in zip(got, want, strict=True))
    torch.set_default_dtype(torch.bfloat16)
    return Run(item.outcome, _input_proj(layer), ran, diff)


def _layer_class(spec: _build.Spec) -> type[nn.Module]:
    cls = getattr(importlib.import_module(spec.module), spec.layer_class)
    assert isinstance(cls, type)
    return cls


def family(spec: _build.Spec, emit: Emit, *, declines: bool = False) -> list[str]:
    """Conforms, then runs the runner both ways against upstream.

    A: the runner's unfused composition (pinned) against the model's own
    forward, identical. B: the rule's value (``fused_qkv`` where the
    capability allows) through the same runner. ``declines``: fused_qkv must
    fall back to unfused.
    """
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS.get((_layer_class(spec), norm_then_attention))
    if declaration is None:
        return [f"{spec.name}: not declared"]
    meta = _build.build(spec)
    with set_current_vllm_config(_build.current_config()):
        conformed = norm_attn.conform(meta, declaration)
    a = _install_and_run(spec, declaration, "input_proj=unfused")
    b = _install_and_run(spec, declaration, "")
    emit(
        f"    {spec.name:30} declared -> conforms ({_props(conformed.config)})\n"
        f"    {'':30}   A runner unfused vs upstream: max diff {a.max_diff:.1e}\n"
        f"    {'':30}   B {b.outcome}; fused runs {b.fused_runs}; "
        f"max diff {b.max_diff:.1e}"
    )
    failures: list[str] = []
    if a.fused_runs or a.max_diff:
        failures.append(
            f"{spec.name}: runner unfused differs from upstream ({a.max_diff:.1e})"
        )
    if declines:
        fell_back = b.input_proj is runner_mod.InputProj.UNFUSED
        if not fell_back or b.fused_runs or b.max_diff:
            failures.append(f"{spec.name}: fused_qkv was expected to fall back")
    elif b.fused_runs != 1 or b.max_diff > TOLERANCE:
        failures.append(
            f"{spec.name}: fused runs {b.fused_runs}, diff {b.max_diff:.1e}"
        )
    return failures


def _props(c: runner_mod.AttnConfig) -> str:
    gqa = f"{c.num_heads}q/{c.num_kv_heads}kv x {c.head_dim}"
    return ", ".join(
        [gqa]
        + (["per-head q/k norm"] if c.qk_norm else [])
        + (["output gate"] if c.output_gate else [])
    )


def families(emit: Emit) -> list[str]:
    """Check 1: each family's declaration, run A and run B."""
    failures: list[str] = []
    for spec in FAMILIES:
        try:
            failures += family(spec, emit)
        except BlockConformanceError as e:
            failures.append(f"{spec.name}: {e}")
    return failures


def new_models(emit: Emit) -> list[str]:
    """Check 2: current models #3096 does not target, from real configs."""
    failures: list[str] = []
    for c in NEW_MODELS:
        emit(
            f"    {c.family} ({c.repo}), not a #3096 target: declaration 1 call; "
            "per-model code 0"
        )
        failures += family(spec_of(c), emit, declines=c is subblock_reach.GPT_OSS_20B)
    return failures


def planted(emit: Emit) -> list[str]:
    """Check 3: a planted declaration that does not conform stops startup."""
    failures: list[str] = []
    for cls, where in (
        (subblock_fixtures.ScaledAttention, "the layer"),
        (subblock_fixtures.ScaledQueryAttention, "attention"),
    ):
        failures += _planted(cls, where, emit)
    return failures


def _planted(planted_cls: type[nn.Module], where: str, emit: Emit) -> list[str]:
    layer = _build.build(FAMILIES[0])
    layer.__class__ = planted_cls
    if planted_cls is subblock_fixtures.ScaledQueryAttention:
        subblock_fixtures.scale_query(
            as_kind(layer.get_submodule("self_attn"), LlamaAttention)
        )
    declaration = blocks.BlockDeclaration.of(
        planted_cls, subblock_fixtures.llama_binding
    )
    key = (planted_cls, norm_then_attention)
    blocks.DECLARATIONS[key] = declaration
    try:
        with set_current_vllm_config(_build.current_config()):
            installer.install(nn.ModuleDict({_LAYER: layer}), _context())
    except BlockConformanceError as e:
        emit(f"    planted in {where}: startup error: {e}")
        if PLANT_MISMATCH:  # CI: the planted layer counts as a declared model.
            return [f"declared {planted_cls.__name__} (planted) does not conform"]
        return []
    finally:
        del blocks.DECLARATIONS[key]
    return [f"the planted {planted_cls.__name__} declaration did not stop startup"]


def source_less(spec: _build.Spec) -> list[str]:
    """Code objects the installed forward runs whose source cannot be found.

    torch.compile's AOT cache (``torch/_dynamo/package.py``, ``add_code``)
    calls ``inspect.getsourcelines`` on every code object it traces; one
    without a source file (``exec``, ``compile``, a generated method) fails
    the compile on the TPU. This runs the same lookup on a CPU.
    """
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS[(_layer_class(spec), norm_then_attention)]
    torch.set_default_dtype(torch.float32)
    layer = _runnable(spec)
    positions = torch.arange(_TOKENS)
    hidden, residual = torch.randn(_TOKENS, HIDDEN), torch.randn(_TOKENS, HIDDEN)
    codes: set[CodeType] = set()

    def tracer(frame: FrameType, event: str, arg: object) -> None:
        del arg
        if event == "call":  # str-ok: sys.settrace's event name
            codes.add(frame.f_code)

    with set_current_vllm_config(_build.current_config()), torch.no_grad():
        impl = installer.IMPLEMENTATIONS[norm_then_attention]
        _ = impl.install(_LAYER, layer, declaration, _context())
        sys.settrace(tracer)
        try:
            layer(positions=positions, hidden_states=hidden, residual=residual)
            layer(positions=positions, hidden_states=hidden, residual=None)
        finally:
            sys.settrace(None)
    torch.set_default_dtype(torch.bfloat16)
    missing: list[str] = []
    for code in codes:
        if code.co_filename.startswith("<frozen "):
            continue  # Frozen stdlib (os.getenv); dynamo never traces into it.
        try:
            _ = inspect.getsourcelines(code)
        except (OSError, TypeError):
            missing.append(f"{code.co_filename}:{code.co_firstlineno} {code.co_name}")
    return sorted(missing)


def source_lookup(emit: Emit) -> list[str]:
    """Check 5: every code object the installed forward runs has source."""
    failures: list[str] = []
    for spec in (*FAMILIES, *(spec_of(c) for c in NEW_MODELS)):
        missing = source_less(spec)
        emit(f"    {spec.name:30} {len(missing)} code object(s) without source")
        failures += [f"{spec.name}: {m}" for m in missing]
    return failures


@dataclasses.dataclass(frozen=True)
class Counted:
    """One kind of per-model knowledge in #3096, and its sites."""

    what: str
    sites: tuple[str, ...]


# Counted from PR #3096's diff (vllm-project/vllm-torchtpu#3096, open), which
# wires a Pallas fused_qkv kernel into these layers through patches.
PR_3096 = (
    Counted(
        "per-model forwards",
        (
            "fused_attn_proj_patch.py: _patched_qwen3_moe_forward",
            "fused_attn_proj_patch.py: _patched_qwen3_next_forward "
            "(Qwen3-Next, Qwen3.5)",
            "fused_attn_proj_patch.py: _patched_gemma4_attention_forward",
            "qwen3_5_patch.py: _overlap_forward (#1492's), edited to call it",
        ),
    ),
    Counted(
        "structural eligibility checks (here: conformance)",
        (
            "layers/adapter/fused_attn_proj.py: unsupported norm",
            "layers/adapter/fused_attn_proj.py: sequence-parallel attention input",
            "layers/adapter/fused_attn_proj.py: layer_scale",
            "layers/adapter/fused_attn_proj.py: separate in_proj_qkv / in_proj_z (GDN)",
            "layers/adapter/fused_attn_proj.py: first layer, residual None (call time)",
        ),
    ),
    Counted(
        "kernel limits (here: the primitive's supports(), not this check)",
        (
            "quant method",
            "weight not stored (K, N)",
            "fp8 not per-channel",
            "not batched RPA",
            "head_dim 64 or DCP",
            "KV sharing or sinks",
            "in_proj_qkvz not fp8",
        ),
    ),
)


def counts(emit: Emit) -> list[str]:
    """Check 4: the same fusion as #3096, by count."""
    for counted in PR_3096:
        emit(f"    #3096 {counted.what}: {len(counted.sites)}")
    emit(
        "    here: 1 forward (AttentionRunner) with 1 fused branch, 7 declarations, "
        "1 conformance check (norm_attn.conform)"
    )
    return []


CHECKS = (
    Check(
        "1 each family: declared -> conforms -> AttentionRunner, against upstream",
        families,
    ),
    Check("2 current models #3096 does not target, real configs", new_models),
    Check("3 a declaration that does not conform stops startup", planted),
    Check("4 the same fusion as #3096, by count", counts),
    Check(
        "5 every code object the installed forward runs has source (for torch.compile)",
        source_lookup,
    ),
)
