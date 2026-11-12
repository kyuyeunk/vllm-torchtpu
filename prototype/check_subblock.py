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

Run on a CPU against vLLM v0.33.0::

    python -m prototype.check_subblock

For each family the model owner declares ``norm_then_attention`` (one
``declare_block`` call in ``models/vllm/<model>/registration.py``); the check
then confirms the declaration, installs the plan value ``fused_qkv`` (written
once, against the pattern's roles) and compares the layer's output with
upstream's on random weights. A planted layer with an extra multiply between
attention and the second norm is declined with the edge and line. Exits 1 if
anything is not as expected.
"""

import dataclasses
import inspect
import sys

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer

from prototype import _build, subblock_reach
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import norm_attn
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.norm_attn import runner as runner_mod
from vllm_torchtpu.blocks.norm_attn.pattern import (
    NormAttnParts,
    norm_then_attention,
    pre_norm_parts,
)
from vllm_torchtpu.blocks.parts import as_kind

# Comment in to see CI fail: the planted layer becomes a declared model.
PLANT_MISMATCH = False
# PLANT_MISMATCH = True

TOLERANCE = 1e-5  # float32 on a CPU; the fused value only reorders one add.

HIDDEN, HEAD_DIM = 128, 128  # head_dim 128: every condition of fused_qkv holds


def _sized(spec: _build.Spec) -> _build.Spec:
    """The spec with hidden 128 and head_dim 128 (4 query, 2 KV heads)."""
    make = spec.config

    def config():  # noqa: ANN202
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
        next(s for s in _build.SPECS if s.name == "Qwen3-MoE"),
        _build.SPECS[0],  # Qwen3.5-MoE, full attention
    )
)


# Current models #3096 does not target, from their real configs (shrunk):
# each gets the fusion from one declare_block call in its registration.py.
NEW_MODELS = tuple(
    c
    for c in subblock_reach.CANDIDATES
    if c.repo
    in (
        "openai/gpt-oss-20b",
        "mistralai/Ministral-8B-Instruct-2410",
        "Qwen/Qwen3-32B",
        "zai-org/GLM-4.5-Air",
    )
)


def _spec(c: subblock_reach.Candidate) -> _build.Spec:
    cfg = subblock_reach._config(c)  # noqa: SLF001 - the shrunk real config
    return _build.Spec(c.repo, lambda: cfg, c.module, c.layer_class, c.extra, c.model)


class ScaledAttention(LlamaDecoderLayer):
    """Planted: scales attention's output before the second norm."""

    def forward(self, positions, hidden_states, residual):  # noqa: ANN001, ANN201, D102
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = hidden_states * 2.0
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class ScaledQueryAttention(LlamaDecoderLayer):
    """Planted: its attention scales q after RoPE (see ``_scale_query``)."""


def _scale_query(attn: nn.Module) -> None:
    """Gives ``attn`` a forward the runner does not compute: q * 0.5 after RoPE."""

    def forward(positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = attn.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        q, k = attn.rotary_emb(positions, q, k)
        q = q * 0.5
        out, _ = attn.o_proj(attn.attn(q, k, v))
        return out

    attn.forward = forward


def _core(heads: int, kv_heads: int, head_dim: int):  # noqa: ANN202
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


def _runnable(spec: _build.Spec) -> nn.Module:
    """The family's layer on a CPU in float32 with random weights.

    The attention backend and the MLP (dense or MoE) need TPU or CUDA
    kernels; both are replaced by the same stand-ins in every run, so the
    comparison covers the norms, the residual, the projections, q/k norms,
    RoPE, gates and the output projection.
    """
    torch.manual_seed(0)
    rotary_embedding._ROPE_DICT.clear()  # noqa: SLF001 - drop meta-device ropes
    layer = _build.build(spec, device="cpu")
    for p in layer.parameters():
        nn.init.normal_(p, std=0.05)
    for m in layer.modules():
        if isinstance(m, LinearBase):
            m.cpu_linear = lambda x, w, b: F.linear(x, w, b)
        if isinstance(m, AttentionLayerBase):
            m.forward = _core(4, 2, HEAD_DIM)
    layer.mlp.forward = lambda x, *a, **k: torch.tanh(x)
    return layer


def _context(plan: str = "", chip: str = "") -> blocks.InstallContext:
    return blocks.InstallContext(
        tp=1,
        dtype="float32",
        buckets=(16,),
        override="",
        model_plan="",
        subblock_plan=plan,
        chip=chip,
    )


def _install_and_run(
    spec: _build.Spec, declaration: blocks.BlockDeclaration, plan: str
) -> tuple[str, int, float]:
    """Installs the runner on a fresh CPU layer; (outcome, fused runs, max diff)."""
    torch.set_default_dtype(torch.float32)
    layer = _runnable(spec)
    t = 6
    hidden, residual = torch.randn(t, HIDDEN), torch.randn(t, HIDDEN)
    call = {"positions": torch.arange(t), "hidden_states": hidden, "residual": residual}
    with set_current_vllm_config(_build.current_config()), torch.no_grad():
        want = layer(**call)
        before = runner_mod.FUSED_CALLS.get("l", 0)
        impl = blocks.IMPLEMENTATIONS[norm_then_attention]
        item = impl.install("l", layer, declaration, _context(plan))
        got = layer(**call)
    ran = runner_mod.FUSED_CALLS.get("l", 0) - before
    diff = max((g - w).abs().max().item() for g, w in zip(got, want, strict=True))
    torch.set_default_dtype(torch.bfloat16)
    return item.outcome, ran, diff


def family(spec: _build.Spec, declines: bool = False) -> list[str]:
    """Conforms, then runs the runner both ways against upstream.

    A: the runner's unfused composition (pinned) against the model's own
    forward, identical. B: the rule's value (``fused_qkv`` where the
    capability allows) through the same runner. ``declines``: fused_qkv must
    fall back to unfused.
    """
    blocks.load_declarations()
    cls = getattr(__import__(spec.module, fromlist=["x"]), spec.layer_class)
    declaration = blocks.DECLARATIONS.get((cls, norm_then_attention))
    if declaration is None:
        return [f"{spec.name}: not declared"]
    meta = _build.build(spec)
    with set_current_vllm_config(_build.current_config()):
        roles = norm_attn.conform(meta, declaration)
    _, ran_a, diff_a = _install_and_run(spec, declaration, "input_proj=unfused")
    outcome, ran, diff = _install_and_run(spec, declaration, "")
    print(
        f"    {spec.name:30} declared -> conforms ({_props(roles.config)})\n"
        f"    {'':30}   A runner unfused vs upstream: max diff {diff_a:.1e}\n"
        f"    {'':30}   B {outcome}; fused runs {ran}; max diff {diff:.1e}"
    )
    failures = []
    if ran_a or diff_a:
        failures.append(
            f"{spec.name}: runner unfused differs from upstream ({diff_a:.1e})"
        )
    if declines:
        if "input_proj=unfused" not in outcome or ran or diff:
            failures.append(f"{spec.name}: fused_qkv was expected to fall back")
    elif ran != 1 or diff > TOLERANCE:
        failures.append(f"{spec.name}: fused runs {ran}, diff {diff:.1e}")
    return failures


def _props(c: runner_mod.AttnConfig) -> str:
    gqa = f"{c.num_heads}q/{c.num_kv_heads}kv x {c.head_dim}"
    return ", ".join(
        [gqa]
        + (["per-head q/k norm"] if c.qk_norm else [])
        + (["output gate"] if c.output_gate else [])
    )


def new_models() -> list[str]:
    failures = []
    for c in NEW_MODELS:
        print(
            f"    {c.family} ({c.repo}), not a #3096 target: declaration 1 call; "
            "per-model code 0"
        )
        failures += family(_spec(c), declines=c.repo == "openai/gpt-oss-20b")
    return failures


def planted() -> list[str]:
    failures = []
    for cls, where in (
        (ScaledAttention, "the layer"),
        (ScaledQueryAttention, "attention"),
    ):
        failures += _planted(cls, where)
    return failures


def _llama_parts(layer: LlamaDecoderLayer) -> NormAttnParts:
    """The planted layers' binding function (they reuse Llama's children)."""
    return pre_norm_parts(
        layer, as_kind(layer.self_attn, LlamaAttention), output_gate=False
    )


def _planted(planted_cls: type, where: str) -> list[str]:
    layer = _build.build(FAMILIES[0])
    layer.__class__ = planted_cls
    if planted_cls is ScaledQueryAttention:
        _scale_query(layer.self_attn)
    declaration = blocks.BlockDeclaration.of(planted_cls, _llama_parts)
    key = (planted_cls, norm_then_attention)
    blocks.DECLARATIONS[key] = declaration
    try:
        with set_current_vllm_config(_build.current_config()):
            blocks.install(nn.ModuleDict({"l": layer}), _context())
    except BlockConformanceError as e:
        print(f"    planted in {where}: startup error: {e}")
        if PLANT_MISMATCH:  # CI: the planted layer counts as a declared model.
            return [f"declared {planted_cls.__name__} (planted) does not conform"]
        return []
    finally:
        del blocks.DECLARATIONS[key]
    return [f"the planted {planted_cls.__name__} declaration did not stop startup"]


def _source_less(spec: _build.Spec) -> list[str]:
    """Code objects the installed forward runs whose source cannot be found.

    torch.compile's AOT cache (``torch/_dynamo/package.py``, ``add_code``)
    calls ``inspect.getsourcelines`` on every code object it traces; one
    without a source file (``exec``, ``compile``, a generated method) fails
    the compile on the TPU. This runs the same lookup on a CPU.
    """
    blocks.load_declarations()
    cls = getattr(__import__(spec.module, fromlist=["x"]), spec.layer_class)
    declaration = blocks.DECLARATIONS[(cls, norm_then_attention)]
    torch.set_default_dtype(torch.float32)
    layer = _runnable(spec)
    t = 6
    call = {
        "positions": torch.arange(t),
        "hidden_states": torch.randn(t, HIDDEN),
        "residual": torch.randn(t, HIDDEN),
    }
    codes: set = set()

    def tracer(frame, event, arg):  # noqa: ANN001, ANN202
        if event == "call":
            codes.add(frame.f_code)
        return None

    with set_current_vllm_config(_build.current_config()), torch.no_grad():
        impl = blocks.IMPLEMENTATIONS[norm_then_attention]
        impl.install("l", layer, declaration, _context())
        sys.settrace(tracer)
        try:
            layer(**call)
            layer(**{**call, "residual": None})  # the first layer
        finally:
            sys.settrace(None)
    torch.set_default_dtype(torch.bfloat16)
    missing = []
    for code in codes:
        if code.co_filename.startswith("<frozen "):
            continue  # Frozen stdlib (os.getenv); dynamo never traces into it.
        try:
            inspect.getsourcelines(code)
        except (OSError, TypeError):
            missing.append(f"{code.co_filename}:{code.co_firstlineno} {code.co_name}")
    return sorted(missing)


def source_lookup() -> list[str]:
    failures = []
    for spec in (*FAMILIES, *(_spec(c) for c in NEW_MODELS)):
        missing = _source_less(spec)
        print(f"    {spec.name:30} {len(missing)} code object(s) without source")
        failures += [f"{spec.name}: {m}" for m in missing]
    return failures


# Counted from PR #3096's diff (vllm-project/vllm-torchtpu#3096, open), which
# wires a Pallas fused_qkv kernel into these layers through patches.
PR_3096 = {
    "per-model forwards": [
        "fused_attn_proj_patch.py: _patched_qwen3_moe_forward",
        "fused_attn_proj_patch.py: _patched_qwen3_next_forward (Qwen3-Next, Qwen3.5)",
        "fused_attn_proj_patch.py: _patched_gemma4_attention_forward",
        "qwen3_5_patch.py: _overlap_forward (#1492's), edited to call it",
    ],
    "structural eligibility checks (here: conformance)": [
        "layers/adapter/fused_attn_proj.py: unsupported norm",
        "layers/adapter/fused_attn_proj.py: sequence-parallel attention input",
        "layers/adapter/fused_attn_proj.py: layer_scale",
        "layers/adapter/fused_attn_proj.py: separate in_proj_qkv / in_proj_z (GDN)",
        "layers/adapter/fused_attn_proj.py: first layer, residual None (call time)",
    ],
    "kernel limits (here: the primitive's supports(), not this check)": [
        "quant method",
        "weight not stored (K, N)",
        "fp8 not per-channel",
        "not batched RPA",
        "head_dim 64 or DCP",
        "KV sharing or sinks",
        "in_proj_qkvz not fp8",
    ],
}


def counts() -> list[str]:
    for what, items in PR_3096.items():
        print(f"    #3096 {what}: {len(items)}")
    print(
        "    here: 1 forward (AttentionRunner) with 1 fused branch, 7 declarations, "
        "1 conformance check (norm_attn.conform)"
    )
    return []


def _run(check, title: str) -> bool:  # noqa: ANN001
    print(title)
    failures = check()
    for failure in failures:
        print(f"    FAIL {failure}")
    print("  -> " + ("FAIL" if failures else "PASS"))
    return not failures


def main() -> None:
    ok = True
    print("1 each family: declared -> conforms -> AttentionRunner, against upstream")
    failures = []
    for spec in FAMILIES:
        try:
            failures += family(spec)
        except BlockConformanceError as e:
            failures.append(f"{spec.name}: {e}")
    for failure in failures:
        print(f"    FAIL {failure}")
    print("  -> " + ("FAIL" if failures else "PASS"))
    ok &= not failures
    ok &= _run(new_models, "2 current models #3096 does not target, real configs")
    ok &= _run(planted, "3 a declaration that does not conform stops startup")
    ok &= _run(counts, "4 the same fusion as #3096, by count")
    ok &= _run(
        source_lookup,
        "5 every code object the installed forward runs has source (for torch.compile)",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
