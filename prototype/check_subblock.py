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

import sys

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.models.llama import LlamaDecoderLayer

from prototype import _build, subblock_reach
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import norm_attn
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.norm_attn import forward as fwd
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention

# Comment in to see CI fail: the planted layer becomes a declared model.
PLANT_MISMATCH = False
# PLANT_MISMATCH = True

TOLERANCE = 1e-5  # float32 on a CPU; the fused value only reorders one add.

FAMILIES = (
    _build.Spec(
        "Llama (dense)",
        _build._cfg("LlamaConfig"),  # noqa: SLF001 - the shared tiny sizes
        "vllm.model_executor.models.llama",
        "LlamaDecoderLayer",
    ),
    next(s for s in _build.SPECS if s.name == "Qwen3-MoE"),
    _build.SPECS[0],  # Qwen3.5-MoE, full attention
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
            m.forward = _core(4, 2, 16)
    layer.mlp.forward = lambda x, *a, **k: torch.tanh(x)
    return layer


def _context(plan: str = "") -> blocks.InstallContext:
    return blocks.InstallContext(
        tp=1,
        dtype="float32",
        buckets=(16,),
        override="",
        model_plan="",
        subblock_plan=plan,
    )


def family(spec: _build.Spec, declines: bool = False) -> list[str]:
    """Conforms, installs and compares; ``declines``: fused_qkv must decline."""
    blocks.load_declarations()
    cls = getattr(__import__(spec.module, fromlist=["x"]), spec.layer_class)
    declaration = blocks.DECLARATIONS.get((cls, norm_then_attention))
    if declaration is None:
        return [f"{spec.name}: not declared"]
    meta = _build.build(spec)
    with set_current_vllm_config(_build.current_config()):
        norm_attn.conform(meta, declaration)
    torch.set_default_dtype(torch.float32)
    layer = _runnable(spec)
    t = 6
    hidden, residual = torch.randn(t, 64), torch.randn(t, 64)
    call = {"positions": torch.arange(t), "hidden_states": hidden, "residual": residual}
    with set_current_vllm_config(_build.current_config()), torch.no_grad():
        want = layer(**call)
        before = fwd.FUSED_CALLS.get(cls.__qualname__, 0)
        impl = blocks.IMPLEMENTATIONS[norm_then_attention]
        item = impl.install("l", layer, declaration, _context())
        got = layer(**call)
    ran = fwd.FUSED_CALLS.get(cls.__qualname__, 0) - before
    diff = max((g - w).abs().max().item() for g, w in zip(got, want, strict=True))
    print(
        f"    {spec.name:30} declared -> conforms -> {item.outcome}; "
        f"fused calls {ran}; max |fused - upstream| {diff:.1e}"
    )
    torch.set_default_dtype(torch.bfloat16)
    if declines:
        if "declined" not in item.outcome or ran or diff:
            return [f"{spec.name}: fused_qkv was expected to decline"]
        return []
    if ran != 1 or diff > TOLERANCE:
        return [f"{spec.name}: fused calls {ran}, diff {diff:.1e}"]
    return []


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
    layer = _build.build(FAMILIES[0])
    layer.__class__ = ScaledAttention
    declaration = blocks.BlockDeclaration(
        ScaledAttention,
        norm_then_attention,
        {
            "in_norm": "input_layernorm",
            "attn": "self_attn",
            "post_norm": "post_attention_layernorm",
            "mlp": "mlp",
        },
        lambda _: True,
    )
    key = (ScaledAttention, norm_then_attention)
    blocks.DECLARATIONS[key] = declaration
    try:
        with set_current_vllm_config(_build.current_config()):
            blocks.install(nn.ModuleDict({"l": layer}), _context())
    except BlockConformanceError as e:
        print(f"    startup error: {e}")
        if PLANT_MISMATCH:  # CI: the planted layer counts as a declared model.
            return ["declared ScaledAttention (planted) does not conform"]
        return []
    finally:
        del blocks.DECLARATIONS[key]
    return ["the planted declaration did not stop startup"]


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
        "    here: 1 plan value (FusedNormProjection), 6 declarations, "
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
    print("1 each family: declared -> conforms -> fused_qkv, against upstream")
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
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
