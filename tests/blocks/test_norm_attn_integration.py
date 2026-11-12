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
"""Integration tests: each declared model registers with the runner.

Per declared family: the declaration conforms (or a layer that does not
fails with the edge and line, or the missing role property), the runner is
built from the model's own part objects, and the decoder routes through it.
Layers are built on the meta device from tiny configs. Execution numerics
are ``test_attention_runner.py``'s job (model-free, every role
configuration); here one smoke check runs Llama against upstream.
"""

from typing import Any

import pytest
import torch
from torch import nn
from vllm.config.vllm import set_current_vllm_config

from prototype import _build, check_subblock
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import norm_attn
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnRoles, norm_then_attention
from vllm_torchtpu.blocks.norm_attn.runner import AttentionRunner

SPECS = (
    *check_subblock.FAMILIES,
    *(check_subblock._spec(c) for c in check_subblock.NEW_MODELS),  # noqa: SLF001 - the shrunk real configs
)
GPT_OSS = next(s for s in SPECS if s.name == "openai/gpt-oss-20b")


def _declared(spec: _build.Spec) -> tuple[nn.Module, blocks.BlockDeclaration]:
    """A meta layer of the family, and its model's declaration."""
    layer = _build.build(spec)
    blocks.load_declarations()
    return layer, blocks.DECLARATIONS[(type(layer), norm_then_attention)]


def _runner_of(layer: nn.Module) -> AttentionRunner:
    runner = layer.__dict__["_norm_attn_runner"]
    assert isinstance(runner, AttentionRunner)
    return runner


def _roles_of(layer: nn.Module) -> NormAttnRoles:
    roles = layer.__dict__["_norm_attn_roles"]
    assert isinstance(roles, NormAttnRoles)
    return roles


def _weight(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    assert isinstance(weight, torch.Tensor)
    return weight


def _install(
    layer: nn.Module, declaration: blocks.BlockDeclaration, **context: str
) -> str:
    impl = blocks.IMPLEMENTATIONS[norm_then_attention]
    with set_current_vllm_config(_build.current_config()):
        return impl.install("l", layer, declaration, _context(**context)).outcome


def _context(plan: str = "", chip: str = "", kv: str = "") -> blocks.InstallContext:
    return blocks.InstallContext(
        tp=1,
        dtype="bfloat16",
        buckets=(16,),
        override="",
        model_plan="",
        subblock_plan=plan,
        chip=chip,
        kv_cache_dtype=kv,
    )


# The declaration conforms, or names what does not.


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_declaration_conforms(spec: _build.Spec) -> None:
    layer, declaration = _declared(spec)
    assert declaration.when(layer)
    with set_current_vllm_config(_build.current_config()):
        roles = norm_attn.conform(layer, declaration)
    attention = layer.get_submodule(roles.attention)
    assert roles.qkv_proj == f"{roles.attention}.qkv_proj"
    assert roles.config.head_dim == attention.head_dim
    assert roles.config.qk_norm == (getattr(attention, "q_norm", None) is not None)


@pytest.mark.parametrize(
    ("planted", "edge"),
    [
        (check_subblock.ScaledAttention, "aten.mul between attn and post_norm"),
        (check_subblock.ScaledQueryAttention, "aten.mul between rope and attn"),
    ],
    ids=["in_the_layer", "in_attention"],
)
def test_nonconforming_layer_names_edge_and_line(planted: type, edge: str) -> None:
    """A declared layer computing something else stops startup at its line."""
    layer = _build.build(check_subblock.FAMILIES[0])
    layer.__class__ = planted
    if planted is check_subblock.ScaledQueryAttention:
        check_subblock._scale_query(layer.get_submodule("self_attn"))  # noqa: SLF001 - the planted forward
    declaration = blocks.BlockDeclaration(
        planted,
        norm_then_attention,
        {
            "in_norm": "input_layernorm",
            "attn": "self_attn",
            "post_norm": "post_attention_layernorm",
            "mlp": "mlp",
        },
        lambda _: True,
    )
    with (
        set_current_vllm_config(_build.current_config()),
        pytest.raises(
            BlockConformanceError, match=rf"{edge} at check_subblock\.py:\d+"
        ),
    ):
        norm_attn.conform(layer, declaration)


def _without_rotary(attention: nn.Module) -> None:
    del attention.rotary_emb


def _lone_q_norm(attention: nn.Module) -> None:
    attention.q_norm = nn.Identity()


@pytest.mark.parametrize(
    ("edit", "missing"),
    [(_without_rotary, "rotary_emb"), (_lone_q_norm, "q_norm without its pair")],
    ids=["no_rotary", "q_norm_alone"],
)
def test_missing_role_property_is_named(
    edit: Any,  # noqa: ANN401 - a test case's edit
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(norm_attn, "_conformed", {})  # Conform this layer afresh.
    layer, declaration = _declared(check_subblock.FAMILIES[0])
    edit(layer.get_submodule("self_attn"))
    with (
        set_current_vllm_config(_build.current_config()),
        pytest.raises(BlockConformanceError, match=f"runner implements: {missing}"),
    ):
        norm_attn.conform(layer, declaration)


# The runner holds the model's own parts, and the decoder routes through it.


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_runner_holds_the_models_parts(spec: _build.Spec) -> None:
    layer, declaration = _declared(spec)
    state = dict(layer.state_dict(keep_vars=True))
    _install(layer, declaration)
    runner, roles = _runner_of(layer), _roles_of(layer)
    attention = layer.get_submodule(roles.attention)
    parts = dict(roles.parts)
    assert runner.input_norm is layer.get_submodule(roles.input_norm)
    assert runner.qkv_proj is attention.qkv_proj
    assert runner.o_proj is attention.o_proj
    assert runner.rotary is attention.rotary_emb
    assert runner.attn is attention.get_submodule(parts["attn"])
    assert runner.q_norm is getattr(attention, "q_norm", None)
    assert runner.k_norm is getattr(attention, "k_norm", None)
    # No copies, no new parameters: the model's state_dict is the same tensors.
    after = layer.state_dict(keep_vars=True)
    assert after.keys() == state.keys()
    assert all(after[k] is v for k, v in state.items())
    assert list(runner.parameters()) == []


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_decoder_routes_through_the_runner(
    spec: _build.Spec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed forward is runner, then the layer's post norm and MLP."""
    layer, declaration = _declared(spec)
    _install(layer, declaration)
    assert isinstance(layer, norm_attn.NormAttnDecoder)
    roles = _roles_of(layer)
    calls: list[tuple[torch.Tensor, ...]] = []
    meta = {"device": "meta", "dtype": torch.bfloat16}
    hidden_size = int(_weight(layer.get_submodule(roles.input_norm)).shape[0])
    attn_out, summed, normed = (torch.empty(4, hidden_size, **meta) for _ in range(3))

    def runner(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> tuple:
        calls.append((h, r, p))
        return attn_out, summed

    def post_norm(x: torch.Tensor, r: torch.Tensor) -> tuple:
        assert x is attn_out
        assert r is summed
        return normed, r

    monkeypatch.setattr(_runner_of(layer), "forward", runner)
    monkeypatch.setattr(layer.get_submodule(roles.post_norm), "forward", post_norm)
    monkeypatch.setattr(layer.get_submodule(roles.mlp), "forward", torch.neg)
    hidden, residual = (
        torch.empty(4, hidden_size, **meta),
        torch.empty(4, hidden_size, **meta),
    )
    positions = torch.empty(4, device="meta", dtype=torch.long)
    call = {
        roles.hidden_param: hidden,
        roles.residual_param: residual,
        roles.positions_param: positions,
    }
    out, out_residual = layer(**call)
    assert len(calls) == 1
    assert all(
        a is b for a, b in zip(calls[0], (hidden, residual, positions), strict=True)
    )
    assert out.shape == normed.shape
    assert out_residual is summed


def test_scopes_nest_on_qwen3_5() -> None:
    """Qwen3.5 declares both scopes; the block runs its front half in the runner."""
    layer = _build.build(check_subblock.FAMILIES[2])  # Qwen3.5-MoE, full attention
    context = blocks.InstallContext(
        tp=4,
        dtype="bfloat16",
        buckets=(16, 8192),
        override="collectives=reduce_scatter_all_gather",
        model_plan="",
    )
    with set_current_vllm_config(_build.current_config()):
        report = blocks.install(torch.nn.ModuleDict({"l": layer}), context)
    outcomes = [item.outcome for item in report]
    assert len(outcomes) == 2, outcomes
    assert outcomes[1].startswith("runner inside block"), outcomes
    runner = _runner_of(layer)
    assert runner.qkv_proj is layer.get_submodule("self_attn.qkv_proj")
    assert runner.input_proj == "fused_qkv"


# The plan reads the platform from the install context.


def test_capability_mismatch_falls_back_unless_pinned() -> None:
    """gpt-oss conforms, but fused_qkv cannot serve sinks: the rule falls back."""
    layer, declaration = _declared(GPT_OSS)
    assert "input_proj=unfused (rule)" in _install(layer, declaration)


def test_pin_that_cannot_run_stops_startup() -> None:
    """A user pin to fused_qkv on gpt-oss (sinks) is a startup error."""
    layer, declaration = _declared(GPT_OSS)
    with pytest.raises(ValueError, match="sinks"):
        _install(layer, declaration, plan="input_proj=fused_qkv")


def _fp8_llama() -> tuple[nn.Module, blocks.BlockDeclaration]:
    """A meta Llama layer whose qkv_proj holds FP8 weights, and its declaration."""
    layer, declaration = _declared(check_subblock.FAMILIES[0])
    qkv = layer.get_submodule("self_attn.qkv_proj")
    fp8 = _weight(qkv).to(torch.float8_e4m3fn)
    qkv.weight = nn.Parameter(fp8, requires_grad=False)
    return layer, declaration


@pytest.mark.parametrize(("chip", "matmul"), [("v6e", "bf16"), ("7x", "fp8")])
def test_fp8_variant_follows_the_chip(chip: str, matmul: str) -> None:
    """FP8 weights take the fp8 variant only where the chip has FP8 matmul."""
    layer, declaration = _fp8_llama()
    outcome = _install(layer, declaration, chip=chip)
    assert f"input_proj=fused_qkv matmul={matmul}" in outcome


def test_fp8_pin_on_a_chip_without_fp8_matmul_stops_startup() -> None:
    """A user pin to matmul=fp8 on v6e is a startup error."""
    layer, declaration = _fp8_llama()
    with pytest.raises(ValueError, match="chip v6e has none"):
        _install(layer, declaration, plan="matmul=fp8", chip="v6e")


def test_bf16_kv_cache_falls_back() -> None:
    """A bf16 KV cache (the platform's setting) leaves the rule unfused."""
    layer, declaration = _declared(check_subblock.FAMILIES[0])
    assert "input_proj=unfused (rule)" in _install(layer, declaration, kv="bfloat16")


# One execution smoke check, and the compile cache's source lookup.


def test_smoke_llama_against_upstream() -> None:
    """Llama through the runner, unfused and fused, against its own forward."""
    assert not check_subblock.family(check_subblock.FAMILIES[0])


def test_installed_forward_has_source() -> None:
    """torch.compile's AOT cache needs source for every code object it traces."""
    for spec in check_subblock.FAMILIES:
        assert check_subblock._source_less(spec) == []  # noqa: SLF001
