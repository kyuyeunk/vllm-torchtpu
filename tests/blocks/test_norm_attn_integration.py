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

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn
from vllm.config.vllm import set_current_vllm_config

from prototype import _build, check_subblock
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import installer
from vllm_torchtpu.blocks.attn_moe.plan import Collectives
from vllm_torchtpu.blocks.attn_moe.runner import AttnMoERunner
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.context import InstallContext
from vllm_torchtpu.blocks.norm_attn import install as norm_attn
from vllm_torchtpu.blocks.norm_attn import plan as subblock_plan
from vllm_torchtpu.blocks.norm_attn.pattern import NormAttnParts, norm_then_attention
from vllm_torchtpu.blocks.norm_attn.runner import AttentionRunner, InputProj
from vllm_torchtpu.blocks.stream import routed_runner

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


def _routed(layer: nn.Module) -> norm_attn.NormAttnLayer:
    routed = routed_runner(layer)
    assert isinstance(routed, norm_attn.NormAttnLayer)
    return routed


def _parts_of(layer: nn.Module, declaration: blocks.BlockDeclaration) -> NormAttnParts:
    parts = declaration.parts(layer)
    assert isinstance(parts, NormAttnParts)
    return parts


def _weight(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    assert isinstance(weight, torch.Tensor)
    return weight


def _install(
    layer: nn.Module, declaration: blocks.BlockDeclaration, **context: str
) -> str:
    impl = installer.IMPLEMENTATIONS[norm_then_attention]
    with set_current_vllm_config(_build.current_config()):
        return impl.install("l", layer, declaration, _context(**context)).outcome


def _context(plan: str = "", chip: str = "", kv: str = "") -> InstallContext:
    """The test's strings, parsed as the runner parses the platform's."""
    return InstallContext(
        tp=1,
        dtype=torch.bfloat16,
        buckets=(16,),
        subblock_pins=subblock_plan.parse_pins(plan),
        chip=pltpu.ChipVersion(chip) if chip else None,
        fp8_kv_cache=subblock_plan.is_fp8_e4m3(kv) if kv else None,
    )


# The declaration conforms, or names what does not.


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_declaration_conforms(spec: _build.Spec) -> None:
    """Declaration conforms."""
    layer, declaration = _declared(spec)
    assert declaration.when(layer)
    with set_current_vllm_config(_build.current_config()):
        config = norm_attn.conform(layer, declaration).config
    parts = _parts_of(layer, declaration)
    assert config.head_dim == parts.attention.head_dim
    assert config.qk_norm == (parts.q_norm is not None)


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
    declaration = blocks.BlockDeclaration.of(planted, check_subblock._llama_parts)  # noqa: SLF001 - the planted layers' binding
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


def test_binding_reads_a_missing_part_and_fails() -> None:
    """Attention without rotary_emb: the binding function's own read fails.

    For a real class the type checker reports this read before anything runs
    (the attribute is not on the class); here it is deleted at run time.
    """
    layer, declaration = _declared(check_subblock.FAMILIES[0])
    _without_rotary(layer.get_submodule("self_attn"))
    with pytest.raises(AttributeError, match="rotary_emb"):
        declaration.parts(layer)


def test_part_of_the_wrong_kind_is_named_by_attribute() -> None:
    """A lone q_norm: the parts' kind check names the attribute."""
    layer, declaration = _declared(check_subblock.FAMILIES[0])
    _lone_q_norm(layer.get_submodule("self_attn"))
    with pytest.raises(
        BlockConformanceError,
        match=r"q_norm=self_attn\.q_norm and k_norm come together",
    ):
        declaration.parts(layer)


# The runner holds the model's own parts, and the decoder routes through it.


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_runner_holds_the_models_parts(spec: _build.Spec) -> None:
    """Runner holds the models parts."""
    layer, declaration = _declared(spec)
    state = dict(layer.state_dict(keep_vars=True))
    _install(layer, declaration)
    runner = _routed(layer).runner
    parts = _parts_of(layer, declaration)
    attention = parts.attention
    assert runner.input_norm is parts.input_norm
    assert runner.qkv_proj is attention.qkv_proj
    assert runner.o_proj is attention.o_proj
    assert runner.rotary is attention.rotary_emb
    assert runner.attn is parts.attn
    assert runner.q_norm is parts.q_norm
    assert runner.k_norm is parts.k_norm
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
    cls = type(layer)
    _install(layer, declaration)
    assert type(layer) is cls  # No mixin: the runner is the instance's forward.
    parts = _parts_of(layer, declaration)
    calls: list[tuple[torch.Tensor, ...]] = []
    meta = {"device": "meta", "dtype": torch.bfloat16}
    hidden_size = int(_weight(parts.input_norm).shape[0])
    attn_out, summed, normed = (torch.empty(4, hidden_size, **meta) for _ in range(3))

    def runner(h: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> tuple:
        calls.append((h, r, p))
        return attn_out, summed

    def post_norm(x: torch.Tensor, r: torch.Tensor) -> tuple:
        assert x is attn_out
        assert r is summed
        return normed, r

    monkeypatch.setattr(_routed(layer).runner, "forward", runner)
    monkeypatch.setattr(parts.post_norm, "forward", post_norm)
    monkeypatch.setattr(parts.mlp, "forward", torch.neg)
    hidden, residual = (
        torch.empty(4, hidden_size, **meta),
        torch.empty(4, hidden_size, **meta),
    )
    positions = torch.empty(4, device="meta", dtype=torch.long)
    # Every surveyed forward names the stream hidden_states/residual/positions.
    out, out_residual = layer(
        hidden_states=hidden, residual=residual, positions=positions
    )
    assert len(calls) == 1
    assert all(
        a is b for a, b in zip(calls[0], (hidden, residual, positions), strict=True)
    )
    assert out.shape == normed.shape
    assert out_residual is summed


def test_scopes_nest_on_qwen3_5() -> None:
    """Qwen3.5 declares both scopes; the block builds the attention runner itself."""
    layer = _build.build(check_subblock.FAMILIES[2])  # Qwen3.5-MoE, full attention
    context = InstallContext(
        tp=4,
        dtype=torch.bfloat16,
        buckets=(16, 8192),
        block_pin=Collectives.REDUCE_SCATTER_ALL_GATHER,
    )
    with set_current_vllm_config(_build.current_config()):
        report = installer.install(torch.nn.ModuleDict({"l": layer}), context)
    outcomes = [item.outcome for item in report]
    assert len(outcomes) == 1, outcomes  # The block installed the sub-block.
    assert "nested AttentionRunner" in outcomes[0], outcomes
    block = routed_runner(layer)
    assert isinstance(block, AttnMoERunner)
    runner = block.attention
    assert isinstance(runner, AttentionRunner)
    assert runner.qkv_proj is layer.get_submodule("self_attn.qkv_proj")
    assert runner.input_proj is InputProj.FUSED_QKV
    assert runner.output_reduced is False


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
