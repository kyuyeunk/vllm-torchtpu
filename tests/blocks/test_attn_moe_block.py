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
"""Integration tests: each declared model registers with the block runner.

``test_every_declaration_conforms`` is the merge gate: every declaration in
``models/vllm/*/registration.py`` must conform. The install tests check that
the runner is built from the model's own parts, that the nested
``AttentionRunner`` is constructed with the block's contract (attention
unreduced) instead of a field write, and what the interim adapter still
writes. Execution numerics and the collective order are
``test_attn_moe_runner.py``'s job (model-free).
"""

import pytest
import torch
from torch import nn
from vllm.config import set_current_vllm_config

from prototype import _build, fixtures
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import conformance, norm_attn
from vllm_torchtpu.blocks.attn_moe import conform, contract_adapter
from vllm_torchtpu.blocks.attn_moe import pattern as pattern_mod
from vllm_torchtpu.blocks.attn_moe.runner import AttnMoERunner, ModelAttention
from vllm_torchtpu.blocks.norm_attn.runner import AttentionRunner
from vllm_torchtpu.blocks.stream import routed_runner

QWEN3_5_FULL, QWEN3_5_LINEAR = _build.SPECS[:2]
GLM = next(s for s in _build.SPECS if s.name == "GLM-4 MoE")
DECLARED = (QWEN3_5_FULL, QWEN3_5_LINEAR, GLM)


@pytest.mark.parametrize("spec", DECLARED, ids=lambda s: s.name)
def test_every_declaration_conforms(spec: _build.Spec) -> None:
    layer = _build.build(spec)
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS[(type(layer), pattern_mod.attn_then_moe)]
    assert declaration.when(layer)
    with set_current_vllm_config(_build.current_config()):
        roles = conform(layer, declaration)
    assert roles.out_proj.endswith(("o_proj", "out_proj"))


def test_mismatch_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared layer that does not conform is a startup error."""
    layer = fixtures.ReorderedNorms(_build.build(_build.SPECS[0]))
    declaration = blocks.BlockDeclaration(
        fixtures.ReorderedNorms,
        pattern_mod.attn_then_moe,
        fixtures.RENAMED_ROLES,
        lambda _: True,
    )
    monkeypatch.setitem(
        blocks.DECLARATIONS,
        (fixtures.ReorderedNorms, pattern_mod.attn_then_moe),
        declaration,
    )
    context = blocks.InstallContext(
        tp=4,
        dtype="bfloat16",
        buckets=(16, 8192),
        override="",
        model_plan="",
    )
    with (
        set_current_vllm_config(_build.current_config()),
        pytest.raises(conformance.BlockConformanceError, match="post_norm"),
    ):
        blocks.install(nn.ModuleDict({"layer": layer}), context)
    assert type(layer) is fixtures.ReorderedNorms


OVERLAP_PIN = "collectives=reduce_scatter_all_gather"


def _context(override: str = OVERLAP_PIN, buckets: tuple[int, ...] = (16, 8192)):  # noqa: ANN202
    return blocks.InstallContext(
        tp=4,
        dtype="bfloat16",
        buckets=buckets,
        override=override,
        model_plan="",
    )


def _install(layer: nn.Module, context: blocks.InstallContext) -> list[str]:
    with set_current_vllm_config(_build.current_config()):
        report = blocks.install(nn.ModuleDict({"l": layer}), context)
    return [item.outcome for item in report]


def _block_runner(layer: nn.Module) -> AttnMoERunner:
    runner = routed_runner(layer)
    assert isinstance(runner, AttnMoERunner)
    return runner


@pytest.mark.parametrize("spec", [QWEN3_5_FULL, GLM], ids=lambda s: s.name)
def test_block_builds_attention_with_its_contract(spec: _build.Spec) -> None:
    """Both scopes declared: the block constructs the nested runner unreduced.

    Attention's contract is a constructor argument, so ``o_proj`` keeps the
    value the model built and no ledger entry names it. The MoE child's
    contracts are still the interim writes.
    """
    layer = _build.build(spec)
    cls = type(layer)
    attention = layer.get_submodule("self_attn")
    experts = layer.get_submodule("mlp.experts")
    contract_adapter.LEDGER.clear()
    outcomes = _install(layer, _context())
    assert len(outcomes) == 1, outcomes
    assert outcomes[0].startswith("AttnMoERunner installed"), outcomes
    assert "nested AttentionRunner input_proj=" in outcomes[0], outcomes
    assert "output_reduced=False" in outcomes[0], outcomes
    assert type(layer) is cls  # No mixin, no __class__ rebinding.
    runner = _block_runner(layer)
    front = runner.front
    assert isinstance(front, AttentionRunner)
    assert front.output_reduced is False
    assert front.o_proj is attention.o_proj
    assert front.qkv_proj is attention.qkv_proj
    assert attention.o_proj.reduce_results is True  # Not written.
    assert runner.post_norm is layer.get_submodule("post_attention_layernorm")
    assert runner.router is layer.get_submodule("mlp.gate")
    assert runner.experts is experts
    assert list(runner.parameters()) == []
    written = {(e.child, e.field) for e in contract_adapter.LEDGER}
    assert written == {("experts", "gate"), ("moe_config", "skip_final_all_reduce")}
    assert experts.gate is None
    assert experts.moe_config.skip_final_all_reduce is True


def test_model_attention_front_keeps_the_interim_write() -> None:
    """Qwen3.5's linear attention declares no sub-block: its o_proj is written."""
    layer = _build.build(QWEN3_5_LINEAR)
    contract_adapter.LEDGER.clear()
    _install(layer, _context())
    assert isinstance(_block_runner(layer).front, ModelAttention)
    assert ("out_proj", "reduce_results") in {
        (e.child, e.field) for e in contract_adapter.LEDGER
    }
    out_proj = layer.get_submodule(conform_roles(layer).out_proj)
    assert out_proj.reduce_results is False
    with pytest.raises(contract_adapter.BlockContractError):
        out_proj.reduce_results = True


def conform_roles(layer: nn.Module) -> pattern_mod.AttnMoERoles:
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS[(type(layer), pattern_mod.attn_then_moe)]
    with set_current_vllm_config(_build.current_config()):
        return conform(layer, declaration)


@pytest.mark.parametrize("spec", [QWEN3_5_FULL, GLM], ids=lambda s: s.name)
def test_upstream_collectives_leave_the_sub_block_on_its_own(spec: _build.Spec) -> None:
    """No bucket overlaps: the block stays upstream, the sub-block reduces itself."""
    layer = _build.build(spec)
    outcomes = _install(layer, _context(override="", buckets=(16,)))
    assert "all_reduce on every bucket; sub-block runner installed" in outcomes[0]
    assert isinstance(layer, norm_attn.NormAttnDecoder)
    runner = layer.__dict__["_norm_attn_runner"]
    assert isinstance(runner, AttentionRunner)
    assert runner.output_reduced is True
    assert routed_runner(layer) is None


def _with_o_proj_bias(layer: nn.Module) -> nn.Module:
    o_proj = layer.get_submodule("self_attn.o_proj")
    o_proj.bias = nn.Parameter(torch.zeros(o_proj.output_size, device="meta"))
    return layer


def test_pinned_overlap_without_unreduced_attention_raises() -> None:
    """Pinned: attention that cannot skip its reduce stops startup."""
    layer = _with_o_proj_bias(_build.build(GLM))
    with pytest.raises(ValueError, match="adds a bias"):
        _install(layer, _context())


def test_rule_overlap_without_unreduced_attention_falls_back() -> None:
    """Chosen by the rule: the block falls back to upstream's collectives."""
    layer = _with_o_proj_bias(_build.build(GLM))
    floor = 8 * 1024 * 1024 * 4 // (64 * 2)  # The rule's floor at TP 4, hidden 64.
    outcomes = _install(layer, _context(override="", buckets=(16, floor)))
    assert "all_reduce on every bucket (rule fell back" in outcomes[0], outcomes
    assert "adds a bias" in outcomes[0], outcomes
    assert routed_runner(layer) is None
