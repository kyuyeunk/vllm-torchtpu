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
"""Execution tests: AttentionRunner on synthetic binding, no model.

The runner is built from tiny binding (``synthetic.py``), each with the
contract of the vLLM building block it stands for: an RMSNorm taking ``(x, residual)``,
linears returning ``(out, bias)``, a NeoX rotary on ``(positions, q, k)``,
a causal GQA core on ``(q, k, v)``. Every role configuration the runner
implements (per-head q/k norm, output gate, GQA ratio, head_dim, QKV bias,
partial rotary, first layer) is checked against a straight-line reference,
and the fused value against the unfused one. The plan's choice and
``fused_qkv``'s capability are enumerated over synthetic attentions, chips
and KV cache settings.

Model registration (that a model's layers conform, and that the runner
holds their binding) is ``test_norm_attn_integration.py``'s job. This file
imports nothing from ``vllm.model_executor.models``;
``test_imports_no_model`` checks that.
"""

import sys

_PRELOADED = frozenset(sys.modules)  # Before this file's imports; see the last test.

import ast  # noqa: E402 - after the snapshot above
import dataclasses  # noqa: E402 - after the snapshot above
import pathlib  # noqa: E402 - after the snapshot above

import pytest  # noqa: E402 - after the snapshot above
import torch  # noqa: E402 - after the snapshot above
from jax.experimental.pallas import tpu as pltpu  # noqa: E402 - after the snapshot
from synthetic import (  # noqa: E402 - ditto
    Roles,
    build_runner,
    make_inputs,
    reference_forward,
    setup_parts,
)
from torch import nn  # noqa: E402 - after the snapshot above

from vllm_torchtpu.blocks.norm_attn import plan as plan_mod  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn import runner as runner_mod  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn.runner import (  # noqa: E402 - ditto
    AttnConfig,
    InputProj,
    reference_fused_qkv,
)

_LOADED = frozenset(sys.modules) - _PRELOADED  # What this file's imports loaded.

# The fused reference calls the same binding with the platform's rounding
# points (the residual sum rounded to the stream's dtype, as the fused
# add-norm does), so it equals unfused bit for bit, here in bf16. A kernel
# would be held to bf16 tolerance instead (atol 1e-2, rtol 1.6e-2).
FUSED_ATOL = FUSED_RTOL = 0.0


@pytest.fixture(autouse=True, scope="module")
def _one_thread() -> object:
    """Runs on one thread: the binding are tiny, and a many-core pool costs more."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


ROLE_GRID = [
    Roles(qk_norm, gate, gqa, head_dim, bias, rotary)
    for qk_norm in (False, True)
    for gate in (False, True)
    for gqa in (1, 4, 8)
    for head_dim in (64, 128)
    for bias in (False, True)
    for rotary in ("full", "partial")
]


def _id(roles: Roles) -> str:
    return "-".join(
        [
            f"gqa{roles.gqa}",
            f"hd{roles.head_dim}",
            *(["qknorm"] if roles.qk_norm else []),
            *(["gate"] if roles.output_gate else []),
            *(["bias"] if roles.qkv_bias else []),
            f"rope_{roles.rotary}",
        ]
    )


# (a) and (b): every role configuration, both plan values.


@pytest.mark.parametrize("first", [True, False], ids=["first_layer", "residual"])
@pytest.mark.parametrize("roles", ROLE_GRID, ids=_id)
def test_unfused_equals_reference(roles: Roles, first: bool) -> None:  # noqa: FBT001 - a pytest parameter, passed by name
    """Unfused equals reference."""
    norm, attention = setup_parts(roles)
    runner = build_runner(norm, attention, roles.config())
    hidden, residual, positions = make_inputs(first)
    with torch.no_grad():
        got = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
        want = reference_forward(roles, norm, attention, hidden, residual, positions)
    torch.testing.assert_close(got, want)  # float32 defaults: atol 1e-5, rtol 1.3e-6


@pytest.mark.parametrize("first", [True, False], ids=["first_layer", "residual"])
@pytest.mark.parametrize("roles", ROLE_GRID, ids=_id)
def test_fused_equals_unfused(roles: Roles, first: bool) -> None:  # noqa: FBT001 - a pytest parameter, passed by name
    """Fused equals unfused."""
    norm, attention = setup_parts(roles, torch.bfloat16)
    runner = build_runner(
        norm, attention, roles.config(), InputProj.FUSED_QKV, reference_fused_qkv
    )
    hidden, residual, positions = make_inputs(first, torch.bfloat16)
    before = runner_mod.FUSED_CALLS.get("layer", 0)
    with torch.no_grad():
        got = runner._fused_qkv(hidden, residual, positions)  # noqa: SLF001 - the value under test
        want = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
    torch.testing.assert_close(got, want, atol=FUSED_ATOL, rtol=FUSED_RTOL)
    assert runner_mod.FUSED_CALLS["layer"] == before + 1


def test_fused_op_that_declines_runs_unfused() -> None:
    """A fused op returning None for a call leaves that call to unfused."""
    roles = Roles()
    norm, attention = setup_parts(roles)
    runner = build_runner(
        norm, attention, roles.config(), InputProj.FUSED_QKV, lambda *_: None
    )
    hidden, residual, positions = make_inputs(first=False)
    with torch.no_grad():
        got = runner(hidden, residual, positions)
        want = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_runner_is_composition_only() -> None:
    """The runner is not a module and holds the model's own part objects."""
    roles = Roles(qk_norm=True)
    norm, attention = setup_parts(roles)
    runner = build_runner(norm, attention, roles.config())
    assert not isinstance(runner, nn.Module)
    assert runner.qkv_proj is attention.qkv_proj
    assert runner.q_norm is attention.q_norm


def _facts(**changes: object) -> plan_mod.AttentionFacts:
    """A supported attention's facts (8 heads of 128 over 2 KV heads), changed."""
    base = plan_mod.AttentionFacts(
        config=Roles().config(),
        sinks=False,
        kv_sharing=False,
        q_norm=None,
        k_norm=None,
        neox_rope=True,
        qkv_quant_supported=True,
        qkv_quant="UnquantizedLinearMethod",
        fp8_weights=False,
        o_proj_split_input=True,
        o_proj_adds_bias=False,
    )
    return dataclasses.replace(base, **changes)  # pyright: ignore[reportArgumentType]


def _config(**changes: object) -> AttnConfig:
    return dataclasses.replace(Roles().config(), **changes)  # pyright: ignore[reportArgumentType]


_RMS = plan_mod.NormFacts(rms=True, width=128, kind="RMSNorm")
_FP8 = _facts(fp8_weights=True)
_V7X, _V6E = pltpu.ChipVersion("7x"), pltpu.ChipVersion("v6e")


# (c): the plan, resolved once; rule versus pin.


@pytest.mark.parametrize(
    ("override", "facts", "expected"),
    [
        ("", _facts(), "_fused_qkv"),  # The rule's value.
        ("", _facts(sinks=True), "_unfused"),  # The rule falls back, quietly.
        ("input_proj=unfused", _facts(), "_unfused"),  # A pin beats the rule.
        ("input_proj=fused_qkv", _facts(), "_fused_qkv"),
    ],
    ids=["rule", "rule_falls_back", "pin_unfused", "pin_fused"],
)
def test_select_forward(
    override: str, facts: plan_mod.AttentionFacts, expected: str
) -> None:
    """Select forward."""
    plan, declined = plan_mod.choose(plan_mod.parse_pins(override), facts)
    assert bool(declined) == facts.sinks
    roles = Roles()
    norm, attention = setup_parts(roles)
    runner = build_runner(
        norm, attention, roles.config(), plan.input_proj, reference_fused_qkv
    )
    assert runner._forward_entry == getattr(runner, expected)  # noqa: SLF001 - the selection under test


@pytest.mark.parametrize(
    ("override", "facts", "match"),
    [
        ("input_proj=fused_qkv", _facts(sinks=True), "sinks"),
        ("matmul=fp8", _facts(), "needs FP8 weights"),
        ("input_proj=fused_qkv,matmul=bf16", _facts(kv_sharing=True), "KV sharing"),
        ("input_proj=bogus", _facts(), "valid: unfused, fused_qkv"),
        ("tile=8", _facts(), "unknown key 'tile'"),
    ],
    ids=[
        "pin_on_sinks",
        "pin_fp8_bf16_weights",
        "pin_kv_sharing",
        "bad_value",
        "bad_key",
    ],
)
def test_pin_that_cannot_run_raises(
    override: str, facts: plan_mod.AttentionFacts, match: str
) -> None:
    """Pin that cannot run raises."""
    with pytest.raises(ValueError, match=match):
        plan_mod.choose(plan_mod.parse_pins(override), facts, where="Synthetic")


def test_fp8_pin_on_v6e_raises() -> None:
    """Fp8 pin on v6e raises."""
    none, fp8 = plan_mod.Pins(), plan_mod.Pins(matmul=plan_mod.Matmul.FP8)
    assert plan_mod.choose(none, _FP8, chip=_V7X)[0].matmul is plan_mod.Matmul.FP8
    assert plan_mod.choose(none, _FP8, chip=_V6E)[0].matmul is plan_mod.Matmul.BF16
    with pytest.raises(ValueError, match="chip v6e has none"):
        plan_mod.choose(fp8, _FP8, chip=_V6E)


def test_fused_value_without_an_op_raises() -> None:
    """Fused value without an op raises."""
    roles = Roles()
    norm, attention = setup_parts(roles)
    with pytest.raises(ValueError, match="fused_qkv needs a fused op"):
        build_runner(norm, attention, roles.config(), InputProj.FUSED_QKV, None)


# (d): fused_qkv's capability over the model's, the chip's and the platform's
# descriptions.


def _case(  # noqa: PLR0913 - the binding and options a runner or reference takes
    name: str,
    facts: plan_mod.AttentionFacts,
    reason: str | None = None,
    *,
    matmul: plan_mod.Matmul = plan_mod.Matmul.BF16,
    chip: pltpu.ChipVersion | None = None,
    kv: str = "",
) -> object:
    """A capability case; the KV string is parsed as the platform's is."""
    fp8_kv = plan_mod.is_fp8_e4m3(kv) if kv else None
    return pytest.param(facts, matmul, chip, fp8_kv, reason, id=name)


_F8 = plan_mod.Matmul.FP8
CAPABILITY = [
    _case("supported", _facts()),
    _case("supported_v6e_fp8_kv", _facts(), chip=_V6E, kv="fp8"),
    _case("supported_7x_fp8_e4m3_kv", _facts(), chip=_V7X, kv="fp8_e4m3"),
    _case(
        "qk_norm_gate",
        _facts(
            config=_config(qk_norm=True, output_gate=True), q_norm=_RMS, k_norm=_RMS
        ),
    ),
    _case("sinks", _facts(sinks=True), "sinks need RPA v3"),
    _case("kv_sharing", _facts(kv_sharing=True), "KV sharing"),
    _case("head_dim_64", _facts(config=_config(head_dim=64)), "head_dim 64 runs on"),
    _case("head_dim_80", _facts(config=_config(head_dim=80)), "not a multiple of 32"),
    _case("gqa_1", _facts(config=_config(num_heads=2)), "1 query heads per KV head"),
    _case(
        "gate_head_dim_160",
        _facts(config=_config(head_dim=160, output_gate=True)),
        "an output gate needs head_dim % 128",
    ),
    _case(
        "qk_norm_not_rms",
        _facts(q_norm=plan_mod.NormFacts(rms=False, width=128, kind="WholeNorm")),
        "q_norm is WholeNorm",
    ),
    _case(  # Over all 8 heads: the norm needs the whole projection.
        "qk_norm_tp_wide",
        _facts(q_norm=plan_mod.NormFacts(rms=True, width=8 * 128, kind="RMSNorm")),
        "TP-wide norm",
    ),
    _case("gptj_rope", _facts(neox_rope=False), "GPT-J-style RoPE"),
    _case(
        "awq",
        _facts(qkv_quant_supported=False, qkv_quant="AwqLinearMethod"),
        "quantized by AwqLinearMethod",
    ),
    _case("bf16_kv", _facts(), "KV cache is not FP8", chip=_V7X, kv="bfloat16"),
    _case("auto_kv", _facts(), "KV cache is not FP8", kv="auto"),
    _case("fp8_e5m2_kv", _facts(), "KV cache is not FP8", kv="fp8_e5m2"),
    _case("fp8_matmul_bf16_weights", _facts(), "needs FP8 weights", matmul=_F8),
    _case("fp8_matmul_7x", _FP8, matmul=_F8, chip=_V7X, kv="fp8"),
    _case("fp8_matmul_v6e", _FP8, "chip v6e has none", matmul=_F8, chip=_V6E, kv="fp8"),
    _case("fp8_matmul_off_tpu", _FP8, matmul=_F8),
]


@pytest.mark.parametrize(("facts", "matmul", "chip", "fp8_kv", "reason"), CAPABILITY)
def test_fused_qkv_supports(
    facts: plan_mod.AttentionFacts,
    matmul: plan_mod.Matmul,
    chip: pltpu.ChipVersion | None,
    fp8_kv: bool | None,  # noqa: FBT001 - a pytest parameter, passed by name
    reason: str | None,
) -> None:
    """Fused qkv supports."""
    why = plan_mod.fused_qkv_supports(
        facts, matmul=matmul, chip=chip, fp8_kv_cache=fp8_kv
    )
    if reason is None:
        assert why is None
    else:
        assert why is not None
        assert reason in why, why


# The split: this file and the runner import no model.

_MODELS = "vllm.model_executor.models"
# vLLM's model package infrastructure, no model class: the registry (model
# names to lazily imported modules), the interfaces, config.py's config
# fix-ups, and the shared helpers (utils, vision) that vLLM's Fp8LinearMethod
# import chain loads, which the capability check matches by type.
_MODEL_INFRASTRUCTURE = frozenset(
    {_MODELS}
    | {
        f"{_MODELS}.{m}"
        for m in (
            "registry",
            "interfaces",
            "interfaces_base",
            "config",
            "utils",
            "vision",
        )
    }
)


def _imports(path: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_imports_no_model() -> None:
    """No import of vllm.model_executor.models, statically or at runtime.

    Statically: neither this file nor ``runner.py`` names a model module, and
    ``runner.py`` imports nothing from vLLM at all. At runtime: the modules
    this file's imports loaded include no model implementation, only vLLM's
    model package infrastructure (``_MODEL_INFRASTRUCTURE``). The runtime
    half sees only modules no earlier file loaded; pytest collects this
    file first in ``tests/blocks``.
    """
    this, runner = pathlib.Path(__file__), pathlib.Path(runner_mod.__file__)
    assert not {n for n in _imports(this) if n.startswith(_MODELS)}
    assert not {n for n in _imports(runner) if n.split(".")[0] == "vllm"}
    assert not {m for m in _LOADED if m.startswith(_MODELS)} - _MODEL_INFRASTRUCTURE
