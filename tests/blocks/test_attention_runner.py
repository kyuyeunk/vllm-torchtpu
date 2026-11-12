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
"""Execution tests: AttentionRunner on synthetic parts, no model.

The runner is built from tiny parts (``synthetic.py``), each with the
contract of the vLLM building block it stands for: an RMSNorm taking ``(x, residual)``,
linears returning ``(out, bias)``, a NeoX rotary on ``(positions, q, k)``,
a causal GQA core on ``(q, k, v)``. Every role configuration the runner
implements (per-head q/k norm, output gate, GQA ratio, head_dim, QKV bias,
partial rotary, first layer) is checked against a straight-line reference,
and the fused value against the unfused one. The plan's choice and
``fused_qkv``'s capability are enumerated over synthetic attentions, chips
and KV cache settings.

Model registration (that a model's layers conform, and that the runner
holds their parts) is ``test_norm_attn_integration.py``'s job. This file
imports nothing from ``vllm.model_executor.models``;
``test_imports_no_model`` checks that.
"""

import sys

_PRELOADED = frozenset(sys.modules)  # Before this file's imports; see the last test.

import ast  # noqa: E402 - after the snapshot above
import pathlib  # noqa: E402 - after the snapshot above
from collections.abc import Callable  # noqa: E402 - after the snapshot above

import pytest  # noqa: E402 - after the snapshot above
import torch  # noqa: E402 - after the snapshot above
from synthetic import (  # noqa: E402 - ditto
    HIDDEN,
    Attention,
    AwqLinearMethod,
    HeadNorm,
    InputNorm,
    Roles,
    WholeNorm,
    build_runner,
    make_inputs,
    reference_forward,
    setup_parts,
)
from torch import nn  # noqa: E402 - after the snapshot above

from vllm_torchtpu.blocks.norm_attn import plan as plan_mod  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn import runner as runner_mod  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn.runner import (  # noqa: E402 - ditto
    reference_fused_qkv,
)

_LOADED = frozenset(sys.modules) - _PRELOADED  # What this file's imports loaded.

# The fused reference calls the same parts with the platform's rounding
# points (the residual sum rounded to the stream's dtype, as the fused
# add-norm does), so it equals unfused bit for bit, here in bf16. A kernel
# would be held to bf16 tolerance instead (atol 1e-2, rtol 1.6e-2).
FUSED_ATOL = FUSED_RTOL = 0.0


@pytest.fixture(autouse=True, scope="module")
def _one_thread() -> object:
    """Runs on one thread: the parts are tiny, and a many-core pool costs more."""
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
def test_unfused_equals_reference(roles: Roles, first: bool) -> None:
    norm, attention = setup_parts(roles)
    runner = build_runner(norm, attention, roles.config())
    hidden, residual, positions = make_inputs(first)
    with torch.no_grad():
        got = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
        want = reference_forward(roles, norm, attention, hidden, residual, positions)
    torch.testing.assert_close(got, want)  # float32 defaults: atol 1e-5, rtol 1.3e-6


@pytest.mark.parametrize("first", [True, False], ids=["first_layer", "residual"])
@pytest.mark.parametrize("roles", ROLE_GRID, ids=_id)
def test_fused_equals_unfused(roles: Roles, first: bool) -> None:
    norm, attention = setup_parts(roles, torch.bfloat16)
    runner = build_runner(
        norm, attention, roles.config(), "fused_qkv", reference_fused_qkv
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
    runner = build_runner(norm, attention, roles.config(), "fused_qkv", lambda *_: None)
    hidden, residual, positions = make_inputs(first=False)
    with torch.no_grad():
        got = runner(hidden, residual, positions)
        want = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_runner_owns_no_parameters() -> None:
    """The parts are references: the runner adds nothing to a state_dict."""
    roles = Roles(qk_norm=True)
    norm, attention = setup_parts(roles)
    runner = build_runner(norm, attention, roles.config())
    assert list(runner.parameters()) == []
    assert runner.qkv_proj is attention.qkv_proj
    assert runner.q_norm is attention.q_norm


def _fp8(attention: Attention) -> Attention:
    weight = attention.qkv_proj.weight
    attention.qkv_proj.weight = nn.Parameter(
        weight.data.to(torch.float8_e4m3fn), requires_grad=False
    )
    return attention


def _awq(attention: Attention) -> Attention:
    attention.qkv_proj.quant_method = AwqLinearMethod()
    return attention


# (c): the plan, resolved once; rule versus pin.


@pytest.mark.parametrize(
    ("override", "attention_kwargs", "expected"),
    [
        ("", {}, "_fused_qkv"),  # The rule's value.
        ("", {"sinks": True}, "_unfused"),  # The rule falls back, quietly.
        ("input_proj=unfused", {}, "_unfused"),  # A pin beats the rule.
        ("input_proj=fused_qkv", {}, "_fused_qkv"),
    ],
    ids=["rule", "rule_falls_back", "pin_unfused", "pin_fused"],
)
def test_select_forward(override: str, attention_kwargs: dict, expected: str) -> None:
    torch.manual_seed(0)
    roles = Roles()
    attention = Attention(roles, **attention_kwargs)
    plan, pinned, declined = plan_mod.choose(override, attention)
    assert pinned == frozenset(dict(p.split("=") for p in override.split(",") if p))
    assert bool(declined) == bool(attention_kwargs)
    norm = InputNorm(HIDDEN)
    runner = build_runner(
        norm, attention, roles.config(), plan.input_proj, reference_fused_qkv
    )
    assert runner._forward_entry == getattr(runner, expected)  # noqa: SLF001 - the selection under test


@pytest.mark.parametrize(
    ("override", "attention_kwargs", "chip", "match"),
    [
        ("input_proj=fused_qkv", {"sinks": True}, "", "sinks"),
        ("matmul=fp8", {}, "", "needs FP8 weights"),
        ("input_proj=fused_qkv,matmul=bf16", {"kv_sharing": "l0"}, "", "KV sharing"),
        ("input_proj=bogus", {}, "", "not one of"),
        ("tile=8", {}, "", "unknown keys"),
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
    override: str, attention_kwargs: dict, chip: str, match: str
) -> None:
    attention = Attention(Roles(), **attention_kwargs)
    with pytest.raises(ValueError, match=match):
        plan_mod.choose(override, attention, chip=chip, where="Synthetic")


def test_fp8_pin_on_v6e_raises() -> None:
    attention = _fp8(Attention(Roles()))
    assert plan_mod.choose("", attention, chip="7x")[0].matmul == "fp8"
    assert plan_mod.choose("", attention, chip="v6e")[0].matmul == "bf16"
    with pytest.raises(ValueError, match="chip v6e has none"):
        plan_mod.choose("matmul=fp8", attention, chip="v6e")


def test_fused_value_without_an_op_raises() -> None:
    roles = Roles()
    norm, attention = setup_parts(roles)
    with pytest.raises(ValueError, match="fused_qkv needs a fused op"):
        build_runner(norm, attention, roles.config(), "fused_qkv", None)


# (d): fused_qkv's capability over the model's, the chip's and the platform's
# descriptions.


def _case(
    name: str,
    make: Callable[[], Attention],
    reason: str | None = None,
    *,
    matmul: str = "bf16",
    chip: str = "",
    kv: str = "",
) -> object:
    return pytest.param(make, matmul, chip, kv, reason, id=name)


def _plain() -> Attention:
    return Attention(Roles())


CAPABILITY = [
    _case("supported", _plain),
    _case("supported_v6e_fp8_kv", _plain, chip="v6e", kv="fp8"),
    _case("supported_7x_fp8_e4m3_kv", _plain, chip="7x", kv="fp8_e4m3"),
    _case("qk_norm_gate", lambda: Attention(Roles(qk_norm=True, output_gate=True))),
    _case("sinks", lambda: Attention(Roles(), sinks=True), "sinks need RPA v3"),
    _case("kv_sharing", lambda: Attention(Roles(), kv_sharing="l0"), "KV sharing"),
    _case("head_dim_64", lambda: Attention(Roles(head_dim=64)), "head_dim 64 runs on"),
    _case("head_dim_80", lambda: Attention(Roles(head_dim=80)), "not a multiple of 32"),
    _case("gqa_1", lambda: Attention(Roles(gqa=1)), "1 query heads per KV head"),
    _case(
        "gate_head_dim_160",
        lambda: Attention(Roles(head_dim=160, output_gate=True)),
        "an output gate needs head_dim % 128",
    ),
    _case(
        "qk_norm_not_rms",
        lambda: Attention(Roles(qk_norm=True), q_norm=WholeNorm(128)),
        "q_norm is WholeNorm",
    ),
    _case(
        "qk_norm_tp_wide",  # Over all 8 heads: the norm needs the whole projection.
        lambda: Attention(Roles(qk_norm=True), q_norm=HeadNorm(8 * 128)),
        "TP-wide norm",
    ),
    _case("gptj_rope", lambda: Attention(Roles(), neox=False), "GPT-J-style RoPE"),
    _case("awq", lambda: _awq(_plain()), "quantized by AwqLinearMethod"),
    _case("bf16_kv", _plain, "KV cache bfloat16", chip="7x", kv="bfloat16"),
    _case("auto_kv", _plain, "KV cache auto", kv="auto"),
    _case("fp8_e5m2_kv", _plain, "KV cache fp8_e5m2", kv="fp8_e5m2"),
    _case("fp8_matmul_bf16_weights", _plain, "needs FP8 weights", matmul="fp8"),
    _case("fp8_matmul_7x", lambda: _fp8(_plain()), matmul="fp8", chip="7x", kv="fp8"),
    _case(
        "fp8_matmul_v6e",
        lambda: _fp8(_plain()),
        "chip v6e has none",
        matmul="fp8",
        chip="v6e",
        kv="fp8",
    ),
    _case("fp8_matmul_off_tpu", lambda: _fp8(_plain()), matmul="fp8"),
]


@pytest.mark.parametrize(
    ("make", "matmul", "chip", "kv_cache_dtype", "reason"), CAPABILITY
)
def test_fused_qkv_supports(
    make: Callable[[], Attention],
    matmul: plan_mod.Matmul,
    chip: str,
    kv_cache_dtype: str,
    reason: str | None,
) -> None:
    torch.manual_seed(0)
    why = plan_mod.fused_qkv_supports(
        make(), matmul=matmul, chip=chip, kv_cache_dtype=kv_cache_dtype
    )
    if reason is None:
        assert why is None
    else:
        assert why is not None
        assert reason in why, why


# The split: this file and the runner import no model.

_MODELS = "vllm.model_executor.models"
# vLLM's model package infrastructure, no model class: the registry (model
# names to lazily imported modules), the interfaces, and config.py's config
# fix-ups. vllm_torchtpu.layers loads them through vllm.entrypoints.
_MODEL_INFRASTRUCTURE = frozenset(
    {_MODELS}
    | {f"{_MODELS}.{m}" for m in ("registry", "interfaces", "interfaces_base")}
    | {f"{_MODELS}.config"}
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
    assert not {n for n in _imports(runner) if n.startswith("vllm")}
    assert not {m for m in _LOADED if m.startswith(_MODELS)} - _MODEL_INFRASTRUCTURE
