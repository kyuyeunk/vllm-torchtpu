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

The runner is built from tiny parts written here, each with the contract of
the vLLM building block it stands for: an RMSNorm taking ``(x, residual)``,
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
import dataclasses  # noqa: E402 - after the snapshot above
import pathlib  # noqa: E402 - after the snapshot above
from collections.abc import Callable  # noqa: E402 - after the snapshot above

import pytest  # noqa: E402 - after the snapshot above
import torch  # noqa: E402 - after the snapshot above
import torch.nn.functional as F  # noqa: E402, N812 - after the snapshot; torch's alias
from torch import nn  # noqa: E402 - after the snapshot above
from vllm.model_executor.layers import layernorm  # noqa: E402 - after the snapshot

from vllm_torchtpu.blocks.norm_attn import plan as plan_mod  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn import runner as runner_mod  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn.runner import (  # noqa: E402 - ditto
    AttentionRunner,
    AttnConfig,
    reference_fused_qkv,
)

_LOADED = frozenset(sys.modules) - _PRELOADED  # What this file's imports loaded.

HIDDEN = 64
TOKENS = 7
NUM_KV_HEADS = 2
EPS = 1e-6
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


# The synthetic parts.


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return (normed * weight.float()).to(x.dtype)


class InputNorm(nn.Module):
    """vLLM's RMSNorm contract: ``x -> normed``, ``(x, residual) -> (normed, sum)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rms(x, self.weight)
        summed = (x.float() + residual.float()).to(x.dtype)
        return _rms(summed, self.weight), summed


class HeadNorm(layernorm.RMSNorm):
    """A per-head RMSNorm over ``(..., heads, head_dim)``.

    Subclasses vLLM's RMSNorm only for its type, which ``fused_qkv``'s
    capability reads; the forward is this file's, and ``CustomOp.__init__``
    (which needs a vLLM config) is skipped.
    """

    def __init__(self, dim: int) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return _rms(x, self.weight)


class WholeNorm(nn.Module):
    """A q/k norm that is not an RMSNorm (stands for a TP-wide LayerNorm)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class UnquantizedLinearMethod:
    """Stands for vLLM's; the capability reads the quant method's class name."""


class AwqLinearMethod:
    """A quant method ``fused_qkv`` does not implement."""


class Linear(nn.Module):
    """vLLM's linear contract: ``x -> (out, output_bias)``, the bias already added."""

    def __init__(self, n_in: int, n_out: int, *, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) / n_in**0.5)
        self.bias = nn.Parameter(0.1 * torch.randn(n_out)) if bias else None
        self.quant_method: object = UnquantizedLinearMethod()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight, self.bias), None


class NeoxRotary(nn.Module):
    """NeoX RoPE on the first ``rotary_dim`` values of each head, on flat q and k."""

    def __init__(self, head_dim: int, rotary_dim: int, *, neox: bool = True) -> None:
        super().__init__()
        self.head_dim, self.rotary_dim = head_dim, rotary_dim
        self.is_neox_style = neox

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(shape[0], -1, self.head_dim)
        rot, keep = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        half = self.rotary_dim // 2
        freqs = 10000.0 ** (-torch.arange(half, dtype=torch.float32) / half)
        angle = positions.float()[:, None] * freqs[None, :]
        cos = torch.cat([angle.cos(), angle.cos()], -1)[:, None, :]
        sin = torch.cat([angle.sin(), angle.sin()], -1)[:, None, :]
        rotated = torch.cat([-rot[..., half:], rot[..., :half]], -1)
        out = (rot.float() * cos + rotated.float() * sin).to(x.dtype)
        return torch.cat([out, keep], -1).view(shape)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rotate(q, positions), self._rotate(k, positions)


class CausalGqa(nn.Module):
    """A causal GQA core on flat ``(q, k, v)``, standing in for the backend."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        sinks: bool = False,
        kv_sharing: str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            num_heads,
            num_kv_heads,
            head_dim,
        )
        self.sinks = nn.Parameter(torch.zeros(num_heads)) if sinks else None
        self.kv_sharing_target_layer_name = kv_sharing

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        t, h, d = q.shape[0], self.num_heads, self.head_dim
        q = q.view(t, h, d).transpose(0, 1)
        k = k.view(t, self.num_kv_heads, d).transpose(0, 1)
        v = v.view(t, self.num_kv_heads, d).transpose(0, 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return out.transpose(0, 1).reshape(t, h * d)


@dataclasses.dataclass(frozen=True)
class Roles:
    """One role configuration of the sub-block."""

    qk_norm: bool = False
    output_gate: bool = False
    gqa: int = 4
    head_dim: int = 128
    qkv_bias: bool = False
    rotary: str = "full"  # or "partial": the first half of each head.

    def config(self) -> AttnConfig:
        return AttnConfig(
            num_heads=NUM_KV_HEADS * self.gqa,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=self.head_dim,
            qk_norm=self.qk_norm,
            output_gate=self.output_gate,
        )


class Attention(nn.Module):
    """A synthetic attention role: the parts, and the properties vLLM's carry.

    ``fused_qkv_supports`` reads the properties; the runner receives the parts.
    """

    def __init__(
        self,
        roles: Roles,
        *,
        sinks: bool = False,
        kv_sharing: str | None = None,
        q_norm: nn.Module | None = None,
        neox: bool = True,
    ) -> None:
        super().__init__()
        c = roles.config()
        self.num_heads, self.num_kv_heads, self.head_dim = (
            c.num_heads,
            c.num_kv_heads,
            c.head_dim,
        )
        self.attn_output_gate = roles.output_gate
        q_width = c.q_size * (2 if roles.output_gate else 1)
        self.qkv_proj = Linear(HIDDEN, q_width + 2 * c.kv_size, bias=roles.qkv_bias)
        self.o_proj = Linear(c.q_size, HIDDEN)
        rotary_dim = c.head_dim if roles.rotary == "full" else c.head_dim // 2
        self.rotary_emb = NeoxRotary(c.head_dim, rotary_dim, neox=neox)
        self.attn = CausalGqa(
            c.num_heads, c.num_kv_heads, c.head_dim, sinks=sinks, kv_sharing=kv_sharing
        )
        self.q_norm = self.k_norm = None
        if roles.qk_norm:
            self.q_norm = q_norm or HeadNorm(c.head_dim)
            self.k_norm = HeadNorm(c.head_dim)


def _setup(
    roles: Roles, dtype: torch.dtype = torch.float32
) -> tuple[InputNorm, Attention]:
    torch.manual_seed(0)
    return InputNorm(HIDDEN).to(dtype), Attention(roles).to(dtype)


def _runner(
    norm: nn.Module,
    attention: Attention,
    config: AttnConfig,
    input_proj: str = "unfused",
    fused_op: runner_mod.FusedOp | None = None,
) -> AttentionRunner:
    return AttentionRunner(
        "layer",
        input_norm=norm,
        qkv_proj=attention.qkv_proj,
        attn=attention.attn,
        o_proj=attention.o_proj,
        rotary=attention.rotary_emb,
        q_norm=attention.q_norm,
        k_norm=attention.k_norm,
        config=config,
        input_proj=input_proj,
        fused_op=fused_op,
    )


def _inputs(
    first: bool, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    torch.manual_seed(1)
    hidden = torch.randn(TOKENS, HIDDEN, dtype=dtype)
    residual = None if first else torch.randn(TOKENS, HIDDEN, dtype=dtype)
    return hidden, residual, torch.arange(TOKENS) * 3


def reference_forward(
    roles: Roles,
    norm: InputNorm,
    attention: Attention,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The sub-block in straight-line float32 torch, from the parts' weights."""
    c = roles.config()
    h, kvh, d, t = c.num_heads, c.num_kv_heads, c.head_dim, hidden.shape[0]
    stream = hidden if residual is None else hidden + residual

    def rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS) * w

    x = rms(stream, norm.weight)
    qkv = x @ attention.qkv_proj.weight.T
    if attention.qkv_proj.bias is not None:
        qkv = qkv + attention.qkv_proj.bias
    q_end = h * d * (2 if roles.output_gate else 1)
    q, k, v = qkv[:, :q_end], qkv[:, q_end : q_end + kvh * d], qkv[:, q_end + kvh * d :]
    gate = None
    if roles.output_gate:  # Per head: [q (d values), gate (d values)].
        per_head = q.reshape(t, h, 2 * d)
        q, gate = per_head[..., :d], per_head[..., d:].reshape(t, h * d)
    q, k, v = q.reshape(t, h, d), k.reshape(t, kvh, d), v.reshape(t, kvh, d)
    if roles.qk_norm:
        q_norm, k_norm = attention.q_norm, attention.k_norm
        assert isinstance(q_norm, HeadNorm)
        assert isinstance(k_norm, HeadNorm)
        q, k = rms(q, q_norm.weight), rms(k, k_norm.weight)
    r = d if roles.rotary == "full" else d // 2
    half = r // 2
    inv_freq = 1.0 / 10000.0 ** (torch.arange(half, dtype=torch.float32) / half)
    angle = positions.float()[:, None, None] * inv_freq
    cos, sin = angle.cos(), angle.sin()

    def rope(x: torch.Tensor) -> torch.Tensor:
        x1, x2, rest = x[..., :half], x[..., half:r], x[..., r:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], -1)

    q, k = rope(q), rope(k)
    k = k.repeat_interleave(h // kvh, dim=1)
    v = v.repeat_interleave(h // kvh, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / d**0.5
    causal = torch.ones(t, t, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(t, h * d)
    if gate is not None:
        out = out * torch.sigmoid(gate)
    return out @ attention.o_proj.weight.T, stream


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
    norm, attention = _setup(roles)
    runner = _runner(norm, attention, roles.config())
    hidden, residual, positions = _inputs(first)
    with torch.no_grad():
        got = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
        want = reference_forward(roles, norm, attention, hidden, residual, positions)
    torch.testing.assert_close(got, want)  # float32 defaults: atol 1e-5, rtol 1.3e-6


@pytest.mark.parametrize("first", [True, False], ids=["first_layer", "residual"])
@pytest.mark.parametrize("roles", ROLE_GRID, ids=_id)
def test_fused_equals_unfused(roles: Roles, first: bool) -> None:
    norm, attention = _setup(roles, torch.bfloat16)
    runner = _runner(norm, attention, roles.config(), "fused_qkv", reference_fused_qkv)
    hidden, residual, positions = _inputs(first, torch.bfloat16)
    before = runner_mod.FUSED_CALLS.get("layer", 0)
    with torch.no_grad():
        got = runner._fused_qkv(hidden, residual, positions)  # noqa: SLF001 - the value under test
        want = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
    torch.testing.assert_close(got, want, atol=FUSED_ATOL, rtol=FUSED_RTOL)
    assert runner_mod.FUSED_CALLS["layer"] == before + 1


def test_fused_op_that_declines_runs_unfused() -> None:
    """A fused op returning None for a call leaves that call to unfused."""
    roles = Roles()
    norm, attention = _setup(roles)
    runner = _runner(norm, attention, roles.config(), "fused_qkv", lambda *_: None)
    hidden, residual, positions = _inputs(first=False)
    with torch.no_grad():
        got = runner(hidden, residual, positions)
        want = runner._unfused(hidden, residual, positions)  # noqa: SLF001 - the value under test
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_runner_owns_no_parameters() -> None:
    """The parts are references: the runner adds nothing to a state_dict."""
    roles = Roles(qk_norm=True)
    norm, attention = _setup(roles)
    runner = _runner(norm, attention, roles.config())
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
    runner = _runner(
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
    norm, attention = _setup(roles)
    with pytest.raises(ValueError, match="fused_qkv needs a fused op"):
        _runner(norm, attention, roles.config(), "fused_qkv", None)


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
