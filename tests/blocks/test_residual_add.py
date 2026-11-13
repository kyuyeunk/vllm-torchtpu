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
"""The fused value's ``residual_add`` field, with no model.

PR #3096's first binding (scratch bf66fb51b) added the residual in torch
before the kernel and returned that sum as the new residual; the fixed one
(dd9241d1d) passes the residual into the kernel, which returns the new
residual it wrote. Both compute the same output, so the choice is a plan
field (``ResidualAdd``): ``folded`` or ``outside``. These tests hold the
field to the plan's rules: both values give the same output (plan
invariance), the fused op declares the values it implements (the first
layer, which has no residual, runs ``outside`` only), the rule folds where
it can, and a pin that cannot be served stops startup.
"""

import dataclasses

import pytest
import torch
from torch import nn

from tests.blocks.synthetic import Roles, build_runner, make_inputs, setup_parts
from vllm_torchtpu.blocks.norm_attn import plan as plan_mod
from vllm_torchtpu.blocks.norm_attn import runner as runner_mod
from vllm_torchtpu.blocks.norm_attn.runner import (
    REFERENCE_CAPABILITY,
    AttentionRunner,
    AttnConfig,
    FusedCapability,
    InputProj,
    ResidualAdd,
    reference_fused_qkv,
)

Tensor = torch.Tensor
FOLDED, OUTSIDE = ResidualAdd.FOLDED, ResidualAdd.OUTSIDE

ROLES = [
    Roles(qk_norm=qk, output_gate=gate, gqa=gqa)
    for qk in (False, True)
    for gate in (False, True)
    for gqa in (1, 4)
]


def _id(roles: Roles) -> str:
    return f"qk{int(roles.qk_norm)}-gate{int(roles.output_gate)}-gqa{roles.gqa}"


@dataclasses.dataclass
class Kernel:
    """A stand-in for #3096's kernel, built with or without the residual.

    It computes what the kernel computes, from the same parts: with the
    residual it adds, normalizes and returns ``(out, new_residual)``;
    without, it normalizes its input and returns ``(out, input)``
    (``fused_attn_proj.py:353``). Each call's residual is recorded.
    """

    inner: AttentionRunner
    has_residual: bool
    residuals: list[Tensor | None] = dataclasses.field(
        default_factory=list[Tensor | None]
    )
    returned: list[Tensor] = dataclasses.field(default_factory=list[Tensor])

    def __call__(
        self, hidden: Tensor, residual: Tensor | None, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """``(hidden, residual | None, positions) -> (out, new residual)``."""
        if (residual is not None) != self.has_residual:
            msg = f"built with has_residual={self.has_residual}"
            raise AssertionError(msg)
        self.residuals.append(residual)
        out = self.inner(hidden, residual, positions)
        self.returned.append(out[1])
        return out


class Pr3096Binding:
    """#3096's kernel as the fused op, adding nothing itself.

    Called with the residual (``folded``) it runs the kernel built with
    ``has_residual=True`` and returns its new residual; called without
    (``outside``, the first layer) it runs the one built without.
    """

    def __init__(self, with_residual: Kernel | None, without: Kernel) -> None:
        """Keeps the kernels; ``with_residual`` None builds only the first."""
        self.with_residual: Kernel | None = with_residual
        self.without: Kernel = without

    @property
    def capability(self) -> FusedCapability:
        """The binding author's declaration: what its kernels implement."""
        values = {OUTSIDE} | ({FOLDED} if self.with_residual is not None else set())
        return FusedCapability(residual_add=frozenset(values), unreduced=False)

    def __call__(
        self,
        runner: AttentionRunner,
        hidden: Tensor,
        residual: Tensor | None,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        """``op(runner, hidden, residual, positions) -> (out, residual) | None``."""
        del runner
        if residual is None:
            return self.without(hidden, None, positions)
        if self.with_residual is None:  # The runner never calls it so.
            msg = "folded called on a binding that declares outside only"
            raise AssertionError(msg)
        return self.with_residual(hidden, residual, positions)


def _pr3096(
    norm: nn.Module, attention: nn.Module, roles: Roles, *, folded: bool = True
) -> Pr3096Binding:
    inner = build_runner(norm, attention, roles.config())  # pyright: ignore[reportArgumentType] - synthetic parts
    with_residual = Kernel(inner, has_residual=True) if folded else None
    return Pr3096Binding(with_residual, Kernel(inner, has_residual=False))


# Plan invariance: both values, both ops, every role configuration.


@pytest.mark.parametrize("first", [True, False], ids=["first_layer", "residual"])
@pytest.mark.parametrize("residual_add", list(ResidualAdd), ids=lambda v: v.value)
@pytest.mark.parametrize("roles", ROLES, ids=_id)
def test_both_values_equal_unfused(
    roles: Roles,
    residual_add: ResidualAdd,
    first: bool,  # noqa: FBT001 - a pytest parameter, passed by name
) -> None:
    """Folded and outside give unfused's output, with the reference and #3096's op.

    The add outside rounds the sum to the stream's dtype once, as the
    platform's fused add-norm does, so in bf16 all three agree exactly; a
    real kernel would be held to bf16 tolerance instead.
    """
    norm, attention = setup_parts(roles, torch.bfloat16)
    hidden, residual, positions = make_inputs(first, torch.bfloat16)
    unfused = build_runner(norm, attention, roles.config())
    pr3096 = _pr3096(norm, attention, roles)
    ops: tuple[tuple[runner_mod.FusedOp, FusedCapability], ...] = (
        (reference_fused_qkv, REFERENCE_CAPABILITY),
        (pr3096, pr3096.capability),
    )
    with torch.no_grad():
        want = unfused(hidden, residual, positions)
        for op, capability in ops:
            fused = build_runner(
                norm,
                attention,
                roles.config(),
                InputProj.FUSED_QKV,
                op,
                residual_add=residual_add,
                capability=capability,
            )
            got = fused(hidden, residual, positions)
            torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)


def test_folded_hands_the_residual_to_the_op_and_keeps_its_new_residual() -> None:
    """Folded: the kernel gets the residual and its new residual is the stream."""
    roles = Roles()
    norm, attention = setup_parts(roles, torch.bfloat16)
    pr3096 = _pr3096(norm, attention, roles)
    runner = build_runner(
        norm,
        attention,
        roles.config(),
        InputProj.FUSED_QKV,
        pr3096,
        residual_add=FOLDED,
        capability=pr3096.capability,
    )
    hidden, residual, positions = make_inputs(False, torch.bfloat16)  # noqa: FBT003 - make_inputs takes it positionally
    with torch.no_grad():
        _, stream = runner(hidden, residual, positions)
    kernel = pr3096.with_residual
    assert kernel is not None
    assert kernel.residuals == [residual]
    assert stream is kernel.returned[0]
    assert not pr3096.without.residuals


def test_outside_adds_before_the_op_and_keeps_its_own_sum() -> None:
    """Outside: the op gets the summed stream and no residual."""
    roles = Roles()
    norm, attention = setup_parts(roles, torch.bfloat16)
    pr3096 = _pr3096(norm, attention, roles)
    runner = build_runner(
        norm,
        attention,
        roles.config(),
        InputProj.FUSED_QKV,
        pr3096,
        residual_add=OUTSIDE,
        capability=pr3096.capability,
    )
    hidden, residual, positions = make_inputs(False, torch.bfloat16)  # noqa: FBT003 - make_inputs takes it positionally
    assert residual is not None
    with torch.no_grad():
        _, stream = runner(hidden, residual, positions)
    assert pr3096.without.residuals == [None]
    assert pr3096.with_residual is not None
    assert not pr3096.with_residual.residuals
    torch.testing.assert_close(
        stream, AttentionRunner.add_outside(hidden, residual), atol=0.0, rtol=0.0
    )


# Capability: what the op declares, and the first layer.


@pytest.mark.parametrize(
    ("value", "declared", "first_layer", "why"),
    [
        (FOLDED, frozenset({FOLDED, OUTSIDE}), False, None),
        (OUTSIDE, frozenset({FOLDED, OUTSIDE}), False, None),
        (FOLDED, frozenset({OUTSIDE}), False, "does not implement residual_add=folded"),
        (
            OUTSIDE,
            frozenset({FOLDED}),
            False,
            "does not implement residual_add=outside",
        ),
        (FOLDED, frozenset({FOLDED, OUTSIDE}), True, "first layer has no residual"),
        (OUTSIDE, frozenset({FOLDED, OUTSIDE}), True, None),
        (FOLDED, frozenset({FOLDED}), True, "first layer has no residual"),
        (OUTSIDE, frozenset({OUTSIDE}), True, None),
    ],
)
def test_residual_add_supports(
    value: ResidualAdd,
    declared: frozenset[ResidualAdd],
    first_layer: bool,  # noqa: FBT001 - a pytest parameter, passed by name
    why: str | None,
) -> None:
    """The first layer takes outside only; otherwise what the op declares."""
    got = plan_mod.residual_add_supports(value, declared, first_layer=first_layer)
    assert (got is None) == (why is None), got
    if why is not None and got is not None:
        assert why in got  # str-ok: the reason's text is the subject


def test_the_first_layer_runs_outside_under_a_folded_plan() -> None:
    """With no residual, the folded plan's call is outside's: no residual passed."""
    roles = Roles()
    norm, attention = setup_parts(roles, torch.bfloat16)
    pr3096 = _pr3096(norm, attention, roles)
    runner = build_runner(
        norm,
        attention,
        roles.config(),
        InputProj.FUSED_QKV,
        pr3096,
        residual_add=FOLDED,
        capability=pr3096.capability,
    )
    hidden, _, positions = make_inputs(True, torch.bfloat16)  # noqa: FBT003 - make_inputs takes it positionally
    with torch.no_grad():
        _, stream = runner(hidden, None, positions)
    assert pr3096.without.residuals == [None]
    assert stream is hidden


def test_the_first_layer_runs_unfused_with_a_folded_only_op() -> None:
    """An op that implements folded only cannot serve the first layer: unfused does."""
    roles = Roles()
    norm, attention = setup_parts(roles, torch.bfloat16)
    calls: list[Tensor | None] = []

    def folded_only(
        runner: AttentionRunner,
        hidden: Tensor,
        residual: Tensor | None,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        calls.append(residual)
        return reference_fused_qkv(runner, hidden, residual, positions)

    capability = FusedCapability(residual_add=frozenset({FOLDED}), unreduced=True)
    runner = build_runner(
        norm,
        attention,
        roles.config(),
        InputProj.FUSED_QKV,
        folded_only,
        residual_add=FOLDED,
        capability=capability,
    )
    unfused = build_runner(norm, attention, roles.config())
    hidden, _, positions = make_inputs(True, torch.bfloat16)  # noqa: FBT003 - make_inputs takes it positionally
    before = runner_mod.FUSED_CALLS.get(runner.layer_name, 0)
    with torch.no_grad():
        got = runner(hidden, None, positions)
        want = unfused(hidden, None, positions)
    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    assert not calls
    assert runner_mod.FUSED_CALLS.get(runner.layer_name, 0) == before


@pytest.mark.parametrize(
    ("residual_add", "capability", "match"),
    [
        (
            FOLDED,
            FusedCapability(frozenset({OUTSIDE}), unreduced=True),
            "residual_add=folded",
        ),
        (OUTSIDE, None, "needs its op's capability"),
    ],
    ids=["undeclared_value", "no_declaration"],
)
def test_a_runner_built_against_its_ops_declaration_raises(
    residual_add: ResidualAdd, capability: FusedCapability | None, match: str
) -> None:
    """The runner refuses a fused value its op does not declare."""
    roles = Roles()
    norm, attention = setup_parts(roles)
    binding = runner_mod.AttentionBinding(
        input_norm=norm,
        qkv_proj=attention.qkv_proj,
        attn=attention.attn,
        o_proj=attention.o_proj,
        rotary=attention.rotary_emb,
    )
    with pytest.raises(ValueError, match=match):
        AttentionRunner(
            "layer",
            binding,  # pyright: ignore[reportArgumentType] - synthetic parts
            config=roles.config(),
            input_proj=InputProj.FUSED_QKV,
            residual_add=residual_add,
            fused_op=reference_fused_qkv,
            fused_capability=capability,
        )


# The rule and the pins.


_FACTS = plan_mod.AttentionFacts(
    config=AttnConfig(8, 2, 128, qk_norm=False, output_gate=False),
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
_OUTSIDE_ONLY = frozenset({OUTSIDE})


def test_the_rule_folds_where_the_op_can() -> None:
    """Folded by default: it avoids the duplicated residual sum."""
    plan, declined = plan_mod.choose(plan_mod.Pins(), _FACTS)
    assert plan == plan_mod.NormAttnPlan(
        InputProj.FUSED_QKV, plan_mod.Matmul.BF16, FOLDED
    )
    assert not declined


def test_the_rule_adds_outside_where_the_op_cannot_fold() -> None:
    """An op without folded: the rule's value falls back to outside, quietly."""
    plan, declined = plan_mod.choose(
        plan_mod.Pins(), _FACTS, fused_residual_add=_OUTSIDE_ONLY
    )
    assert plan.input_proj is InputProj.FUSED_QKV
    assert plan.residual_add is OUTSIDE
    assert declined == [
        "fused_qkv/bf16/folded declined: the fused op does not implement "
        "residual_add=folded"
    ]


def test_a_pin_the_op_cannot_serve_raises() -> None:
    """``residual_add=folded`` on the fused value of an op without it stops startup.

    As with ``matmul``, a pin on the field alone also allows unfused, whose
    input norm takes the residual itself; pinning the fused value too
    leaves nothing to fall back to.
    """
    pins = plan_mod.parse_pins("input_proj=fused_qkv,residual_add=folded")
    with pytest.raises(ValueError, match="does not implement residual_add=folded"):
        plan_mod.choose(
            pins, _FACTS, where="Synthetic", fused_residual_add=_OUTSIDE_ONLY
        )
    alone = plan_mod.parse_pins("residual_add=folded")
    plan, _ = plan_mod.choose(alone, _FACTS, fused_residual_add=_OUTSIDE_ONLY)
    assert plan == plan_mod.UNFUSED


def test_a_pin_to_outside_wins_over_the_rule() -> None:
    """``residual_add=outside`` is honoured where the op could fold."""
    plan, _ = plan_mod.choose(plan_mod.parse_pins("residual_add=outside"), _FACTS)
    assert plan.residual_add is OUTSIDE
    assert plan.input_proj is InputProj.FUSED_QKV


def test_parse_pins() -> None:
    """The one parse point reads the field and prints it back."""
    pins = plan_mod.parse_pins("input_proj=fused_qkv,residual_add=outside")
    assert pins == plan_mod.Pins(InputProj.FUSED_QKV, None, OUTSIDE)
    assert plan_mod.parse_pins(str(pins)) == pins
    assert plan_mod.Pins(residual_add=FOLDED)
    assert not plan_mod.Pins()


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("residual_add=inside", "residual_add=inside; valid: folded, outside"),
        ("residual=folded", "valid: input_proj, matmul, residual_add"),
    ],
    ids=["bad_value", "bad_key"],
)
def test_parse_pins_errors(text: str, match: str) -> None:
    """A bad value or key lists the valid ones."""
    with pytest.raises(ValueError, match=match):
        plan_mod.parse_pins(text)
