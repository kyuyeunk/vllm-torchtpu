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
"""The Qwen3NextAttention boundary checks pass, and each planted break is caught.

Each planted violation trips exactly its own check, with the failure line the
prototype README records, and leaves the other two checks passing.
"""

from collections.abc import Callable

import pytest

from tests.layers.attention_layer_plan import contract_check
from tests.layers.attention_layer_plan.contract_check import Planted

_Check = Callable[[frozenset[Planted]], list[str]]
_CHECKS: tuple[_Check, ...] = (
    contract_check.check_structural,
    contract_check.check_outside_write,
    contract_check.check_guard,
)
_UNFUSED_EAGER = (
    "layer=unfused,qk_norm_rope_gate=eager,gate=eager_mul,o_proj_input=none"
)
_UNFUSED_TRITON = (
    "layer=unfused,qk_norm_rope_gate=fused_triton,gate=eager_mul,o_proj_input=none"
)
# (planted violation, the check it trips, that check's failure lines).
_CAUGHT: tuple[tuple[Planted, _Check, list[str]], ...] = (
    (
        Planted.SHAPE,
        contract_check.check_structural,
        [
            f"cuda gated=True {_UNFUSED_TRITON}: ((8, 64), 'float32')"
            " != ('num_tokens', 64) bfloat16"
        ],
    ),
    (
        Planted.WRITE,
        contract_check.check_outside_write,
        [
            "layer.use_fused_output_gate = True was accepted; the plan still says "
            + _UNFUSED_EAGER
        ],
    ),
    (
        Planted.SIGNATURE,
        contract_check.check_guard,
        [
            "signatures: Qwen3NextAttention.forward changed upstream;"
            " TpuQwen3NextAttention expects (self, positions, hidden_states),"
            " found (self, positions, hidden_states, residual=None)"
        ],
    ),
    (
        Planted.PARAM,
        contract_check.check_guard,
        ["parameter names: TpuQwen3NextAttention lacks parameters ['q_norm.weight']"],
    ),
    (
        Planted.OWNER,
        contract_check.check_guard,
        ["owner entry: expansion point Qwen3NextAttention has no owner"],
    ),
)


@pytest.mark.parametrize("check", _CHECKS, ids=[c.__name__ for c in _CHECKS])
def test_check_passes_unplanted(check: _Check) -> None:
    """With nothing planted, every check passes."""
    assert check(frozenset()) == []


@pytest.mark.parametrize(
    ("planted", "check", "want"),
    _CAUGHT,
    ids=[f"{p.value}-{c.__name__}" for p, c, _ in _CAUGHT],
)
def test_planted_violation_is_caught(
    planted: Planted, check: _Check, want: list[str]
) -> None:
    """The planted violation's own check reports it."""
    assert check(frozenset({planted})) == want


_UNTRIPPED = [
    pytest.param(planted, other, id=f"{planted.value}-{other.__name__}")
    for planted, tripped, _ in _CAUGHT
    for other in _CHECKS
    if other is not tripped
]


@pytest.mark.parametrize(("planted", "check"), _UNTRIPPED)
def test_planted_violation_leaves_other_checks(planted: Planted, check: _Check) -> None:
    """A planted violation does not trip the checks it does not target."""
    assert check(frozenset({planted})) == []
