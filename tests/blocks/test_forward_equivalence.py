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
"""The comparison ``python -m prototype.forward_equivalence`` prints, per layer.

Every declared layer's block forward equals upstream's on a CPU (relative
difference of hidden + residual within ``TOLERANCE``) under each plan value;
the layers no model declares are reported as such.
"""

import pytest

from tests.blocks import _build
from tests.blocks.forward_equivalence import Outcome, check

# The recorded outcomes, in _build.SPECS order.
_RECORDED = (
    Outcome.EQUAL,  # Qwen3.5-MoE (full attention)
    Outcome.EQUAL,  # Qwen3.5-MoE (linear attention)
    Outcome.EQUAL,  # Qwen3-Next
    Outcome.EQUAL,  # Qwen3-MoE
    Outcome.EQUAL,  # Qwen2-MoE
    Outcome.EQUAL,  # GLM-4 MoE
    Outcome.EQUAL,  # Mixtral
    Outcome.EQUAL,  # MiniMax-M2
    Outcome.EQUAL,  # Llama4 (MoE layer)
    Outcome.NOT_DECLARED,  # GPT-OSS
    Outcome.EQUAL,  # DeepSeek-V3
    Outcome.EQUAL,  # Bailing-MoE
    Outcome.NOT_DECLARED,  # Kimi-Linear (nvidia)
    Outcome.NOT_DECLARED,  # Jamba (attention layer)
    Outcome.NOT_DECLARED,  # Jamba (Mamba layer)
    Outcome.NOT_DECLARED,  # Nemotron-H (MoE layer)
)


@pytest.mark.parametrize(
    ("spec", "recorded"),
    list(zip(_build.SPECS, _RECORDED, strict=True)),
    ids=[s.name for s in _build.SPECS],
)
def test_outcome_is_recorded(spec: _build.Spec, recorded: Outcome) -> None:
    """The layer's outcome is the recorded one."""
    outcome, _ = check(spec)
    assert outcome is recorded
