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
"""The proposals ``python -m prototype.suggest_blocks`` prints, verdict by verdict.

Each surveyed layer (``_build.SPECS``) gets the verdict the suggestion tool
recorded (its ``--json`` output).
"""

import pytest

from tests.blocks import _build
from tests.blocks._report import discard
from tests.blocks.suggest_blocks import Suggestion, suggest

# The tool's verdicts, in _build.SPECS order.
_RECORDED = (
    Suggestion.CONFORMS,  # Qwen3.5-MoE (full attention)
    Suggestion.CONFORMS,  # Qwen3.5-MoE (linear attention)
    Suggestion.CONFORMS,  # Qwen3-Next
    Suggestion.CONFORMS,  # Qwen3-MoE
    Suggestion.CONFORMS,  # Qwen2-MoE
    Suggestion.CONFORMS,  # GLM-4 MoE
    Suggestion.CONFORMS,  # Mixtral
    Suggestion.CONFORMS,  # MiniMax-M2
    Suggestion.CONFORMS,  # Llama4 (MoE layer)
    Suggestion.DECLINES,  # GPT-OSS
    Suggestion.CONFORMS,  # DeepSeek-V3
    Suggestion.CONFORMS,  # Bailing-MoE
    Suggestion.DECLINES,  # Kimi-Linear (nvidia)
    Suggestion.DECLINES,  # Jamba (attention layer)
    Suggestion.DECLINES,  # Jamba (Mamba layer)
    Suggestion.DECLINES,  # Nemotron-H (MoE layer)
)


@pytest.mark.parametrize(
    ("spec", "recorded"),
    list(zip(_build.SPECS, _RECORDED, strict=True)),
    ids=[s.name for s in _build.SPECS],
)
def test_verdict_is_recorded(spec: _build.Spec, recorded: Suggestion) -> None:
    """The layer's verdict is the recorded one."""
    assert suggest(spec, discard) is recorded
