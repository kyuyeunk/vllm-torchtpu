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
recorded. On this branch the tool raises for three layers (the demo exits 1
at the first, DeepSeek-V3): the conformance stub does not bind DeepSeek's
``llama_4_scaling`` or Jamba Mamba's ``output``, and Jamba's attention layer
has no RowParallelLinear to propose. Those are strict xfails naming the error.
"""

import pytest

from tests.blocks import _build
from tests.blocks._report import discard
from tests.blocks.suggest_blocks import Suggestion, suggest

# The tool's verdicts in _build.SPECS order, or the error it raises first.
_RECORDED: tuple[Suggestion | type[Exception], ...] = (
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
    TypeError,  # DeepSeek-V3: the stub does not bind llama_4_scaling
    Suggestion.DECLINES,  # Bailing-MoE
    Suggestion.DECLINES,  # Kimi-Linear (nvidia)
    IndexError,  # Jamba (attention layer): no RowParallelLinear in attention
    TypeError,  # Jamba (Mamba layer): the stub does not bind output
    Suggestion.NOT_BUILT,  # Nemotron-H (MoE layer)
)


def _case(spec: _build.Spec, recorded: Suggestion | type[Exception]) -> object:
    if isinstance(recorded, Suggestion):
        return pytest.param(spec, recorded, id=spec.name)
    raises = pytest.mark.xfail(raises=recorded, strict=True, reason="tool raises")
    return pytest.param(spec, None, id=spec.name, marks=raises)


@pytest.mark.parametrize(
    ("spec", "recorded"),
    [_case(s, r) for s, r in zip(_build.SPECS, _RECORDED, strict=True)],
)
def test_verdict_is_recorded(spec: _build.Spec, recorded: Suggestion | None) -> None:
    """The layer's verdict is the recorded one (or the recorded error)."""
    assert suggest(spec, discard) is recorded
