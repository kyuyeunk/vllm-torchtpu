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
"""The survey ``python -m prototype.subblock_reach`` prints, verdict by verdict.

Each current model's verdict against norm_then_attention, from its real
``config.json``, is the one the README's reach table records.
"""

import pytest

from tests.blocks.subblock_reach import CANDIDATES, Candidate, Verdict, check

# The README's table, in CANDIDATES order.
_RECORDED = (
    Verdict.DECLINES,  # MiniMaxAI/MiniMax-M2.5
    Verdict.CONFORMS,  # zai-org/GLM-4.6
    Verdict.CONFORMS,  # zai-org/GLM-4.5-Air
    Verdict.CONFORMS,  # mistralai/Ministral-8B-Instruct-2410
    Verdict.CONFORMS,  # mistralai/Mistral-Small-3.2-24B-Instruct-2506
    Verdict.CONFORMS,  # openai/gpt-oss-20b
    Verdict.CONFORMS,  # meta-llama/Llama-3.3-70B-Instruct
    Verdict.DECLINES,  # meta-llama/Llama-4-Scout-17B-16E-Instruct
    Verdict.CONFORMS,  # microsoft/phi-4
    Verdict.NOT_BUILDABLE,  # ibm-granite/granite-4.0-h-small
    Verdict.CONFORMS,  # baidu/ERNIE-4.5-21B-A3B-PT
    Verdict.DECLINES,  # LGAI-EXAONE/EXAONE-4.0-32B
    Verdict.DECLINES,  # deepseek-ai/DeepSeek-V3.1
    Verdict.DECLINES,  # moonshotai/Kimi-K2-Instruct
    Verdict.CONFORMS,  # Qwen/Qwen3-30B-A3B
    Verdict.CONFORMS,  # Qwen/Qwen3-32B
    Verdict.CONFORMS,  # Qwen/Qwen3.5-35B-A3B
    Verdict.DECLINES,  # google/gemma-3-27b-it
    Verdict.DECLINES,  # google/gemma-4-31b-it
    Verdict.NOT_BUILDABLE,  # CohereForAI/c4ai-command-r-v01
)


@pytest.mark.parametrize(
    ("candidate", "recorded"),
    list(zip(CANDIDATES, _RECORDED, strict=True)),
    ids=[c.repo for c in CANDIDATES],
)
def test_verdict_is_recorded(candidate: Candidate, recorded: Verdict) -> None:
    """The candidate's verdict is the recorded one."""
    verdict, _ = check(candidate)
    assert verdict is recorded
