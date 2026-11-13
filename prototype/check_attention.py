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
"""Prints the prototype's three checks on the Qwen3NextAttention boundary.

The checks live in `tests/layers/attention_layer_plan/contract_check.py` and
are asserted by `test_attention_layer_contract.py`; this prints them. Each
check has a planted violation: comment one in below and rerun to see it
caught.

Run: PYTHONPATH=.:src:<vllm-v0.33.0> python -m prototype.check_attention
"""

# isort: off
from tests.layers.attention_layer_plan import cpu_layer  # noqa: F401  # pyright: ignore[reportUnusedImport]  # first: no TPU runtime

# isort: on
import sys

from tests.layers.attention_layer_plan import contract_check
from tests.layers.attention_layer_plan.contract_check import Planted

# Comment in one line to plant that violation (see `Planted` for each).
PLANTED: frozenset[Planted] = frozenset()
# PLANTED = frozenset({Planted.SHAPE})  # noqa: ERA001  # comment-in switch
# PLANTED = frozenset({Planted.WRITE})  # noqa: ERA001  # comment-in switch
# PLANTED = frozenset({Planted.SIGNATURE})  # noqa: ERA001  # comment-in switch
# PLANTED = frozenset({Planted.PARAM})  # noqa: ERA001  # comment-in switch
# PLANTED = frozenset({Planted.OWNER})  # noqa: ERA001  # comment-in switch

if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(contract_check.main(PLANTED))
