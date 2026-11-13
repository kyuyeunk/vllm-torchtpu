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
"""Prints plan invariance for the attention layer, on a CPU.

The check lives in `tests/layers/attention_layer_plan/plan_check.py` and is
asserted by `test_attention_layer_plan_invariance.py`; this prints it.

Run: PYTHONPATH=.:src:<vllm-v0.33.0> python -m prototype.check_attention_plans
"""

# isort: off
from tests.layers.attention_layer_plan import cpu_layer  # noqa: F401  # pyright: ignore[reportUnusedImport]  # first: no TPU runtime

# isort: on
from tests.layers.attention_layer_plan import plan_check

if __name__ == "__main__":  # str-ok: the module entry-point idiom
    raise SystemExit(plan_check.main())
