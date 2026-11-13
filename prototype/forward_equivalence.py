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
"""Prints each declared layer's block forward against upstream's (README table).

Both forwards run on a CPU at TP 1 under each plan value and their residual
streams are compared; the check is ``tests/blocks/forward_equivalence.py`` and
``test_forward_equivalence.py`` asserts each layer's outcome::

    python -m prototype.forward_equivalence
"""

from tests.blocks import forward_equivalence

if __name__ == "__main__":  # str-ok: the module entry-point idiom
    forward_equivalence.report(print)
