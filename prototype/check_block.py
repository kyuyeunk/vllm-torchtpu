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
"""Prints the attn_then_moe block checks on a CPU (README, "Running it").

Conformance, plan per bucket, outside writes and a declared mismatch, from
``tests/blocks/block_check.py``; ``test_block_check.py`` asserts the same
checks. Exits 1 if any check fails::

    python -m prototype.check_block
"""

import sys

from tests.blocks import block_check
from tests.blocks._report import run_checks

if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(0 if run_checks(block_check.CHECKS, print) else 1)
