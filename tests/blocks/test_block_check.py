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
"""The attn_then_moe block checks ``python -m prototype.check_block`` prints.

Each numbered check of ``block_check.py`` returns no failures: the declared
and planted layers conform or fail as expected, the plan table resolves (or
raises) per case, an outside write raises and a mismatch stops startup.
"""

import pytest

from tests.blocks import block_check
from tests.blocks._report import Check, discard


@pytest.mark.parametrize(
    "check", block_check.CHECKS, ids=[c.name for c in block_check.CHECKS]
)
def test_block_check_passes(check: Check) -> None:
    """The check finds nothing wrong."""
    assert check.run(discard) == []
