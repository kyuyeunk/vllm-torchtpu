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
"""The sub-block checks ``python -m prototype.check_subblock`` prints.

Each numbered check of ``subblock_check.py`` returns no failures: every
family and current model conforms and matches upstream (fused where the
capability allows), planted declarations stop startup, and the installed
forward runs only code with source.
"""

import pytest

from tests.blocks import subblock_check
from tests.blocks._report import Check, discard


@pytest.mark.parametrize(
    "check", subblock_check.CHECKS, ids=[c.name for c in subblock_check.CHECKS]
)
def test_subblock_check_passes(check: Check) -> None:
    """The check finds nothing wrong."""
    assert check.run(discard) == []
