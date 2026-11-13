# Copyright 2025 Google LLC
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
"""Prints the handoff report of README section 2.

The check lives in `tests/layers/core/handoff_check.py`; the assertions in
`tests/layers/core/test_handoff.py`.
"""

from tests.layers.core import handoff_check


def main() -> None:
    """Prints one line per resolved-type receiver, then the violation count."""
    found = handoff_check.receivers()
    for receiver in found:
        print(receiver.line())
    print(f"violations: {sum(not r.ok for r in found)}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
