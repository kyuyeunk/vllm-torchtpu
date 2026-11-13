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
"""Prints the dead-field report of README section 3.

Loads libtpu, so run it under `flock /tmp/vt-tpu.lock`. The check lives in
`tests/layers/core/dead_fields_check.py`; the assertions in
`tests/layers/core/test_dead_fields.py`.
"""

from tests.layers.core import dead_fields_check


def main() -> None:
    """Prints the traced plan count and the fields no flip ever changes."""
    programs = dead_fields_check.programs_on_v6e()
    print(f"plans traced on v6e: {len(programs)}")
    print(f"fields never read: {dead_fields_check.fields_never_read(programs)}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
