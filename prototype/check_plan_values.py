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
"""Prints the numerics tier's findings for the MoE combine, on the host's TPU.

    flock /tmp/vt-tpu.lock env PYTHONPATH=src python -m prototype.check_plan_values

The inputs, callbacks, tolerance and known findings are in
`tests/layers/core/moe_combine_plan_values_check.py`; the test that asserts the
findings equal the known set is `tests/layers/core/test_moe_combine_plan_check.py`.
"""

import jax

from tests.layers.core import moe_combine_plan_values_check


def main() -> None:
    """Prints the chip, then the findings with their error sizes."""
    if jax.default_backend() != "tpu":  # str-ok: JAX names its backend by string
        raise SystemExit(f"needs a TPU, got {jax.default_backend()}")
    findings = moe_combine_plan_values_check.value_findings()
    kind = jax.devices()[0].device_kind  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] - JAX types devices loosely
    print(f"chip: {kind}")
    for line in moe_combine_plan_values_check.lines(findings):
        print(line)


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
