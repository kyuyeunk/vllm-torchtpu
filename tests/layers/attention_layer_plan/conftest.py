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
"""Shared setup for the attention-layer plan checks.

`cpu_layer` is imported first, before any test module, so vLLM resolves the
CPU platform and the plugin's TPU runtime never loads. The `tpu` and
`compile_tier` markers are skipped by `tests/conftest.py` unless
``--run-tpu`` or ``--run-compile-tier`` is passed.
"""

# isort: off
from tests.layers.attention_layer_plan import cpu_layer  # noqa: F401  # pyright: ignore[reportUnusedImport]  # first: no TPU runtime

# isort: on
from collections.abc import Iterator

import pytest

from tests.layers.attention_layer_plan import layer_builder


@pytest.fixture(scope="session", autouse=True)
def _runtime() -> Iterator[None]:
    """Opens the single-rank world every layer build needs, once per session."""
    with layer_builder.runtime():
        yield
