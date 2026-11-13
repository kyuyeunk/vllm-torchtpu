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
"""Pins vLLM's platform to CPU before any model code loads.

Importing this module keeps torch from autoloading the TPU backend and makes
vLLM's current platform the CPU one. ``tests/blocks/conftest.py`` imports it
for the tests, and ``prototype/__init__.py`` for the demo scripts, so both run
on a CPU and never load libtpu.
"""

import os

# The environment is the process boundary these switches cross.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # pyright: ignore[reportUnusedCallResult]  # str-ok: env var
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pyright: ignore[reportUnusedCallResult]  # str-ok: env var
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")  # pyright: ignore[reportUnusedCallResult]  # str-ok: env var

import vllm.platforms
from vllm.platforms.cpu import CpuPlatform

vllm.platforms._current_platform = CpuPlatform()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] - pin before use
