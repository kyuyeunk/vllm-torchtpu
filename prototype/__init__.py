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
"""Prototype: a block-scope matcher for the attention-then-MoE overlap.

Everything here runs on a CPU. Importing the package keeps torch from
autoloading the TPU backend and pins vLLM's platform to CPU before any model
code loads.
"""

import os

# The environment is the process boundary these switches cross.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # str-ok: env var
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # str-ok: env var
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")  # str-ok: env var

import vllm.platforms  # noqa: E402 - after the environment above
from vllm.platforms.cpu import CpuPlatform  # noqa: E402

vllm.platforms._current_platform = CpuPlatform()  # noqa: SLF001 - pin before use
