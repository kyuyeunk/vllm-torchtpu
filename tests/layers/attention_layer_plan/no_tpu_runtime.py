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
"""Keeps vLLM from detecting the TPU; import it before torch or vLLM.

It selects vLLM's CPU platform (no libtpu, no plugin platform) and maps the
plugin's packages without running their ``__init__`` files, which load the
TPU runtime. `cpu_layer` imports it first.
"""

import os
import pathlib
import sys
import types

_ = os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")  # str-ok: env var
_ = os.environ.setdefault("VLLM_PLUGINS", "")  # str-ok: env var
_ = os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # str-ok: env var
_ = os.environ.setdefault("JAX_PLATFORMS", "cpu")  # str-ok: env var
sys.modules["libtpu"] = None  # pyright: ignore[reportArgumentType]  # str-ok: None blocks the import

# The plugin's package __init__ files load the TPU runtime; the plan modules
# do not need them, so the packages are mapped without running them.
_SRC = pathlib.Path(__file__).resolve().parents[3] / "src" / "vllm_torchtpu"
for _name, _sub in (
    ("vllm_torchtpu", ""),
    ("vllm_torchtpu.layers", "layers"),
    ("vllm_torchtpu.models", "models"),
    ("vllm_torchtpu.models.vllm", "models/vllm"),
):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_SRC / _sub)]
        sys.modules[_name] = _module
