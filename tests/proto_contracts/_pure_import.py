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
"""Imports the pure contract modules without the package `__init__`s.

`vllm_torchtpu/__init__.py` and `vllm_torchtpu/layers/__init__.py` import
torch, torch_tpu and vLLM. The contract modules import none of them, so the
tests register empty namespace stand-ins for the parent packages and let the
normal file finder load only the modules under test. Run these tests with the
standard library only, e.g.

    python3.12 -I -m unittest discover -s tests/proto_contracts -t tests/proto_contracts
"""

import pathlib
import sys
import types

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu"
_FORBIDDEN = ("torch", "torch_tpu", "jax", "jaxlib", "vllm", "libtpu")


def install() -> None:
    """Registers stand-ins for the parent packages of the contract modules."""
    for name, rel in (
        ("vllm_torchtpu", ""),
        ("vllm_torchtpu.layers", "layers"),
        ("vllm_torchtpu.layers.core", "layers/core"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / rel)]
        sys.modules[name] = module


def loaded_runtime_modules() -> list[str]:
    """Names of any heavy runtime modules that got imported.

    Returns:
        Module names from torch, torch_tpu, JAX, vLLM or libtpu.
    """
    return sorted(name for name in sys.modules if name.split(".")[0] in _FORBIDDEN)


install()
