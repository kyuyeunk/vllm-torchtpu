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
"""CPU only: imports the contract modules with no TPU runtime.

These tests run on a CPU and must never touch a TPU. JAX (on its CPU backend),
torch and vLLM are allowed; the TPU runtime is not: `torch_tpu`, and the
`libtpu.so` shared library. (JAX imports the small `libtpu` Python package on
any host, only to locate that library, so the check looks at the library.)

`vllm_torchtpu/__init__.py` and `vllm_torchtpu/layers/__init__.py` import
`torch_tpu`, so the tests register empty stand-ins for those two packages and
let the normal file finder load only the modules under test. Run them with
the project's Python, e.g.

    .venv/bin/python -I -m unittest discover \
        -s tests/proto_contracts -t tests/proto_contracts
"""

import os
import pathlib
import sys
import types

# Before anything imports JAX or torch: CPU backends only.
os.environ.update(JAX_PLATFORMS="cpu", TORCH_DEVICE_BACKEND_AUTOLOAD="0")

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu"
_TPU_RUNTIME_MODULES = ("torch_tpu",)
_TPU_RUNTIME_LIBRARY = "libtpu.so"


def install() -> None:
    """Registers stand-ins for the two packages whose `__init__` loads torch_tpu."""
    for name, rel in (("vllm_torchtpu", ""), ("vllm_torchtpu.layers", "layers")):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(_SRC / rel)]
        sys.modules[name] = module


def loaded_tpu_runtime() -> list[str]:
    """The parts of the TPU runtime this process loaded.

    Returns:
        `torch_tpu` module names, plus the library path if `libtpu.so` is
        mapped into the process.
    """
    loaded = sorted(
        name for name in sys.modules if name.split(".")[0] in _TPU_RUNTIME_MODULES
    )
    with pathlib.Path("/proc/self/maps").open() as maps:
        loaded += sorted(
            {
                line.split()[-1]
                for line in maps
                if pathlib.Path(line.split()[-1]).name == _TPU_RUNTIME_LIBRARY
            }
        )
    return loaded


def jax_backend() -> str:
    """The backend JAX runs on; the tests assert it is the CPU."""
    import jax  # noqa: PLC0415 - after the environment is set above

    return jax.default_backend()


install()
