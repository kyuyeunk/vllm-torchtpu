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
"""The TPU chip as one immutable value (data contract C0).

Plans take a `TpuTarget` instead of calling `pltpu.get_tpu_info()` in the
middle of a decision, so a decision can be tested on CPU against synthetic
chips. This module imports no JAX or TPU runtime at module level; only
`TpuTarget.from_runtime` touches the runtime, and it imports JAX lazily.

The value holds no rank, mesh position or other per-process fact: two ranks
on the same chip must build equal values, or a plan keyed on it would compile
structurally different programs per rank.
"""

import dataclasses
import functools
from typing import Any


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SparseCoreInfo:
    """SparseCore geometry, per TensorCore, as reported by the runtime.

    Attributes:
        num_cores: SparseCores per TensorCore.
        num_subcores: Vector subcores per SparseCore.
        num_lanes: SIMD lanes per vector subcore.
    """

    num_cores: int
    num_subcores: int
    num_lanes: int

    def __post_init__(self) -> None:
        if min(self.num_cores, self.num_subcores, self.num_lanes) <= 0:
            raise ValueError(f"SparseCore geometry must be positive: {self}")

    @property
    def block_rows(self) -> int:
        """Rows one SparseCore pass covers: lanes x cores x subcores."""
        return self.num_lanes * self.num_cores * self.num_subcores


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TpuTarget:
    """Per-TensorCore chip facts the MoE plans read.

    All sizes are per TensorCore, matching `pltpu.TpuInfo`.

    Attributes:
        generation: TPU generation number (e.g. 6 for v6e, 7 for TPU7x).
        tensorcores_per_device: TensorCores in one logical device.
        vmem_bytes_per_core: TensorCore VMEM capacity in bytes.
        num_lanes: TensorCore vector lanes.
        num_sublanes: TensorCore vector sublanes. Not in the design sketch;
            needed to reproduce `TpuInfo.get_sublane_tiling`.
        mxu_column_size: MXU column width.
        sparse_core: SparseCore geometry, or None on chips without one.
    """

    generation: int
    tensorcores_per_device: int
    vmem_bytes_per_core: int
    num_lanes: int
    num_sublanes: int
    mxu_column_size: int
    sparse_core: SparseCoreInfo | None

    def sublane_tiling(self, bitwidth: int) -> int:
        """Sublane tiling for an element bit width.

        A copy of `jax._src.tpu_info.TpuInfo.get_sublane_tiling` (jax 0.11.2),
        including its `NotImplementedError` past generation 8.

        Args:
            bitwidth: Element width in bits.

        Returns:
            The second-minor tile size XLA uses for that width.

        Raises:
            NotImplementedError: For generations JAX does not describe.
        """
        if self.generation < 7:
            if bitwidth in (16, 32):
                return self.num_sublanes
            return self.num_sublanes * (32 // bitwidth)
        if self.generation in (7, 8):
            return self.num_sublanes * (32 // bitwidth)
        raise NotImplementedError("TPU generation is not supported")

    @classmethod
    def from_tpu_info(cls, info: Any) -> "TpuTarget":
        """Adapts a `pltpu.TpuInfo` (or any object with its attributes).

        Missing integer attributes read as 0. Real `TpuInfo` values have all of
        them; the tolerance exists because existing tests fake `get_tpu_info`
        with a `SimpleNamespace` holding only the fields the code under test
        used to read. No plan reads a field that such a fake leaves out on the
        paths those tests exercise.

        Args:
            info: A `pltpu.TpuInfo`, or a duck-typed stand-in.

        Returns:
            The equivalent `TpuTarget`.
        """
        sc = getattr(info, "sparse_core", None)
        return cls(
            generation=int(getattr(info, "generation", 0)),
            tensorcores_per_device=int(getattr(info, "num_cores", 0)),
            vmem_bytes_per_core=int(getattr(info, "vmem_capacity_bytes", 0)),
            num_lanes=int(getattr(info, "num_lanes", 0)),
            num_sublanes=int(getattr(info, "num_sublanes", 0)),
            mxu_column_size=int(getattr(info, "mxu_column_size", 0)),
            sparse_core=(
                None
                if sc is None
                else SparseCoreInfo(
                    num_cores=int(sc.num_cores),
                    num_subcores=int(sc.num_subcores),
                    num_lanes=int(sc.num_lanes),
                )
            ),
        )

    @classmethod
    def from_runtime(cls) -> "TpuTarget":
        """Reads the chip from the initialized JAX runtime.

        Not cached here: `pltpu.get_tpu_info` already caches per trace
        context, and looking it up at call time keeps test patches of
        `pltpu.get_tpu_info` effective.

        Returns:
            The chip this process runs on.

        Raises:
            ValueError: On a device kind JAX does not describe (e.g. a CPU
                host), exactly as `pltpu.get_tpu_info` does.
        """
        from jax.experimental.pallas import tpu as pltpu

        return cls.from_tpu_info(pltpu.get_tpu_info())

    @classmethod
    def runtime_or_none(cls) -> "TpuTarget | None":
        """`from_runtime`, or None where `get_tpu_info` raises ValueError.

        Returns:
            The chip, or None on hosts JAX has no TPU description for.
        """
        try:
            return cls.from_runtime()
        except ValueError:
            # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
            return None


@functools.cache
def current() -> TpuTarget:
    """The process's chip, read once and cached.

    For host-side callers that run after device initialization and outside
    any abstract-mesh context. Trace-time callers in this scratch build use
    `TpuTarget.from_runtime` instead (see its docstring). A ValueError from a
    non-TPU host is not cached.

    Returns:
        The chip this process runs on.
    """
    return TpuTarget.from_runtime()
