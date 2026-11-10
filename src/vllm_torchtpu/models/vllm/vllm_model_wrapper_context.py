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

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from jax.sharding import Mesh


@dataclass
class VllmModelWrapperContext:
    # TODO need discussion, ullm use this context to manager kv_caches
    # while vllm binds the KV cache to the attention layer
    mesh: Mesh
    kv_caches: list[torch.Tensor] | None = None
    layer_name_to_kvcache_index: dict[str, int] | None = None
    vllm_config: object | None = None
    # Full-model block-major KV cache bundle shared across layers. When set,
    # attention forward execution routes through the bundled RPA custom op with
    # in-place buffer donation, bypassing the per-layer tensor views bound by vLLM.
    kv_cache_bundle: torch.Tensor | None = None


_vllm_model_wrapper_context: VllmModelWrapperContext | None = None


def get_vllm_model_wrapper_context() -> VllmModelWrapperContext:
    assert _vllm_model_wrapper_context is not None, (
        "VllmModelWrapperContext is not set. "
        "Please use `set_vllm_model_wrapper_context` to set the "
        "VllmModelWrapperContext."
    )
    return _vllm_model_wrapper_context


def get_pallas_op_mesh() -> Mesh | None:
    """The wrapper-context mesh to hand to ``pallas.jax_op(mesh=...)``.

    torch_tpu (since 6c32c72bab, "enforces world-sized meshes for Pallas
    distributed kernels") rejects a kernel mesh that does not span the whole
    torch.distributed world, so passing the per-worker mesh fails every
    multi-chip run at op build time::

        ValueError: Pallas kernel mesh size (1) must match world size (16).

    That mesh is one device (see TPURunner._create_mesh_for_parallelism),
    which cannot describe cross-chip work anyway: without a mesh torch_tpu
    ignores the partition specs and runs the kernel on this rank's device,
    the same device the one-device mesh named. A multi-device mesh is passed
    through unchanged.
    """
    mesh = get_vllm_model_wrapper_context().mesh
    if mesh is None or mesh.size == 1:
        return None
    return mesh


# Cache for `get_pallas_op_for_mesh`, keyed by (base op name, kernel mesh).
_pallas_ops_by_mesh: dict[tuple[str, Mesh | None], Any] = {}


def get_pallas_op_for_mesh(
    name: str, register: Callable[[str, Mesh | None], Any]
) -> Any:
    """Returns the op registered as ``name`` for the current kernel mesh.

    On first use for a mesh, calls ``register(op_name, mesh)`` with the mesh
    from ``get_pallas_op_mesh``. A torch op name can be registered only once,
    so each mesh after the first gets ``name`` with a numeric suffix.

    Args:
        name: Base torch op name, e.g. ``"pallas::deepseek_v4_o_proj"``.
        register: Registers and returns the op under the given name on the
            given mesh.

    Returns:
        The registered op.
    """
    mesh = get_pallas_op_mesh()
    op = _pallas_ops_by_mesh.get((name, mesh))
    if op is None:
        clashes = sum(base == name for base, _ in _pallas_ops_by_mesh)
        op = register(f"{name}_{clashes}" if clashes else name, mesh)
        _pallas_ops_by_mesh[(name, mesh)] = op
    return op


@contextmanager
def set_vllm_model_wrapper_context(
    *,
    kv_caches: list[torch.Tensor] | None = None,
    mesh: Mesh,
    layer_name_to_kvcache_index: dict[str, int] | None = None,
    vllm_config: object | None = None,
    kv_cache_bundle: torch.Tensor | None = None,
):
    global _vllm_model_wrapper_context
    prev_context = _vllm_model_wrapper_context
    _vllm_model_wrapper_context = VllmModelWrapperContext(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        vllm_config=vllm_config,
        kv_cache_bundle=kv_cache_bundle,
    )

    try:
        yield
    finally:
        _vllm_model_wrapper_context = prev_context
