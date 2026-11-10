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
"""Declared output specs vs `jax.eval_shape` of the kernels they describe.

Shapes come from each op's own tests.
"""

import jax
import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.layers.adapter import fused_moe, linear_common, router_topk
from vllm_torchtpu.layers.adapter.jax_op_spec import (
    check_fake_matches_kernel,
    jax_dtype_of,
    jax_op_with_fake,
    like,
    torch_dtype_of,
)


def _meta(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="meta")


# Qwen3-Coder-30B-A3B's local EP shard, as in test_fused_moe_gmm.py.
_HIDDEN, _INTERMEDIATE, _EXPERTS, _TOPK = 2048, 768, 16, 8


@pytest.mark.parametrize(
    ("num_tokens", "with_bias", "onehot_threshold"),
    [(4, True, 0), (64, False, 0), (256, True, 32)],
)
def test_fused_moe_spec(num_tokens, with_bias, onehot_threshold):
    bf16 = torch.bfloat16
    check_fake_matches_kernel(
        fused_moe._fused_moe_op,
        _meta(num_tokens, _HIDDEN, dtype=bf16),
        _meta(_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, dtype=bf16),
        _meta(_EXPERTS, _INTERMEDIATE, _HIDDEN, dtype=bf16),
        None,
        None,
        _meta(_EXPERTS, 1, 2 * _INTERMEDIATE, dtype=bf16) if with_bias else None,
        _meta(_EXPERTS, 1, _HIDDEN, dtype=bf16) if with_bias else None,
        _meta(num_tokens, _TOPK, dtype=torch.float32),
        _meta(num_tokens, _TOPK, dtype=torch.int32),
        _meta(dtype=torch.int32),
        _TOPK,
        "silu",
        False,  # use_ep
        False,  # use_sparse_core
        onehot_threshold,
        None,  # rhs_quant_dtype
        False,  # skip_padded_tokens
    )


@pytest.mark.parametrize("rows", [8, 1000, 4384])
def test_router_topk_spec(rows):
    check_fake_matches_kernel(
        router_topk._op, _meta(rows, 512, dtype=torch.float32), 10
    )


@pytest.mark.parametrize(
    ("rows", "w_s_shape", "w8a16_max_rows"),
    [
        (16, (128,), 0),  # per-channel W8A8, as in test_fp8_w8a16.py
        (16, (128,), 64),  # per-channel W8A16
        (64, (1, 2, 1, 128), 0),  # blockwise, 128-wide blocks via gmm_v2
    ],
)
def test_quantized_matmul_spec(rows, w_s_shape, w8a16_max_rows):
    check_fake_matches_kernel(
        linear_common._quantized_matmul_op,
        _meta(rows, 256, dtype=torch.bfloat16),
        _meta(256, 128, dtype=torch.float8_e4m3fn),
        _meta(*w_s_shape, dtype=torch.float32),
        w8a16_max_rows,
    )


@pytest.mark.cpu_test
def test_mismatched_spec_fails():
    @jax_op_with_fake("vt_jax_op_spec_test::row_sums", out=like("x"))
    def row_sums(x: jax.Array) -> jax.Array:
        return x.sum(axis=1)

    with pytest.raises(AssertionError, match="fake gives"):
        check_fake_matches_kernel(row_sums, _meta(4, 8, dtype=torch.float32))


@pytest.mark.cpu_test
def test_dtype_map_round_trips_and_reads_names():
    for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.int32, torch.bool):
        assert torch_dtype_of(jax_dtype_of(dtype)) == dtype
    assert torch_dtype_of("float8_e4m3fn") == torch.float8_e4m3fn
    assert torch_dtype_of(jnp.bfloat16) == torch.bfloat16
    with pytest.raises(ValueError, match="int4"):
        torch_dtype_of("int4")
