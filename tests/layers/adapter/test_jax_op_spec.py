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
"""Spec-built fakes from `jax_op_spec`, evaluated on meta and fake tensors."""

import jax
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

from vllm_torchtpu.layers.adapter.jax_op_spec import (
    dim,
    jax_op_with_fake,
    like,
    like_dtype,
    make_fake,
    registered_op,
    spec,
    static,
)

pytestmark = pytest.mark.cpu_test


def _meta(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device="meta")


def _kernel(x: jax.Array, w: jax.Array, topk: int, block: int = 4) -> jax.Array:
    del w, topk, block
    return x


def test_like_keeps_shape_and_dtype_of_any_rank():
    fake = make_fake(_kernel, like("x"))
    for shape in [(7,), (3, 5), (2, 3, 4, 5)]:
        out = fake(_meta(*shape, dtype=torch.float16), _meta(1), 2)
        assert out.shape == shape
        assert out.dtype == torch.float16
        assert out.device.type == "meta"


def test_spec_mixes_literals_dims_and_statics():
    fake = make_fake(
        _kernel,
        spec(
            shape=(1, dim("x", 0), "topk", dim("w", -1)),
            dtype=like_dtype("w"),
        ),
    )
    out = fake(_meta(6, 8), _meta(8, 32, dtype=torch.float32), 3)
    assert out.shape == (1, 6, 3, 32)
    assert out.dtype == torch.float32


def test_arithmetic_on_dims_and_statics():
    fake = make_fake(
        _kernel,
        spec(
            shape=(
                dim("x", 0) * "topk",
                dim("w", 0) // static("block") + 1,
                2 * static("topk") - 1,
                64 // static("block"),
            ),
            dtype=torch.int32,
        ),
    )
    out = fake(_meta(5, 8), _meta(16, 2), 3, block=8)
    assert out.shape == (15, 3, 5, 8)


def test_defaults_bind_as_statics():
    fake = make_fake(_kernel, spec(shape=(dim("x", 0), "block"), dtype=torch.float32))
    assert fake(_meta(5, 8), _meta(1), 3).shape == (5, 4)
    assert fake(_meta(5, 8), _meta(1), topk=3, block=16).shape == (5, 16)


def test_tuple_of_specs_returns_one_tensor_per_output():
    fake = make_fake(
        _kernel,
        (
            spec(shape=(dim("x", 0), "topk"), dtype=torch.float32),
            (like("w"), spec(shape=(), dtype=torch.int32)),
        ),
    )
    weights, (w_like, scalar) = fake(_meta(9, 4), _meta(2, 3), 5)
    assert weights.shape == (9, 5)
    assert weights.dtype == torch.float32
    assert w_like.shape == (2, 3)
    assert scalar.shape == ()
    assert scalar.dtype == torch.int32


def test_device_comes_from_the_first_tensor_argument():
    def kernel(x: jax.Array | None, y: jax.Array) -> jax.Array:
        del x
        return y

    fake = make_fake(kernel, like("y"))
    assert fake(None, torch.empty(2, 3)).device.type == "cpu"


def test_callable_spec_is_the_fake():
    def custom(x, w, topk, block=4):
        del w, topk, block
        return x.new_empty(x.shape[::-1])

    assert make_fake(_kernel, custom) is custom


def test_symbolic_token_dimension_propagates():
    mode = FakeTensorMode(shape_env=ShapeEnv())
    context = StatelessSymbolicContext(
        dynamic_sizes=[DimDynamic.DYNAMIC, DimDynamic.STATIC]
    )
    x = mode.from_tensor(torch.empty(16, 8), symbolic_context=context)
    w = mode.from_tensor(torch.empty(8, 32))
    fake = make_fake(
        _kernel, spec(shape=(dim("x", 0), dim("w", 1)), dtype=torch.float32)
    )
    with mode:
        out = fake(x, w, 2)
    assert isinstance(out.shape[0], torch.SymInt)
    assert out.shape[1] == 32


def test_decorator_registers_op_and_fake():
    @jax_op_with_fake(
        "vt_jax_op_spec_test::scaled_rows",
        out=spec(shape=(dim("x", 0), "cols"), dtype=like_dtype("x")),
    )
    def scaled_rows(x: jax.Array, cols: int) -> jax.Array:
        return x[:, :cols] * 2

    out = scaled_rows(_meta(4, 16), 8)
    assert out.shape == (4, 8)
    assert out.dtype == torch.bfloat16
    record = registered_op(scaled_rows)
    assert list(record.signature.parameters) == ["x", "cols"]


def test_registered_op_rejects_other_ops():
    @torch.library.custom_op("vt_jax_op_spec_test::plain", mutates_args=())
    def plain(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    with pytest.raises(ValueError, match="not registered"):
        registered_op(plain)
