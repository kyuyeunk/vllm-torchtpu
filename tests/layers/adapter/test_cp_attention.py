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
"""CPU tests for DCP layout propagation into the DCP op's static arguments.

Real TPU pass correctness is covered by test_attention_dcp_hnd.py. Here the
DCP op and collectives are replaced so the static arguments that both adapter
entry points bind can be checked across layers and layouts.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import (
    DeviceConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import (
    AttentionScope,
    KVLayout,
)
from vllm_torchtpu.layers.adapter import attention, cp_attention
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


@pytest.fixture
def cpu_vllm_config_context():
    # CPU CI does not register torch's TPU device type. These tests only
    # exercise layout propagation with CPU tensors and mocked kernels.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl(cls=attention.PallasBatchedRPAAttentionBackendImpl, head_size=256):
    return cls(
        num_heads=8,
        head_size=head_size,
        scale=head_size**-0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=2,
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


def _group(rank):
    return SimpleNamespace(
        world_size=2,
        rank_in_group=rank,
        all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim),
        reduce_scatter=lambda tensor, dim: (2 * tensor).chunk(2, dim)[rank],
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("compiled", [False, True])
def test_dcp_layout_survives_prebuild_forward_and_registry_reuse(
    monkeypatch, rank, compiled, cpu_vllm_config_context
):
    calls = []

    def fake_op(cache, query, *args, **static_kwargs):
        scope = static_kwargs["attention_scope"]
        calls.append((static_kwargs["kv_layout"], scope, query.shape))
        value = 1 if scope == AttentionScope.CACHE_ONLY else 3
        return cache, torch.full_like(query, value), torch.zeros(query.shape[:2])

    group = _group(rank)
    monkeypatch.setattr(attention, "_get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_OP", fake_op)
    monkeypatch.setattr(cp_attention, "synchronize_tensors", lambda *a, **k: None)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    implementations = []
    # The third, NHD layer must not inherit the HND layer's static arguments.
    for layout in ("LBNHC", "LBHNC", "LBNHC"):
        get_current_vllm_config().cache_config.kv_cache_layout = layout
        impl = _impl()
        with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
            impl.initialize_kernel(layer)
        implementations.append(impl)

    def statics(impl):
        return (
            impl.rpa_dcp_cache_kernel.keywords,
            impl.rpa_dcp_new_kernel.keywords,
        )

    # Layers with the same layout bind the same static arguments.
    assert statics(implementations[0]) == statics(implementations[2])
    assert statics(implementations[0]) != statics(implementations[1])

    metadata = SimpleNamespace(
        seq_lens=torch.tensor([2, 3, 4], dtype=torch.int32),
        block_tables=torch.zeros(3, 2, dtype=torch.int32),
        query_start_loc=torch.arange(4, dtype=torch.int32),
        request_distribution=torch.tensor([3, 3, 3], dtype=torch.int32),
    )
    query = torch.ones(3, 8, 256)
    key = torch.ones(3, 1, 256)
    for impl, expected_layout in zip(
        implementations,
        (
            KVLayout.HEAD_ALONG_SUBLANE,
            KVLayout.SEQ_ALONG_LANE,
            KVLayout.HEAD_ALONG_SUBLANE,
        ),
    ):
        # The per-layer layout must survive even if the current config changes.
        get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
        forward = impl._run_dcp_forward
        if compiled:
            forward = torch.compile(forward, backend="eager", fullgraph=True)
        output = forward(layer, query, key, key, torch.empty(1), metadata)
        assert calls[-2:] == [
            (expected_layout, AttentionScope.CACHE_ONLY, (3, 16, 256)),
            (expected_layout, AttentionScope.NEW_TOKENS_ONLY, (3, 8, 256)),
        ]
        # Two cached partials (value=1, weight=1 each) and one new partial
        # (value=3, weight=1), including both production LSE merges.
        torch.testing.assert_close(output, torch.full_like(query, 5 / 3))
    assert all(
        type(keywords["kv_layout"]) is KVLayout
        for impl in implementations
        for keywords in statics(impl)
    )
    assert all(type(layout) is KVLayout for layout, _, _ in calls)


def test_dcp_kernel_default_layout_reuses_explicit_nhd():
    kwargs = dict(
        sliding_window=None,
        sm_scale=0.0625,
        logits_soft_cap=None,
        q_scale=None,
        k_scale=None,
        v_scale=None,
        cp_group_size=2,
        cp_rank=cp_attention.make_cp_rank_tensor(0, "cpu"),
    )
    implicit = cp_attention.build_dcp_kernels(**kwargs)
    explicit = cp_attention.build_dcp_kernels(
        **kwargs, kv_layout=KVLayout.HEAD_ALONG_SUBLANE
    )
    assert all(a.keywords == b.keywords for a, b in zip(implicit, explicit))


# Rank operands the sink op saw, in call order. The op body runs at call time
# in both eager and compiled forwards, so it records the runtime value even
# when one compiled graph serves several ranks.
_SUNK_RANKS: list[list[int]] = []


@torch.library.custom_op("vt_test::dcp_rank_sink", mutates_args=())
def _dcp_rank_sink(query: torch.Tensor, cp_rank: torch.Tensor) -> torch.Tensor:
    """Stands in for the DCP op's use of the rank: an opaque op operand."""
    _SUNK_RANKS.append(cp_rank.tolist())
    return torch.zeros_like(query)


@_dcp_rank_sink.register_fake
def _(query, cp_rank):
    del cp_rank  # Unused by the fake.
    return torch.empty_like(query)


@pytest.mark.parametrize("compiled", [False, True])
def test_dcp_ranks_share_static_arguments(
    monkeypatch, compiled, cpu_vllm_config_context
):
    """Both DCP ranks call the op with identical static arguments.

    The rank differs only in the tensor operand, which a compiled forward must
    take as a graph input rather than fold into the graph as a constant. The
    fake op feeds the rank to an opaque op, as the real op does: a tensor the
    graph never consumes is pruned from the inputs whether or not it is a
    constant.
    """
    statics = []
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    def fake_op(cache, query, *args, **static_kwargs):
        if not torch.compiler.is_compiling():
            statics.append(static_kwargs)
        lse = torch.zeros(query.shape[:2])
        return cache, _dcp_rank_sink(query, args[-1]), lse

    # One group for both ranks, with rank-independent collectives, so only the
    # rank operand differs between the ranks' forwards.
    group = SimpleNamespace(
        world_size=2,
        rank_in_group=0,
        all_gather=lambda tensor, dim: torch.cat([tensor, tensor], dim=dim),
        reduce_scatter=lambda tensor, dim: tensor.chunk(2, dim)[0],
    )
    monkeypatch.setattr(attention, "_get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "get_dcp_group", lambda: group)
    monkeypatch.setattr(cp_attention, "_DCP_KERNEL_OP", fake_op)
    monkeypatch.setattr(cp_attention, "synchronize_tensors", lambda *a, **k: None)
    _SUNK_RANKS.clear()
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    metadata = SimpleNamespace(
        seq_lens=torch.tensor([2, 3, 4], dtype=torch.int32),
        block_tables=torch.zeros(3, 2, dtype=torch.int32),
        query_start_loc=torch.arange(4, dtype=torch.int32),
        request_distribution=torch.tensor([3, 3, 3], dtype=torch.int32),
    )
    query = torch.ones(3, 8, 256)
    key = torch.ones(3, 1, 256)

    def run(impl):
        return impl._run_dcp_forward(layer, query, key, key, torch.empty(1), metadata)

    if compiled:
        run = torch.compile(run, backend=backend, fullgraph=True)
    for rank in (0, 1):
        group.rank_in_group = rank
        impl = _impl()
        with set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()):
            impl.initialize_kernel(layer)
        assert impl.rpa_dcp_cache_kernel.args[0].tolist() == [rank]
        run(impl)

    assert _SUNK_RANKS == [[0], [0], [1], [1]]
    if compiled:
        # One graph serves both ranks: a Python-int rank anywhere in the
        # forward would fail a guard and recompile, and the sinks read the
        # rank from a graph input.
        assert len(graphs) == 1
        sinks = [
            node
            for node in graphs[0].graph.nodes
            if node.target is torch.ops.vt_test.dcp_rank_sink.default
        ]
        assert len(sinks) == 2
        assert all(node.args[1].op == "placeholder" for node in sinks)
    else:
        assert len(statics) == 4
        assert statics[:2] == statics[2:]


@pytest.mark.parametrize(
    "cls,head_size",
    [
        (attention.PallasAttentionBackendImpl, 256),
        (attention.PallasBatchedRPAAttentionBackendImpl, 64),
    ],
)
def test_dcp_hnd_rejects_backends_that_allocate_nhd(
    cls, head_size, cpu_vllm_config_context
):
    get_current_vllm_config().cache_config.kv_cache_layout = "LBHNC"
    impl = _impl(cls, head_size)
    layer = SimpleNamespace(_k_scale_float=0.5, _v_scale_float=0.25)
    with (
        set_vllm_model_wrapper_context(mesh=None, vllm_config=_config()),
        pytest.raises(NotImplementedError, match="DCP with HND requires"),
    ):
        impl.initialize_kernel(layer)
