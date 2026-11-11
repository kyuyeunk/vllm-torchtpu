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
"""The merge-time conformance check, and the block forward's collectives.

``test_every_declaration_conforms`` is the merge gate: every declaration in
``models/vllm/*/registration.py`` must conform. The forward tests port PR
#1492's call-order and warmup-gate tests to the parts-based forward.
"""

import pytest
import torch
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.model_executor.models.qwen3_5 import Qwen3_5RMSNorm

from prototype import _build
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks.attn_moe import conform, forward
from vllm_torchtpu.blocks.attn_moe import pattern as pattern_mod
from vllm_torchtpu.blocks.attn_moe import plan as plan_mod
from vllm_torchtpu.blocks.stream import (
    attend_with_positions,
    hidden_residual_positions,
)
from vllm_torchtpu.compilation import shape_variants


@pytest.mark.parametrize("spec", _build.SPECS[:2], ids=lambda s: s.name)
def test_every_declaration_conforms(spec: _build.Spec) -> None:
    layer = _build.build(spec)
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS[type(layer)]
    assert declaration.parts(layer) is not None
    with set_current_vllm_config(_build.current_config()):
        parts = conform(layer, declaration)
    assert any(m is parts.out_proj for m in parts.attention.modules())


class _CpuRMSNorm(Qwen3_5RMSNorm):
    def dispatch_forward(self, compile_native: bool = False):  # noqa: ANN201, ARG002
        return self.forward_native


class _Fn(nn.Module):
    def __init__(self, fn) -> None:  # noqa: ANN001
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return self.fn(*args, **kwargs)


def _layer(
    d: int, tp: int, table: plan_mod.BlockPlanTable, gate_rows: list[int]
) -> nn.Module:
    layer = forward.AttentionMoEBlock()
    layer.input_layernorm = _CpuRMSNorm(d)
    layer.post_attention_layernorm = _CpuRMSNorm(d)
    layer.self_attn = _Fn(lambda hidden_states, positions: hidden_states)
    layer.mlp = nn.Module()
    layer.mlp.gate = _Fn(
        lambda h: (gate_rows.append(h.shape[0]), torch.zeros(h.shape[0], 4), None)[1:]
    )
    layer.mlp.experts = _Fn(lambda hidden_states, router_logits: hidden_states)
    parts = pattern_mod.ForwardParts(
        input_norm=layer.input_layernorm,
        post_norm=layer.post_attention_layernorm,
        router=layer.mlp.gate,
        experts=layer.mlp.experts,
        stream=hidden_residual_positions(layer),
        attend=attend_with_positions(layer.self_attn),
    )
    layer.bind_block(parts, table, tp)
    return layer


def _record(monkeypatch: pytest.MonkeyPatch, calls: list[str], tp: int) -> None:
    monkeypatch.setattr(
        forward,
        "tensor_model_parallel_reduce_scatter",
        lambda x, dim=0: (calls.append("rs"), x[: x.shape[0] // tp])[1],
    )
    monkeypatch.setattr(
        forward,
        "tensor_model_parallel_all_gather",
        lambda x, dim=0: (calls.append("ag"), torch.cat([x] * tp, dim=0))[1],
    )
    monkeypatch.setattr(
        forward,
        "tensor_model_parallel_all_reduce",
        lambda x: (calls.append("ar"), x)[1],
    )


def test_call_order_by_bucket_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rule's table picks ["ar", "ar"] or #1492's ["rs", "ag", "ag", "ar"]."""
    calls: list[str] = []
    gate_rows: list[int] = []
    tp, d = 2, 4096
    _record(monkeypatch, calls, tp)
    key = pattern_mod.BlockKey(
        tp=tp,
        hidden=d,
        dtype=torch.bfloat16,
        attention_kind=nn.Module,
        router_mode=plan_mod.RouterMode.EXPLICIT,
        has_shared_experts=True,
        sp_moe=False,
        fused_output_reduced=False,
        moe_path=plan_mod.MoEPath.LATE_ALL_REDUCE,
    )
    table, _ = plan_mod.resolve_table(key, (2047, 2048))
    layer = _layer(d, tp, table, gate_rows)
    layer(torch.ones((2047, d), dtype=torch.bfloat16), None, torch.arange(2047))
    assert (calls, gate_rows) == (["ar", "ar"], [2047])
    calls.clear()
    gate_rows.clear()
    layer(torch.ones((2048, d), dtype=torch.bfloat16), None, torch.arange(2048))
    assert (calls, gate_rows) == (["rs", "ag", "ag", "ar"], [1024])


def test_compile_outside_warmup_keeps_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1492's warmup gate: a compile outside warmup keeps the all-reduce path."""
    calls: list[str] = []
    _record(monkeypatch, calls, 2)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    table = plan_mod.BlockPlanTable(((16, plan_mod.OVERLAP, plan_mod.Source.RULE),))
    layer = _layer(16, 2, table, [])
    x = torch.ones((16, 16))
    layer(x, None, torch.arange(16))
    assert calls == ["ar", "ar"]
    calls.clear()
    with shape_variants.warmup():
        layer(x, None, torch.arange(16))
    assert calls == ["rs", "ag", "ag", "ar"]
