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
"""The norm_then_attention sub-block: declarations conform, mismatches decline."""

import pytest
from torch import nn
from vllm.config import set_current_vllm_config

from prototype import _build, check_subblock
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks import norm_attn
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention


@pytest.mark.parametrize("spec", check_subblock.FAMILIES, ids=lambda s: s.name)
def test_every_declaration_conforms(spec: _build.Spec) -> None:
    layer = _build.build(spec)
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS[(type(layer), norm_then_attention)]
    assert declaration.when(layer)
    with set_current_vllm_config(_build.current_config()):
        roles = norm_attn.conform(layer, declaration)
    assert roles.qkv_proj == "self_attn.qkv_proj"


def test_fused_value_matches_upstream() -> None:
    """The one fused value runs and matches upstream on every family."""
    assert all(not check_subblock.family(spec) for spec in check_subblock.FAMILIES)


def test_mismatch_declines_at_install() -> None:
    assert not check_subblock.planted()
    layer = _build.build(check_subblock.FAMILIES[0])
    layer.__class__ = check_subblock.ScaledAttention
    key = (check_subblock.ScaledAttention, norm_then_attention)
    blocks.DECLARATIONS[key] = blocks.BlockDeclaration(
        check_subblock.ScaledAttention,
        norm_then_attention,
        {
            "in_norm": "input_layernorm",
            "attn": "self_attn",
            "post_norm": "post_attention_layernorm",
            "mlp": "mlp",
        },
        lambda _: True,
    )
    try:
        with set_current_vllm_config(_build.current_config()):
            (item,) = blocks.install(
                nn.ModuleDict({"l": layer}), check_subblock._context()
            )  # noqa: SLF001
    finally:
        del blocks.DECLARATIONS[key]
    assert item.outcome.startswith("declined: ")
    assert type(layer) is check_subblock.ScaledAttention
