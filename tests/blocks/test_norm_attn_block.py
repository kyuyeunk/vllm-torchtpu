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


def test_mismatch_stops_startup() -> None:
    """A declared layer that does not conform is a startup error."""
    assert not check_subblock.planted()


def test_capability_mismatch_falls_back_unless_pinned() -> None:
    """gpt-oss conforms, but fused_qkv cannot serve sinks: the rule falls back."""
    (c,) = [m for m in check_subblock.NEW_MODELS if m.repo == "openai/gpt-oss-20b"]
    assert not check_subblock.family(check_subblock._spec(c), declines=True)  # noqa: SLF001


def test_pin_that_cannot_run_stops_startup() -> None:
    """A user pin to fused_qkv on gpt-oss (sinks) is a startup error."""
    (c,) = [m for m in check_subblock.NEW_MODELS if m.repo == "openai/gpt-oss-20b"]
    spec = check_subblock._spec(c)  # noqa: SLF001
    layer = _build.build(spec)
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS[(type(layer), norm_then_attention)]
    impl = blocks.IMPLEMENTATIONS[norm_then_attention]
    context = check_subblock._context("input_proj=fused_qkv")  # noqa: SLF001
    with (
        set_current_vllm_config(_build.current_config()),
        pytest.raises(ValueError, match="sinks"),
    ):
        impl.install("l", layer, declaration, context)
