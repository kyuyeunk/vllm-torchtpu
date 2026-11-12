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
"""The sub-block's plan: whether the input norm joins attention's projection.

Two plan values compute the same output within the residual stream's
tolerance:

- ``unfused``: the input RMSNorm adds the residual and normalizes, then
  attention's QKV projection reads the normed tokens (upstream).
- ``fused_qkv``: one function adds, normalizes and projects (PR #3096's
  fused_qkv kernel; here a plain-torch reference written against the roles).

The rule picks ``fused_qkv`` for every layer that conforms. The user's
override (``TPU_SUBBLOCK_PLAN``) comes first, as for every plan.
"""

import dataclasses
from typing import Literal, get_args

from vllm_torchtpu.layers.core.plan_marker import plan_of
from vllm_torchtpu.layers.core.resolution import require_no_default

InputProj = Literal["unfused", "fused_qkv"]


@plan_of("blocks.norm_then_attention")
@require_no_default
@dataclasses.dataclass(frozen=True)
class NormAttnPlan:
    """How one conforming sub-block reaches attention's projection."""

    input_proj: InputProj


RULE = NormAttnPlan(input_proj="fused_qkv")


def resolve(override: str) -> tuple[NormAttnPlan, str]:
    """The plan and its source: the user's override, else the rule.

    Raises:
        ValueError: The override names an unknown key or value.
    """
    if not override:
        return RULE, "rule"
    pins = dict(item.split("=", 1) for item in override.split(","))
    if unknown := sorted(set(pins) - {"input_proj"}):
        raise ValueError(f"TPU_SUBBLOCK_PLAN: unknown keys {unknown}")
    value = pins["input_proj"]
    if value not in get_args(InputProj):
        raise ValueError(
            f"TPU_SUBBLOCK_PLAN: input_proj={value} is not one of "
            f"{list(get_args(InputProj))}"
        )
    return NormAttnPlan(input_proj=value), "user"  # type: ignore[arg-type]
