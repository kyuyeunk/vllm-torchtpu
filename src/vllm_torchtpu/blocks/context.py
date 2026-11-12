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
"""What an implementation reads at install time, and what it reports."""

import dataclasses
from typing import Protocol

import torch
from jax.experimental.pallas import tpu as pltpu
from torch import nn

from vllm_torchtpu.blocks import BlockDeclaration, Pattern
from vllm_torchtpu.blocks.attn_moe.plan import Collectives
from vllm_torchtpu.blocks.norm_attn.plan import Pins


@dataclasses.dataclass(frozen=True)
class InstallContext:
    """What every implementation reads at install time, passed as typed data.

    Strings from the environment and the model config are parsed before this
    is built (``attn_moe.plan.parse_pin``, ``norm_attn.plan.parse_pins``).

    Attributes:
        block_pin, block_model_pin: The user's and the model plan's pin of the
            block's collectives, or None.
        subblock_pins: The user's pins of the sub-block's fields; None pins none.
        chip: The TPU generation; None off-TPU (no chip facts).
        fp8_kv_cache: Whether the KV cache is e4m3; None when unknown (the
            TPU runner passes None while the fused value is the plain-torch
            reference, which writes no KV cache; a kernel build passes it).
    """

    tp: int
    dtype: torch.dtype
    buckets: tuple[int, ...]
    block_pin: Collectives | None = None
    block_model_pin: Collectives | None = None
    subblock_pins: Pins | None = None
    chip: pltpu.ChipVersion | None = None
    fp8_kv_cache: bool | None = None


@dataclasses.dataclass(frozen=True)
class Installed:
    """One layer's install outcome, for the startup report."""

    name: str
    layer_class: str
    outcome: str


class BlockImplementation(Protocol):
    """A pattern's installer (``attn_moe.install``, ``norm_attn.install``)."""

    nests: tuple[Pattern, ...]
    """Patterns this one builds inside itself when a layer declares both."""

    def install(
        self,
        name: str,
        layer: nn.Module,
        declaration: BlockDeclaration,
        context: InstallContext,
        nested: dict[Pattern, BlockDeclaration] | None = None,
    ) -> Installed:
        """The part's ``install``."""
        ...
