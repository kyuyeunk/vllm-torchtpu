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
"""Checks and installs every declared block of a model, after weights load."""

import logging

from torch import nn

from vllm_torchtpu.blocks import Pattern, declarations_for, load_declarations
from vllm_torchtpu.blocks.attn_moe.install import AttnThenMoE
from vllm_torchtpu.blocks.attn_moe.pattern import attn_then_moe
from vllm_torchtpu.blocks.binding import named_modules
from vllm_torchtpu.blocks.conformance import BlockConformanceError
from vllm_torchtpu.blocks.context import BlockImplementation, InstallContext, Installed
from vllm_torchtpu.blocks.norm_attn.install import NormThenAttention
from vllm_torchtpu.blocks.norm_attn.pattern import norm_then_attention

# vllm_torchtpu.logger's proxy resolves to vLLM's logger of this name.
logger = logging.getLogger("vllm." + __name__)

IMPLEMENTATIONS: dict[Pattern, BlockImplementation] = {
    attn_then_moe: AttnThenMoE(),
    norm_then_attention: NormThenAttention(),
}
"""The framework's implementation of each pattern."""


def install(model: nn.Module, context: InstallContext) -> list[Installed]:
    """Checks and installs every declared block in ``model``.

    A declaration is the model owner's explicit claim, so a layer that does
    not conform to its declared pattern stops startup, naming the edge and
    line. A pinned plan value whose capability does not cover a conforming
    layer stops startup too; a value the rule picked falls back to the next
    capable one (the default) quietly.
    """
    load_declarations()
    report: list[Installed] = []
    for name, module in named_modules(model):
        cls = type(module)
        try:
            declared = [d for d in declarations_for(cls) if d.when(module)]
        except BlockConformanceError as e:
            msg = f"{name} ({cls.__qualname__}): {e}"
            raise BlockConformanceError(msg) from e
        inner = {p for d in declared for p in IMPLEMENTATIONS[d.pattern].nests}
        nested = {d.pattern: d for d in declared if d.pattern in inner}
        for declaration in declared:
            if declaration.pattern in inner:
                continue  # The enclosing pattern's installer builds it.
            impl = IMPLEMENTATIONS[declaration.pattern]
            try:
                report.append(impl.install(name, module, declaration, context, nested))
            except BlockConformanceError as e:
                msg = (
                    f"{name} ({cls.__qualname__}) is declared "
                    f"{declaration.pattern.__name__}: {e}"
                )
                raise BlockConformanceError(msg) from e
    counts: dict[str, int] = {}
    for item in report:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    for outcome, count in sorted(counts.items()):
        logger.info("block install: %d layer(s): %s", count, outcome)
    return report
