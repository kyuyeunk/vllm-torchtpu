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
"""Hands the block's resolved child contracts to children built without them.

The block needs three child decisions: attention's output is not reduced,
the MoE runner does not all-reduce, and the runner takes router logits
instead of holding the router.

- **Attention, at construction.** Where the layer also declares
  ``norm_then_attention``, the block builds the nested ``AttentionRunner``
  with ``output_reduced=False``; nothing here touches ``o_proj``. Only a
  front half that is the model's own attention module (Qwen3.5's
  linear-attention layers, which declare no sub-block) calls its projection
  itself, and for that one this adapter still writes
  ``o_proj.reduce_results``.
- **The MoE, after construction (interim).** The model builds its MoE child;
  vLLM exposes the decisions as constructor arguments
  (``FusedMoEFactory(reduce_results=...)``) but models hard-code them, so the
  block cannot pass them down. ``bind_child_contracts`` writes them once,
  from the block installer, before any compile and before anything reads the
  fields (``arm_runner_models``, the MoE decode arming). Each write goes to a
  ledger and the fields become read-only, so a later write raises instead of
  silently changing a decision the plan made. The exit is a
  construction-time contract scope that ``FusedMoE.__init__`` reads, or
  models passing ``reduce_results`` through.
"""

import dataclasses
from typing import override

from torch import nn
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.blocks.parts import as_kind


@dataclasses.dataclass(frozen=True)
class ChildContracts:
    """What the plan hands each child. No defaults: the plan states each."""

    experts_hold_router: bool
    experts_reduce: bool


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    """A ledger line: one write the adapter made, and who decided it."""

    layer: str
    child: str
    field: str
    before: str
    after: str
    source: str


LEDGER: list[LedgerEntry] = []

_guarded: dict[tuple[type, str], type] = {}


class BlockContractError(AttributeError):
    """A child decision owned by a block plan was written after binding."""


def _guarded_class(base: type, field: str) -> type:
    """``base`` with ``field`` made read-only (the block plan decided it)."""

    class Guarded(base):  # pyright: ignore[reportUntypedBaseClass] - the child's own class
        @override
        def __setattr__(self, name: str, value: object) -> None:
            # Python's attribute protocol hands the name over as a string.
            if name == field:  # str-ok: the guarded attribute's own name
                msg = (
                    f"{base.__qualname__}.{name} is decided by the "
                    "attn_then_moe block plan; pin TPU_BLOCK_PLAN instead"
                )
                raise BlockContractError(msg)
            super().__setattr__(name, value)  # pyright: ignore[reportUnknownMemberType] - the child's own __setattr__

    Guarded.__name__ = base.__name__
    Guarded.__qualname__ = base.__qualname__
    return Guarded


def _guard(obj: object, field: str) -> None:
    """Makes ``obj.field`` read-only by rebinding its class."""
    key = (type(obj), field)
    if key not in _guarded:
        _guarded[key] = _guarded_class(*key)
    obj.__class__ = _guarded[key]


def _render(value: object) -> str:
    """A module by its class (its repr is its whole tree); a flag by its repr."""
    return type(value).__name__ if isinstance(value, nn.Module) else repr(value)


def bind_child_contracts(
    layer_name: str,
    out_proj: RowParallelLinear | None,
    experts: nn.Module,
    contracts: ChildContracts,
    source: str,
) -> None:
    """Applies the plan's child contracts once, then freezes the fields.

    Args:
        layer_name: The decoder layer's module path, for the ledger.
        out_proj: Attention's output projection, only where the front half is
            the model's own attention module; None where the nested runner
            was built with the contract.
        experts: The MoE runner.
        contracts: What the plan resolved for the children.
        source: Which plan source decided, for the ledger.
    """
    runner = as_kind(experts, MoERunner)
    moe_config = runner.moe_config

    def record(child: str, field: str, *, before: object, after: object) -> None:
        LEDGER.append(
            LedgerEntry(
                layer_name, child, field, _render(before), _render(after), source
            )
        )

    gate = runner.gate
    runner.gate = gate if contracts.experts_hold_router else None
    record("experts", "gate", before=gate, after=runner.gate)
    skip = moe_config.skip_final_all_reduce
    moe_config.skip_final_all_reduce = not contracts.experts_reduce
    after = not contracts.experts_reduce
    record("moe_config", "skip_final_all_reduce", before=skip, after=after)
    if out_proj is not None:  # The model's attention reduces unless told not to.
        reduces = out_proj.reduce_results
        out_proj.reduce_results = False
        record("out_proj", "reduce_results", before=reduces, after=False)
    if out_proj is not None:
        _guard(out_proj, "reduce_results")
    _guard(runner, "gate")
    _guard(moe_config, "skip_final_all_reduce")
