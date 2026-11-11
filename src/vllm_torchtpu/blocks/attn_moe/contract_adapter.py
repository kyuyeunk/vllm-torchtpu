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

The overlap plan needs three child decisions: attention's output projection
does not all-reduce, the MoE runner does not all-reduce, and the runner takes
router logits instead of holding the router. vLLM exposes the first two as
constructor arguments (``RowParallelLinear(reduce_results=...)``,
``FusedMoEFactory(reduce_results=...)``), but most models hard-code them, so
the block cannot pass them down at construction.

``bind_child_contracts`` is the one sanctioned place that sets those fields
after construction. It runs once, from the block installer, before any compile
and before anything that reads the fields (``arm_runner_models``, the MoE
decode arming). Each write is recorded in a ledger, and the fields become
read-only afterwards, so a later write raises instead of silently changing a
decision the plan made. This is an interim: the exit is a construction-time
contract scope that linear and MoE ``__init__`` read, and upstream passing
``reduce_results`` through.
"""

import dataclasses
from typing import Any

from torch import nn

# The fields this adapter may write, per child role.
_FIELDS = {
    "out_proj": ("reduce_results",),
    "experts": ("gate",),
    "moe_config": ("skip_final_all_reduce",),
}


@dataclasses.dataclass(frozen=True)
class ChildContracts:
    """What the plan hands each child. No defaults: the plan states each."""

    out_proj_reduces: bool
    experts_hold_router: bool
    experts_reduce: bool


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    layer: str
    child: str
    field: str
    before: str
    after: str
    source: str


LEDGER: list[LedgerEntry] = []

_guarded: dict[type, type] = {}


class BlockContractError(AttributeError):
    """A child decision owned by a block plan was written after binding."""


def _guarded_class(base: type) -> type:
    """``base`` with the block-owned fields made read-only."""

    class Guarded(base):
        def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
            if name in getattr(self, "_block_owned", ()):
                raise BlockContractError(
                    f"{type(self).__qualname__}.{name} is decided by the "
                    f"{self._block_owner} block plan; pin TPU_BLOCK_PLAN instead"
                )
            super().__setattr__(name, value)

    Guarded.__name__ = base.__name__
    Guarded.__qualname__ = base.__qualname__
    return Guarded


def _guard(obj: Any, fields: tuple[str, ...], owner: str) -> None:  # noqa: ANN401
    """Makes ``fields`` of ``obj`` read-only by rebinding its class."""
    base = type(obj)
    if base not in _guarded:
        _guarded[base] = _guarded_class(base)
    object.__setattr__(obj, "_block_owned", fields)
    object.__setattr__(obj, "_block_owner", owner)
    obj.__class__ = _guarded[base]


def bind_child_contracts(
    layer_name: str,
    out_proj: nn.Module,
    experts: nn.Module,
    contracts: ChildContracts,
    source: str,
) -> None:
    """Applies the plan's child contracts once, then freezes the fields.

    Args:
        layer_name: The decoder layer's module path, for the ledger.
        out_proj: Attention's output projection (a RowParallelLinear).
        experts: The MoE runner.
        contracts: What the plan resolved for the children.
        source: Which plan source decided, for the ledger.
    """
    moe_config = experts.moe_config
    writes = (
        ("out_proj", out_proj, "reduce_results", contracts.out_proj_reduces),
        (
            "experts",
            experts,
            "gate",
            experts.gate if contracts.experts_hold_router else None,
        ),
        (
            "moe_config",
            moe_config,
            "skip_final_all_reduce",
            not contracts.experts_reduce,
        ),
    )
    for child, obj, field, value in writes:
        before = getattr(obj, field)
        setattr(obj, field, value)
        LEDGER.append(
            LedgerEntry(
                layer_name,
                child,
                field,
                type(before).__name__ if field == "gate" else repr(before),
                type(value).__name__ if field == "gate" else repr(value),
                source,
            )
        )
    _guard(out_proj, _FIELDS["out_proj"], "attn_then_moe")
    _guard(experts, _FIELDS["experts"], "attn_then_moe")
    _guard(moe_config, _FIELDS["moe_config"], "attn_then_moe")
