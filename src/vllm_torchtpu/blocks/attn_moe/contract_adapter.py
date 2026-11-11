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
from typing import Any, ClassVar

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import RowParallelLinear


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


class BlockContractError(AttributeError):
    """A child decision owned by a block plan was written after binding."""


class _ReadOnly:
    """Mixin making ``owned`` attributes read-only; the block plan decided them.

    Python's attribute protocol hands ``__setattr__`` the attribute's name, so
    the guard holds the names of the fields this adapter wrote.
    """

    owned: ClassVar[frozenset[str]] = frozenset()

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        if name in self.owned:
            raise BlockContractError(
                f"{type(self).__qualname__}.{name} is decided by the "
                "attn_then_moe block plan; pin TPU_BLOCK_PLAN instead"
            )
        super().__setattr__(name, value)


_guarded: dict[tuple[type, frozenset[str]], type] = {}


def _guard(obj: object, owned: frozenset[str]) -> None:
    """Makes ``owned`` attributes of ``obj`` read-only by rebinding its class."""
    base = type(obj)
    key = (base, owned)
    if key not in _guarded:

        class Guarded(_ReadOnly, base):  # type: ignore[misc,valid-type]
            pass

        Guarded.owned = owned
        Guarded.__name__ = base.__name__
        Guarded.__qualname__ = base.__qualname__
        _guarded[key] = Guarded
    obj.__class__ = _guarded[key]


def bind_child_contracts(
    layer_name: str,
    out_proj: RowParallelLinear,
    experts: MoERunner,
    contracts: ChildContracts,
    source: str,
) -> None:
    """Applies the plan's child contracts once, then freezes the fields.

    Args:
        layer_name: The decoder layer's module path, for the ledger.
        out_proj: Attention's output projection.
        experts: The MoE runner.
        contracts: What the plan resolved for the children.
        source: Which plan source decided, for the ledger.
    """
    moe_config = experts.moe_config
    reduces = out_proj.reduce_results
    out_proj.reduce_results = contracts.out_proj_reduces
    gate = experts.gate
    experts.gate = gate if contracts.experts_hold_router else None
    skips = moe_config.skip_final_all_reduce
    moe_config.skip_final_all_reduce = not contracts.experts_reduce
    LEDGER.extend(
        (
            LedgerEntry(
                layer_name,
                "out_proj",
                "reduce_results",
                repr(reduces),
                repr(out_proj.reduce_results),
                source,
            ),
            LedgerEntry(
                layer_name,
                "experts",
                "gate",
                type(gate).__name__,
                type(experts.gate).__name__,
                source,
            ),
            LedgerEntry(
                layer_name,
                "moe_config",
                "skip_final_all_reduce",
                repr(skips),
                repr(moe_config.skip_final_all_reduce),
                source,
            ),
        )
    )
    _guard(out_proj, frozenset({"reduce_results"}))
    _guard(experts, frozenset({"gate"}))
    _guard(moe_config, frozenset({"skip_final_all_reduce"}))
