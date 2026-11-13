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
"""Shows the MoE decision module choosing, on a CPU, with no TPU runtime.

Prints the plan `moe_plan.plan_bucket` returns for a few calls on three
chips, then the six places that still decide the combine, with their current
line ranges. The chips are `pltpu.get_tpu_info_for_chip` records; JAX runs on
its CPU backend:

    .venv/bin/python -m prototype.moe_decision_module
"""

import os
import pathlib
import re
import sys
import types
from typing import NamedTuple

# Before JAX is imported: the CPU backend only.
os.environ.update(JAX_PLATFORMS="cpu")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "vllm_torchtpu"

# `vllm_torchtpu/__init__.py` and `layers/__init__.py` import torch_tpu, the
# TPU runtime; the decision module and its helpers do not. Register empty
# stand-ins for those two packages so only the modules shown here load.
for _name, _rel in (("vllm_torchtpu", ""), ("vllm_torchtpu.layers", "layers")):
    _module = types.ModuleType(_name)
    _module.__path__ = [str(_SRC / _rel)]
    sys.modules[_name] = _module

import jax.numpy as jnp  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu.layers.core import moe_plan  # noqa: E402


class _Chip(NamedTuple):
    """A chip and the name it is printed under."""

    label: str
    target: pltpu.TpuInfo


_V5E = _Chip(
    "v5e (no SparseCore)", pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V5E, 1)
)
_V6E = _Chip("v6e", pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1))
_7X = _Chip("7x", pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1))

# (chip, tokens, topk, local experts, expert parallel)
_CALLS = (
    (_V6E, 8, 8, 128, False),
    (_V6E, 4096, 8, 128, False),
    (_7X, 16, 8, 32, True),
    (_V6E, 32, 8, 32, True),
    (_7X, 8192, 8, 32, True),
    (_V5E, 8192, 8, 32, True),
)

# The six places that still decide the combine: (file, first line, last line,
# what it decides). Line ranges are found in the current source.
_SITES = (
    (
        "layers/core/fused_moe_gmm.py",
        r"^def _select_ragged_gather_reduce",
        r"^ragged_gather_reduce = ",
        "EP gather-reduce version, fixed at import from the environment",
    ),
    (
        "layers/core/fused_moe_gmm.py",
        r"^def resolve_onehot_permute_threshold",
        r"^    return moe_plan.auto_onehot_permute_threshold",
        "environment override of the one-hot threshold",
    ),
    (
        "layers/core/moe_plan.py",
        r"^def auto_onehot_permute_threshold",
        None,
        "automatic one-hot threshold from the chip",
    ),
    (
        "layers/core/moe_plan.py",
        r"^def plan_combine",
        None,
        "which combine runs, including the owner-output variant",
    ),
    (
        "kernels/sparse_core/ragged_gather_reduce_fallback.py",
        r"^def takes_tensorcore_path",
        None,
        "the 0.6 x VMEM fallback, one rule for the kernels and the plan",
    ),
    (
        "kernels/megablox/moe_onehot_unpermute_admission.py",
        r"^def chip_admits",
        None,
        "owner-output capability, shared by the kernel and the plan",
    ),
)


def _out(op: moe_plan.MoeOpIn, hidden: int = 2048) -> moe_plan.GmmOutSpec:
    """The combine's input facts for a bf16 MoE layer of this call."""
    routes = op.routed_rows
    # jnp's scalar types are Any.
    return moe_plan.GmmOutSpec(
        routes=routes,
        rows=routes,
        hidden=hidden,
        dtype=jnp.dtype(jnp.bfloat16),  # pyright: ignore[reportAny]
        token_indices_shape=(routes,),
        token_indices_dtype=jnp.dtype(jnp.int32),  # pyright: ignore[reportAny]
        topk_weights_shape=(routes,),
        topk_weights_dtype=jnp.dtype(jnp.float32),  # pyright: ignore[reportAny]
    )


def _process(target: pltpu.TpuInfo) -> moe_plan.MoeProcessPlan:
    """Today's defaults: SparseCore on where present, auto one-hot threshold."""
    return moe_plan.MoeProcessPlan(
        use_sparse_core=target.sparse_core is not None,
        onehot_threshold=moe_plan.auto_onehot_permute_threshold(target),
        single_sort=True,
        counting_sort=True,
        dense_expert_threshold=0,
        ep_gather_reduce_version=moe_plan.GatherReduceVersion.V2,
    )


def _span(path: pathlib.Path, start: str, end: str | None) -> tuple[int, int]:
    """First and last line of a top-level block, found by pattern."""
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if re.match(start, line))
    if end is not None:
        last = next(i for i in range(first, len(lines)) if re.match(end, lines[i]))
        return first + 1, last + 1
    last = first
    for i in range(first + 1, len(lines)):
        if lines[i] and not lines[i][0].isspace() and not lines[i].startswith(")"):
            break
        if lines[i].strip():
            last = i
    return first + 1, last + 1


def main() -> None:
    """Prints the plans, then the six deciding places."""
    print("Plans from moe_plan.plan_bucket (bf16, hidden 2048):")
    for chip, tokens, topk, experts, use_ep in _CALLS:
        target = chip.target
        op = moe_plan.MoeOpIn(
            num_tokens=tokens, topk=topk, local_num_experts=experts, use_ep=use_ep
        )
        plan = moe_plan.plan_bucket(
            _process(target),
            op,
            _out(op),
            target,
            dense_expert_admitted=False,
            owner_output=False,
        )
        if plan.routing is None or plan.combine is None:
            msg = "plan_bucket left the routing or the combine unplanned"
            raise ValueError(msg)
        mode = "EP" if use_ep else "TP"
        version = plan.combine.ep_gather_reduce_version
        print(
            f"  {chip.label:20} {mode} {tokens:5} tokens x top-{topk}:\n"
            f"    sort={plan.routing.sort.value} "
            f"permute={plan.routing.permute.value} "
            f"combine={plan.combine.combine.value}"
            + ("" if version is None else f" (EP gather-reduce {version.value})")
        )
    print("\nThe six places that still decide the combine:")
    for rel, start, end, what in _SITES:
        first, last = _span(_SRC / rel, start, end)
        print(f"  {rel}:{first}-{last}\n    {what}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
