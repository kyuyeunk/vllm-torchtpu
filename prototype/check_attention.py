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
"""Runs the prototype's three checks on the Qwen3NextAttention boundary.

1. Structural: every capable plan value, for each described chip, produces
   the contract's output shape and dtype (meta device, on a CPU).
2. Outside writes: setting an upstream decision attribute raises.
3. Guard: upstream still matches what the replacement was written against
   (signatures, parameter names), and the expansion point has an owner.

Each check has a planted violation; comment one in below and rerun to see it
caught. Run: JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.check_attention
"""

import contextlib
import dataclasses
import enum
import sys
from collections.abc import Callable, Iterator

import torch
from torch import nn
from vllm.model_executor.models import qwen3_next
from vllm.platforms.interface import PlatformEnum

from prototype import _layer
from vllm_torchtpu import patch_registry
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.layers import expansion
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod


class Planted(enum.Enum):
    """A violation to plant; comment one in below."""

    SHAPE = "shape"
    WRITE = "write"
    SIGNATURE = "signature"
    PARAM = "param"
    OWNER = "owner"


PLANTED: set[Planted] = set()
# PLANTED = {Planted.SHAPE}      # 1: the fused_triton QK path returns float32
# PLANTED = {Planted.WRITE}      # 2: use_fused_output_gate's read-only view is gone
# PLANTED = {Planted.SIGNATURE}  # 3: upstream forward gains a parameter
# PLANTED = {Planted.PARAM}      # 3: upstream renames q_norm
# PLANTED = {Planted.OWNER}      # 3: the expansion point's registry entry has no owner

Replacement = layer_mod.TpuQwen3NextAttention


class _AttentionByContract(nn.Module):
    """The attention kernel, stubbed by its own contract: q-shaped output."""

    def forward(self, q, k, v):
        del k, v
        return torch.empty_like(q)


def _fake_qk_rmsnorm_rope_gate(q_gate, k, *args, **kwargs):
    """The Triton kernel's declared outputs: (q, k, gate), as its docstring."""
    del kwargs
    num_q_heads, head_dim = args[5], args[7]
    q = q_gate.new_empty(q_gate.shape[0], num_q_heads * head_dim)
    return q, torch.empty_like(k), torch.empty_like(q)


@contextlib.contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def check_structural() -> list[str]:
    failures = []
    stack = contextlib.ExitStack()
    stack.enter_context(
        _patched(qwen3_next, "fused_qk_rmsnorm_rope_gate", _fake_qk_rmsnorm_rope_gate)
    )
    if Planted.SHAPE in PLANTED:
        forward = Replacement.forward

        def planted(self, positions, hidden_states):
            out = forward(self, positions, hidden_states)
            fused = (
                isinstance(self.plan, alp.UnfusedLayerPlan)
                and self.plan.qk_norm_rope_gate is alp.QkPath.FUSED_TRITON
            )
            return out.float() if fused else out

        stack.enter_context(_patched(Replacement, "forward", planted))
    with stack:
        for chip in (PlatformEnum.TPU, PlatformEnum.CUDA):
            for gated in (True, False):
                config = _layer.tiny_config(attn_output_gate=gated)
                probe = _layer.build(config, chip=chip)
                caps = layer_mod.describe_children(probe)
                for plan in alp.enumerate_plans(probe.desc):
                    if alp.plan_declines(plan, probe.desc, caps, alp.ChipDesc(chip)):
                        continue
                    layer = _layer.build(
                        config, chip=chip, override=alp.plan_name(plan)
                    )
                    layer.attn = _AttentionByContract()
                    out = layer(
                        torch.zeros(8, dtype=torch.int64, device="meta"),
                        torch.empty(8, config.hidden_size, device="meta"),
                    )
                    want = layer.contract.output
                    got = (tuple(out.shape), alp.dtype_name(out.dtype))
                    chip_name = alp.platform_name(chip)
                    line = f"{chip_name} gated={gated} {alp.plan_name(plan)}: {got}"
                    print(f"    {line}")
                    shape_ok = tuple(out.shape) == (8, want.shape[1])
                    if not shape_ok or out.dtype != want.dtype:
                        want_dtype = alp.dtype_name(want.dtype)
                        failures.append(f"{line} != {want.shape} {want_dtype}")
    return failures


def check_outside_write() -> list[str]:
    stack = contextlib.ExitStack()
    if Planted.WRITE in PLANTED:
        stack.enter_context(_patched(Replacement, "use_fused_output_gate", False))
    with stack:
        layer = _layer.build(_layer.tiny_config())
        try:
            layer.use_fused_output_gate = True  # what Qwen3.5's decoder layer does
        except AttributeError as e:
            print(f"    raised: {e}")
            return []
        return [
            "layer.use_fused_output_gate = True was accepted; the plan still says "
            f"{alp.plan_name(layer.plan)}"
        ]


def check_guard() -> list[str]:
    failures = []
    stack = contextlib.ExitStack()
    if Planted.SIGNATURE in PLANTED:

        def forward(self, positions, hidden_states, residual=None):
            del residual
            return hidden_states

        stack.enter_context(_patched(qwen3_next.Qwen3NextAttention, "forward", forward))
    entries = dict(patch_registry.EXPANSION_POINTS)
    if Planted.OWNER in PLANTED:
        key = Replacement.expansion_key
        entries[key] = dataclasses.replace(entries[key], owner="")
    with stack:
        steps: list[tuple[str, Callable[[], None]]] = [
            (
                "signatures",
                lambda: expansion._guard_signatures(
                    qwen3_next.Qwen3NextAttention, Replacement
                ),
            ),
            ("parameter names", _check_parameters),
            ("owner entry", lambda: _check_owner(entries)),
        ]
        for name, step in steps:
            try:
                step()
                print(f"    {name}: ok")
            except (expansion.ExpansionGuardError, LookupError) as e:
                failures.append(f"{name}: {e}")
    return failures


def _check_parameters() -> None:
    layer = _layer.build(_layer.tiny_config())
    if Planted.PARAM in PLANTED:
        layer.q_norm_renamed = layer.q_norm
        del layer.q_norm
    expansion.check_parameter_names(layer, Replacement.upstream_parameter_names)


def _check_owner(
    entries: dict[patch_registry.ExpansionKey, patch_registry.ExpansionPoint],
) -> None:
    for key in sorted(expansion.registered_keys, key=lambda k: k.value):
        entry = entries.get(key)
        if entry is None or not entry.owner.strip():
            raise LookupError(f"expansion point {key.value} has no owner")
        print(f"    {key.value}: owner={entry.owner!r}")


def main() -> int:
    print(f"planted: {sorted(p.value for p in PLANTED) or 'none'}")
    failed = False
    with _layer.runtime():
        for title, check in (
            ("1 structural: every capable plan meets the contract", check_structural),
            ("2 outside write raises", check_outside_write),
            ("3 guard: upstream matches, owner present", check_guard),
        ):
            print(title)
            failures = check()
            for failure in failures:
                print(f"    FAIL {failure}")
            print(f"  -> {'FAIL' if failures else 'PASS'}")
            failed |= bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(main())
