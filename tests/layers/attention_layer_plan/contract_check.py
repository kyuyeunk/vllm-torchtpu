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
"""The three checks on the Qwen3NextAttention boundary.

1. Structural: every capable plan value, for each described chip, produces
   the contract's output shape and dtype (meta device, on a CPU).
2. Outside writes: setting an upstream decision attribute raises.
3. Guard: upstream still matches what the replacement was written against
   (signatures, parameter names), and the expansion point has an owner.

Each check takes the set of planted violations, so a test or the demo can
show each one caught. `test_attention_layer_contract.py` asserts the
results; `prototype/check_attention.py` prints them.
"""

# isort: off
from tests.layers.attention_layer_plan import cpu_layer  # first: no TPU runtime

# isort: on
import contextlib
import dataclasses
import enum
import typing
from collections.abc import Callable, Generator

import torch
from torch import nn
from vllm.model_executor.models import qwen3_next
from vllm.platforms.interface import PlatformEnum

from tests.layers.attention_layer_plan import layer_builder
from vllm_torchtpu import patch_registry
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.layers import expansion
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod


class Planted(enum.Enum):
    """A violation to plant."""

    SHAPE = "shape"  # 1: the fused_triton QK path returns float32
    WRITE = "write"  # 2: use_fused_output_gate's read-only view is gone
    SIGNATURE = "signature"  # 3: upstream forward gains a parameter
    PARAM = "param"  # 3: upstream renames q_norm
    OWNER = "owner"  # 3: the expansion point's registry entry has no owner


Replacement = layer_mod.TpuQwen3NextAttention


class _AttentionByContract(nn.Module):
    """The attention kernel, stubbed by its own contract: q-shaped output."""

    @typing.override
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Returns an empty q-shaped tensor.

        Args:
            q: Queries.
            k: Keys, unused.
            v: Values, unused.

        Returns:
            An empty tensor shaped like `q`.
        """
        del k, v
        return torch.empty_like(q)


def _fake_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor, k: torch.Tensor, *args: int, **kwargs: object
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Triton kernel's declared outputs: (q, k, gate), as its docstring."""
    del kwargs
    num_q_heads, head_dim = args[5], args[7]
    q = q_gate.new_empty(q_gate.shape[0], num_q_heads * head_dim)
    return q, torch.empty_like(k), torch.empty_like(q)


@contextlib.contextmanager
def _patched(owner: object, name: str, value: object) -> Generator[None]:
    old = typing.cast(
        object,
        owner.__dict__[name] if isinstance(owner, type) else getattr(owner, name),
    )
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _plant_shape(stack: contextlib.ExitStack) -> None:
    forward = Replacement.forward

    def planted(
        self: layer_mod.TpuQwen3NextAttention,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        out = forward(self, positions, hidden_states)
        fused = (
            isinstance(self.plan, alp.UnfusedLayerPlan)
            and self.plan.qk_norm_rope_gate is alp.QkPath.FUSED_TRITON
        )
        return out.float() if fused else out

    stack.enter_context(_patched(Replacement, "forward", planted))


def check_structural(planted: frozenset[Planted]) -> list[str]:
    """Runs every capable plan value on meta and compares with the contract.

    Args:
        planted: The violations to plant.

    Returns:
        One line per plan value whose output breaks the contract.
    """
    failures: list[str] = []
    stack = contextlib.ExitStack()
    stack.enter_context(
        _patched(qwen3_next, "fused_qk_rmsnorm_rope_gate", _fake_qk_rmsnorm_rope_gate)
    )
    if Planted.SHAPE in planted:
        _plant_shape(stack)
    with stack:
        for chip in (PlatformEnum.TPU, PlatformEnum.CUDA):
            for gated in (True, False):
                config = layer_builder.tiny_config(attn_output_gate=gated)
                probe = layer_builder.build(config, chip=chip)
                caps = layer_mod.describe_children(probe)
                for plan in alp.enumerate_plans(probe.desc):
                    if alp.plan_declines(plan, probe.desc, caps, alp.ChipDesc(chip)):
                        continue
                    layer = layer_builder.build(
                        config, chip=chip, override=alp.plan_name(plan)
                    )
                    layer.attn = _AttentionByContract()
                    cpu_layer.use_cpu_gemm(layer)
                    out: torch.Tensor = layer(
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


def check_outside_write(planted: frozenset[Planted]) -> list[str]:
    """Writes an upstream decision attribute from outside the layer.

    Args:
        planted: The violations to plant.

    Returns:
        A failure line if the write was accepted.
    """
    stack = contextlib.ExitStack()
    if Planted.WRITE in planted:
        stack.enter_context(_patched(Replacement, "use_fused_output_gate", value=False))
    with stack:
        layer = layer_builder.build(layer_builder.tiny_config())
        try:
            layer.use_fused_output_gate = True  # what Qwen3.5's decoder layer does
        except AttributeError as e:
            print(f"    raised: {e}")
            return []
        return [
            "layer.use_fused_output_gate = True was accepted; the plan still says "
            f"{alp.plan_name(layer.plan)}"
        ]


@dataclasses.dataclass(frozen=True)
class _GuardStep:
    """One guard step: its printed name and the call that raises on a break."""

    name: str
    run: Callable[[], None]


def check_guard(planted: frozenset[Planted]) -> list[str]:
    """Checks upstream's signatures and parameter names, and the owner entry.

    Args:
        planted: The violations to plant.

    Returns:
        One line per guard step that raised.
    """
    failures: list[str] = []
    stack = contextlib.ExitStack()
    if Planted.SIGNATURE in planted:

        def forward(
            self: qwen3_next.Qwen3NextAttention,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del self, positions, residual
            return hidden_states

        stack.enter_context(_patched(qwen3_next.Qwen3NextAttention, "forward", forward))
    entries = dict(patch_registry.EXPANSION_POINTS)
    if Planted.OWNER in planted:
        key = Replacement.expansion_key
        entries[key] = dataclasses.replace(entries[key], owner="")
    with stack:
        steps = (
            _GuardStep(
                "signatures",
                lambda: expansion._guard_signatures(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # the guard under test
                    qwen3_next.Qwen3NextAttention, Replacement
                ),
            ),
            _GuardStep("parameter names", lambda: _check_parameters(planted)),
            _GuardStep("owner entry", lambda: _check_owner(entries)),
        )
        for step in steps:
            try:
                step.run()
                print(f"    {step.name}: ok")
            except (expansion.ExpansionGuardError, LookupError) as e:
                failures.append(f"{step.name}: {e}")
    return failures


def _check_parameters(planted: frozenset[Planted]) -> None:
    layer = layer_builder.build(layer_builder.tiny_config())
    if Planted.PARAM in planted:
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


@dataclasses.dataclass(frozen=True)
class Check:
    """One of the three checks: its printed title and the check itself."""

    title: str
    run: Callable[[frozenset[Planted]], list[str]]


def checks() -> tuple[Check, ...]:
    """The three checks, in the order they print.

    Returns:
        The checks.
    """
    return (
        Check("1 structural: every capable plan meets the contract", check_structural),
        Check("2 outside write raises", check_outside_write),
        Check("3 guard: upstream matches, owner present", check_guard),
    )


def main(planted: frozenset[Planted]) -> int:
    """Prints each check's result with `planted` planted.

    Args:
        planted: The violations to plant.

    Returns:
        The exit code: 1 if any check failed.
    """
    print(f"planted: {sorted(p.value for p in planted) or 'none'}")
    failed = False
    with layer_builder.runtime():
        for check in checks():
            print(check.title)
            failures = check.run(planted)
            for failure in failures:
                print(f"    FAIL {failure}")
            print(f"  -> {'FAIL' if failures else 'PASS'}")
            failed |= bool(failures)
    return 1 if failed else 0
