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
"""Runs each declared layer both ways on a CPU and compares the residual stream.

For every surveyed layer whose class is declared (``models/vllm/*/
registration.py``), the layer is built on the CPU at tensor-parallel size 1
with small random weights and called twice on the same inputs: as upstream
built it, and with the block forward installed (``apply_block``) under each
plan value. The block's contract is the residual stream ``hidden +
residual``, so the sums are compared. The check runs in float32: in bfloat16
the overlap's different rounding can flip a token's top-k expert choice, a
real numerics difference (the source of #1492's drift) rather than a wiring
one.

Two children are replaced by deterministic stand-ins in both runs, because
the kernels behind them do not run here:

- attention, which needs a KV cache and attention metadata, by a fixed linear
  map of its input;
- the MoE runner (FusedMoE), whose CPU kernels are not built in this vLLM
  tree, by a reference top-k mixture over fixed expert matrices, reading the
  router logits it is passed, or calling the router itself when upstream's
  runner holds it.

What stays real is everything the block forward reorders: the two norms, the
residual handling, the router call and the overlap algebra. At size 1 the
reduce-scatter and all-gather are identities, so the overlap plan exercises
its arithmetic but not its communication; that needs a TPU at size > 1.

``python -m prototype.forward_equivalence`` prints the comparison;
``test_forward_equivalence.py`` asserts each layer's outcome.
"""

import enum

import torch
from torch import nn
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import LinearBase

from tests.blocks import _build
from tests.blocks._report import Emit
from vllm_torchtpu import blocks
from vllm_torchtpu.blocks.attn_moe import apply_block, conform, declared_parts
from vllm_torchtpu.blocks.attn_moe import plan as plan_mod
from vllm_torchtpu.blocks.attn_moe.forward import AttentionMoEBlock
from vllm_torchtpu.blocks.attn_moe.pattern import AttnMoEParts, is_residual_norm
from vllm_torchtpu.blocks.parts import as_kind

_TOKENS, _HIDDEN, _TOP_K = 8, 64, 2
# In float32 the two forwards differ only by the order of additions.
TOLERANCE = 1e-5


def _is_norm(module: nn.Module) -> bool:
    """A norm whose weight multiplies (vLLM's RMSNorm family, a LayerNorm)."""
    return is_residual_norm(module) or isinstance(module, nn.LayerNorm)


def _randomize(
    layer: nn.Module, parts: AttnMoEParts, generator: torch.Generator
) -> None:
    """Random weights; post-load processing for the children that really run."""
    norm_weights = {
        id(p) for m in layer.modules() if _is_norm(m) for p in m.parameters()
    }
    with torch.no_grad():
        for param in layer.parameters():
            noise = (
                torch.randn(param.shape, generator=generator, dtype=torch.float32)
                * 0.05
            )
            param.copy_(noise + (1.0 if id(param) in norm_weights else 0.0))
    # The runner may hold the router, which really runs.
    stood_in = {
        id(m) for m in (*parts.attention.modules(), *parts.experts.modules())
    } - {id(m) for m in parts.router.modules()}
    with set_current_vllm_config(_build.current_config()):
        for module in layer.modules():
            if not isinstance(module, LinearBase) or id(module) in stood_in:
                continue
            if module.quant_method is not None:  # A GateLinear has none.
                module.quant_method.process_weights_after_loading(module)


def _stand_ins(parts: AttnMoEParts, generator: torch.Generator) -> None:
    """Fixed attention and MoE runner, set on the instances (both runs)."""
    router, runner = as_kind(parts.router, GateLinear), parts.experts
    attn_map = (
        torch.randn(_HIDDEN, _HIDDEN, generator=generator, dtype=torch.float32) * 0.1
    )
    # One matrix per router output (vLLM may pad the router past the expert count).
    experts = (
        torch.randn(
            router.output_size,
            _HIDDEN,
            _HIDDEN,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.1
    )

    def attention(*args: object, **kwargs: object) -> torch.Tensor:
        # The normed stream: the one floating tensor of hidden width.
        x = next(
            v
            for v in (*args, *kwargs.values())
            if isinstance(v, torch.Tensor)
            and v.is_floating_point()
            and v.shape[-1] == _HIDDEN
        )
        return x @ attn_map

    def moe(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        # A runner that holds its router is passed the hidden states as
        # placeholder logits and routes itself.
        logits = router_logits
        if logits is None or logits.shape[-1] != experts.shape[0]:
            out = router(hidden_states)
            logits = out[0] if isinstance(out, tuple) else out
        weights, chosen = torch.topk(logits.float().softmax(-1), _TOP_K)
        y = torch.zeros_like(hidden_states, dtype=torch.float32)
        for k in range(_TOP_K):
            picked = experts[chosen[:, k]]  # [tokens, hidden, hidden]
            act = torch.nn.functional.silu(
                torch.einsum("th,thd->td", hidden_states.float(), picked.float())
            )
            y += weights[:, k : k + 1] * act
        return y.to(hidden_states.dtype)

    parts.attention.forward = attention
    runner.forward = moe


def _call(
    parts: AttnMoEParts,
    hidden: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    config = _build.current_config()
    with set_current_vllm_config(config), set_forward_context(None, config):
        out = as_kind(parts.stream.call(hidden, residual, positions), tuple)
    stream, residual = out
    return stream.float() + residual.float()


class Outcome(enum.Enum):
    """How the block forward compares with upstream's for one layer."""

    _value_: str

    EQUAL = "equal"
    DIFFERS = "DIFFERS"
    NOT_DECLARED = "not declared"


def check(spec: _build.Spec) -> tuple[Outcome, dict[plan_mod.Collectives, float]]:
    """(outcome, relative max abs stream difference per plan value) for one spec."""
    probe = _build.build(spec)
    blocks.load_declarations()
    declaration = blocks.DECLARATIONS.get(type(probe))
    if declaration is None or declaration.parts(probe) is None:
        return Outcome.NOT_DECLARED, {}
    with set_current_vllm_config(_build.current_config()):
        conform(probe, declaration)

    generator = torch.Generator().manual_seed(0)
    layer = _build.build(spec, device="cpu", dtype=torch.float32)
    parts = declared_parts(layer, declaration)
    _randomize(layer, parts, generator)
    _stand_ins(parts, generator)
    hidden = torch.randn(_TOKENS, _HIDDEN, generator=generator, dtype=torch.float32)
    residual = torch.randn(_TOKENS, _HIDDEN, generator=generator, dtype=torch.float32)
    positions = torch.arange(_TOKENS)
    want = _call(parts, hidden, residual, positions)
    scale = float(want.abs().max())
    diffs: dict[plan_mod.Collectives, float] = {}
    for plan in (plan_mod.DEFAULT, plan_mod.OVERLAP):
        table = plan_mod.BlockPlanTable(entries=((1, plan, plan_mod.Source.USER),))
        if not isinstance(layer, AttentionMoEBlock):
            apply_block("layer", layer, parts, table, tp=1)
        as_kind(layer, AttentionMoEBlock).bind_block(parts, table, 1)
        got = _call(parts, hidden, residual, positions)
        diffs[plan.collectives] = float((got - want).abs().max()) / scale
    worst = max(diffs.values())
    return (Outcome.EQUAL if worst <= TOLERANCE else Outcome.DIFFERS), diffs


def report(emit: Emit) -> None:
    """Emits each surveyed layer's outcome and differences per plan value."""
    emit(
        f"=== block forward vs upstream forward, TP 1, CPU "
        f"(relative max abs difference of hidden + residual; tolerance {TOLERANCE:.0e})"
    )
    for spec in _build.SPECS:
        try:
            outcome, diffs = check(spec)
        except Exception as e:  # noqa: BLE001 - report and continue
            emit(f"  {spec.name}: error: {type(e).__name__}: {e}")
            continue
        if not diffs:
            emit(f"  {spec.name}: {outcome.value}")
            continue
        shown = ", ".join(f"{k.value} {v:.2e}" for k, v in diffs.items())
        emit(f"  {spec.name}: {outcome.value} ({shown})")
