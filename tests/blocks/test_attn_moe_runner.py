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
"""Execution tests: AttnMoERunner on synthetic binding, no model.

The block runner is built from tiny binding: the nested ``AttentionRunner``
over ``synthetic.py``'s attention binding, an RMSNorm, a router linear and a
dense top-k expert mixture. Each block plan value (upstream's ``all_reduce``,
PR #1492's ``reduce_scatter_all_gather``) runs with each attention plan value
(``unfused``, ``fused_qkv``), and each is checked against the upstream
composition in straight-line torch at TP=1, where the collectives are the
identity. The block's contract with attention (``output_reduced=False``) and
the capability that guards it are checked here too; a model's layers
conforming and routing through the runner is ``test_attn_moe_block.py``'s job.
"""

import dataclasses
import sys

_PRELOADED = frozenset(sys.modules)  # Before this file's imports; see the last test.

import pytest  # noqa: E402 - after the snapshot above
import torch  # noqa: E402 - after the snapshot above
import torch.nn.functional as F  # noqa: E402, N812 - after the snapshot; torch's alias
from synthetic import (  # noqa: E402 - after the snapshot above
    HIDDEN,
    TOKENS,
    Attention,
    InputNorm,
    Linear,
    Roles,
    build_runner,
    make_inputs,
    reference_forward,
    setup_parts,
)
from torch import nn  # noqa: E402 - after the snapshot above

from vllm_torchtpu.blocks.attn_moe import plan as block_plan  # noqa: E402 - ditto
from vllm_torchtpu.blocks.attn_moe import runner as block_runner  # noqa: E402 - ditto
from vllm_torchtpu.blocks.attn_moe.runner import (  # noqa: E402 - ditto
    AttnMoERunner,
    MoEBinding,
)
from vllm_torchtpu.blocks.norm_attn import plan as subblock_plan  # noqa: E402 - ditto
from vllm_torchtpu.blocks.norm_attn.runner import (  # noqa: E402 - ditto
    AttentionRunner,
    InputProj,
    ResidualAdd,
    reference_fused_qkv,
)
from vllm_torchtpu.blocks.stream import Stream  # noqa: E402 - ditto
from vllm_torchtpu.compilation import shape_variants  # noqa: E402 - ditto

_LOADED = frozenset(sys.modules) - _PRELOADED  # What this file's imports loaded.

NUM_EXPERTS, TOP_K = 4, 2
ATOL = 1e-5  # float32; the overlap sums the stream in another order.
# head_dim 128: a configuration fused_qkv's capability accepts, so the
# unreduced-output condition is the one under test.
SUBBLOCK = Roles(qk_norm=True, gqa=4, head_dim=128)
ALL_REDUCE = block_plan.Collectives.ALL_REDUCE
OVERLAP = block_plan.Collectives.REDUCE_SCATTER_ALL_GATHER
UNFUSED, FUSED = InputProj.UNFUSED, InputProj.FUSED_QKV


def _unpack(
    positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    return hidden_states, residual, positions


def _no_call(
    h: torch.Tensor, r: torch.Tensor | None, p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del h, r, p
    msg = "the runner never calls the layer's own forward"
    raise AssertionError(msg)


# A layer whose forward is (positions, hidden_states, residual), as Llama's.
STREAM = Stream(call=_no_call, unpack=_unpack)


@pytest.fixture(autouse=True, scope="module")
def _one_thread() -> object:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def collectives(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """TP=1: every collective is the identity; records the call order."""
    calls: list[str] = []

    def record(name: str):  # noqa: ANN202
        def op(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
            del dim
            calls.append(name)
            return x

        return op

    for name, short in (
        ("tensor_model_parallel_all_reduce", "ar"),
        ("tensor_model_parallel_reduce_scatter", "rs"),
        ("tensor_model_parallel_all_gather", "ag"),
    ):
        monkeypatch.setattr(block_runner, name, record(short))
    return calls


# The MoE's synthetic binding.


class Experts(nn.Module):
    """A dense top-k mixture taking ``(hidden_states, router_logits)``, as vLLM's."""

    def __init__(self) -> None:
        """Builds the synthetic part."""
        super().__init__()
        self.w = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, HIDDEN) / HIDDEN**0.5)

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """The part's call contract, in plain torch."""
        weights, ids = router_logits.softmax(-1).topk(TOP_K, dim=-1)
        per_expert = torch.einsum("td,edh->teh", hidden_states, self.w)
        picked = per_expert.gather(1, ids[..., None].expand(-1, -1, HIDDEN))
        return (weights[..., None] * picked).sum(1)


def _moe() -> tuple[InputNorm, Linear, Experts]:
    """float32 explicitly: an earlier file's meta builds set a bf16 default."""
    torch.manual_seed(2)
    binding = InputNorm(HIDDEN), Linear(HIDDEN, NUM_EXPERTS), Experts()
    return binding[0].float(), binding[1].float(), binding[2].float()


def _attention(input_proj: InputProj) -> tuple[InputNorm, Attention, AttentionRunner]:
    norm, attention = setup_parts(SUBBLOCK)
    fused = reference_fused_qkv if input_proj is FUSED else None
    attention_part = build_runner(
        norm, attention, SUBBLOCK.config(), input_proj, fused, output_reduced=False
    )
    return norm, attention, attention_part


def _table(collectives: block_plan.Collectives) -> block_plan.BlockPlanTable:
    plan = block_plan.AttnMoEBlockPlan(collectives)
    return block_plan.BlockPlanTable(((1, plan, block_plan.Source.USER),))


Parts = tuple[InputNorm, Attention, InputNorm, Linear, "Experts"]


def _block(
    collectives: block_plan.Collectives, input_proj: InputProj
) -> tuple[AttnMoERunner, Parts]:
    norm, attention, attention_part = _attention(input_proj)
    post_norm, router, experts = _moe()
    block = AttnMoERunner(
        "layer",
        MoEBinding(post_norm=post_norm, router=router, experts=experts),
        attention=attention_part,
        table=_table(collectives),
        tp=1,
        stream=STREAM,
    )
    return block, (norm, attention, post_norm, router, experts)


def reference_block(
    binding: "Parts",
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream's layer in straight-line torch: attention, norm, route, experts."""
    norm, attention, post_norm, router, experts = binding
    attn, stream = reference_forward(
        SUBBLOCK, norm, attention, hidden, residual, positions
    )
    stream = attn + stream
    h = stream / torch.sqrt(stream.pow(2).mean(-1, keepdim=True) + 1e-6)
    h = h * post_norm.weight
    logits = F.linear(h, router.weight)
    return experts(h, logits), stream


# Every block value x every attention value, at TP=1.

GRID = [
    (collectives, input_proj)
    for collectives in (ALL_REDUCE, OVERLAP)
    for input_proj in (UNFUSED, FUSED)
]


@pytest.mark.parametrize("first", [True, False], ids=["first_layer", "residual"])
@pytest.mark.parametrize(("collectives_value", "input_proj"), GRID)
def test_block_equals_upstream(
    collectives: list[str],
    collectives_value: block_plan.Collectives,
    input_proj: InputProj,
    first: bool,  # noqa: FBT001 - a pytest parameter, passed by name
) -> None:
    """The residual stream (``out + residual``) matches upstream's layer."""
    block, binding = _block(collectives_value, input_proj)
    hidden, residual, positions = make_inputs(first)
    with torch.no_grad():
        out, res = block(positions, hidden, residual)
        want_out, want_res = reference_block(binding, hidden, residual, positions)
    torch.testing.assert_close(out + res, want_out + want_res, atol=ATOL, rtol=0)
    if collectives_value is ALL_REDUCE:  # Upstream's split of the stream, too.
        torch.testing.assert_close(res, want_res, atol=ATOL, rtol=0)
    want = ["ar", "ar"] if collectives_value is ALL_REDUCE else ["rs", "ag", "ag", "ar"]
    assert collectives == want


def test_the_layer_calls_by_name() -> None:
    """The runner stands in for the layer's forward, positional or by keyword."""
    block, _ = _block(ALL_REDUCE, UNFUSED)
    hidden, residual, positions = make_inputs(first=False)
    assert block.stream.unpack(positions, hidden, residual) == (
        hidden,
        residual,
        positions,
    )
    got = block.stream.unpack(hidden_states=hidden, positions=positions, residual=None)
    assert got == (hidden, None, positions)


@pytest.mark.parametrize("collectives_value", [ALL_REDUCE, OVERLAP])
def test_select_forward(collectives_value: block_plan.Collectives) -> None:
    """One value everywhere runs it directly; a table with the overlap looks up."""
    block, _ = _block(collectives_value, UNFUSED)
    entry = block._forward_entry  # noqa: SLF001 - the selection under test
    want = block._all_reduce if collectives_value is ALL_REDUCE else block._by_bucket  # noqa: SLF001
    assert entry == want


def test_by_bucket(collectives: list[str]) -> None:
    """The table's runs pick the value per token count; eager runs are counted."""
    _, _, attention_part = _attention(UNFUSED)
    post_norm, router, experts = _moe()
    rule = block_plan.Source.RULE
    table = block_plan.BlockPlanTable(
        ((4, block_plan.DEFAULT, rule), (TOKENS, block_plan.OVERLAP, rule))
    )
    block = AttnMoERunner(
        "bucketed",
        MoEBinding(post_norm=post_norm, router=router, experts=experts),
        attention=attention_part,
        table=table,
        tp=1,
        stream=STREAM,
    )
    hidden, residual, positions = make_inputs(first=False)
    assert residual is not None
    with torch.no_grad():
        block(positions[:4], hidden[:4], residual[:4])
        block(positions, hidden, residual)
    assert collectives == ["ar", "ar", "rs", "ag", "ag", "ar"]
    assert block_runner.CALLS["bucketed"] == {
        block_plan.DEFAULT: 1,
        block_plan.OVERLAP: 1,
    }


def test_compile_outside_warmup_keeps_all_reduce(
    collectives: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1492's warmup gate: a compile outside warmup keeps the all-reduce path."""
    block, _ = _block(OVERLAP, UNFUSED)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    hidden, residual, positions = make_inputs(first=False)
    with torch.no_grad():
        block(positions, hidden, residual)
        assert collectives == ["ar", "ar"]
        collectives.clear()
        with shape_variants.warmup():
            block(positions, hidden, residual)
    assert collectives == ["rs", "ag", "ag", "ar"]


def test_runner_is_composition_only() -> None:
    """Runner owns no parameters."""
    block, binding = _block(OVERLAP, FUSED)
    assert not isinstance(block, nn.Module)
    assert not isinstance(block.attention, nn.Module)
    del binding


# The contract: the block builds attention unreduced; the capability guards it.


@pytest.mark.parametrize("input_proj", [UNFUSED, FUSED])
def test_unreduced_attention_equals_reduced_at_tp1(input_proj: InputProj) -> None:
    """``output_reduced=False`` skips only the reduction; at TP=1 that is nothing."""
    norm, attention = setup_parts(SUBBLOCK)
    fused = reference_fused_qkv if input_proj is FUSED else None
    reduced = build_runner(norm, attention, SUBBLOCK.config(), input_proj, fused)
    unreduced = build_runner(
        norm, attention, SUBBLOCK.config(), input_proj, fused, output_reduced=False
    )
    hidden, residual, positions = make_inputs(first=False)
    with torch.no_grad():
        torch.testing.assert_close(
            unreduced(hidden, residual, positions), reduced(hidden, residual, positions)
        )


def test_unreduced_never_calls_the_projection_forward() -> None:
    """The flag is passed per call; the projection's own reduce is not consulted."""
    norm, attention = setup_parts(SUBBLOCK)
    runner = build_runner(norm, attention, SUBBLOCK.config(), output_reduced=False)

    def refuse(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "o_proj.forward runs its all-reduce"
        raise AssertionError(msg)

    object.__setattr__(attention.o_proj, "forward", refuse)
    hidden, residual, positions = make_inputs(first=False)
    with torch.no_grad():
        runner(hidden, residual, positions)


def _facts(**changes: object) -> subblock_plan.AttentionFacts:
    """A supported attention's facts, changed."""
    base = subblock_plan.AttentionFacts(
        config=SUBBLOCK.config(),
        sinks=False,
        kv_sharing=False,
        q_norm=None,
        k_norm=None,
        neox_rope=True,
        qkv_quant_supported=True,
        qkv_quant="UnquantizedLinearMethod",
        fp8_weights=False,
        o_proj_split_input=True,
        o_proj_adds_bias=False,
    )
    return dataclasses.replace(base, **changes)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("facts", "input_proj", "fused_unreduced", "why"),
    [
        (_facts(), UNFUSED, False, None),
        (_facts(), FUSED, True, None),
        (_facts(), FUSED, False, "always all-reduces"),
        (_facts(o_proj_adds_bias=True), UNFUSED, True, "adds a bias"),
        (_facts(o_proj_split_input=False), UNFUSED, True, "TP-split input"),
    ],
    ids=["unfused", "fused", "fused_op_reduces", "o_proj_bias", "o_proj_replicated"],
)
def test_unreduced_output_supports(
    facts: subblock_plan.AttentionFacts,
    input_proj: InputProj,
    fused_unreduced: bool,  # noqa: FBT001 - a pytest parameter, passed by name
    why: str | None,
) -> None:
    """Unreduced output supports."""
    plan = subblock_plan.NormAttnPlan(
        input_proj, subblock_plan.Matmul.BF16, ResidualAdd.FOLDED
    )
    got = subblock_plan.unreduced_output_supports(
        plan, facts, fused_unreduced=fused_unreduced
    )
    assert (got is None) if why is None else (why in (got or ""))


def test_rule_falls_back_when_the_fused_op_cannot_skip_the_reduce() -> None:
    """Chosen automatically: fused_qkv declines quietly, unfused serves the block."""
    plan, declined = subblock_plan.choose(
        subblock_plan.Pins(), _facts(), output_reduced=False, fused_unreduced=False
    )
    assert plan.input_proj is UNFUSED
    assert any("always all-reduces" in d for d in declined)


def test_pin_that_cannot_skip_the_reduce_raises() -> None:
    """Pinned: the same decline stops startup."""
    with pytest.raises(ValueError, match="always all-reduces"):
        subblock_plan.choose(
            subblock_plan.Pins(input_proj=FUSED),
            _facts(),
            output_reduced=False,
            fused_unreduced=False,
            where="layer",
        )


def test_reduced_contract_ignores_the_capability() -> None:
    """A layer on its own (reduced output) keeps fused_qkv whatever the op says."""
    plan, _ = subblock_plan.choose(
        subblock_plan.Pins(), _facts(o_proj_adds_bias=True), fused_unreduced=False
    )
    assert plan.input_proj is FUSED


def test_imports_no_model() -> None:
    """This file's imports load no model implementation."""
    models = "vllm.model_executor.models"
    infrastructure = {models} | {
        f"{models}.{m}" for m in ("registry", "interfaces", "interfaces_base", "config")
    }
    assert not {m for m in _LOADED if m.startswith(models)} - infrastructure
