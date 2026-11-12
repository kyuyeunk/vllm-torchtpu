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
"""Oracle table test: CollectivePlan v1 against the d3991ebf9 formulas.

The `_reference_*` functions are the d3991ebf9 expressions from
`quantization/fp8.py`, `quantization/unquantized.py`, `pipelined_fused_moe.py`,
`fused_moe.py` and `runner/tpu_runner.py`, verbatim except that env reads and
the per-layer predicates are parameters.
"""

import enum
import itertools
import math
import unittest

import _pure_import

from vllm_torchtpu.layers.core import moe_collective_plan as mcp
from vllm_torchtpu.layers.core.moe_collective_plan import CollectiveRule


class _Route(enum.Enum):
    """Which MoE entry point the dispatch calls."""

    PIPELINED = enum.auto()
    FUSED_MOE_GMM = enum.auto()


def _reference_fp8(pipelined, fused, adaptive, fp8_ag, skip):
    """fp8.py:804-833 and :1182-1222 (kernel skip via fused_moe.py:208-209)."""
    supports_internal_mk = pipelined or fused or adaptive or fp8_ag
    owns_collectives = pipelined or fp8_ag
    zeroed = owns_collectives if skip else None
    route = _route(owns_collectives)
    quantize_ag = owns_collectives and fp8_ag  # pipelined_fused_moe.py:254
    return supports_internal_mk, route, zeroed, skip, quantize_ag


def _reference_unquantized(pipelined, fp8_ag, skip):
    """unquantized.py:253-256 and :379-382 (kernel skip via fused_moe.py)."""
    supports_internal_mk = pipelined
    route = _route(pipelined)
    quantize_ag = pipelined and fp8_ag
    return supports_internal_mk, route, None, skip, quantize_ag


def _reference_nvfp4(pipelined, fused, adaptive, fp8_ag, skip):
    """nvfp4.py:329-337 and :576."""
    return (
        pipelined or fused or adaptive,
        _route(pipelined),
        None,
        skip,
        (pipelined and fp8_ag),
    )


def _reference_pipelined_only(pipelined, fp8_ag, skip):
    """mxfp4.py:226-229/:359, w4a16.py:134-137/:402 (same as unquantized)."""
    return pipelined, _route(pipelined), None, skip, pipelined and fp8_ag


def _reference_mxfp4_dsv4(pipelined, fused, fp8_ag, skip):
    """mxfp4.py:378-383; dispatch falls back to the base class (:359)."""
    return pipelined or fused, _route(pipelined), None, skip, pipelined and fp8_ag


def _reference_w4an_mxfp4(pipelined, fp8_ag):
    """w4an_mxfp4.py:62-65, :272 (prebuild), :292, :318 (zero), :336 (kwargs)."""
    return (
        pipelined,
        _route(pipelined),
        pipelined,
        True,
        (pipelined and fp8_ag),
    )


def _route(owns):
    return _Route.PIPELINED if owns else _Route.FUSED_MOE_GMM


def _new(plan: mcp.CollectivePlan):
    zeroed = plan.stages_collectives if plan.zero_padded_routing_weights else None
    route = _route(plan.stages_collectives)
    return (
        plan.claims_collectives,
        route,
        zeroed,
        plan.kernel_skip_padded_tokens,
        plan.quantize_all_gather,
    )


def _reference_fixed_chunk_count(buckets, parallel_size, chunk_size):
    if chunk_size <= 0 or parallel_size <= 1 or not buckets:
        return 1
    needed = -(-max(buckets) * parallel_size // chunk_size)
    common = math.gcd(*buckets)
    count = needed
    while count < common and common % count:
        count += 1
    return min(count, common)


def _reference_calculate_moe_chunks(seq_len, parallel_size, chunk_size):
    if chunk_size <= 0 or parallel_size <= 1 or seq_len <= 0:
        return 1, seq_len
    total_tokens = seq_len * parallel_size
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    if seq_len % num_chunks != 0:
        raise ValueError(
            f"Local sequence length ({seq_len}) is not divisible by chunk count "
            f"({num_chunks})."
        )
    chunk_size_local = seq_len // num_chunks
    return num_chunks, chunk_size_local


def _reference_collection_chunks(seq_len, parallel_size, _fixed_chunk_buckets, env):
    """pipelined_fused_moe.py:83-92, TPU_MOE_COLLECTION_CHUNK_SIZE as `env`."""
    if _fixed_chunk_buckets is not None and parallel_size > 1:
        num_chunks = _reference_fixed_chunk_count(
            _fixed_chunk_buckets, parallel_size, env
        )
        return num_chunks, seq_len // num_chunks
    return _reference_calculate_moe_chunks(seq_len, parallel_size, env)


def _outcome(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        return (ValueError, str(e))


class CollectivePlanOracleTest(unittest.TestCase):
    """The collective formulas and chunking against d3991ebf9."""

    def test_no_tpu_runtime(self):
        self.assertEqual(_pure_import.loaded_tpu_runtime(), [])

    def test_ownership_formulas(self):
        cases = 0
        for pipelined, fused, adaptive, fp8_ag, skip in itertools.product(
            (False, True), repeat=5
        ):
            overrides = mcp.CollectiveOverrides(
                chunk_pipelining=pipelined,
                fp8_all_gather=fp8_ag,
                skip_padded_tokens=skip,
            )
            fp8 = mcp.collective_plan(
                mcp.MoeLayerInfo(
                    rule=CollectiveRule.FP8_ALL_GATHER,
                    ep_fused_supported=fused,
                    adaptive_supported=adaptive,
                ),
                overrides,
            )
            self.assertEqual(
                _new(fp8), _reference_fp8(pipelined, fused, adaptive, fp8_ag, skip)
            )
            unq = mcp.collective_plan(
                mcp.MoeLayerInfo(
                    rule=CollectiveRule.PIPELINING_ONLY,
                    ep_fused_supported=fused,
                    adaptive_supported=adaptive,
                ),
                overrides,
            )
            self.assertEqual(_new(unq), _reference_unquantized(pipelined, fp8_ag, skip))
            for uses_mxfp4 in (False, True):
                # tpu_runner.py:5588.
                self.assertEqual(
                    mcp.needs_token_padding_state(
                        skip_padded_tokens=skip, uses_mxfp4_moe=uses_mxfp4
                    ),
                    skip or uses_mxfp4,
                )
            per_path: tuple[tuple[CollectiveRule, object], ...] = (
                (
                    CollectiveRule.FUSED_KERNEL,
                    _reference_nvfp4(pipelined, fused, adaptive, fp8_ag, skip),
                ),
                (
                    CollectiveRule.PIPELINING_ONLY,
                    _reference_pipelined_only(pipelined, fp8_ag, skip),
                ),
                (
                    CollectiveRule.PIPELINING_ONLY,
                    _reference_pipelined_only(pipelined, fp8_ag, skip),
                ),
                (
                    CollectiveRule.FUSED_EP_KERNEL,
                    _reference_mxfp4_dsv4(pipelined, fused, fp8_ag, skip),
                ),
                (
                    CollectiveRule.ALWAYS_SKIP_PADDED,
                    _reference_w4an_mxfp4(pipelined, fp8_ag),
                ),
            )
            for rule, expected in per_path:
                plan = mcp.collective_plan(
                    mcp.MoeLayerInfo(
                        rule=rule,
                        ep_fused_supported=fused,
                        adaptive_supported=adaptive,
                    ),
                    overrides,
                )
                self.assertEqual(_new(plan), expected, rule)
                cases += 1
            # moe_tp_decode.py:203 mirrors the fp8 dispatch formula.
            self.assertEqual(fp8.stages_collectives, pipelined or fp8_ag)
            # moe_runner.py:124-125 and :136-138.
            self.assertEqual(
                mcp.fused_kernel_armed(
                    ep_fused_supported=fused, adaptive_supported=adaptive
                ),
                fused or adaptive,
            )
            cases += 2
        print(f"\n[ownership] {cases} cases")

    def test_all2all_patch(self):
        # __init__.py:280-289, with the chunk size read from the env.
        saved = mcp.chunk_size_from_envs
        cases = 0
        try:
            for use_ep, pcp, dp, sp, chunk in itertools.product(
                (False, True), (1, 2, 4), (1, 2), (False, True), (-1, 0, 1, 4096)
            ):
                mcp.chunk_size_from_envs = lambda chunk=chunk: chunk
                expected = use_ep and pcp > 1 and dp == 1 and not sp and chunk <= 0
                self.assertEqual(
                    mcp.disables_all2all_for_explicit_pcp(
                        use_ep=use_ep, pcp_size=pcp, dp_size=dp, is_sequence_parallel=sp
                    ),
                    expected,
                )
                cases += 1
        finally:
            mcp.chunk_size_from_envs = saved
        print(f"\n[all2all patch] {cases} cases")

    def test_chunking(self):
        seq_lens = (0, 1, 2, 3, 7, 8, 16, 31, 64, 96, 128, 255, 256, 1000, 1024, 4096)
        parallels = (1, 2, 3, 4, 8, 16)
        chunk_sizes = (-1, 0, 1, 7, 16, 100, 512, 4096, 16384)
        bucket_sets = (
            None,
            (),
            (8,),
            (8, 16, 32),
            (16, 48),
            (12, 18, 30),
            (64, 128, 256, 512),
            (7, 11),
        )
        cases = 0
        for seq_len, parallel, chunk, buckets in itertools.product(
            seq_lens, parallels, chunk_sizes, bucket_sets
        ):
            self.assertEqual(
                _outcome(
                    mcp.collection_chunks,
                    seq_len,
                    parallel,
                    chunk_size=chunk,
                    fixed_buckets=buckets,
                ),
                _outcome(
                    _reference_collection_chunks, seq_len, parallel, buckets, chunk
                ),
                (seq_len, parallel, chunk, buckets),
            )
            cases += 1
        print(f"\n[chunking] {cases} cases")

    def test_stages_implies_claims(self):
        with self.assertRaises(ValueError):
            mcp.CollectivePlan(
                claims_collectives=False,
                stages_collectives=True,
                quantize_all_gather=False,
                zero_padded_routing_weights=False,
                kernel_skip_padded_tokens=False,
            )


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    unittest.main()
