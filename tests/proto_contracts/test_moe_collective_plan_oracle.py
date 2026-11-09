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

import itertools
import math
import unittest

import _pure_import

from vllm_torchtpu.layers.core import moe_collective_plan as mcp


def _reference_fp8(pipelined, fused, adaptive, fp8_ag, skip):
    """fp8.py:804-833 and :1182-1222 (kernel skip via fused_moe.py:208-209)."""
    supports_internal_mk = pipelined or fused or adaptive or fp8_ag
    owns_collectives = pipelined or fp8_ag
    zeroed = ("zero", owns_collectives) if skip else None
    route = "pipelined" if owns_collectives else "fused_moe_gmm"
    quantize_ag = owns_collectives and fp8_ag  # pipelined_fused_moe.py:254
    return supports_internal_mk, route, zeroed, skip, quantize_ag


def _reference_unquantized(pipelined, fp8_ag, skip):
    """unquantized.py:253-256 and :379-382 (kernel skip via fused_moe.py)."""
    supports_internal_mk = pipelined
    route = "pipelined" if pipelined else "fused_moe_gmm"
    quantize_ag = pipelined and fp8_ag
    return supports_internal_mk, route, None, skip, quantize_ag


def _new(plan: mcp.CollectivePlan):
    zeroed = (
        ("zero", plan.stages_collectives) if plan.zero_padded_routing_weights else None
    )
    route = "pipelined" if plan.stages_collectives else "fused_moe_gmm"
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
        return ("ValueError", str(e))


class CollectivePlanOracleTest(unittest.TestCase):
    """C8 v1 formulas and chunking against d3991ebf9."""

    def test_no_runtime_imported(self):
        self.assertEqual(_pure_import.loaded_runtime_modules(), [])

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
                    quant="fp8", ep_fused_supported=fused, adaptive_supported=adaptive
                ),
                overrides,
            )
            self.assertEqual(
                _new(fp8), _reference_fp8(pipelined, fused, adaptive, fp8_ag, skip)
            )
            unq = mcp.collective_plan(
                mcp.MoeLayerInfo(
                    quant="unquantized",
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
            cases += 2
        print(f"\n[ownership] {cases} cases")

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


if __name__ == "__main__":
    unittest.main()
