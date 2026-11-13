#!/usr/bin/env python3
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
"""Fail when pool-layout facts are read around the record, or it gains a writer.

Two checks over ``src/vllm_torchtpu``, both by AST, importing nothing:

- ``reads``: a ratchet of direct reads of pool-layout facts outside the
  record (``cache_config.block_size``, ``mamba_block_size``,
  ``mamba_page_size_padded``, and ``VLLM_KV_CACHE_LAYOUT``), counted per
  ``(file, function)``. A new direct read fails; moving one onto
  ``pool_layout()`` fails until its entry is removed from ``known_reads``,
  so the list only shrinks.
- ``writers``: only ``pool_layout.py`` names the stage marker attribute.

    tools/check_pool_layout_reads.py           # both checks
    tools/check_pool_layout_reads.py reads     # the ratchet only
    tools/check_pool_layout_reads.py writers   # the single writer only
"""

from __future__ import annotations

import argparse
import ast
import collections
import enum
import pathlib
import sys
from typing import NamedTuple

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "src" / "vllm_torchtpu"
_RECORD = "pool_layout.py"
_MIN_CHAIN = 2  # an owner and the fact read off it


class Check(enum.Enum):
    """One of the two checks; the value is its name on the command line."""

    READS = "reads"
    WRITERS = "writers"


class Site(NamedTuple):
    """A function that reads a pool-layout fact directly."""

    file: str
    function: str


def producer_files() -> frozenset[str]:
    """Files that produce the facts, so may read them directly."""
    return frozenset(
        {
            _RECORD,
            "platforms/tpu_block_size_utils.py",
            "platforms/tpu_platform.py",
        }
    )


def fact_names() -> frozenset[str]:
    """The `cache_config` fields the record guards."""
    return frozenset({"block_size", "mamba_block_size", "mamba_page_size_padded"})


def known_reads() -> collections.Counter[Site]:
    """The direct reads that remain, each `Site` with its count."""
    return collections.Counter(
        {
            Site("__init__.py", "_dcp_fine_prefix_match_unit"): 1,
            Site(
                "__init__.py",
                "_patch_vllm_dcp_fine_prefix_hits.resolve_kv_cache_block_sizes",
            ): 1,
            Site(
                "__init__.py",
                "_patch_vllm_mamba_split_scheduler_block_size"
                ".split_with_scheduler_block_size",
            ): 1,
            Site(
                "core/experimental/srpt_prefill_scheduler.py",
                "SrptPrefillScheduler.schedule",
            ): 1,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPUConnectorScheduler.__init__",
            ): 1,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPURaidenConnectorWorker._admit_raiden_glm_kv_cache",
            ): 1,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPURaidenConnectorWorker._max_request_blocks",
            ): 1,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPURaidenConnectorWorker._register_stage3_request_blocks",
            ): 1,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPURaidenConnectorWorker._register_stage3_request_row_spans",
            ): 1,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPURaidenConnectorWorker._stage3_fa_skip_bytes",
            ): 2,
            Site(
                "distributed/kv_transfer/tpu_connector.py",
                "TPURaidenConnectorWorker._validate_glm_dcp_destination",
            ): 1,
            Site(
                "distributed/kv_transfer/zmq_shm_base.py",
                "ZmqShmKvConnectorBase._build_pool_spec",
            ): 1,
            Site(
                "distributed/kv_transfer/zmq_shm_base.py",
                "ZmqShmKvConnectorBase._warmup_block_sizes",
            ): 1,
            Site(
                "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py",
                "VllmDeepseekV4SWACache.__init__",
            ): 1,
            Site(
                "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_attention_op.py",
                "VllmDeepseekV4SWACache.get_kv_cache_spec",
            ): 1,
            Site(
                "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_compressor.py",
                "VllmCompressorStateCache.get_kv_cache_spec",
            ): 1,
            Site(
                "layers/adapter/custom_ops/deepseek_v4/deepseek_v4_indexer.py",
                "VllmDeepseekV4IndexerCache.get_kv_cache_spec",
            ): 1,
            Site(
                "layers/adapter/custom_ops/kda_attention_op.py",
                "build_kimi_pooled_kda_op.pooled_core",
            ): 1,
            Site("layers/core/gdn_attention.py", "_is_seq_along_lane_pool"): 1,
            Site("megakernel/adapter/plugin.py", "RunnerView.cache_block_size"): 1,
            Site(
                "models/vllm/deepseek_v4/attention.py",
                "VllmDeepseekV4MLAAttention._build_attn_op",
            ): 1,
            Site(
                "models/vllm/deepseek_v4/attention.py",
                "VllmDeepseekV4MLAAttention.get_kv_cache_spec",
            ): 1,
            Site(
                "runner/kv_cache_manager.py",
                "KVCacheManager._update_mamba_page_size_padded",
            ): 1,
            Site("runner/kv_cache_manager.py", "KVCacheManager.get_kv_cache_spec"): 1,
            # The placeholder that sizes the runner's placeholder block table.
            Site("runner/tpu_runner.py", "TPUModelRunner.__init__"): 1,
            Site(
                "runner/tpu_runner.py", "TPUModelRunner._add_pipeline_chunk_buckets"
            ): 2,
            Site("runner/tpu_runner.py", "TPUModelRunner._attention_kernel_shapes"): 1,
            Site(
                "runner/tpu_runner.py", "TPUModelRunner.pp_handoff_hbm_reserve_bytes"
            ): 1,
        }
    )


def _chain(node: ast.expr) -> list[str]:
    """The dotted names of an attribute chain, outermost first."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _is_direct_read(node: ast.AST) -> bool:
    """Whether `node` loads a guarded fact off a `cache_config`, or the env var."""
    if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
        return False
    chain = _chain(node)
    on_cache_config = (
        node.attr in fact_names()  # str-ok: scans source names
        and len(chain) >= _MIN_CHAIN
        and chain[-2].endswith("cache_config")
    )
    on_env = node.attr == "VLLM_KV_CACHE_LAYOUT"  # str-ok: scans source names
    return on_cache_config or on_env


def _reads_in(tree: ast.AST, rel: str, found: collections.Counter[Site]) -> None:
    """Count each direct read in `tree` against its enclosing function."""
    stack: list[str] = []

    def visit(node: ast.AST) -> None:
        named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if named:
            stack.append(node.name)
        if _is_direct_read(node):
            found[Site(rel, ".".join(stack) or "<module>")] += 1
        for child in ast.iter_child_nodes(node):
            visit(child)
        if named:
            _ = stack.pop()

    visit(tree)


def find_direct_reads(package: pathlib.Path) -> collections.Counter[Site]:
    """Every direct read of a pool-layout fact outside the producers."""
    found: collections.Counter[Site] = collections.Counter()
    producers = producer_files()
    for path in sorted(package.rglob("*.py")):
        rel = path.relative_to(package).as_posix()
        if rel in producers:  # str-ok: file paths are the scanned data
            continue
        _reads_in(ast.parse(path.read_text()), rel, found)
    return found


def marker_attribute(package: pathlib.Path) -> str:
    """The stage marker's attribute name, read from `pool_layout._ATTR`."""
    tree = ast.parse((package / _RECORD).read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and [_chain(t) for t in node.targets] == [["_ATTR"]]
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    message = f"no _ATTR string assignment in {_RECORD}"
    raise LookupError(message)


def find_writers(package: pathlib.Path) -> list[str]:
    """Files that name the stage marker attribute."""
    attr = marker_attribute(package)
    return [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if attr in path.read_text()  # str-ok: scans source text for the marker
    ]


def _report(line: str) -> None:
    """Write one line of the report to stdout."""
    print(line)  # noqa: T201 - the CLI's report is its output


def check_reads(package: pathlib.Path) -> bool:
    """Print the direct reads; return whether they equal `known_reads`."""
    found = find_direct_reads(package)
    known = known_reads()
    for site, count in sorted(found.items()):
        _report(f"{site.file}:{site.function}: {count}")
    new = {k: v for k, v in found.items() if v > known[k]}
    gone = {k: v for k, v in known.items() if found[k] < v}
    if new:
        _report(f"read these through pool_layout(), not cache_config: {new}")
    if gone:
        _report(f"moved onto the record; drop from known_reads: {gone}")
    _report(f"reads: {found.total()} in {len(found)} functions")
    return not new and not gone


def check_writers(package: pathlib.Path) -> bool:
    """Print the marker's writers; return whether only the record writes it."""
    writers = find_writers(package)
    _report(f"writers: {writers}")
    return writers == [_RECORD]


def run(check: Check, package: pathlib.Path) -> bool:
    """Run one check over `package`; return whether it passed."""
    match check:
        case Check.READS:
            return check_reads(package)
        case Check.WRITERS:
            return check_writers(package)


class _Args(argparse.Namespace):
    """The parsed command line."""

    def __init__(self) -> None:
        """Start from no checks named and this branch's package."""
        super().__init__()
        self.checks: list[Check] = []
        self.package: pathlib.Path = PACKAGE


def main(argv: list[str]) -> int:
    """Run the requested checks; return 1 if any failed."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    _ = parser.add_argument(
        "checks", nargs="*", type=Check, metavar="{reads,writers}"
    )
    _ = parser.add_argument("--package", type=pathlib.Path, default=PACKAGE)
    args = parser.parse_args(argv, namespace=_Args())
    results = [run(check, args.package) for check in args.checks or list(Check)]
    return 0 if all(results) else 1


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    sys.exit(main(sys.argv[1:]))
