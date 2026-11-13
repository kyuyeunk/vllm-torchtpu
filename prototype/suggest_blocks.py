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
"""Prints proposed attn_then_moe declarations for vLLM's MoE decoder layers.

A suggestion tool: it never applies anything. The proposals come from
``tests/blocks/suggest_blocks.py``; ``test_suggest_blocks.py`` asserts each
layer's verdict::

    python -m prototype.suggest_blocks
    python -m prototype.suggest_blocks --json   # then the verdicts as JSON
"""

import argparse

from tests.blocks import suggest_blocks


class _Args(argparse.Namespace):
    json: bool = False


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--json", action="store_true", help="print the verdicts")
    args = parser.parse_args(namespace=_Args())
    suggest_blocks.report(print, as_json=args.json)
