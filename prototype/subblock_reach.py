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
"""Prints which current models conform to norm_then_attention (README table).

Fetches only each candidate's Hugging Face ``config.json`` (no weights); the
survey is ``tests/blocks/subblock_reach.py`` and ``test_subblock_reach.py``
asserts its verdicts::

    python -m prototype.subblock_reach
    python -m prototype.subblock_reach --main   # against a vLLM main checkout
"""

import argparse

from tests.blocks import subblock_reach


class _Args(argparse.Namespace):
    main: bool = False


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--main",
        action="store_true",
        help="survey MAIN_CANDIDATES (families vLLM v0.33.0 lacks)",
    )
    args = parser.parse_args(namespace=_Args())
    subblock_reach.report(
        subblock_reach.MAIN_CANDIDATES if args.main else subblock_reach.CANDIDATES,
        print,
    )
