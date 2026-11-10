# Copyright 2025 Google LLC
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
"""Every receiver of a MoE combine decision accepts only the resolved type.

A resolved type (`resolution.require_no_default`) has no field defaults, so
importing `moe_plan` already fails if `MoeCombinePlan` gains one. This check
covers the receiving side: every parameter in `moe_plan` and `fused_moe_gmm`
annotated with a resolved type takes exactly that type, with no default, so no
child can be handed "not decided yet" and decide it again. It reads
annotations only, not what each caller passes.
"""

import inspect
import types
import typing

from vllm_torchtpu.layers.core import fused_moe_gmm, moe_plan
from vllm_torchtpu.layers.core.resolution import is_resolved


def _names_resolved(hint: object) -> bool:
    if is_resolved(hint):
        return True
    return isinstance(hint, types.UnionType) and any(
        is_resolved(a) for a in typing.get_args(hint)
    )


def main() -> None:
    bad = 0
    for module in (moe_plan, fused_moe_gmm):
        for name, fn in vars(module).items():
            if not inspect.isfunction(fn) or fn.__module__ != module.__name__:
                continue
            hints = typing.get_type_hints(fn)
            for param in inspect.signature(fn).parameters.values():
                hint = hints.get(param.name)
                if not _names_resolved(hint):
                    continue
                ok = is_resolved(hint) and param.default is inspect.Parameter.empty
                bad += not ok
                shown = hint.__name__ if isinstance(hint, type) else hint
                print(f"{'ok ' if ok else 'BAD'} {name}({param.name}: {shown})")
    print(f"violations: {bad}")


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
