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

Shared by `test_handoff.py` and the `prototype.check_handoff` demo.
"""

import dataclasses
import inspect
import types
import typing

from vllm_torchtpu.layers.core import fused_moe_gmm, moe_plan
from vllm_torchtpu.layers.core.resolution import is_resolved


@dataclasses.dataclass(frozen=True)
class Receiver:
    """One parameter annotated with a resolved type, or a union holding one.

    Attributes:
        function: The function that takes the parameter.
        param: The parameter name.
        shown: The annotation as the report prints it.
        ok: Whether the parameter takes exactly the resolved type, no default.
    """

    function: str
    param: str
    shown: str
    ok: bool

    def line(self) -> str:
        """The report line: `ok ` or `BAD`, then the signature fragment."""
        mark = "ok " if self.ok else "BAD"
        return f"{mark} {self.function}({self.param}: {self.shown})"


def _names_resolved(hint: object) -> bool:
    if is_resolved(hint):
        return True
    if not isinstance(hint, types.UnionType):
        return False
    args: tuple[object, ...] = typing.get_args(hint)
    return any(is_resolved(a) for a in args)


def _shown(hint: object) -> str:
    return hint.__name__ if isinstance(hint, type) else str(hint)


def _has_no_default(param: inspect.Parameter) -> bool:
    default: object = param.default  # pyright: ignore[reportAny]  # typeshed types it Any.
    return default is inspect.Parameter.empty


def _module_receivers(module: types.ModuleType) -> list[Receiver]:
    found: list[Receiver] = []
    members: dict[str, object] = vars(module)
    for name, fn in members.items():
        if not inspect.isfunction(fn) or fn.__module__ != module.__name__:
            continue
        hints: dict[str, object] = typing.get_type_hints(fn)
        found.extend(
            Receiver(
                function=name,
                param=param.name,
                shown=_shown(hints[param.name]),
                ok=is_resolved(hints[param.name]) and _has_no_default(param),
            )
            for param in inspect.signature(fn).parameters.values()
            if param.name in hints and _names_resolved(hints[param.name])
        )
    return found


def receivers() -> list[Receiver]:
    """Every resolved-type parameter in `moe_plan` and `fused_moe_gmm`, in order."""
    return [r for m in (moe_plan, fused_moe_gmm) for r in _module_receivers(m)]
