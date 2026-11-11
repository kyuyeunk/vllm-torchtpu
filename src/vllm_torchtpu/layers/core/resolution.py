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
"""Resolved types: what one level hands the level below, every field decided.

Each level owns its decisions and hands them down resolved (DATA_CONTRACTS
section 3). A resolved type has no field defaults, so whoever builds one
states every decision, and a child that accepts only resolved types cannot be
handed a decision nobody made. After tokamax's routing_moe
(`common/dataclass_util.py`).
"""

import dataclasses

_RESOLVED: set[type] = set()


def require_no_default[T: type](cls: T) -> T:
    """Rejects a dataclass field with a default, and marks `cls` resolved."""
    for f in dataclasses.fields(cls):
        if (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        ):
            raise ValueError(
                f"{cls.__name__}.{f.name} must not have a default, so whoever "
                "builds it states every decision"
            )
    _RESOLVED.add(cls)
    return cls


def is_resolved(cls: object) -> bool:
    """Whether `cls` is marked by `require_no_default`."""
    return cls in _RESOLVED
