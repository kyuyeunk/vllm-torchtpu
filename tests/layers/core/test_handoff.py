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
"""Every receiver of a MoE combine decision accepts only the resolved type."""

import pytest

from tests.layers.core import handoff_check


def test_receivers_found() -> None:
    """The check sees the one resolved-type receiver today, so it is not vacuous."""
    found = [(r.function, r.param) for r in handoff_check.receivers()]
    assert found == [("moe_combine", "plan")]


def _receiver_id(receiver: handoff_check.Receiver) -> str:
    return f"{receiver.function}-{receiver.param}"


@pytest.mark.parametrize("receiver", handoff_check.receivers(), ids=_receiver_id)
def test_receiver_takes_resolved_type_without_default(
    receiver: handoff_check.Receiver,
) -> None:
    """A child cannot be handed "not decided yet" and decide it again."""
    assert receiver.ok, receiver.line()
