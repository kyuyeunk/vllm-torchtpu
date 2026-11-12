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
"""The attention-then-MoE block: PR #1492's collective overlap, framework-owned.

``pattern`` is the reference forward and the parts' kinds; ``plan`` resolves
which collectives join the two halves per bucket; ``runner`` is the one
forward every installed block runs; ``install`` conforms a declared layer
and builds the runner from its declared parts; ``contract_adapter`` writes
the MoE child's contracts, which the model built without them.
"""
