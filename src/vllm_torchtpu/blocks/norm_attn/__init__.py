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
"""The pre-norm attention sub-block: one fusion for every model that declares it.

``pattern`` is the reference forward and the binding' kinds; ``plan`` picks the
value and holds the kernels' capabilities; ``runner`` is the one forward
every declared layer runs, shaped like vLLM's ``MoERunner``; ``install``
conforms a declared layer and builds the runner from its declared binding.
"""
