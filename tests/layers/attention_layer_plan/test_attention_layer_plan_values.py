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
"""Plan invariance for the attention layer, numerics tier, on a TPU.

Each plan value declared capable on a TPU runs on the device, and its output
is compared with the float32 eager reference computed on the CPU from the same
weights, within the layer contract's tolerance (the tolerance the contract
states; `layer_contract` sets it for bfloat16 at atol 2e-2, rtol 1.6e-2).

The layer is built by `cpu_layer` (no plugin platform) and then moved to the
TPU; its attention child stays `RefAttention`, so this checks the layer's own
plan, not the attention kernel. Marked `tpu`: a default CPU run skips it.
"""

# isort: off
from tests.layers.attention_layer_plan import cpu_layer  # first: no TPU runtime

# isort: on
import pytest
import torch

from tests.layers.attention_layer_plan.plan_check import CASES, CHIP, TOKENS, LayerCase
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod


@pytest.mark.tpu
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_capable_values_match_cpu_reference(case: LayerCase) -> None:
    """Every value declared capable on a TPU is within tolerance on the device."""
    import torch_tpu  # noqa: F401, PLC0415  # pyright: ignore[reportMissingTypeStubs, reportUnusedImport]  # registers the "tpu" device; real TPU only

    config = case.config()
    probe = cpu_layer.build_real(config)
    caps = layer_mod.describe_children(probe)
    capable = [
        v
        for v in alp.enumerate_plans(probe.desc)
        if not alp.plan_declines(v, probe.desc, caps, CHIP)
    ]
    print(f"== {case.name}: {len(capable)} capable plan value(s) on a TPU")
    failures = 0
    for value in capable:
        reference = cpu_layer.build_real(config, value).float()
        device = cpu_layer.build_real(config, value).to("tpu")
        atol, rtol = device.contract.atol, device.contract.rtol
        worst = 0.0
        for t in TOKENS:
            positions, hidden = cpu_layer.inputs(config, t)
            with torch.no_grad():
                ref: torch.Tensor = reference(positions, hidden.float())
                got: torch.Tensor = device(positions.to("tpu"), hidden.to("tpu")).cpu()
            diff = (got.float() - ref).abs().max().item()
            worst = max(worst, diff)
            if not torch.allclose(got.float(), ref, atol=atol, rtol=rtol):
                failures += 1
        print(
            f"  {alp.plan_name(value)}: max |diff| {worst:.1e}"
            f" (atol {atol}, rtol {rtol})"
        )
    assert failures == 0, f"outside tolerance: {failures}"
