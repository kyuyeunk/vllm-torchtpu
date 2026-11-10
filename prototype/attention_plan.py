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
"""Prints, per config, the layer contract, the resolved plan and every plan value.

Prototype. For each config this shows:

- the contract: what crosses the Qwen3NextAttention boundary;
- the resolved plan, and the submodule contracts it hands down;
- every plan value, with its capability or each decline reason, including
  the whole layer as one fused kernel.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src python -m prototype.attention_plan
"""

import dataclasses

from vllm.platforms.interface import PlatformEnum

from prototype import _layer
from vllm_torchtpu.layers import attention_layer_plan as alp
from vllm_torchtpu.models.vllm import qwen3_next_attention as layer_mod

# (title, config overrides, chip, o_proj capability override or None).
TPU, CUDA = PlatformEnum.TPU, PlatformEnum.CUDA
CONFIGS = [
    ("gated output, TPU", {}, TPU, None),
    ("gated output, CUDA description", {}, CUDA, None),
    ("no output gate, TPU", {"attn_output_gate": False}, TPU, None),
    (
        "gated output, CUDA, o_proj declares FP8 static input (newer vLLM)",
        {},
        CUDA,
        frozenset({alp.Quant.NONE, alp.Quant.FP8_STATIC_TENSOR_SYM}),
    ),
]


def _spec(spec: alp.TensorSpec) -> str:
    shape = ", ".join(str(d) for d in spec.shape)
    return f"({shape}) {alp.dtype_name(spec.dtype)}, {spec.sharding.value}"


def _children(plan: alp.AttentionLayerPlan) -> str:
    """The submodule contracts this plan value hands down."""
    if isinstance(plan, alp.FusedLayerPlan):
        return "no submodule contracts (no intermediate)"

    def fmt(f: alp.ActivationFormat) -> str:
        scale = "" if f.scale is alp.Scale.NONE else f" @{f.scale.value}"
        return f"{alp.dtype_name(f.dtype)}/{f.quant.value}{scale}"

    return (
        f"attn output {fmt(plan.attn_output)}; gate {plan.gate.value}; "
        f"o_proj input {fmt(plan.o_proj_input)}"
    )


def show(title, overrides, chip, o_proj_inputs) -> None:
    layer = _layer.build(_layer.tiny_config(**overrides), chip=chip)
    caps = layer_mod.describe_children(layer)
    if o_proj_inputs is not None:
        caps = dataclasses.replace(caps, o_proj_inputs=o_proj_inputs)
    plan, _ = alp.resolve_plan(layer.desc, caps, alp.ChipDesc(chip))
    c = layer.contract
    print(f"=== {title}")
    print("contract")
    print(f"  positions      {_spec(c.positions)}")
    print(f"  hidden_states  {_spec(c.hidden_states)}")
    print(f"  output         {_spec(c.output)}")
    print(f"  tolerance      atol={c.atol} rtol={c.rtol}")
    print("  weights        " + ", ".join(f"{n}{list(s)}" for n, s in c.weights))
    print(f"resolved plan    {alp.plan_name(plan)}")
    print(f"  hands down     {_children(plan)}")
    print("plan values")
    for value in alp.enumerate_plans(layer.desc):
        reasons = alp.plan_declines(value, layer.desc, caps, alp.ChipDesc(chip))
        mark = "capable " if not reasons else "declines"
        print(f"  {mark} {alp.plan_name(value)}")
        print(f"           hands down: {_children(value)}")
        for reason in reasons:
            print(f"           - {reason}")
    print()


# (user override, model plan): partial plans, completed or rejected
# when the layer is built.
OVERRIDES = [
    ("gate=eager_mul", ""),
    ("gate=eager_mul", "qk_norm_rope_gate=eager,gate=fused_mul"),
    ("layer=unfused,gate=fused_mul_quant,o_proj_input=none", ""),
    ("output=float32", ""),
]


def show_overrides() -> None:
    print("=== overrides, gated output, TPU")
    for user, model in OVERRIDES:
        config = _layer.tiny_config(tpu_attention_layer_plan=model)
        try:
            layer = _layer.build(config, override=user)
        except ValueError as e:
            result = f"raises ValueError: {e}"
        else:
            src = layer.plan_sources
            result = ", ".join(
                f"{k.value}={v.value} ({src[k].value})"
                for k, v in alp.plan_fields(layer.plan).items()
            )
            result = f"{alp.plan_name(layer.plan)}\n    from {result}"
        print(f"  user={user!r} model plan={model!r}\n    {result}")


def main() -> None:
    with _layer.runtime():
        for config in CONFIGS:
            show(*config)
        show_overrides()


if __name__ == "__main__":  # str-ok: the module entry-point idiom
    main()
