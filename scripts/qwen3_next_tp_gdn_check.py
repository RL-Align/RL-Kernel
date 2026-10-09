#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4 GDN gate on the official checkpoint's layer-0 weights; four GPUs, torchrun.

    torchrun --nproc-per-node 4 scripts/qwen3_next_tp_gdn_check.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --output <dir>

A one-layer integration gate, not full-model evidence. Requires, bitwise: the
HF load/export round trip of the actual parameter storage; full prefill ==
chunked prefill, output and convolution/recurrent state handoff, for 8-1024
tokens; the training response recompute == the no-grad forward; a gradient that
reaches the prompt through the state; identical replicated norm gradients.
"""

import json

import torch
import torch.distributed as dist
from qwen3_next_gate_common import check_gradients, run_gate
from safetensors import safe_open

from rl_engine.integrations.qwen3_next_provider import GDNProviderConfig
from rl_engine.integrations.qwen3_next_tp_gdn import _GLOBAL_SHAPES, TP4GDN, assemble_gdn_weights
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def load_first_gdn(checkpoint):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for name in _GLOBAL_SHAPES:
        key = f"model.layers.0.linear_attn.{name}"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as reader:
            result[name] = reader.get_tensor(key)
    return result


def gdn_gate(checkpoint, generator):
    weights = load_first_gdn(checkpoint)
    module = TP4GDN(group=dist.group.WORLD, device="cuda", config=GDNProviderConfig())
    module.load_hf_weights(weights)
    # Validate actual parameter storage, not just the pure conversion helper.
    shards = [dict() for _ in range(4)]
    for name, parameter in module.named_parameters():
        gathered = [torch.empty_like(parameter) for _ in range(4)]
        dist.all_gather(gathered, parameter.detach())
        for peer, value in enumerate(gathered):
            shards[peer][name] = value.cpu()
    for name, restored in assemble_gdn_weights(shards).items():
        exact(restored, weights[name], name=f"loaded weight {name}")
    del shards, weights
    cases = []
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        for length in (8, 64, 256, 1024):
            hidden = (
                torch.randn(length, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
                * 0.1
            )
            initial = module.initial_state(2)
            whole, expected_state = module(hidden, initial, indices, torch.tensor([0, length]))
            cut = min(65, length - 1)
            first, state = module(hidden[:cut], initial, indices, torch.tensor([0, cut]))
            last, state = module(hidden[cut:], state, indices, torch.tensor([0, length - cut]))
            exact(torch.cat((first, last)), whole, name=f"TP4 layer length={length}")
            exact(state.convolution, expected_state.convolution, name="convolution handoff")
            exact(state.recurrent, expected_state.recurrent, name="recurrent handoff")
            cases.append(f"full-chunk-{length}")
    hidden = (
        torch.randn(80, 2048, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
    ).requires_grad_()
    initial = module.initial_state(2)
    with torch.no_grad():
        expected, _ = module(hidden, initial, indices, torch.tensor([0, 80]))
    _, prompt_state = module(hidden[:64], initial, indices, torch.tensor([0, 64]))
    actual, _ = module(hidden[64:], prompt_state, indices, torch.tensor([0, 16]))
    exact(actual, expected[64:], name="independent training response")
    actual.float().square().mean().backward()
    if (
        hidden.grad is None
        or not torch.isfinite(hidden.grad).all()
        or not hidden.grad[:64].count_nonzero()
    ):
        raise AssertionError("Response gradient did not reach prompt input")
    grad_stats = check_gradients(module)
    norm_grads = [torch.empty_like(module.norm.weight.grad) for _ in range(4)]
    dist.all_gather(norm_grads, module.norm.weight.grad)
    for other in norm_grads:
        exact(other, module.norm.weight.grad, name="replicated norm gradient")
    cases.extend(("prompt-gradient", "replicated-norm-gradient", "hf-load-export"))
    return {"cases": cases, "gradient_max_abs": grad_stats}


if __name__ == "__main__":
    run_gate(__doc__, "tp4_gdn_layer0", gdn_gate)
