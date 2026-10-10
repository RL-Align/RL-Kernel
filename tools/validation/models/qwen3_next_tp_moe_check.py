#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4 MoE gate on the official checkpoint's layer-0 weights; four GPUs, torchrun.

    TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 \\
        tools/validation/models/qwen3_next_tp_moe_check.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --output <dir>

Each rank writes ``rank-<r>.json``. The gate requires, bitwise:

* the HF load -> local shard -> all-gather -> HF export round trip of all 512 experts;
* identical top-10 routes on every rank (the router is replicated);
* full batch == concatenated chunks, and reordered tokens == reordered output,
  for 8, 64, 256 and 1024 tokens;
* the training forward (autograd on) == the no-grad forward;
* identical replicated router and shared-expert-gate gradients on every rank,
  and a finite, nonzero gradient for every parameter.
"""

import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from rl_engine.models.qwen3_next.qwen3_next_forward import shared_router, stable_top10_routes
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import (
    TP4MoE,
    assemble_moe_weight,
    moe_hf_shapes,
)
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact
from tools.validation.models.qwen3_next_gate_common import (
    CheckpointWeights,
    check_gradients,
    gather,
    run_gate,
    timed,
)

TOKEN_COUNTS = (8, 64, 256, 1024)


def moe_gate(checkpoint, generator):
    module = TP4MoE(group=dist.group.WORLD, device="cuda")
    with ExitStack() as stack:
        weights = CheckpointWeights(checkpoint, "model.layers.0.mlp.", moe_hf_shapes(), stack)
        module.load_hf_weights(weights)
        for name, value in module.export_local_hf_weights().items():
            restored = assemble_moe_weight(name, [peer.cpu() for peer in gather(value)])
            exact(restored, weights[name], name=f"MoE HF {name}")
    cases = ["hf-load-export-all512experts"]
    forward_us = {}
    with torch.no_grad():
        for length in TOKEN_COUNTS:
            hidden = (
                torch.randn(length, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
                * 0.1
            )
            # Every rank draws the same tokens from the same generator; verify it.
            for peer in gather(hidden):
                exact(peer, hidden, name="replicated input")
            routes = stable_top10_routes(shared_router(hidden, module.gate.weight))
            for peer in gather(routes.indices):
                exact(peer, routes.indices, name=f"routes across ranks {length}")
            module(hidden)
            whole, forward_us[str(length)] = timed(lambda: module(hidden))
            cut = min(65, length - 1)
            exact(
                torch.cat((module(hidden[:cut]), module(hidden[cut:]))),
                whole,
                name=f"MoE full/chunk {length}",
            )
            order = torch.arange(length - 1, -1, -1, device="cuda")
            exact(module(hidden[order]), whole[order], name=f"MoE reorder {length}")
            cases.extend((f"routes-replicated-{length}", f"full-chunk-reorder-{length}"))
    hidden = (
        torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
    ).requires_grad_()
    with torch.no_grad():
        expected = module(hidden)
    output = module(hidden)
    exact(output, expected, name="MoE independent training forward")
    output.float().square().mean().backward()
    if (
        hidden.grad is None
        or not torch.isfinite(hidden.grad).all()
        or not hidden.grad.count_nonzero()
    ):
        raise AssertionError("MoE input gradient is missing")
    gradients = check_gradients(module)
    for parameter in (module.gate.weight, module.shared_expert_gate.weight):
        for peer in gather(parameter.grad):
            exact(peer, parameter.grad, name="MoE replicated router/gate gradient")
    cases.extend(("training-forward", "all-parameter-gradient", "replicated-router-gradient"))
    return {"cases": cases, "gradient_max_abs": gradients, "forward_us": forward_us}


if __name__ == "__main__":
    run_gate(
        __doc__,
        "tp4_moe_block_layer0",
        lambda checkpoint, generator: {"moe": moe_gate(checkpoint, generator)},
    )
