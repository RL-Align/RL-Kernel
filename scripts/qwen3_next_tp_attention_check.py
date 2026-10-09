#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4 full-attention gate on the official checkpoint's layer-3 weights; four GPUs, torchrun.

    torchrun --nproc-per-node 4 scripts/qwen3_next_tp_attention_check.py \\
        --checkpoint <Qwen3-Next-80B-A3B-Instruct> --output <dir>

Requires, bitwise: the HF load/export round trip with the KV head replicated on
each rank pair; full prefill == chunked prefill (output and KV continuation) for
8-1024 tokens; a variable-length batch of four == each sequence alone, in any
order; a sixteen-token decode == the prefill rows; the training recompute == the
no-grad forward; and identical KV-pair and QK-norm gradients across replicas.
"""

from contextlib import ExitStack

import torch
import torch.distributed as dist
from qwen3_next_gate_common import CheckpointWeights, check_gradients, gather, run_gate

from rl_engine.integrations.qwen3_next_tp_blocks import (
    ATTENTION_HF_SHAPES,
    TP4FullAttention,
    assemble_attention_weights,
)
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def attention_gate(checkpoint, generator):
    module = TP4FullAttention(group=dist.group.WORLD, device="cuda")
    with ExitStack() as stack:
        weights = CheckpointWeights(
            checkpoint, "model.layers.3.self_attn.", ATTENTION_HF_SHAPES, stack
        )
        module.load_hf_weights(weights)
        shards = [dict() for _ in range(4)]
        for name, tensor in module.export_local_hf_weights().items():
            for rank, value in enumerate(gather(tensor)):
                shards[rank][name] = value.cpu()
        for name, tensor in assemble_attention_weights(shards).items():
            exact(tensor, weights[name], name=f"attention HF {name}")
    del shards, weights
    cases = ["hf-load-export"]
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        for length in (8, 64, 256, 1024):
            hidden = (
                torch.randn(length, 2048, device="cuda", dtype=torch.bfloat16, generator=generator)
                * 0.1
            )
            state = module.initial_state(2)
            whole, expected = module(hidden, state, slots, torch.tensor([0, length]))
            cut = min(65, length - 1)
            first, state = module(hidden[:cut], state, slots, torch.tensor([0, cut]))
            last, state = module(hidden[cut:], state, slots, torch.tensor([0, length - cut]))
            exact(torch.cat((first, last)), whole, name=f"attention full/chunk {length}")
            exact(state.keys[1], expected.keys[1], name="attention key continuation")
            exact(state.values[1], expected.values[1], name="attention value continuation")
            cases.append(f"full-chunk-{length}")
        hidden = (
            torch.randn(88, 2048, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
        )
        packed, state = module(
            hidden,
            module.initial_state(5),
            torch.tensor([1, 2, 3, 4], device="cuda"),
            torch.tensor([0, 8, 24, 48, 88]),
        )
        order = [3, 1, 0, 2]
        segments = [hidden[:8], hidden[8:24], hidden[24:48], hidden[48:]]
        ends = [0]
        for index in order:
            ends.append(ends[-1] + segments[index].shape[0])
        reordered, _ = module(
            torch.cat([segments[i] for i in order]),
            module.initial_state(5),
            torch.tensor([i + 1 for i in order], device="cuda"),
            torch.tensor(ends),
        )
        expected = packed.split((8, 16, 24, 40))
        exact(
            reordered,
            torch.cat([expected[i] for i in order]),
            name="attention variable-length batch4 reorder",
        )
        for index, segment in enumerate(segments):
            single, _ = module(
                segment, module.initial_state(2), slots, torch.tensor([0, segment.shape[0]])
            )
            exact(single, expected[index], name="attention batch4/single")
        cases.append("batch4-varlen-reorder")
    hidden = (
        torch.randn(80, 2048, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
    ).requires_grad_()
    with torch.no_grad():
        expected, _ = module(hidden, module.initial_state(2), slots, torch.tensor([0, 80]))
        _, decode_state = module(hidden[:64], module.initial_state(2), slots, torch.tensor([0, 64]))
        decoded = []
        for token in hidden[64:].split(1):
            value, decode_state = module(token, decode_state, slots, torch.tensor([0, 1]))
            decoded.append(value)
        exact(torch.cat(decoded), expected[64:], name="attention sixteen-token decode")
    _, state = module(hidden[:64], module.initial_state(2), slots, torch.tensor([0, 64]))
    output, _ = module(hidden[64:], state, slots, torch.tensor([0, 16]))
    exact(output, expected[64:], name="attention independent response recompute")
    output.float().square().mean().backward()
    if (
        hidden.grad is None
        or not torch.isfinite(hidden.grad).all()
        or not hidden.grad[:64].count_nonzero()
    ):
        raise AssertionError("Attention response gradient did not reach the prompt")
    gradients = check_gradients(module)
    for projection in (module.k_proj, module.v_proj):
        peers = gather(projection.weight.grad)
        exact(peers[0], peers[1], name="KV replica gradient pair0")
        exact(peers[2], peers[3], name="KV replica gradient pair1")
    for norm in (module.q_norm, module.k_norm):
        for peer in gather(norm.weight.grad):
            exact(peer, norm.weight.grad, name="QK norm replicated gradient")
    cases.extend(
        ("sixteen-token-decode", "prompt-gradient", "kv-pair-gradient", "qk-norm-gradient")
    )
    return {"cases": cases, "gradient_max_abs": gradients}


if __name__ == "__main__":
    run_gate(
        __doc__,
        "tp4_full_attention_layer3",
        lambda checkpoint, generator: {"attention": attention_gate(checkpoint, generator)},
    )
