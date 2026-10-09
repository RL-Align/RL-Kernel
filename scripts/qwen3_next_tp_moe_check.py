#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4 MoE gate on the official checkpoint's layer-0 weights; four GPUs, torchrun.

    TRITON_F32_DEFAULT=ieee torchrun --nproc-per-node 4 scripts/qwen3_next_tp_moe_check.py \\
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

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from rl_engine.integrations.qwen3_next_forward import (
    FORWARD_PROVIDER_ID,
    shared_router,
    stable_top10_routes,
)
from rl_engine.integrations.qwen3_next_tp_blocks import TP4MoE, assemble_moe_weight, moe_hf_shapes
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact

ROOT = Path(__file__).resolve().parents[1]
TOKEN_COUNTS = (8, 64, 256, 1024)


class CheckpointWeights(Mapping):
    """Keep shard headers open while reading only the requested layer's tensors."""

    def __init__(self, checkpoint, prefix, names, stack):
        self.root, self.prefix, self.names, self.stack = checkpoint, prefix, tuple(names), stack
        self.index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.readers = {}

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, name):
        if name not in self.names:
            raise KeyError(name)
        key = self.prefix + name
        shard = self.index[key]
        if shard not in self.readers:
            self.readers[shard] = self.stack.enter_context(
                safe_open(self.root / shard, framework="pt", device="cpu")
            )
        return self.readers[shard].get_tensor(key)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def gather(value):
    peers = [torch.empty_like(value) for _ in range(4)]
    dist.all_gather(peers, value.contiguous())
    return peers


def check_gradients(module):
    result = {}
    for name, parameter in module.named_parameters():
        grad = parameter.grad
        if grad is None or not torch.isfinite(grad).all() or not grad.count_nonzero():
            raise AssertionError(f"Missing finite nonzero gradient: {name}")
        result[name] = float(grad.float().abs().max())
    return result


def timed(fn):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) * 1000.0


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


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() < 4:
        raise RuntimeError("The TP4 MoE gate requires exactly four ranks on four GPUs")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=15), device_id=torch.device("cuda", rank)
    )
    try:
        generator = torch.Generator(device="cuda").manual_seed(1234)
        moe = moe_gate(args.checkpoint, generator)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / f"rank-{rank}.json",
            {
                "status": "passed",
                "scope": "tp4_moe_block_layer0",
                "rank": rank,
                "rl_kernel_commit": _git("rev-parse", "HEAD"),
                "tracked_tree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
                "provider": FORWARD_PROVIDER_ID,
                "environment": {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "vllm": version("vllm"),
                    "nccl": ".".join(map(str, torch.cuda.nccl.version())),
                },
                "moe": moe,
            },
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
