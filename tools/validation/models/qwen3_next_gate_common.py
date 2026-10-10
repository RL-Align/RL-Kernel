# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared pieces of the Qwen3-Next TP4 real-weight gates (``scripts/qwen3_next_tp_*_check.py``).

Each gate is launched as ``torchrun --nproc-per-node 4 scripts/<gate>.py
--checkpoint <dir> --output <dir>`` and writes one ``rank-<r>.json``.
"""

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]


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


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def provenance():
    from rl_engine.models.qwen3_next.qwen3_next_forward import FORWARD_PROVIDER_ID

    return {
        "rl_kernel_commit": _git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "provider": FORWARD_PROVIDER_ID,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "vllm": version("vllm"),
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
        },
    }


def run_gate(description, scope, gate):
    """Parse ``--checkpoint/--output``, run ``gate(checkpoint, generator)`` on four ranks."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4 or torch.cuda.device_count() < 4:
        raise RuntimeError(f"The {scope} gate requires exactly four ranks on four GPUs")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=15), device_id=torch.device("cuda", rank)
    )
    try:
        generator = torch.Generator(device="cuda").manual_seed(1234)
        result = gate(args.checkpoint, generator)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"rank-{rank}.json"
        partial = path.with_suffix(".json.partial")
        payload = {"status": "passed", "scope": scope, "rank": rank, **provenance(), **result}
        partial.write_text(json.dumps(payload, indent=2) + "\n")
        partial.replace(path)
        dist.barrier()
    finally:
        dist.destroy_process_group()
