# SPDX-License-Identifier: Apache-2.0
"""Small explicit-ABI benchmark; does not mutate any live framework request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch

from rl_engine.backends.rocm.attention.recent128_cache import VllmRecent128Backend
from rl_engine.contracts.operators.recent128 import CacheRequest, CacheVersion, Recent128Contract
from rl_engine.models.deepseek_v4.recent128 import (
    Recent128Batch,
    Recent128Cache,
    Recent128Workspace,
)


def time_us(function, iterations):
    for _ in range(5):
        function()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iterations)
    return statistics.median(samples)


def benchmark(batch_size, iterations, backend):
    contract = Recent128Contract(1)
    version = CacheVersion("benchmark", "none", "ue8m0")
    requests = tuple(CacheRequest(f"bench-{i}") for i in range(batch_size))
    device = torch.device("cuda")
    tables = torch.arange(batch_size * 3, dtype=torch.int32, device=device).view(batch_size, 3)
    metadata = SimpleNamespace(
        block_size=64,
        block_table=tables,
        slot_mapping=tables[:, 2].to(torch.int64) * 64,
        seq_lens=torch.full((batch_size,), 129, dtype=torch.int32, device=device),
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
    )
    started = time.perf_counter()
    batch = Recent128Batch.from_metadata(
        metadata,
        requests=requests,
        version=version,
        contract=contract,
    )
    torch.cuda.synchronize()
    metadata_bind_us = (time.perf_counter() - started) * 1e6
    cache = torch.zeros((batch_size * 3, contract.page_bytes), dtype=torch.uint8, device=device)
    full = torch.randn((batch_size * 129, 512), dtype=torch.bfloat16, device=device)
    full_slots = (
        (
            torch.arange(batch_size, device=device)[:, None] * 192
            + torch.arange(129, device=device)[None, :]
        )
        .flatten()
        .to(torch.int64)
    )
    from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache

    quantize_and_insert_k_cache(full, cache, full_slots, block_size=64, use_fnuz=False)
    session = Recent128Cache(cache, contract, layer_id="bench-layer", backend=backend)
    for request in requests:
        session.register(request, version, cached_length=128)
    kv = full.view(batch_size, 129, 512)[:, -1].contiguous()
    workspace = Recent128Workspace.allocate(batch)
    session.update(kv, batch, workspace=workspace)

    def native_read():
        backend.gather(cache, batch, workspace, contract)
        backend.indices(batch, workspace, contract)

    return {
        "batch": batch_size,
        "query_tokens_per_request": 1,
        "workspace_width": contract.workspace_width,
        "workspace_bytes": workspace.kv.numel() * workspace.kv.element_size(),
        "metadata_bind_us_once_per_step_shared_across_layers": metadata_bind_us,
        "upstream_write_us": time_us(lambda: backend.write(kv, cache, batch, contract), iterations),
        "upstream_gather_and_indices_us": time_us(native_read, iterations),
        "contract_replay_us": time_us(
            lambda: session.replay(batch, workspace=workspace),
            iterations,
        ),
        "metric": "median of 5 synchronized wall-time loops; includes Python launch overhead",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    backend = VllmRecent128Backend()
    result = {
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(),
        "backend": backend.provenance,
        "scope": "cache microbenchmark, NOT end-to-end model throughput",
        "cases": [benchmark(b, args.iterations, backend) for b in (1, 8, 32)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
