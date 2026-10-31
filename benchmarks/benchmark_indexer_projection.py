# SPDX-License-Identifier: Apache-2.0
"""Item-13 library microbenchmarks, not end-to-end DSV4 throughput."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.backends.rocm.gemm.indexer_projection import AiterIndexerProjectionBackend
from rl_engine.contracts.operators.indexer_projection import IndexerProjectionContract
from rl_engine.models.deepseek_v4.indexer_projection import IndexerProjection, PreparedIndexerFP8Q


def microseconds(function, repeats):
    for _ in range(5):
        function()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6 / repeats)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    from vllm.model_executor.layers import batch_invariant as vllm_bi

    backend = AiterIndexerProjectionBackend()
    torch.manual_seed(1313)
    c = IndexerProjectionContract()
    fp8_contract = replace(c, q_precision="block_fp8_ue8m0")
    bf16_op = IndexerProjection(c, layer_id="bench", backend=backend)
    fp8_op = IndexerProjection(fp8_contract, layer_id="bench", backend=backend)
    wq = torch.randn((8192, 1024), device="cuda", dtype=torch.bfloat16)
    ww = torch.randn((64, 4096), device="cuda", dtype=torch.bfloat16)
    packed = PreparedIndexerFP8Q.from_checkpoint(
        wq.to(torch.float8_e4m3fn),
        torch.ones((64, 8), device="cuda", dtype=torch.float32),
        version="bench-v1",
        backend=backend,
    )
    cases = []
    with torch.no_grad():
        for m in (1, 8, 128, 512):
            qr = torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16)
            hidden = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16)
            calls = {
                "q_bf16_aiter_us": partial(
                    backend.linear, qr, wq, output_dtype=torch.bfloat16, role="q"
                ),
                "q_bf16_vllm_bi_us": partial(vllm_bi.linear_batch_invariant, qr, wq),
                "weights_bf16_aiter_us": partial(
                    backend.linear, hidden, ww, output_dtype=torch.bfloat16, role="weights"
                ),
                "weights_bf16_vllm_bi_us": partial(vllm_bi.linear_batch_invariant, hidden, ww),
                "weights_fp32_aiter_us": partial(
                    backend.linear, hidden, ww, output_dtype=torch.float32, role="weights"
                ),
                "bf16_q_fp32_weights_chain_us": partial(bf16_op, qr, hidden, wq, ww),
                "fp8_q_fp32_weights_chain_us": partial(
                    fp8_op,
                    qr,
                    hidden,
                    None,
                    ww,
                    packed_q=packed,
                    q_weight_version="bench-v1",
                ),
            }
            cases.append(
                {"M": m, **{name: microseconds(fn, args.repeats) for name, fn in calls.items()}}
            )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "backend": backend.provenance,
        "vllm_bi_reference_sha256": hashlib.sha256(Path(vllm_bi.__file__).read_bytes()).hexdigest(),
        "contract": asdict(c),
        "plans": {
            "q": asdict(backend.plan(1024, 8192, "q")),
            "weights": asdict(backend.plan(4096, 64, "weights")),
            "fp8_q": asdict(backend.plan(1024, 8192, "q", "block_fp8_ue8m0")),
        },
        "metric": "median of five synchronized wall-time loops; Python dispatch included",
        "caveat": (
            "Different reduction contracts; native-vLLM comparisons use BF16 outputs only. "
            "No model-throughput claim."
        ),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
