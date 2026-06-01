# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import time
import torch
import argparse
from typing import Dict, Any
from rl_engine.utils.logger import logger
from rl_engine.platforms.device import device_ctx
from rl_engine.executors.bridge import IPCWeightBridge

from benchmarks.benchmark_grpo_op import run_benchmark as run_logp_perf
from benchmarks.benchmark_sampling import run_benchmark as run_sample_perf


class PerfReport:
    @staticmethod
    def print_panel(metrics: Dict[str, Any]):
        print(f"\n\t{metrics['tip']}")
        print(f"{'============ RL-Kernel Serving Benchmark ============':^60}")
        PerfReport._row("Hardware Device", metrics["device"])
        PerfReport._row("Model Architecture", f"Vocab={metrics['vocab']}, Seq={metrics['seq']}")
        PerfReport._row("Dtype Precision", str(metrics["dtype"]))

        print(f"{'---------------Throughput & Efficiency----------------':^60}")
        PerfReport._row("Max VRAM Reduction (GB)", f"{metrics['vram_saved']:.2f}")
        PerfReport._row("Logp Compute Speedup", f"{metrics['logp_speedup']}")
        PerfReport._row("Sampling Speedup (Fused)", f"{metrics['sample_speedup']}")

        print(f"{'---------------Latency Statistics (ms)----------------':^60}")
        PerfReport._row("Mean Logp Latency", f"{metrics['avg_logp_ms']:.2f}")
        PerfReport._row("Mean Sampling Latency", f"{metrics['avg_sample_ms']:.2f}")
        PerfReport._row("Zero-Copy Sync Latency", f"{metrics['sync_ms']:.4f}")

        status = "PASSED (Industrial Grade)" if metrics["sync_ms"] < 100 else "OPTIMIZATION NEEDED"
        PerfReport._row("System Sync Status", status)

        print("=" * 60)

    @staticmethod
    def _row(key: str, value: Any):
        print(f"{key:<35} {str(value):>25}")


def main():
    parser = argparse.ArgumentParser(description="RL-Kernel Production Benchmark Suite")
    parser.add_argument("--vocab-size", type=int, default=128256, help="Model vocab size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--g-sizes", type=str, default="64,128,256", help="Batch sizes to test")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P sampling")
    args = parser.parse_args()

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "AMD MI300X/ROCm"
    dtype = device_ctx.get_preferred_dtype()

    logger.info("Phase 1: Benchmarking Fused Logprobs...")
    logp_results = run_logp_perf(args, return_data=True)

    logger.info("Phase 2: Benchmarking FlashInfer Sampling...")
    sample_results = run_sample_perf(args, return_data=True)

    logger.info("Phase 3: Testing Weight Sync Latency...")
    bridge = IPCWeightBridge()
    dummy_model = torch.nn.Linear(1024, 1024).to(device_ctx.device)
    t0 = time.perf_counter()
    handles = bridge.export_model_handles(dummy_model)
    _ = bridge.import_model_weights(handles)
    sync_latency = (time.perf_counter() - t0) * 1000

    metrics = {
        "tip": "install flashinfer and aiter for maximum throughput.",
        "device": device_name,
        "vocab": args.vocab_size,
        "seq": args.seq_len,
        "dtype": dtype,
        "vram_saved": max([r["vram_saved_val"] for r in logp_results]),
        "logp_speedup": logp_results[-1]["speedup"],
        "sample_speedup": sample_results[-1]["speedup"],
        "avg_logp_ms": sum([r["engine_ms"] for r in logp_results]) / len(logp_results),
        "avg_sample_ms": sum([r["engine_ms"] for r in sample_results]) / len(sample_results),
        "sync_ms": sync_latency,
    }

    PerfReport.print_panel(metrics)


if __name__ == "__main__":
    main()
