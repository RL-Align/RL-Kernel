# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Real-model logprob memory/latency benchmark against Qwen3-30B-A3B.

Downloads real weights from the HF Hub (default: Qwen/Qwen3-30B-A3B) and runs a real
forward pass to obtain real hidden states and the model's real lm_head weight -- unlike
benchmarks/benchmark_linear_logp.py, which uses synthetic random tensors. Compares native
(materializing log_softmax + gather) against the dispatched rl_engine linear_logp kernel
as token count scales into the model's remaining VRAM headroom.

Usage:
    python benchmarks/benchmark_qwen3_moe_real_model.py
    python benchmarks/benchmark_qwen3_moe_real_model.py --model Qwen/Qwen3-30B-A3B \
        --n-configs 2048,4096,8192,12288,16384
"""

import argparse
import time

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_engine.kernels.registry import kernel_registry
from rl_engine.utils.logger import logger

DEFAULT_N_CONFIGS = [2048, 4096, 8192, 12288, 16384, 20480, 24576]
DEFAULT_PROMPT = (
    "Explain the tradeoffs between memory usage and compute latency when "
    "computing log probabilities over a large vocabulary in reinforcement "
    "learning post-training for large language models."
)


def gb(x):
    return x / (1024**3)


def native_logprob(hidden, weight, target):
    """Standard full log_softmax + gather -- O(N*V) extra memory."""
    logits = torch.nn.functional.linear(hidden, weight)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)


def measure(fn, warmup=2, iters=5):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    return gb(peak - base), (t1 - t0) / iters * 1000


def run_benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU.")
    dtype = getattr(torch, args.dtype)

    torch.cuda.reset_peak_memory_stats()
    baseline_before_load = torch.cuda.memory_allocated()

    t0 = time.time()
    logger.info(f"Loading {args.model} in {dtype} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map={"": 0})
    model.eval()
    torch.cuda.synchronize()
    load_s = time.time() - t0

    weight_gb = gb(torch.cuda.memory_allocated() - baseline_before_load)
    total_gb = gb(torch.cuda.get_device_properties(0).total_memory)
    headroom_gb = total_gb - weight_gb
    logger.info(
        f"Loaded in {load_s:.1f}s. Weight VRAM: {weight_gb:.2f} GB / {total_gb:.2f} GB "
        f"total -> headroom {headroom_gb:.2f} GB"
    )

    hidden_size = model.config.hidden_size
    vocab_size = model.config.vocab_size
    lm_head_weight = model.lm_head.weight.detach()
    logger.info(
        f"hidden_size={hidden_size} vocab_size={vocab_size} "
        f"lm_head shape={tuple(lm_head_weight.shape)}"
    )

    enc = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    logger.info("Running a real forward pass to obtain real hidden states...")
    with torch.no_grad():
        out = model.model(**enc, output_hidden_states=False, use_cache=False)
    real_hidden = out.last_hidden_state.detach()  # [1, seq, H]
    real_hidden = real_hidden.reshape(-1, hidden_size).to(dtype).contiguous()
    logger.info(f"Real hidden states shape: {tuple(real_hidden.shape)}")

    linear_logp_op = kernel_registry.get_op("linear_logp")
    logger.info(f"Dispatched linear_logp backend: {type(linear_logp_op).__name__}")

    def make_batch(n):
        reps = (n + real_hidden.shape[0] - 1) // real_hidden.shape[0]
        hidden = real_hidden.repeat(reps, 1)[:n].clone()
        target = torch.randint(0, vocab_size, (n,), device="cuda")
        return hidden, target

    rows = []
    for n in args.n_configs:
        try:
            hidden, target = make_batch(n)
        except torch.cuda.OutOfMemoryError:
            rows.append([n, "OOM (input alloc)", "OOM (input alloc)", "N/A", "N/A", "N/A"])
            torch.cuda.empty_cache()
            continue

        try:
            native_extra, native_ms = measure(
                lambda: native_logprob(hidden, lm_head_weight, target)
            )
            native_str, native_ms_str = f"{native_extra:.2f} GB", f"{native_ms:.2f} ms"
        except torch.cuda.OutOfMemoryError:
            native_str, native_ms_str = "OOM", "N/A"
        torch.cuda.empty_cache()

        try:
            kernel_extra, kernel_ms = measure(
                lambda: linear_logp_op(hidden, lm_head_weight, target)
            )
            kernel_str, kernel_ms_str = f"{kernel_extra:.2f} GB", f"{kernel_ms:.2f} ms"
        except torch.cuda.OutOfMemoryError:
            kernel_str, kernel_ms_str = "OOM", "N/A"
        torch.cuda.empty_cache()

        rows.append(
            [
                n,
                native_str,
                kernel_str,
                native_ms_str,
                kernel_ms_str,
                f"current alloc: {gb(torch.cuda.memory_allocated()):.1f} GB",
            ]
        )
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print(f"{args.model} REAL MODEL LOGPROB BENCHMARK on {torch.cuda.get_device_name(0)}")
    print(
        f"Weight VRAM: {weight_gb:.2f} GB | Total: {total_gb:.2f} GB | "
        f"Headroom: {headroom_gb:.2f} GB"
    )
    print(f"linear_logp backend: {type(linear_logp_op).__name__}")
    print("=" * 100)
    print(
        tabulate(
            rows,
            headers=[
                "N tokens",
                "Native extra VRAM",
                "RL-Kernel extra VRAM",
                "Native ms",
                "RL-Kernel ms",
                "note",
            ],
            tablefmt="github",
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--n-configs",
        type=str,
        default=None,
        help="Comma-separated token counts, e.g. '2048,4096,8192'.",
    )
    args = parser.parse_args()
    args.n_configs = (
        [int(x) for x in args.n_configs.split(",")] if args.n_configs else DEFAULT_N_CONFIGS
    )
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
