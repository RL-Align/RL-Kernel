# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Profile one strict (or native) vLLM decode workload on ROCm with torch.profiler.

Starts an in-process ``vllm.LLM`` with the same engine settings the Vime
launcher uses for the PR377 workload (TP4, HIP Graph ``FULL_AND_PIECEWISE``,
capture size 32, AITER FA backend), warms the strict route, then records a
profiler trace of a decode-heavy generation through ``VLLM_TORCH_PROFILER_DIR``.
Summarize the per-rank traces afterwards with ``summarize_rollout_trace.py``.

Example::

    RL_KERNEL_CASE=R/R RL_KERNEL_ROCM_ATTENTION_BACKEND=triton \
    HIP_VISIBLE_DEVICES=0,1,2,3 python benchmarks/profiling/profile_rocm_rollout_decode.py \
        --trace-dir /tmp/rollout-trace --prompt-tokens 4000 --decode-tokens 64 --batch 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def configure_environment(case: str, trace_dir: Path, capture_size: int) -> None:
    # vLLM workers are separate processes: make them import this checkout's
    # rl_engine (and its vLLM plugin) rather than whatever editable install
    # is registered in site-packages.
    root = str(Path(__file__).resolve().parents[2])
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = root + (os.pathsep + existing if existing else "")
    # AITER JIT builds resolve GPU_ARCHS=native to nothing inside workers.
    if os.environ.get("GPU_ARCHS", "native") == "native":
        os.environ["GPU_ARCHS"] = os.environ.get("PYTORCH_ROCM_ARCH", "gfx942")
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
    os.environ.setdefault("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "ROCM_AITER_FA")
    os.environ.setdefault("RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE", str(capture_size))
    os.environ.setdefault("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", "151936")
    os.environ.setdefault("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE", "152064")
    os.environ.setdefault("RL_KERNEL_ROCM_FIXED_PAGED_TILE", "128")
    os.environ.setdefault("RL_KERNEL_ROCM_PAGED_KV_MAX_TOKENS", "8192")
    os.environ["RL_KERNEL_ATTENTION_CASE"] = case
    os.environ["RL_KERNEL_FFN_CASE"] = case
    os.environ["RL_KERNEL_LOGP_CASE"] = case
    os.environ["RL_KERNEL_VLLM_INTEGRATION"] = "1"
    os.environ.setdefault("RL_KERNEL_READBACK_DIR", str(trace_dir / "readbacks"))
    os.environ.setdefault("RL_KERNEL_MISMATCH_SIDECAR_DIR", str(trace_dir / "sidecars"))
    os.environ["VLLM_TORCH_PROFILER_DIR"] = str(trace_dir)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        os.environ.pop(name, None)
    Path(os.environ["RL_KERNEL_READBACK_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["RL_KERNEL_MISMATCH_SIDECAR_DIR"]).mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/app/model/Qwen3-8B")
    parser.add_argument("--case", default=os.environ.get("RL_KERNEL_CASE", "R/R"))
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.38)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--capture-size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=4000)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup-decode-tokens", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    args = parser.parse_args()

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    configure_environment(args.case, args.trace_dir, args.capture_size)

    import torch
    from vllm import LLM, SamplingParams

    engine_limits = {}
    if args.max_num_seqs is not None:
        engine_limits["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        engine_limits["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(
        model=args.model,
        **engine_limits,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        seed=1234,
        trust_remote_code=True,
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "max_cudagraph_capture_size": args.capture_size,
        },
    )
    generator = torch.Generator().manual_seed(7)
    vocab = 150000
    prompts = [
        {
            "prompt_token_ids": torch.randint(
                1000, vocab, (args.prompt_tokens,), generator=generator
            ).tolist()
        }
        for _ in range(args.batch)
    ]
    warm = SamplingParams(
        max_tokens=args.warmup_decode_tokens, temperature=1.0, ignore_eos=True, seed=1
    )
    t0 = time.time()
    llm.generate(prompts, warm)
    print(f"warmup generate: {time.time() - t0:.1f}s", flush=True)

    params = SamplingParams(max_tokens=args.decode_tokens, temperature=1.0, ignore_eos=True, seed=2)
    llm.start_profile()
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0
    llm.stop_profile()
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    summary = {
        "case": args.case,
        "batch": args.batch,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "generated_tokens": tokens,
        "elapsed_s": elapsed,
        "step_ms_estimate": 1000.0 * elapsed / max(args.decode_tokens, 1),
        "attention_backend": os.environ.get("RL_KERNEL_ROCM_ATTENTION_BACKEND", "ck"),
        "det_gemm_backend": os.environ.get("RL_KERNEL_DET_GEMM_BACKEND", "auto"),
    }
    (args.trace_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    time.sleep(5)  # let worker profiler exports finish
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
