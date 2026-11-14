#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Real-checkpoint rollout diagnostic. This does not perform VIME or L2 validation."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--non-strict-smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "qwen3_next" or config.get("num_hidden_layers") != 48:
        raise SystemExit("Expected the full Qwen3-Next checkpoint")
    os.environ["VLLM_BATCH_INVARIANT"] = "0" if args.non_strict_smoke else "1"
    os.environ["VLLM_PLUGINS"] = ""
    from vllm import LLM, SamplingParams

    model = LLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.93,
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=1234,
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float32",
        kernel_config={"moe_backend": "triton", "enable_flashinfer_autotune": False},
    )
    outputs = model.generate(
        ["The capital of France is", "Calculate 7 plus 5. The result is"],
        SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8, logprobs=1),
    )
    results = []
    for output in outputs:
        response = output.outputs[0]
        results.append(
            {
                "prompt_token_ids": output.prompt_token_ids,
                "response_token_ids": list(response.token_ids),
                "selected_logprobs": [
                    row[token].logprob for row, token in zip(response.logprobs, response.token_ids)
                ],
                "text": response.text,
            }
        )
    payload = {
        "scope": "rollout_smoke_only",
        "full_model_l2": "not_evaluated",
        "batch_invariant_requested": not args.non_strict_smoke,
        "moe_backend": "triton",
        "model": str(args.model.resolve()),
        "tensor_parallel_size": args.tensor_parallel_size,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("REAL_CHECKPOINT_ROLLOUT_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
