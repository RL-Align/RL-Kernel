# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Isolated vLLM runtime for the real-framework PPO integration check.

JSON control messages reference locally generated tensor files. No network
service, framework patch, synthetic behavior logprob or pretrained model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def install_weights(worker, filename: str, version: int):
    """Executed by vLLM's real collective_rpc on each inference worker (TP=1)."""
    weights = torch.load(filename, map_location="cpu", weights_only=True)
    model = worker.model_runner.model
    loaded = model.load_weights(weights.items())
    assert loaded == set(weights), (loaded, set(weights))
    parameters = dict(model.named_parameters())
    for name, expected in weights.items():
        if name.endswith(".weight") and any(x in name for x in (".c_attn.", ".c_proj.", ".c_fc.")):
            expected = expected.t()
        actual = parameters[name].detach().cpu()
        # vLLM pads the vocabulary internally; only real vocabulary rows matter.
        if name in ("transformer.wte.weight", "lm_head.weight"):
            actual = actual[: expected.shape[0]]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()
    worker.ppo_actor_version = version
    return {"version": version, "verified_tensors": len(weights)}


class PPOWeightLoader:
    def install_ppo_weights(self, filename: str, version: int):
        return install_weights(self, filename, version)


def main():
    from importlib.metadata import version

    from transformers import GPT2Config
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--response-fd", type=int, required=True)
    args = parser.parse_args()
    responses = os.fdopen(args.response_fd, "w", buffering=1)
    args.directory.mkdir(parents=True, exist_ok=True)
    config = GPT2Config(
        vocab_size=32,
        n_positions=64,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=256,
        activation_function="gelu",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=31,
    )
    config.architectures = ["GPT2LMHeadModel"]
    config.save_pretrained(args.directory)
    started = time.perf_counter()
    llm = LLM(
        model=str(args.directory),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="float32",
        tensor_parallel_size=1,
        max_model_len=64,
        enforce_eager=True,
        enable_prefix_caching=True,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.1,
        kv_cache_memory_bytes=64 * 1024**2,
        attention_backend="TRITON_ATTN",
        worker_extension_cls="tests.distributed.tp.vllm_ppo_sidecar.PPOWeightLoader",
        disable_log_stats=True,
        seed=44,
    )

    def emit(value):
        print(json.dumps(value), file=responses, flush=True)

    emit(
        {
            "ready": True,
            "vllm": version("vllm"),
            "torch": torch.__version__,
            "startup_seconds": time.perf_counter() - started,
        }
    )
    active = -1
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        try:
            if request["op"] == "close":
                emit({"closed": True})
                break
            if request["op"] == "install":
                assert request["version"] == active + 1
                acknowledgments = llm.collective_rpc(
                    "install_ppo_weights", args=(request["filename"], request["version"])
                )
                assert all(ack["version"] == request["version"] for ack in acknowledgments)
                reset = llm.reset_prefix_cache()
                if reset is False:
                    raise RuntimeError("vLLM refused prefix-cache reset")
                active = request["version"]
                emit(
                    {
                        "installed": active,
                        "workers": acknowledgments,
                        "seconds": time.perf_counter() - started,
                    }
                )
            elif request["op"] == "sample":
                assert request["version"] == active
                params = SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=4,
                    stop_token_ids=[31],
                    logprobs=0,
                    detokenize=False,
                    seed=123 + active,
                )
                outputs = llm.generate(
                    [{"prompt_token_ids": p} for p in request["prompts"]], params, use_tqdm=False
                )
                generation_seconds = time.perf_counter() - started
                assert all(output.num_cached_tokens == 0 for output in outputs)
                probe_started = time.perf_counter()
                replay = llm.generate(
                    [{"prompt_token_ids": p} for p in request["prompts"]], params, use_tqdm=False
                )
                assert all(output.num_cached_tokens >= 16 for output in replay)
                for cold, cached in zip(outputs, replay, strict=True):
                    assert cold.outputs[0].token_ids == cached.outputs[0].token_ids
                    cold_lp = [
                        x[t].logprob
                        for t, x in zip(
                            cold.outputs[0].token_ids, cold.outputs[0].logprobs, strict=True
                        )
                    ]
                    cached_lp = [
                        x[t].logprob
                        for t, x in zip(
                            cached.outputs[0].token_ids, cached.outputs[0].logprobs, strict=True
                        )
                    ]
                    torch.testing.assert_close(
                        torch.tensor(cold_lp), torch.tensor(cached_lp), atol=2e-6, rtol=2e-6
                    )
                probe_seconds = time.perf_counter() - probe_started
                sequences = []
                for output in outputs:
                    completion = output.outputs[0]
                    assert completion.finish_reason in ("stop", "length")
                    assert completion.logprobs is not None
                    sequences.append(
                        {
                            "prompt": output.prompt_token_ids,
                            "tokens": completion.token_ids,
                            "logprobs": [
                                lp[token].logprob
                                for token, lp in zip(
                                    completion.token_ids, completion.logprobs, strict=True
                                )
                            ],
                            "terminated": completion.finish_reason == "stop",
                        }
                    )
                emit(
                    {
                        "version": active,
                        "sequences": sequences,
                        "seconds": generation_seconds,
                        "cache_probe_seconds": probe_seconds,
                        "cached_tokens_after_install": [o.num_cached_tokens for o in outputs],
                        "cached_tokens_on_replay": [o.num_cached_tokens for o in replay],
                    }
                )
            else:
                raise ValueError("unknown operation")
        except Exception as exc:
            emit({"error": f"{type(exc).__name__}: {exc}"})
            raise
    # Engine lifetime ends with the isolated process, not the trainer's groups.
    del llm


if __name__ == "__main__":
    main()
