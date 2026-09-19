# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""torchrun --standalone --nproc-per-node=8 this_file.py"""

import json
import os

import torch
import torch.distributed as dist

from rl_engine.integrations.linear_logp import LinearLogpWrapper
from rl_engine.integrations.sampling import sampling_keep_mask, vocab_parallel_sampling_keep_mask
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    os.environ["RL_KERNEL_STRICT_CANONICAL_TP"] = "8"
    os.environ["RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE"] = "152576"
    torch.manual_seed(2026)
    raw = torch.randn(3, 152064, dtype=torch.bfloat16, device="cuda")
    real = 151936
    raw[:, real:] = float("-inf")
    target = raw[:, :real].argmax(dim=1)
    results = []
    for temperature, p, k in [(0.7, 0.95, -1), (1.3, 0.99, 128), (1.0, 1.0, -1), (0.0, 0.5, 1)]:
        effective_temperature = 1.0 if temperature < 1e-5 else temperature
        expected = sampling_keep_mask(
            raw[:, :real],
            temperature=temperature,
            top_p=p if p < 1 else None,
            top_k=k if k > 0 else None,
        )
        native = apply_top_k_top_p(
            raw[:, :real].float() / effective_temperature,
            torch.full((3,), k, device="cuda", dtype=torch.int32)
            if k > 0 and temperature
            else None,
            torch.full((3,), p, device="cuda") if p < 1 and temperature else None,
        )
        assert torch.equal(expected, torch.isfinite(native)), "support differs from vLLM"
        reference = None
        for tp in (1, 2, 4, 8):
            group = None
            for start in range(0, 8, tp):
                created = dist.new_group(list(range(start, start + tp)))
                if start <= rank < start + tp:
                    group = created
            width = raw.size(1) // tp
            offset = rank % tp * width
            local = raw[:, offset : offset + width].contiguous()
            mask = vocab_parallel_sampling_keep_mask(
                local,
                tp_group=group,
                real_vocab_size=real,
                temperature=temperature,
                top_p=p if p < 1 else None,
                top_k=k if k > 0 else None,
                chunk_size=2,
            )
            count = max(0, min(width, real - offset))
            assert torch.equal(mask[:, :count], expected[:, offset : offset + count])
            assert not mask[:, count:].any(), "padded vocabulary must be excluded"
            score = LinearLogpWrapper().from_local_logits(
                local.masked_fill(~mask, float("-inf")),
                target,
                tp_group=group,
                vocab_start_index=offset,
                global_vocab_size=raw.size(1),
                real_vocab_size=real,
                temperature=effective_temperature,
            )
            if reference is None:
                reference = score.clone()
            assert torch.equal(
                score.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
            ), f"cross-TP logprob bits differ at TP{tp}"
            if rank == 0:
                results.append(
                    {
                        "tp": tp,
                        "temperature": temperature,
                        "top_p": p,
                        "top_k": k,
                        "nucleus_widths": expected.sum(1).tolist(),
                        "bitwise_equal": True,
                    }
                )
            dist.barrier()
    if rank == 0:
        print("CONFIGURABLE_SAMPLING_RESULT=" + json.dumps(results), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
