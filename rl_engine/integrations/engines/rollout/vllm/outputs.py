# SPDX-License-Identifier: Apache-2.0
"""Explicit eager TP1 DSV4 vocabulary-head and raw-score worker boundaries."""

import inspect

import torch

from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp
from rl_engine.backends.shared.triton.logprob.logp import TritonLogpOp


def bind_output_head(wrapper, worker):
    if worker.role != "rollout" or hasattr(wrapper, "_rlk_output_worker"):
        raise ValueError("expected an unbound rollout wrapper")
    weight = wrapper.lm_head.weight
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("DSV4 output head requires the original BF16 vocabulary weight")
    if weight.shape[0] != wrapper.config.vocab_size:
        raise ValueError("DSV4 output head currently requires TP=1 and an unpadded vocabulary")
    processor = wrapper.logits_processor
    if getattr(processor, "scale", 1.0) != 1.0 or getattr(processor, "soft_cap", None) is not None:
        raise ValueError("raw RLK DSV4 head does not support scaled or soft-capped logits")
    wrapper._rlk_output_worker = worker
    executed = [False]

    def compute_logits(hidden_states):
        with worker.request(worker.version):
            result = TritonLMHeadOp().forward_fp32(hidden_states, wrapper.lm_head.weight)
            if not executed[0]:
                print(
                    "RLK_ROLLOUT_LM_HEAD_EXECUTED",
                    inspect.getfile(TritonLMHeadOp),
                    worker.version,
                    str(result.dtype),
                    flush=True,
                )
                executed[0] = True
            return worker._record("lm_head", result)

    wrapper.compute_logits = compute_logits
    return worker


def selected_raw_logprobs(wrapper, logits, token_ids, *, weight_version):
    """Full-vocabulary selected scores; returned columns never truncate the distribution."""
    worker = wrapper._rlk_output_worker
    if logits.ndim != 2 or token_ids.ndim != 2 or logits.shape[0] != token_ids.shape[0]:
        raise ValueError("raw rollout scores require aligned [M,V] logits and [M,C] token IDs")
    if token_ids.dtype != torch.int64 or token_ids.shape[1] == 0:
        raise ValueError("raw rollout scores require nonempty int64 target columns")
    with worker.request(weight_version):
        values = torch.stack(
            [TritonLogpOp()(logits, token_ids[:, i]) for i in range(token_ids.shape[1])], -1
        )
        if not getattr(worker, "_rollout_logprob_executed", False):
            print(
                "RLK_ROLLOUT_RAW_LOGPROB_EXECUTED",
                inspect.getfile(TritonLogpOp),
                weight_version,
                str(values.dtype),
                flush=True,
            )
            worker._rollout_logprob_executed = True
        return worker._record("raw_logprob", values)
