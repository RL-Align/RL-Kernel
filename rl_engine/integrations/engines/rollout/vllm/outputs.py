# SPDX-License-Identifier: Apache-2.0
"""Explicit eager sharded DSV4 vocabulary-head and raw-score worker boundaries."""

import inspect
from functools import wraps

import torch

from rl_engine.backends.rocm.gemm.det_gemm import prepare_det_gemm_linear_weight
from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp
from rl_engine.backends.shared.triton.logprob.logp import TritonLogpOp

from .runtime import (
    _invalidate_lm_head_weight_cache,
    _refresh_lm_head_weight_cache,
    _validated_lm_head_weight_cache,
)

# Stateless ops: construct once, not per call.
_LM_HEAD_OP = TritonLMHeadOp()
_LOGP_OP = TritonLogpOp()


def bind_output_head(wrapper, worker):
    if worker.role != "rollout" or hasattr(wrapper, "_rlk_output_worker"):
        raise ValueError("expected an unbound rollout wrapper")
    weight = wrapper.lm_head.weight
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("DSV4 output head requires the original BF16 vocabulary weight")
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp = get_tensor_model_parallel_world_size()
    head = wrapper.lm_head
    if tp not in (1, 2, 4, 8) or head.tp_size != tp:
        raise ValueError("DSV4 head requires the actual native TP1/2/4/8 group")
    if head.org_vocab_size != wrapper.config.vocab_size or head.shard_indices.num_added_elements:
        raise ValueError("DSV4 head requires the original vocabulary without adapter tokens")
    if weight.shape[0] != head.shard_indices.num_elements_padded:
        raise ValueError("DSV4 head weight differs from native shard/padding metadata")
    processor = wrapper.logits_processor
    if getattr(processor, "scale", 1.0) != 1.0 or getattr(processor, "soft_cap", None) is not None:
        raise ValueError("raw RLK DSV4 head does not support scaled or soft-capped logits")
    # Reuse the existing non-persistent, stable-storage inference cache. Both
    # standalone and verl actor updates invoke this wrapper's load_weights;
    # invalidate before the loader because .data writes do not bump _version.
    _refresh_lm_head_weight_cache(wrapper.lm_head, prepare_det_gemm_linear_weight)
    original_load_weights = wrapper.load_weights

    @wraps(original_load_weights)
    def load_weights(*args, **kwargs):
        _invalidate_lm_head_weight_cache(wrapper.lm_head.weight)
        loaded = original_load_weights(*args, **kwargs)
        _refresh_lm_head_weight_cache(wrapper.lm_head, prepare_det_gemm_linear_weight)
        _validated_lm_head_weight_cache(wrapper.lm_head)
        return loaded

    wrapper.load_weights = load_weights
    wrapper._rlk_output_worker = worker
    executed = [False]

    def compute_logits(hidden_states):
        with worker.request(worker.version):
            if torch.is_grad_enabled() and (
                hidden_states.requires_grad or wrapper.lm_head.weight.requires_grad
            ):
                local = _LM_HEAD_OP.forward_fp32(hidden_states, wrapper.lm_head.weight)
            else:
                weight_t = _validated_lm_head_weight_cache(wrapper.lm_head)
                local = _LM_HEAD_OP.forward_fp32_prepared(hidden_states, weight_t)
            result = processor._gather_logits(local) if tp > 1 else local
            if result is None:
                return None  # Native gather only returns scores on its sampling rank.
            # Original tokens occupy the native base-vocabulary region; trim only padding.
            result = result[..., : wrapper.config.vocab_size].contiguous()
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
            [_LOGP_OP(logits, token_ids[:, i]) for i in range(token_ids.shape[1])], -1
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
