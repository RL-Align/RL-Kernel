# SPDX-License-Identifier: Apache-2.0
"""Rollout routed-expert combine in the actor's fixed FP32 expert-ascending order.

vLLM's triton_kernels 3.8 MoE path finishes with FP32 ``index_add_``, which uses
GPU atomics, so each token's 6 expert rows are summed in hardware order. The
training actor sums them with ``combine_rows`` (slots 0..5, expert-ascending).
This module replaces only that final reduction; both matmuls are unchanged.
The shared expert is not part of this combine (vLLM adds it afterwards in BF16).
"""

import importlib
import inspect
import sys

import torch

from rl_engine.contracts.operators.fixed_output_combine import PROFILE

_MODULE = "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe"
_TOP_K = 6
_STATE = {"bound": False, "calls": 0}


def routing_row_metadata(topk_ids):
    """Per-row (token, local expert, valid) for the rows of ``make_routing_data``.

    Mirrors make_routing_data (tk 3.8): stable sort of ``flat_ids.clamp(min=0)``;
    row i belongs to token ``order[i] // topk``. Slots with id -1 (non-local) sort
    as expert 0 and are flagged invalid.
    """
    topk = topk_ids.shape[1]
    flat_ids = topk_ids.reshape(-1)
    sorted_keys, order = torch.sort(flat_ids.clamp(min=0), stable=True)
    tokens = (order // topk).to(torch.int64)
    experts = sorted_keys.to(torch.int64)
    valid = flat_ids[order] >= 0
    return tokens, experts, valid


def fixed_combine_rows(y, tokens, experts, valid, num_tokens, run_id, out):
    """Combine routed rows ``y`` into ``out`` ([M, K]) in the actor's fixed order."""
    from rl_engine.integrations.engines.train.megatron.native_combine import combine_rows

    if y.dim() != 2 or y.shape[0] != num_tokens * _TOP_K:
        raise ValueError("routed combine needs exactly num_tokens * 6 expert rows")
    if y.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("routed combine supports BF16 expert rows and output only")
    if not bool(valid.all()):
        # Non-local (EP) slots would leave a token with fewer than 6 rows.
        raise ValueError("routed combine requires all 6 expert slots local (no -1 slots)")
    with torch.no_grad():
        result = combine_rows(y.contiguous(), tokens, experts, num_tokens, run_id)
    out.copy_(result.reshape(out.shape))
    return out


def _make_rows_class(base):
    # RoutingData has __slots__ (no __dict__/__weakref__): extra attributes need a subclass.
    class RoutingDataWithRows(base):
        __slots__ = ("_rlk_row_tokens", "_rlk_row_experts", "_rlk_row_valid")

    return RoutingDataWithRows


def _rebind_aliases(original, replacement, name):
    # Other modules may have imported the function by name before binding.
    for module in list(sys.modules.values()):
        try:
            if getattr(module, name, None) is original:
                setattr(module, name, replacement)
        except Exception:  # noqa: BLE001
            continue


def bind_rollout_fixed_routed_combine():
    """Idempotent, process-wide. Fails closed on any unsupported vLLM profile."""
    if _STATE["bound"]:
        return
    mod = importlib.import_module(_MODULE)
    if mod.triton_kernels_version != "3.8" or mod._TK38_SCATTER_COMBINE is not False:
        raise RuntimeError("unsupported profile: need triton_kernels 3.8 with index_add combine")
    original_fused = mod.triton_kernel_fused_experts
    original_routing = mod.make_routing_data
    index_add = "acc.index_add_(0, gather_tok.to(torch.int64), y.to(torch.float32))"
    if index_add not in inspect.getsource(original_fused):
        raise RuntimeError("unsupported vLLM: index_add_ combine not found in fused experts")
    if "torch.sort(keys, stable=True)" not in inspect.getsource(original_routing):
        raise RuntimeError("unsupported vLLM: stable sort not found in make_routing_data")
    rows_class = _make_rows_class(mod.RoutingData)

    def make_routing_data(topk_ids, topk_weights, num_local_experts):
        routing, gather_a, gather_b = original_routing(topk_ids, topk_weights, num_local_experts)
        tokens, experts, valid = routing_row_metadata(topk_ids)
        if not torch.equal(tokens.to(routing.gather_tok.dtype), routing.gather_tok):
            raise RuntimeError("recomputed routing rows disagree with make_routing_data")
        wrapped = rows_class(
            routing.ragged,
            routing.gate_scal,
            routing.gather_tok,
            routing.scatter_indx,
            routing.n_expts_act,
            routing.n_valid,
        )
        wrapped._rlk_row_tokens = tokens
        wrapped._rlk_row_experts = experts
        wrapped._rlk_row_valid = valid
        return wrapped, gather_a, gather_b

    def triton_kernel_fused_experts(
        output_tensor,
        hidden_states,
        w1,
        w2,
        routing_data,
        gather_indx,
        scatter_indx,
        topk,
        activation=mod.MoEActivation.SWIGLUOAI,
        quant_config=None,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        apply_router_weight_on_input=False,
        global_num_experts=-1,
        expert_map=None,
        intermediate_cache=None,
        a1q_scale=None,
    ):
        from vllm.distributed import get_tensor_model_parallel_world_size

        assert activation == mod.MoEActivation.SWIGLUOAI, "Only SWIGLUOAI activation is supported"
        if get_tensor_model_parallel_world_size() != 1:
            raise RuntimeError("fixed routed combine supports TP1 only")
        if apply_router_weight_on_input:
            raise RuntimeError("fixed routed combine needs the router weight in the w2 epilogue")
        if topk != _TOP_K:
            raise RuntimeError("fixed routed combine requires top-6 routing")
        if not hasattr(routing_data, "_rlk_row_valid"):
            raise RuntimeError("routing data was not built by the bound make_routing_data")
        # Verbatim vLLM 3.8 branch up to the combine.
        assert quant_config is not None
        M, K = hidden_states.shape[-2:]
        a_ragged = routing_data.ragged
        gather_tok = routing_data.gather_tok
        gammas = routing_data.gate_scal
        act = mod.FusedActivation(
            mod.FnSpecs(
                "swiglu",
                mod.triton_kernels.swiglu.swiglu_fn,
                ("alpha", "limit"),
                reduction_n=2,
            ),
            (swiglu_alpha, swiglu_limit),
        )
        inter = mod.matmul(
            hidden_states,
            w1,
            quant_config.w1_bias,
            a_ragged,
            None,
            gather_tok,
            None,
            quant_config.w1_precision,
            gammas=gammas if apply_router_weight_on_input else None,
            fused_activation=act,
        )
        y = mod.matmul(
            inter,
            w2,
            quant_config.w2_bias,
            a_ragged,
            None,
            None,
            routing_data.scatter_indx,
            quant_config.w2_precision,
            gammas=None if apply_router_weight_on_input else gammas,
        )
        out = output_tensor.view(M, K)
        # Same rows, actor order instead of atomic index_add_. Local == global expert at TP1.
        _STATE["calls"] += 1
        fixed_combine_rows(
            y,
            routing_data._rlk_row_tokens,
            routing_data._rlk_row_experts,
            routing_data._rlk_row_valid,
            M,
            f"vllm-routed-p6-{_STATE['calls']}",
            out,
        )
        return out

    mod.make_routing_data = make_routing_data
    mod.triton_kernel_fused_experts = triton_kernel_fused_experts
    _rebind_aliases(original_routing, make_routing_data, "make_routing_data")
    _rebind_aliases(original_fused, triton_kernel_fused_experts, "triton_kernel_fused_experts")
    _STATE["bound"] = True
    print("RLK_ROLLOUT_FIXED_ROUTED_COMBINE_BOUND", PROFILE, __file__, flush=True)
