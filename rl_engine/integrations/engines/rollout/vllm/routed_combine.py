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
import os
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
    if not _STATE.get("executed"):
        # Bound is not enough: prove the fixed combine is on the executed MoE path.
        print("RLK_ROLLOUT_FIXED_ROUTED_COMBINE_EXECUTED", run_id, tuple(y.shape), flush=True)
        _STATE["executed"] = True
    return out


def _maybe_dump_rows(routing_data, num_tokens, **tensors):
    # Diagnostic only (off by default): per-row expert outputs for actor row comparison.
    dump_dir = os.environ.get("RLK_ROUTED_ROWS_DUMP_DIR")
    if not dump_dir or str(num_tokens) != os.environ.get("RLK_ROUTED_ROWS_DUMP_M"):
        return
    os.makedirs(dump_dir, exist_ok=True)
    payload = {k: v.detach().cpu() for k, v in tensors.items()}
    payload.update(
        tokens=routing_data._rlk_row_tokens.cpu(),
        experts=routing_data._rlk_row_experts.cpu(),
        valid=routing_data._rlk_row_valid.cpu(),
    )
    torch.save(payload, os.path.join(dump_dir, f"call-{_STATE['calls']:05d}-M{num_tokens}.pt"))


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
        _maybe_dump_rows(routing_data, M, inter=inter, y=y, hidden=hidden_states)
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

    # gfx942 selects TRITON_UNFUSED (UnfusedOAITritonExperts): same 3.8 matmuls with a
    # separate activation, finalized by its own inline atomic index_add_.
    unfused_cls = mod.UnfusedOAITritonExperts
    unfused_index_add = "acc.index_add_(0, gather_tok.to(torch.int64), down.to(torch.float32))"
    if unfused_index_add not in inspect.getsource(unfused_cls.apply):
        raise RuntimeError("unsupported vLLM: unfused index_add_ combine not found")

    def unfused_apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        if expert_map is not None:
            raise RuntimeError("fixed routed combine supports no expert map (EP=1) only")
        if self._lora_context is not None:
            raise RuntimeError("fixed routed combine does not support MoE LoRA")
        if apply_router_weight_on_input:
            raise RuntimeError("fixed routed combine needs the router weight in the w2 epilogue")
        quant_config = self.quant_config
        if quant_config is None:
            quant_config = mod.FUSED_MOE_UNQUANTIZED_CONFIG
        local_num_experts = w1.shape[0]
        routing_data, _, _ = self._make_routing_data(topk_ids, topk_weights, local_num_experts)
        if topk_ids.size(1) != _TOP_K or not hasattr(routing_data, "_rlk_row_valid"):
            raise RuntimeError("fixed routed combine requires bound top-6 routing data")
        if tp_size != 1:
            # TP>1: vLLM's I/TP-split matmuls would need a cross-rank float reduction.
            # Run the actor's packed W4A16 kernels on N-sharded original weights instead.
            from .routed_experts_tp import tp_exact_unfused_apply

            tp_exact_unfused_apply(self, output, hidden_states, routing_data, activation, mod)
            return
        # Verbatim vLLM 3.8 unfused branch up to the combine (LoRA excluded above).
        act_out_dim = self.adjust_N_for_activation(w1.shape[2], activation)
        M, K = hidden_states.shape[-2:]
        a_ragged = routing_data.ragged
        gather_tok = routing_data.gather_tok
        gammas = routing_data.gate_scal
        inter = mod.matmul(
            hidden_states,
            w1,
            quant_config.w1_bias,
            a_ragged,
            None,
            gather_tok,
            None,
            quant_config.w1_precision,
            gammas=None,
        )
        act_out = torch.empty(
            (routing_data.n_valid, act_out_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.activation(activation, act_out, inter)
        down = mod.matmul(
            act_out,
            w2,
            quant_config.w2_bias,
            a_ragged,
            None,
            None,
            None,
            quant_config.w2_precision,
            gammas=gammas,
        )
        _STATE["calls"] += 1
        _maybe_dump_rows(routing_data, M, inter=inter, act=act_out, y=down, hidden=hidden_states)
        if not bool(routing_data._rlk_row_valid.all()):
            # vLLM warm-up/dummy batches can carry -1 slots (zero-gamma rows). Keep vLLM's
            # original finalize there and log it; audited scores must show no fallback.
            _STATE["fallbacks"] = _STATE.get("fallbacks", 0) + 1
            print("RLK_ROLLOUT_ROUTED_COMBINE_FALLBACK", _STATE["calls"], "M", M,
                  "invalid", int((~routing_data._rlk_row_valid).sum()), flush=True)
            acc = torch.zeros((M, K), dtype=torch.float32, device=hidden_states.device)
            acc.index_add_(0, gather_tok.to(torch.int64), down.to(torch.float32))
            output.view(M, K).copy_(acc.to(output.dtype))
            return
        fixed_combine_rows(
            down,
            routing_data._rlk_row_tokens,
            routing_data._rlk_row_experts,
            routing_data._rlk_row_valid,
            M,
            f"vllm-unfused-routed-p6-{_STATE['calls']}",
            output.view(M, K),
        )

    unfused_cls.apply = unfused_apply
    mod.make_routing_data = make_routing_data
    mod.triton_kernel_fused_experts = triton_kernel_fused_experts
    _rebind_aliases(original_routing, make_routing_data, "make_routing_data")
    _rebind_aliases(original_fused, triton_kernel_fused_experts, "triton_kernel_fused_experts")
    _STATE["bound"] = True
    print("RLK_ROLLOUT_FIXED_ROUTED_COMBINE_BOUND", PROFILE, __file__, flush=True)
