# SPDX-License-Identifier: Apache-2.0
"""Instance-local P1 binding for NVIDIA MCore's DSV4 HybridModel.

MCore has two HyperConnectionHybridLayer objects per checkpoint layer. Bind
those boundaries, preserving the native attention/MoE dispatch and Parameters.
This adapter deliberately supports eager TP=CP=1 without recompute first.
"""

from __future__ import annotations

from types import FunctionType, MethodType

import torch


def _pre(
    hc, hidden_states, mhc_recompute_manager=None, return_residual=False, *, checkpoint_fn=None
):
    if mhc_recompute_manager is not None or checkpoint_fn is not None:
        raise ValueError("native RLK P1 currently requires eager execution without recompute")
    s, b, _ = hidden_states.shape
    scales = torch.cat((hc.alpha_pre, hc.alpha_post, hc.alpha_res))
    x, p, mix, h = hc._rlk_p1.pre(
        hidden_states.reshape(s * b, 4, hc.hidden_size),
        hc.mapping_proj.weight,
        scales,
        hc.bias,
        hc.sinkhorn_iterations,
    )
    values = (
        h.reshape(s, b, -1),
        mix.reshape(s, b, 4, 4),
        p.reshape(s, b, 4),
        x.reshape_as(hidden_states),
    )
    return values if return_residual else values[:3]


def _post(
    hc,
    h_res,
    original_residual,
    h_post,
    layer_output_with_bias,
    dropout_prob,
    training,
    fused,
    manager=None,
    *,
    checkpoint_fn=None,
):
    if dropout_prob or manager is not None or checkpoint_fn is not None:
        raise ValueError("native RLK P1 requires zero dropout and no recompute manager")
    y, bias = layer_output_with_bias
    if bias is not None:
        raise ValueError("DSV4 P1 requires bias-free branches")
    result = hc._rlk_p1.post(
        y.reshape(-1, hc.hidden_size),
        original_residual.reshape(-1, 4, hc.hidden_size),
        h_post.reshape(-1, 4),
        h_res.reshape(-1, 4, 4),
    )
    return result.reshape_as(original_residual)


def _norm(module, x):
    return module._rlk_p1.norm(x.reshape(-1, x.shape[-1]), module.weight).reshape_as(x)


def _output(layer, hidden, weight=None, runtime_gather_output=None, output_dtype=None):
    from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp

    if output_dtype not in (None, torch.float32):
        raise ValueError("native P1 head emits FP32 logits")
    worker = layer._rlk_p1
    worker._active()
    logits = TritonLMHeadOp().forward_fp32(hidden, layer.weight if weight is None else weight)
    return worker._record("lm_head", logits), None


def selected_raw_logprobs(model, logits, target_ids, *, weight_version):
    """Consume already aligned token rows, with no sampling transforms or dtype casts."""
    from rl_engine.backends.shared.triton.logprob.logp import TritonLogpOp

    worker = model._rlk_p1
    worker._require(weight_version)
    worker._active()
    if target_ids.shape != logits.shape[:-1] or target_ids.dtype != torch.int64:
        raise ValueError("raw logprob requires aligned int64 target IDs")
    values = TritonLogpOp()(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
    return worker._record("raw_logprob", values.reshape_as(target_ids))


def bind_native_training_stage(model, worker):
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.hybrid.layers.hybrid_hyper_connection import (
        HyperConnectionHybridLayer,
    )
    from megatron.core.transformer.identity_op import IdentityOp

    if not isinstance(model, HybridModel) or hasattr(model, "_rlk_p1") or worker.role != "training":
        raise ValueError("expected an unbound NVIDIA HybridModel training stage")
    cfg, stack = model.config, model.decoder
    if (cfg.tensor_model_parallel_size, cfg.context_parallel_size) != (1, 1):
        raise ValueError("native RLK P1 currently supports TP=CP=1")
    if (
        cfg.mhc_num_residual_streams != 4
        or not cfg.enable_mhc_connections
        or cfg.hidden_dropout
        or cfg.attention_dropout
        or cfg.recompute_granularity
        or cfg.fp32_residual_connection
        or cfg.mtp_num_layers
    ):
        raise ValueError("native RLK P1 requires four BF16 streams, zero dropout, no MTP/recompute")
    worker.check_config(cfg.layernorm_epsilon, 1e-6)
    layers = list(stack.layers)
    norms = []
    for layer in layers:
        if not isinstance(layer, HyperConnectionHybridLayer):
            raise ValueError("native DSV4 requires HyperConnectionHybridLayer boundaries")
        hc = layer.hyper_connection
        if hc.n != 4 or hc.mapping_proj.weight.shape != (24, 4 * cfg.hidden_size):
            raise ValueError("unsupported native HyperConnectionModule ABI")
        for name in ("input_layernorm", "pre_mlp_layernorm"):
            norm = getattr(layer.inner_layer, name)
            if not isinstance(norm, IdentityOp):
                if not hasattr(norm, "weight") or getattr(norm, "bias", None) is not None:
                    raise ValueError("native P1 requires a separate bias-free RMSNorm")
                norms.append(norm)
    forward = stack.forward.__func__
    # MCore's learned contraction is a module-global function. Clone only this
    # instance's Python function globals; never replace the MCore module global
    # (which would change other workers). Keep the complete native scheduler.
    if stack.post_process and (
        not cfg.mhc_learned_output_contract
        or "learned_output_contract" not in forward.__code__.co_names
    ):
        raise ValueError("unsupported native HybridStack final-head call site")
    if stack.post_process:
        norms.append(stack.final_norm)
        if (
            model.output_layer.weight.dtype != torch.bfloat16
            or getattr(model.output_layer, "bias", None) is not None
        ):
            raise ValueError("native P1 requires a BF16 bias-free vocabulary head")

    def head(x, fn, base, scale, n, eps):
        worker.check_config(eps, eps)
        shape = x.shape[:-1]
        return worker.head(x.reshape(-1, n, cfg.hidden_size), fn, scale, base).reshape(
            *shape, cfg.hidden_size
        )

    globals_for_instance = dict(forward.__globals__, learned_output_contract=head)
    bound_forward = FunctionType(
        forward.__code__,
        globals_for_instance,
        forward.__name__,
        forward.__defaults__,
        forward.__closure__,
    )
    bound_forward.__kwdefaults__ = forward.__kwdefaults__
    model._rlk_p1 = worker
    stack.forward = MethodType(bound_forward, stack)
    for layer in layers:
        hc = layer.hyper_connection
        hc._rlk_p1 = worker
        hc.forward = MethodType(_pre, hc)
        hc.fused_h_res_h_post_bda = MethodType(_post, hc)
    for norm in norms:
        norm._rlk_p1 = worker
        norm.forward = MethodType(_norm, norm)
    if stack.post_process:
        model.output_layer._rlk_p1 = worker
        model.output_layer.forward = MethodType(_output, model.output_layer)
    return worker


def normalize_native_causal_mask(attention_mask, input_shape=None):
    """Translate verl's explicit full causal mask to MCore CSA's implicit mask.

    Reject padding/document masks rather than silently dropping their semantics.
    This eager adapter deliberately validates tensor contents before execution.
    """
    if attention_mask is None or attention_mask.numel() == 1:
        return attention_mask
    if input_shape is not None and tuple(attention_mask.shape) == tuple(input_shape):
        if not bool((attention_mask == 1).all()):
            raise ValueError("native DSV4 adapter cannot discard padding tokens")
        return None
    if (
        attention_mask.ndim < 2
        or attention_mask.dtype != torch.bool
        or attention_mask.shape[-1] != attention_mask.shape[-2]
    ):
        raise ValueError("native DSV4 requires a square boolean full causal mask")
    length = attention_mask.shape[-1]
    rows = torch.arange(length, device=attention_mask.device)
    causal = rows[:, None] < rows[None, :]
    if not bool((attention_mask == causal).all()):
        raise ValueError("native DSV4 adapter cannot discard padding/document-boundary masks")
    return None
