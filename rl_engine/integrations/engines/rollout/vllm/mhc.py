# SPDX-License-Identifier: Apache-2.0
"""Bind the AMD vLLM DSV4 model at its decoder and final-head boundaries."""
from types import MethodType

from rl_engine.integrations.common.mhc_worker import P1Worker


def _decoder(layer, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
    if any(v is not None for v in (post_mix, res_mix, residual)):
        raise RuntimeError("cannot mix native deferred mHC state with RLK")
    w = layer._rlk_p1
    rounds = layer.hc_sinkhorn_iters
    x, p, b, h = w.pre(x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, rounds)
    y = layer.attn(positions, w.norm(h, layer.attn_norm.weight), None)
    x, p, b, h = w.adjacent(
        y, x, p, b, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, rounds
    )
    y = layer.ffn(w.norm(h, layer.ffn_norm.weight), input_ids)
    return w.post(y, x, p, b), None, None, None


def bind_rollout_stage(model, worker: P1Worker):
    """Call before vLLM compilation/warmup. Keep PP placeholders untouched."""
    if worker.role != "rollout" or hasattr(model, "_rlk_p1"):
        raise ValueError("expected an unbound rollout stage")
    # Do not walk PPMissingLayer instances owned by other stages.
    layers = [model.layers[i] for i in range(model.start_layer, model.end_layer)]
    for layer in layers:
        worker.check_config(layer.rms_norm_eps, layer.hc_eps, layer.hc_post_alpha)
        for name in (
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
            "attn",
            "ffn",
            "attn_norm",
            "ffn_norm",
            "use_fused_mhc",
        ):
            if not hasattr(layer, name):
                raise ValueError(f"unsupported AMD vLLM layer: missing {name}")
        if (
            layer.attn_norm.variance_epsilon != layer.rms_norm_eps
            or layer.ffn_norm.variance_epsilon != layer.rms_norm_eps
        ):
            raise ValueError("unsupported vLLM sublayer norm epsilon")
    head = getattr(model, "hc_head_op", None)
    norm = model.norm
    if hasattr(norm, "weight"):
        if head is None:
            raise ValueError("final stage is missing hc_head_op")
        if norm.variance_epsilon != 1e-6:
            raise ValueError("unsupported final norm epsilon")

        def head_forward(hidden_states, hc_fn, hc_scale, hc_base, rms_eps, hc_eps):
            worker.check_config(rms_eps, hc_eps)
            return worker.head(hidden_states, hc_fn, hc_scale, hc_base)

        head.forward = head_forward
        norm.forward = lambda x: worker.norm(x, norm.weight)
    model._rlk_p1 = worker
    for layer in layers:
        layer._rlk_p1 = worker
        # We return completed residual streams, even when adjacent kernels are fused.
        # The native outer loop must not run a second post on the final layer.
        layer.use_fused_mhc = False
        layer.forward = MethodType(_decoder, layer)
    return worker
