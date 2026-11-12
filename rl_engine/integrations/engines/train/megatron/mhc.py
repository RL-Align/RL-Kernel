# SPDX-License-Identifier: Apache-2.0
"""Bind a local Megatron-Lite DSV4 stage without replacing its parameter objects."""
from types import MethodType

from rl_engine.integrations.common.mhc_worker import P1Worker


def _layer_forward(
    layer, x, *, position_ids, input_ids=None, packed_seq_params=None, attention_metadata=None
):
    # The pinned mLite impl=vllm passes position and packed metadata via attention_metadata.
    p1 = layer._rlk_p1
    config = layer.config
    rounds = config.hc_sinkhorn_iters
    p1.check_config(config.rms_norm_eps, config.hc_eps)
    hc = layer.attn_hc
    x, p, b, h = p1.pre(x, hc.fn, hc.scale, hc.base, rounds)
    y = layer.self_attn(p1.norm(h, layer.input_layernorm.weight), metadata=attention_metadata)
    hc = layer.ffn_hc
    x, p, b, h = p1.adjacent(y, x, p, b, hc.fn, hc.scale, hc.base, rounds)
    y = layer.mlp(p1.norm(h, layer.post_attention_layernorm.weight), input_ids=input_ids)
    return p1.post(y, x, p, b)


def bind_training_stage(model, worker: P1Worker):
    if worker.role != "training" or hasattr(model, "_rlk_p1"):
        raise ValueError("expected an unbound training stage")
    if not getattr(model, "rlk_p1_head_hook", False):
        raise ValueError("apply the pinned mLite P1 instance-head patch before binding")
    layers = list(model.layers.values())
    for layer in layers:
        if layer.config.hc_mult != 4 or not all(
            hasattr(layer, a)
            for a in (
                "_mhc_pre",
                "_mhc_post",
                "self_attn",
                "mlp",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ):
            raise ValueError("unsupported mLite DeepseekV4Layer ABI")
        worker.check_config(layer.config.rms_norm_eps, layer.config.hc_eps)
        for hc in (layer.attn_hc, layer.ffn_hc):
            if not all(hasattr(hc, n) for n in ("fn", "scale", "base")):
                raise ValueError("missing mLite mHC coefficient")
    model._rlk_p1 = worker
    for layer in layers:
        layer._rlk_p1 = worker
        layer.forward = MethodType(_layer_forward, layer)
    return worker
