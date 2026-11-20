# SPDX-License-Identifier: Apache-2.0
"""Scope the original inverse-RoPE call to one loaded native attention instance."""

import types

import torch

from rl_engine.backends.rocm.attention.inverse_rope import inverse_rope_tail


def bind_native_inverse_rope(attention):
    if attention.config.apply_rope_fusion:
        raise ValueError("initial inverse RoPE binding requires ordinary eager native path")
    original = attention.forward.__func__
    if "apply_rotary_pos_emb" not in original.__code__.co_names:
        raise ValueError("unsupported native inverse RoPE call site")

    def rotate(t, freqs, config, *, inverse=False, cu_seqlens=None, cp_group=None, **kwargs):
        if not inverse or cu_seqlens is not None or (cp_group is not None and cp_group.size() != 1):
            raise ValueError("initial native inverse rotation supports ordinary CP1 only")
        positions = (
            torch.arange(t.shape[0], device=t.device, dtype=torch.int64)[:, None]
            .expand(t.shape[:2])
            .contiguous()
        )
        cache = attention._rlk_qkv_preprocess.cos_sin_cache
        result = inverse_rope_tail(t, positions, cache)
        if not getattr(attention, "_rlk_inverse_executed", False):
            print(
                "RLK_NATIVE_INVERSE_ROPE_EXECUTED",
                __file__,
                "vllm._fused_inverse_rope_gptj",
                str(cache.dtype),
                flush=True,
            )
            attention._rlk_inverse_executed = True
        return result

    cloned = types.FunctionType(
        original.__code__,
        dict(original.__globals__, apply_rotary_pos_emb=rotate),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    attention.forward = types.MethodType(cloned, attention)
    return attention
