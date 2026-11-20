# SPDX-License-Identifier: Apache-2.0
"""Instance QKV adapter for original, ordinary TP=PP=EP=CP=1 DSV4 inputs."""

import types

import torch

from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess


def _get_qkv(
    self,
    hidden_states,
    key_value_states,
    position_ids,
    packed_seq_params,
    inference_context=None,
    boundary_hidden=None,
    **kwargs,
):
    if (
        key_value_states is not None
        or packed_seq_params is not None
        or boundary_hidden is not None
        or inference_context is not None
        or kwargs
    ):
        raise ValueError(
            "initial original-cache QKV adapter supports ordinary full teacher forcing only"
        )
    t, b, _ = hidden_states.shape
    q_latent = self.linear_q_down_proj(hidden_states)[0]
    q_latent = self.q_layernorm(q_latent)
    q = self.linear_q_up_proj(q_latent)[0].reshape(
        t, b, self.num_attention_heads_per_partition, 512
    )
    kv = self.kv_layernorm(self.linear_kv_proj(hidden_states)[0])
    positions = (
        torch.arange(t, device=hidden_states.device, dtype=torch.int64)[:, None]
        .expand(t, b)
        .contiguous()
    )
    if position_ids is not None:
        if position_ids.shape != (b, t) or not torch.equal(position_ids.T, positions):
            raise ValueError("initial QKV adapter requires original contiguous unpadded positions")
    q, kv = self._rlk_qkv_preprocess(q, kv, positions)
    key = kv.unsqueeze(-2)
    return q, key, key, q_latent, hidden_states


def bind_native_qkv(attention, hf_config, compression_ratio):
    attention._rlk_qkv_preprocess = DSV4QKVPreprocess(hf_config, compression_ratio)
    attention.get_query_key_value_tensors = types.MethodType(_get_qkv, attention)
    return attention
