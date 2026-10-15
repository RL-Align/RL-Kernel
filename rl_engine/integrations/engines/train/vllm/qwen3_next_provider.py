# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared Qwen3-Next GDN core with explicit provider and state identities.

This is the convolution/recurrence/gated-norm boundary, after the projections
and before the output projection. It is not a full attention or model adapter.
The same call can be made under inference_mode or with training autograd.
"""

from dataclasses import asdict, dataclass

import torch

from rl_engine.integrations.engines.train.vllm.qwen3_next_conv import causal_conv_sequence
from rl_engine.integrations.engines.train.vllm.qwen3_next_gdn import packed_recurrent_sequence


@dataclass(frozen=True)
class GDNProviderConfig:
    """The only implemented profile; other providers cannot silently substitute."""

    version: str = "qwen3-next-gdn-core-v1"
    convolution: str = "vllm-0.30.0-causal-conv-update-equal-length-groups"
    recurrence: str = "vllm-0.30.0-packed-recurrent-decode"
    gated_norm: str = "rl-engine-cuda-rmsnorm-gated"
    recurrent_dtype: str = "float32"
    activation_dtype: str = "bfloat16"
    head_dim: int = 128

    def validate(self):
        if self != GDNProviderConfig():
            raise ValueError("Unsupported GDN provider configuration")

    def identity(self):
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class GDNState:
    convolution: torch.Tensor
    recurrent: torch.Tensor


def shared_gdn_core(
    qkv,
    a,
    b,
    z,
    A_log,
    dt_bias,
    conv_weight,
    norm_weight,
    state,
    indices,
    cu_seqlens,
    *,
    config,
    num_k_heads,
    eps=1e-6,
):
    """Independently recompute packed training inputs or execute rollout inputs.

    qkv is laid out as contiguous Q, K, V groups. z is [tokens, value_heads, 128].
    Both callers must explicitly supply the same GDNProviderConfig. State is
    returned without detachment so a response loss can differentiate through
    the prompt. No global vLLM batch-invariance capability is changed.
    """
    if not isinstance(config, GDNProviderConfig):
        raise ValueError("An explicit GDNProviderConfig is required")
    config.validate()
    if not isinstance(state, GDNState):
        raise ValueError("An explicit convolution and recurrent GDNState is required")
    if state.recurrent.ndim != 4:
        raise ValueError("Recurrent state must have four dimensions")
    if (
        z.shape != (qkv.shape[0], state.recurrent.shape[1], 128)
        or z.dtype != torch.bfloat16
        or z.device != qkv.device
    ):
        raise ValueError("z must be BF16 [tokens, value_heads, 128] on the input device")
    if (
        norm_weight.shape != (128,)
        or norm_weight.dtype != torch.bfloat16
        or norm_weight.device != qkv.device
    ):
        raise ValueError("Gated norm weight must be BF16 [128] on the input device")
    from rl_engine.backends.cuda.norm.rmsnorm import Qwen3NextRMSNormGatedCudaOp

    norm = Qwen3NextRMSNormGatedCudaOp()

    convolved, conv_state = causal_conv_sequence(
        qkv, state.convolution, conv_weight, indices, cu_seqlens
    )
    recurrent, recurrent_state = packed_recurrent_sequence(
        convolved,
        a,
        b,
        A_log,
        dt_bias,
        state.recurrent,
        indices,
        cu_seqlens,
        num_k_heads=num_k_heads,
    )
    normed = norm(recurrent.reshape(-1, 128), norm_weight, z.reshape(-1, 128), eps=eps).reshape_as(
        z
    )
    return normed, GDNState(conv_state, recurrent_state)
