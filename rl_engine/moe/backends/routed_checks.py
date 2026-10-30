# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The per-launch batch check the routed-expert backends share.

Both fused routed backends (CUDA ``sm90_fused_mlp``, Triton
``triton_fused_mlp``) accept the same ``ExpertBatch`` shape and reject the same
things; only the numeric profile and the tile alignment differ. Keeping the
check in one place means the contract is read once, not once per backend.
"""

from __future__ import annotations

import torch

from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import MXTensor


def check_routed_batch(
    batch: ExpertBatch,
    x_q: MXTensor,
    *,
    name: str,
    profile: str,
    align: int,
) -> None:
    """Cheap per-launch checks: host-side only, no device sync, no hashing.

    Full contract validation (``ExpertBatch.validate``) belongs where the batch
    is built: it walks the offsets on device and SHA-256s every weight byte,
    which costs far more than the kernels it guards.

    ``align`` is the backend's tile granularity -- ``hidden`` and ``ffn`` must
    both be multiples of it, so no launch ever needs a partial-K tile.
    """
    m, hidden = batch.x.shape
    if batch.p_s.dtype != torch.float32 or batch.p_s.shape != (m,):
        raise TypeError(f"p_s must be FP32 [{m}], got {batch.p_s.dtype} {tuple(batch.p_s.shape)}")
    if batch.expert_offsets.dtype != torch.int32 or batch.expert_offsets.dim() != 1:
        raise TypeError("expert_offsets must be a 1-D int32 tensor")
    n_experts = batch.expert_offsets.numel() - 1
    if tuple(batch.w1.shape) != (n_experts, 2 * batch.ffn, hidden):
        raise ValueError(f"w1 shape {batch.w1.shape} != {(n_experts, 2 * batch.ffn, hidden)}")
    if tuple(batch.w2.shape) != (n_experts, hidden, batch.ffn):
        raise ValueError(f"w2 shape {batch.w2.shape} != {(n_experts, hidden, batch.ffn)}")
    if batch.w1.elem_format != "e2m1" or batch.w2.elem_format != "e2m1":
        raise TypeError("base weights must be MXFP4 (e2m1)")
    if batch.numeric_profile != profile:
        raise NotImplementedError(
            f"{name} implements {profile!r}; batch declares "
            f"{batch.numeric_profile!r} (fail-closed, no fallback)"
        )
    if batch.lora is not None:
        raise NotImplementedError(f"{name}: LoRA is not supported in v1 (base weights only)")
    if not batch.x.is_cuda:
        raise NotImplementedError(f"{name} requires CUDA tensors, got {batch.x.device}")
    if x_q.elem_format != "e4m3":
        raise TypeError("x_q must be an e4m3 (MXFP8) activation")
    if tuple(x_q.shape) != tuple(batch.x.shape):
        raise ValueError(f"x_q shape {x_q.shape} != x shape {tuple(batch.x.shape)}")
    if batch.hidden % align != 0 or batch.ffn % align != 0:
        raise NotImplementedError(
            f"{name}: hidden and ffn must be multiples of {align}, got "
            f"{batch.hidden} / {batch.ffn}"
        )
