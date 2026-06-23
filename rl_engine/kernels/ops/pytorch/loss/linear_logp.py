# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch

# Backward token-chunk target: process at most this many ``[chunk, V]`` logit
# elements per cuBLAS step so peak backward memory stays ~``chunk*V`` instead of
# ``N*V``.
BWD_CHUNK_ELEMS = 1 << 24


def chunked_linear_logp_backward(
    grad_logp: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    target_1d: torch.Tensor,
    bias_t: torch.Tensor,
    *,
    has_bias: bool,
    lead_shape,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype,
    chunk_elems: int = BWD_CHUNK_ELEMS,
):
    # Liger-style chunked backward shared by the Triton and CUDA SM90 fused ops.
    n, d = hidden_2d.shape
    v = weight.shape[0]
    dt = weight.dtype
    g = grad_logp.reshape(-1).to(torch.float32)

    grad_h = torch.empty_like(hidden_2d, dtype=torch.float32)
    grad_w = torch.zeros(v, d, device=weight.device, dtype=torch.float32)
    grad_b = torch.zeros(v, device=weight.device, dtype=torch.float32) if has_bias else None

    chunk = max(1, min(n, chunk_elems // v))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        x = hidden_2d[i0:i1]  # [C, D]
        logits = torch.matmul(x, weight.t())  # [C, V]
        if has_bias:
            logits = logits + bias_t

        # dz = g * (onehot - softmax(logits)), recomputed from scratch so it is
        # self-normalizing and independent of the forward's saved lse.
        dz = torch.softmax(logits.float(), dim=-1).neg_()  # [C, V] fp32
        rows = torch.arange(i1 - i0, device=dz.device)
        dz[rows, target_1d[i0:i1].long()] += 1.0
        dz *= g[i0:i1].unsqueeze(1)

        dz_dt = dz.to(dt)
        grad_h[i0:i1] = torch.matmul(dz_dt, weight).float()  # [C, D]
        grad_w += torch.matmul(dz_dt.t(), x).float()  # [V, D]
        if grad_b is not None:
            grad_b += dz.sum(0)

    grad_hidden = grad_h.to(hidden_dtype).reshape(tuple(lead_shape) + (d,))
    grad_weight = grad_w.to(weight_dtype)
    grad_bias = grad_b.to(bias_dtype) if grad_b is not None else None
    return grad_hidden, grad_weight, grad_bias


class NativeLinearLogpOp:
    """Naive PyTorch reference for fused linear log-prob.

    Materializes the full ``[N, V]`` logits with a single ``F.linear`` and runs
    ``log_softmax`` + ``gather``. This is the obviously-correct oracle the fused
    kernels are validated against (and the baseline the benchmark measures the
    VRAM win against); it is also the CPU / Triton-less fallback. Differentiable
    w.r.t. ``hidden``, ``lm_head_weight`` and ``bias`` through autograd.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(hidden, lm_head_weight, target_ids, bias)

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Selected-token log-prob ``z[t] - logsumexp(z)``, returned in float32."""
        if hidden.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"hidden leading shape {tuple(hidden.shape[:-1])} must match "
                f"target_ids shape {tuple(target_ids.shape)}"
            )
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )

        lead_shape = hidden.shape[:-1]
        hidden_2d = hidden.reshape(-1, hidden.size(-1))
        logits = torch.nn.functional.linear(hidden_2d, lm_head_weight, bias)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_1d = target_ids.reshape(-1).to(device=logits.device, dtype=torch.long)
        selected = torch.gather(log_probs, dim=-1, index=target_1d.unsqueeze(1)).squeeze(-1)
        return selected.reshape(lead_shape)
