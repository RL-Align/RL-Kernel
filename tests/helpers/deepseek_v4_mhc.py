# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Inputs shared by the batch-invariance tests of the DeepSeek-V4 mHC ops and layer end."""

import torch

N, V = 4, 4096  # DeepSeek-V4-Flash-0731 hc_mult; test vocabulary

# M = 1, 16, 17 (RFC switch points), the WS1 manifest's token counts 8 and 59,
# and powers of two up to 16384.
M_VALUES = [1, 8, 16, 17, 59, 64, 128, 256, 1024, 2048, 4096, 8192, 16384]


def rocm_gpu_available() -> bool:
    """True on a ROCm GPU, where the Triton mHC ops run."""
    return torch.cuda.is_available() and torch.version.hip is not None


def randn(gen, *shape):
    return torch.randn(*shape, generator=gen, device="cuda")


def weights(c):
    """mHC pre, mHC head, final RMSNorm and lm_head weights at hidden size c."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    k = N * c
    mix = 2 * N + N * N
    return {
        "pre_fn": randn(gen, mix, k) / k**0.5,
        "pre_scale": 0.5 + torch.rand(3, generator=gen, device="cuda"),
        "pre_base": 0.1 * randn(gen, mix),
        "head_fn": randn(gen, N, k) / k**0.5,
        "head_scale": 0.5 + torch.rand(1, generator=gen, device="cuda"),
        "head_base": 0.1 * randn(gen, N),
        "norm_weight": 1.0 + 0.1 * randn(gen, c),
        "lm_head_weight": (0.02 * randn(gen, V, c)).to(torch.bfloat16),  # as in the checkpoint
    }


def forward_backward(fn, inputs, grad_out=None):
    """Forward with grad on float inputs, then backward; returns (outputs, input grads).

    Without ``grad_out`` the loss is the sum of the last output.
    """
    leaves = [t.detach().clone().requires_grad_(t.is_floating_point()) for t in inputs]
    outputs = fn(*leaves)
    loss_target = outputs[0] if grad_out is not None else outputs[-1].sum()
    loss_target.backward(grad_out)
    return [o.detach() for o in outputs], [t.grad.detach() for t in leaves if t.grad is not None]
