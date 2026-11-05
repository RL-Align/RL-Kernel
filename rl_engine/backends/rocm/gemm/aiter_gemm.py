# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm batch-invariant linear, y = x @ w^T + bias: integration of aiter's gemm_a16w16.

aiter chooses gemm_a16w16's launch config by (M, N, K), and some of its
choices split K, which makes results change with M. This op passes the config
itself through the public ``config`` argument, chosen per M by
rl_engine/backends/rocm/config_utils.py from
configs/<arch>/gemm_a16w16/GEMM-A16W16-N={N}-K={K}.json, else
FIXED_LINEAR_CONFIG for every M. Batch invariance is checked in
tests/backends/rocm/test_rocm_batch_invariant_linear.py.

Backward (dx, dw, dbias) uses det_gemm's fixed-K-order Triton GEMM in fp32.
"""

from __future__ import annotations

import torch

from rl_engine.backends.rocm.config_utils import get_config
from rl_engine.utils.logger import logger

# Default: aiter 326cffa's gfx942 default for small M (gemm_a16w16/DEFAULT.json,
# entry M_LEQ_64), used for every M of shapes without a tuned file. NUM_KSPLIT=1:
# K is not split.
FIXED_LINEAR_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg",
    "NUM_KSPLIT": 1,
    "kpack": 1,
}

# Parameters of gemm_a16w16 that change the output bits
# (docs/operators/batch-invariant-linear.md).
LINEAR_BIT_PARAMS = ("NUM_KSPLIT", "kpack", "matrix_instr_nonkdim")


SUPPORTED_OUT_DTYPES = (torch.bfloat16, torch.float32)


def check_linear_inputs(x, weight, bias, out_dtype):
    """Raise for an input this op does not support, instead of running it unvalidated:
    x [..., K] and weight [N, K] in bf16, bias [N] in out_dtype, out_dtype bf16 or fp32."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(f"x and weight must be bf16, got {x.dtype} and {weight.dtype}")
    if out_dtype not in SUPPORTED_OUT_DTYPES:
        raise TypeError(f"out_dtype must be one of {SUPPORTED_OUT_DTYPES}, got {out_dtype}")
    if x.dim() < 1 or weight.dim() != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError(
            f"x [..., K] and weight [N, K] do not match: {tuple(x.shape)}, {tuple(weight.shape)}"
        )
    if bias is not None and (bias.dtype != out_dtype or tuple(bias.shape) != (weight.shape[0],)):
        raise ValueError(
            f"bias must be [{weight.shape[0]}] in {out_dtype}, "
            f"got {tuple(bias.shape)} in {bias.dtype}"
        )


def select_config(m: int, n: int, k: int, device: torch.device) -> dict:
    """Launch config for an (M, N, K) GEMM on ``device``'s GPU."""
    config = get_config(
        "gemm_a16w16",
        f"GEMM-A16W16-N={n}-K={k}.json",
        m,
        FIXED_LINEAR_CONFIG,
        LINEAR_BIT_PARAMS,
        device,
    )
    if config["NUM_KSPLIT"] != 1:
        raise ValueError(f"gemm_a16w16 config {config}: only whole-K configs are supported")
    # With NUM_KSPLIT=1 the single K split is the whole K.
    config["SPLITK_BLOCK_SIZE"] = k
    return config


class _LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x2d, weight, bias, out_dtype):
        ctx.save_for_backward(x2d, weight)
        ctx.has_bias = bias is not None
        m, k = x2d.shape
        if m == 0:
            return x2d.new_empty((0, weight.shape[0]), dtype=out_dtype)
        config = select_config(m, weight.shape[0], k, x2d.device)
        return op(x2d, weight, bias, dtype=out_dtype, config=config, backend="triton")

    @staticmethod
    def backward(ctx, grad_out):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm

        x2d, weight = ctx.saved_tensors
        grad_out = grad_out.float().contiguous()
        dx = dw = dbias = None
        if x2d.shape[0] == 0:  # no rows: no launch, zero parameter gradients
            dx = torch.zeros_like(x2d) if ctx.needs_input_grad[1] else None
            dw = torch.zeros_like(weight) if ctx.needs_input_grad[2] else None
            if ctx.has_bias and ctx.needs_input_grad[3]:
                dbias = grad_out.new_zeros(weight.shape[0])
            return None, dx, dw, dbias, None
        # TODO: pass x2d and weight in bf16. The fp32 kernel promotes its inputs
        # itself, so .float() only adds an fp32 copy per call (2.1 GB for lm_head).
        if ctx.needs_input_grad[1]:
            dx = _triton_gemm(grad_out, weight.float(), output_dtype=torch.float32)
            dx = dx.to(x2d.dtype)
        if ctx.needs_input_grad[2]:
            dw = _triton_gemm(grad_out.t(), x2d.float(), output_dtype=torch.float32)
            dw = dw.to(weight.dtype)
        if ctx.has_bias and ctx.needs_input_grad[3]:
            dbias = grad_out.sum(0)
        return None, dx, dw, dbias, None


class AiterBatchInvariantLinearOp:
    """y = x @ weight^T (+ bias) via ``aiter.gemm_a16w16``; per-M configs that keep the bits."""

    def __init__(self):
        if torch.version.hip is None:
            raise RuntimeError("aiter gemm_a16w16 requires a ROCm PyTorch build.")
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError("aiter with the Triton gemm_a16w16 op is required.") from exc
        self.op = gemm_a16w16
        logger.info("Linked aiter gemm_a16w16 with batch-invariant launch configs.")

    def __call__(self, x, weight, bias=None, *, out_dtype=torch.bfloat16):
        """x [..., K] and weight [N, K] in bf16; bias [N]; returns [..., N] in out_dtype."""
        check_linear_inputs(x, weight, bias, out_dtype)
        k = x.shape[-1]
        lead = x.shape[:-1]
        x2d = x.reshape(-1, k).contiguous()
        y = _LinearFn.apply(self.op, x2d, weight.contiguous(), bias, out_dtype)
        return y.reshape(*lead, weight.shape[0])
