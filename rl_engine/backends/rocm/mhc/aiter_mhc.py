# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm mHC pre: integration of aiter's DeepSeek-V4 kernel.

AiterMHCPreOp calls the public aiter op ``mhc_pre_dsv4``, which launches Triton
kernels in forward and recomputes in PyTorch for backward. aiter's own config
splits K once M >= 1024 on gfx942, which changes the result bits with M, so
this op passes the config through the public ``config`` argument, chosen per M
by rl_engine/backends/rocm/config_utils.py from configs/<arch>/mhc_pre, else
FIXED_PRE_CONFIG for every M. Batch invariance is checked in
tests/backends/rocm/test_rocm_mhc_pre.py.
"""

from __future__ import annotations

import torch

from rl_engine.backends.rocm.config_utils import get_config
from rl_engine.utils.logger import logger

# Default for mhc_pre_dsv4: the config aiter 326cffa picks itself for M=1 on
# gfx942 with C=4096 (MHC_FUSED_SINKHORN-C=4096.json, entry M_LEQ_1). It does
# not split K. Used for every M when there is no tuned file.
FIXED_PRE_CONFIG = {
    "NUM_KSPLIT": 1,
    "BLOCK_M": 32,
    "BLOCK_K": 128,
    "BLOCK_C": 128,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 2,
}

# Parameters of mhc_pre_dsv4 that change the output bits (docs/operators/mhc.md):
# equal in every entry of a tuned file. BLOCK_C=256, and BLOCK_M=16 with
# num_warps=8, also change the bits; the tuner never picks them, and the
# batch-invariance tests cover the file.
PRE_BIT_PARAMS = ("NUM_KSPLIT", "BLOCK_K", "kpack")


def _import_aiter_mhc():
    if torch.version.hip is None:
        raise RuntimeError("aiter mHC ops require a ROCm PyTorch build.")
    try:
        from aiter.ops.triton.fusions import mhc_head_dsv4, mhc_post_dsv4, mhc_pre_dsv4
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "aiter with the DeepSeek-V4 mHC ops (aiter PR #5492 or later) is required."
        ) from exc
    return mhc_pre_dsv4, mhc_post_dsv4, mhc_head_dsv4


# DeepSeek-V4's hc_mult; aiter's kernels are only validated with it.
SUPPORTED_N = 4


def _check(name, tensor, dtype, shape=None):
    """Raise for an input these ops do not support, instead of running it unvalidated."""
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")


def _check_streams(residual):
    """(n, C) of a bf16 residual [..., n, C] with n = SUPPORTED_N."""
    _check("residual", residual, torch.bfloat16)
    if residual.dim() < 2 or residual.shape[-2] != SUPPORTED_N:
        raise ValueError(
            f"residual must have shape [..., {SUPPORTED_N}, C], got {tuple(residual.shape)}"
        )
    return residual.shape[-2:]


def check_pre_inputs(residual, fn, scale, base):
    """Inputs of AiterMHCPreOp: bf16 residual [..., n, C]; fp32 fn, scale, base."""
    n, c = _check_streams(residual)
    rows = 2 * n + n * n
    _check("fn", fn, torch.float32, (rows, n * c))
    _check("scale", scale, torch.float32, (3,))
    _check("base", base, torch.float32, (rows,))


class AiterMHCPreOp:
    """mHC pre via ``aiter.mhc_pre_dsv4`` with a per-M launch config; see NativeMHCPreOp."""

    def __init__(self):
        self.op, _, _ = _import_aiter_mhc()
        logger.info("Linked aiter mhc_pre_dsv4 with batch-invariant launch configs.")

    def __call__(
        self,
        residual,
        fn,
        scale,
        base,
        *,
        rms_eps=1e-6,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
    ):
        check_pre_inputs(residual, fn, scale, base)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        residual = residual.reshape(-1, n, c).contiguous()
        config = get_config(
            "mhc_pre",
            f"MHC_PRE-n={n}-C={c}.json",
            residual.shape[0],
            FIXED_PRE_CONFIG,
            PRE_BIT_PARAMS,
            residual.device,
        )
        post, comb, layer_input = self.op(
            residual,
            fn.contiguous(),
            scale.contiguous(),
            base.contiguous(),
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
            config=config,
        )
        return post.reshape(*lead, n), comb.reshape(*lead, n, n), layer_input.reshape(*lead, c)
