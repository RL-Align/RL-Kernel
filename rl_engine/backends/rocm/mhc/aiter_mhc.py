# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ROCm mHC pre, post and head: integration of aiter's DeepSeek-V4 kernels.

Each class calls one public aiter op (``mhc_pre_dsv4``, ``mhc_post_dsv4`` or
``mhc_head_dsv4``), which launches Triton kernels in forward and recomputes in
PyTorch for backward. Batch invariance is checked in
tests/backends/rocm/test_rocm_mhc_batch_invariance.py and, per RFC task, in
tests/backends/rocm/test_rocm_mhc_pre.py; that of the DeepSeek-V4 layer end in
tests/models/deepseek_v4/test_layer_end_batch_invariance.py.

For post aiter chooses the launch config. For pre and head this module passes
the config through the public ``config`` argument, chosen per M by
rl_engine/backends/rocm/config_utils.py from configs/<arch>/mhc_pre and
configs/<arch>/mhc_head, else the op's default config for every M. aiter's own
pre config splits K once M >= 1024 on gfx942, which changes the result bits
with M.
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

# Default for mhc_head_dsv4: aiter 326cffa's own config on gfx942
# (fusions/mhc_dsv4/DEFAULT.json, entry _mhc_head_kernel). Its num_warps=4 and
# num_stages=1 cannot be passed through the public config.
FIXED_HEAD_CONFIG = {"BLOCK_M": 16, "BLOCK_K": 64, "BLOCK_C": 128}

# Parameters of mhc_head_dsv4 that change the output bits (docs/operators/mhc.md).
HEAD_BIT_PARAMS = ("BLOCK_M", "BLOCK_K", "BLOCK_C", "num_warps", "kpack")


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


def check_post_inputs(y, residual, post_mix, comb_mix):
    """Inputs of AiterMHCPostOp: bf16 y [..., C] and residual [..., n, C]; fp32 post_mix
    [..., n] or [..., n, 1] and comb_mix [..., n, n]."""
    n, c = _check_streams(residual)
    lead = tuple(residual.shape[:-2])
    _check("y", y, torch.bfloat16, (*lead, c))
    _check("post_mix", post_mix, torch.float32)
    if tuple(post_mix.shape) not in ((*lead, n), (*lead, n, 1)):
        raise ValueError(f"post_mix must have shape {(*lead, n)}, got {tuple(post_mix.shape)}")
    _check("comb_mix", comb_mix, torch.float32, (*lead, n, n))


def check_head_inputs(residual, fn, scale, base):
    """Inputs of AiterMHCHeadOp: bf16 residual [..., n, C]; fp32 fn, scale, base."""
    n, c = _check_streams(residual)
    _check("fn", fn, torch.float32, (n, n * c))
    _check("scale", scale, torch.float32, (1,))
    _check("base", base, torch.float32, (n,))


class AiterMHCPostOp:
    """mHC post (stream mix-back) via ``aiter.mhc_post_dsv4``; see NativeMHCPostOp."""

    def __init__(self):
        _, self.op, _ = _import_aiter_mhc()
        logger.info("Linked aiter mhc_post_dsv4.")

    def __call__(self, y, residual, post_mix, comb_mix):
        check_post_inputs(y, residual, post_mix, comb_mix)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        out = self.op(
            y.reshape(-1, c).contiguous(),
            residual.reshape(-1, n, c).contiguous(),
            post_mix.reshape(-1, n, 1).contiguous(),
            comb_mix.reshape(-1, n, n).contiguous(),
        )
        return out.reshape(*lead, n, c)


class AiterMHCHeadOp:
    """mHC head (stream contraction) via ``aiter.mhc_head_dsv4`` with a per-M launch
    config; see NativeMHCHeadOp."""

    def __init__(self):
        _, _, self.op = _import_aiter_mhc()
        logger.info("Linked aiter mhc_head_dsv4 with batch-invariant launch configs.")

    def __call__(self, residual, fn, scale, base, *, rms_eps=1e-6, hc_pre_eps=1e-6):
        check_head_inputs(residual, fn, scale, base)
        n, c = residual.shape[-2:]
        lead = residual.shape[:-2]
        rows = residual.reshape(-1, n, c).contiguous()
        config = get_config(
            "mhc_head",
            f"MHC_HEAD-n={n}-C={c}.json",
            rows.shape[0],
            FIXED_HEAD_CONFIG,
            HEAD_BIT_PARAMS,
            rows.device,
        )
        out = self.op(
            rows,
            fn.contiguous(),
            scale.contiguous(),
            base.contiguous(),
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            config=config,
        )
        return out.reshape(*lead, c)
