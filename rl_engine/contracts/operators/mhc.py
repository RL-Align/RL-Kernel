# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 mHC contract: mHC pre, mHC post and mHC head (DSV4 RFC tasks 1-4, 8).

Records the arithmetic of the ROCm backend, aiter ``326cffa``
(``mhc_pre_dsv4``, ``mhc_post_dsv4``, ``mhc_head_dsv4``), as measured in
tests/backends/rocm/test_rocm_mhc_*.py, and where the fp32 PyTorch reference
(rl_engine/reference/mhc/mhc.py, DeepSeek-V4-Flash-0731 ``inference/model.py``)
differs. ``n`` residual streams of hidden size ``C`` per token; leading
dimensions are flattened into the token count M; K = n * C.

mHC pre, per token, x = FP32(flatten(residual)) [K]::

    r      = rsqrt(mean(x^2) + rms_eps)                FP32
    mixes  = (x @ BF16(fn)^T) * r                      BF16 x BF16, FP32 accumulation
    pre    = sigmoid(mixes[:n] * scale[0] + base[:n]) + hc_pre_eps
    post   = hc_post_mult_value * sigmoid(mixes[n:2n] * scale[1] + base[n:2n])
    comb   = sinkhorn(mixes[2n:] * scale[2] + base[2n:], sinkhorn_repeat)  [n, n]
    layer_input = BF16(fma(pre2, x2, pre0 * x0) + fma(pre3, x3, pre1 * x1))

sinkhorn is DeepSeek-V4's asymmetric form: the first round takes a row softmax,
adds hc_sinkhorn_eps, then divides each column by (column sum + eps); each
further round divides rows, then columns, by (sum + eps).

mHC post, per token and destination stream j::

    out[j] = BF16(post[j] * y + sum_h comb[h, j] * residual[h])   FP32

``comb`` is indexed [source stream, destination stream]; the old residual enters
once, through ``comb``.

mHC head, per token, with ``fn`` [n, K] and ``scale`` [1]::

    w   = sigmoid((x @ BF16(fn)^T) * r * scale + base) + hc_pre_eps
    out = BF16(sum_i w[i] * residual[i])                           FP32
"""

from __future__ import annotations

import copy

import torch

PROFILE = "dsv4-mhc-aiter-326cffa-v1"
AITER_COMMIT = "326cffa"

HC_MULT = 4
RESIDUAL_DTYPE = torch.bfloat16
PARAM_DTYPE = torch.float32
MIX_DTYPE = torch.float32
OUTPUT_DTYPE = torch.bfloat16

RMS_EPS = 1.0e-6
HC_PRE_EPS = 1.0e-6
HC_SINKHORN_EPS = 1.0e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20

RMS_FORM = "rsqrt(mean(x^2) + rms_eps) over K = n * C, fp32"
PROJECTION = "fn rounded to bf16; bf16 x bf16 with fp32 accumulation"
REFERENCE_PROJECTION = "fp32 x fp32"
SINKHORN = "dsv4_asymmetric"
STREAM_SUM = "fma(w2, x2, w0 * x0) + fma(w3, x3, w1 * x1) in fp32, one cast to bf16"

# Launch-config parameters that change the output bits, measured on MI325X with
# Triton 3.7 (docs/operators/mhc.md). PRE_BIT_PARAMS and HEAD_BIT_PARAMS equal
# the constants of the same name in rl_engine/backends/rocm/mhc/aiter_mhc.py.
PRE_BIT_PARAMS = ("NUM_KSPLIT", "BLOCK_K", "kpack")
PRE_BIT_COMBINATIONS = ("BLOCK_C=256", "BLOCK_M=16 with num_warps=8")
HEAD_BIT_PARAMS = ("BLOCK_M", "BLOCK_K", "BLOCK_C", "num_warps", "kpack")

BATCH_INVARIANCE = {
    "forward": True,
    "backward": False,
    "validated_on": "gfx942",
    "validated_shape": "DeepSeek-V4-Flash n=4, C=4096",
    "launch_configs": "shipped gfx942 tables (configs/gfx942/mhc_pre, configs/gfx942/mhc_head)",
    "backward_implementation": "aiter PyTorch recompute",
}

OPEN_ITEMS = (
    "RMS form: the RFC states 1 / (sqrt(mean(x^2)) + eps); aiter and the DeepSeek-V4 "
    "reference use rsqrt(mean(x^2) + eps).",
    "Projection precision: aiter rounds fn to bf16; the DeepSeek-V4 reference projects "
    "with fp32 fn.",
)

_SHAPES = {
    "mhc_pre": {
        "residual": "[..., n, C] bf16",
        "fn": "[2n + n^2, n*C] fp32",
        "scale": "[3] fp32",
        "base": "[2n + n^2] fp32",
        "post_mix": "[..., n] fp32",
        "comb_mix": "[..., n, n] fp32, [source, destination]",
        "layer_input": "[..., C] bf16",
    },
    "mhc_post": {
        "y": "[..., C] bf16",
        "residual": "[..., n, C] bf16",
        "post_mix": "[..., n] or [..., n, 1] fp32",
        "comb_mix": "[..., n, n] fp32, [source, destination]",
        "out": "[..., n, C] bf16",
    },
    "mhc_head": {
        "residual": "[..., n, C] bf16",
        "fn": "[n, n*C] fp32",
        "scale": "[1] fp32",
        "base": "[n] fp32",
        "out": "[..., C] bf16",
    },
}

_ARITHMETIC = {
    "mhc_pre": {
        "rms_form": RMS_FORM,
        "projection": PROJECTION,
        "reference_projection": REFERENCE_PROJECTION,
        "pre_weights": "sigmoid(.) + hc_pre_eps",
        "post_weights": "hc_post_mult_value * sigmoid(.)",
        "sinkhorn": SINKHORN,
        "stream_sum": STREAM_SUM,
        "downcast_points": ["layer_input"],
        "bit_params": list(PRE_BIT_PARAMS),
        "bit_combinations": list(PRE_BIT_COMBINATIONS),
    },
    "mhc_post": {
        "formula": "out[j] = post[j] * y + sum_h comb[h, j] * residual[h]",
        "accumulation": "fp32",
        "residual_terms": "once, through comb",
        "downcast_points": ["out"],
        "launch_config": "aiter's own, independent of M",
    },
    "mhc_head": {
        "rms_form": RMS_FORM,
        "projection": PROJECTION,
        "reference_projection": REFERENCE_PROJECTION,
        "weights": "sigmoid(.) + hc_pre_eps",
        "stream_sum": "fp32, one cast to bf16",
        "downcast_points": ["out"],
        "bit_params": list(HEAD_BIT_PARAMS),
    },
}

OPS = tuple(_SHAPES)


class MHCContractError(ValueError):
    """Raised for an mHC operator name outside this contract."""


def pre_fn_rows(n: int = HC_MULT) -> int:
    """Rows of mHC pre's ``fn`` and ``base``: n pre, n post and n * n comb logits."""
    return 2 * n + n * n


def describe(op: str) -> dict[str, object]:
    """Provenance fields of one mHC operator: ``"mhc_pre"``, ``"mhc_post"`` or ``"mhc_head"``."""
    if op not in _SHAPES:
        raise MHCContractError(f"unknown mHC operator {op!r}; want one of {OPS}")
    return {
        "profile": PROFILE,
        "op": op,
        "aiter_commit": AITER_COMMIT,
        "hc_mult": HC_MULT,
        "defaults": {
            "rms_eps": RMS_EPS,
            "hc_pre_eps": HC_PRE_EPS,
            "hc_sinkhorn_eps": HC_SINKHORN_EPS,
            "hc_post_mult_value": HC_POST_MULT_VALUE,
            "sinkhorn_repeat": SINKHORN_REPEAT,
        },
        "shapes": dict(_SHAPES[op]),
        "arithmetic": copy.deepcopy(_ARITHMETIC[op]),
        "batch_invariance": dict(BATCH_INVARIANCE),
        "open_items": list(OPEN_ITEMS),
    }


__all__ = [
    "AITER_COMMIT",
    "BATCH_INVARIANCE",
    "HC_MULT",
    "HC_POST_MULT_VALUE",
    "HC_PRE_EPS",
    "HC_SINKHORN_EPS",
    "HEAD_BIT_PARAMS",
    "MIX_DTYPE",
    "MHCContractError",
    "OPEN_ITEMS",
    "OPS",
    "OUTPUT_DTYPE",
    "PARAM_DTYPE",
    "PRE_BIT_COMBINATIONS",
    "PRE_BIT_PARAMS",
    "PROFILE",
    "RESIDUAL_DTYPE",
    "RMS_EPS",
    "SINKHORN_REPEAT",
    "describe",
    "pre_fn_rows",
]
