# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 mHC contract: mHC pre and mHC post (DSV4 RFC tasks 1-4).

Records the arithmetic of the ROCm backend, RL-Kernel's Triton kernels in
rl_engine/backends/rocm/mhc/triton_mhc.py (copied from aiter ``326cffa``,
``mhc_pre_dsv4``, ``mhc_post_dsv4``, and made batch invariant by
construction), as measured in tests/backends/rocm/test_rocm_mhc_*.py, and
where the PyTorch reference (rl_engine/ops/mhc/training.py, following
DeepSeek-V4-Flash-0731 ``inference/model.py``) differs. ``n`` residual streams
of hidden size ``C`` per token; leading dimensions are flattened into the token
count M; K = n * C.

mHC pre, per token, x = FP32(flatten(residual)) [K]::

    K is cut into NUM_KSPLIT = 4 segments; per segment s:
      sq[s]  = pairwise sum over 128 columns of the element-wise fma(x, x, .)
               accumulated over the K tiles of 128 of the segment
      acc[s] = x[s] @ fn[:, s]^T                 FP32 x FP32 MFMA (ieee)
    r      = 1 / sqrt((((sq[0] + sq[1]) + sq[2]) + sq[3]) / K + rms_eps)
    acc    = ((acc[0] + acc[1]) + acc[2]) + acc[3]
    pre    = sigmoid(r * scale[0] * acc[:n] + base[:n]) + hc_pre_eps
    post   = hc_post_mult_value * sigmoid(r * scale[1] * acc[n:2n] + base[n:2n])
    comb   = sinkhorn(r * scale[2] * acc[2n:] + base[2n:], sinkhorn_repeat)  [n, n]
    layer_input = BF16(fma(pre3, x3, fma(pre2, x2, fma(pre1, x1, pre0 * x0))))

sinkhorn is DeepSeek-V4's asymmetric form: the first round takes a row softmax,
adds hc_sinkhorn_eps, then divides each column by (column sum + eps); each
further round divides rows, then columns, by (sum + eps). Row and column sums
are pairwise: (p0 + p1) + (p2 + p3).

mHC post, per token and destination stream j::

    out[j] = BF16(fma(comb[3, j], residual[3], fma(comb[2, j], residual[2],
                  fma(comb[1, j], residual[1], fma(comb[0, j], residual[0], post[j] * y))))))

in FP32: the product post[j] * y, then one fma per source stream in source-stream
order. ``comb`` is indexed [source stream, destination stream]; the old residual
enters once, through ``comb``.

Backward (Triton, rl_engine/backends/rocm/mhc/triton_mhc.py): the exact
derivative of the arithmetic above, per token, with the bf16 casts taken as the
identity. The forward stores, with grad enabled, s = sigmoid of the pre and
post logits, the raw comb logits, acc and r; the backward
uses those values. Each multiply and add rounds on its own except the written
fma. For mHC pre, with g the gradient of layer_input::

    dpre[i]   = pairwise over 128 columns of fma(g, x_i, .) over C tiles of 128
    dh_pre    = dpre * (s_pre * (1 - s_pre))
    dh_post   = (g_post * hc_post_mult_value) * (s_post * (1 - s_post))
    dh_res    = Sinkhorn backward: each of the 2 * sinkhorn_repeat steps replayed
                from the logits with the forward's code, then walked back with
                dP = (dY - sum(dY * Y)) / (sum(P) + eps) (pairwise sums); the
                first step's softmax as S * (dY - row sum(dY * S))
    dacc      = dh * (r * scale[group]);   t[group] = pairwise sum of dh * acc
    dr        = fma(scale[2], t_res, fma(scale[1], t_post, scale[0] * t_pre))
    dx_i[c]   = BF16(fma(-(dr * r^3) / K, x_i[c],
                     fma(fn[23, iC+c], dacc[23], ... fma(fn[0, iC+c], dacc[0], pre_i * g[c]))))

mHC post::

    dy[c]       = BF16(fma(post3, g3[c], fma(post2, g2[c], fma(post1, g1[c], post0 * g0[c]))))
    dres_h[c]   = BF16(the same with comb[h, j] in place of post[j])
    dpost[j]    = pairwise over 128 columns of fma(g_j, y, .) over C tiles of 128
    dcomb[h, j] = pairwise over 128 columns of fma(g_j, residual_h, .) over C tiles of 128

Parameter gradients (fn, scale, base) are computed only when asked for, through
the canonical backward session: per-row pieces dacc, r * t and dh, reduced in
logical-key order with the fixed-order fp32 det_gemm (fn) and a serial fold
(scale, base).
"""

from __future__ import annotations

import copy

import torch

PROFILE = "dsv4-mhc-rocm-triton-v1"
# The aiter commit the ROCm kernels were copied from.
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
PROJECTION = "fp32 fn; fp32 x fp32 MFMA (ieee), 4 K segments added in segment order"
REFERENCE_PROJECTION = "fp32 x fp32"
SUM_OF_SQUARES = (
    "per K segment: element-wise fma over K tiles of 128, then pairwise over the 128 "
    "columns; 4 K segments added in segment order"
)
SINKHORN = "dsv4_asymmetric, pairwise row and column sums"
STREAM_SUM = "fma(w3, x3, fma(w2, x2, fma(w1, x1, w0 * x0))) in fp32, one cast to bf16"

# The constants that fix the reduction order, equal to those of
# rl_engine/backends/rocm/mhc/triton_mhc.py. No launch parameter changes the bits.
ARITHMETIC_CONSTANTS = {"NUM_KSPLIT": 4, "BLOCK_K": 128, "matrix_instr_nonkdim": 16, "kpack": 1}
# The backward's: the width of the element-wise accumulators of every sum over C.
BACKWARD_ARITHMETIC_CONSTANTS = {"DOT_BLOCK": 128}

# "backward" covers the input gradients (dResidual; for mHC post also dy, dpost_mix,
# dcomb_mix): batch invariant by construction, like the forward. Parameter gradients
# are reduced over tokens in logical-key order by the canonical backward session.
BATCH_INVARIANCE = {
    "forward": True,
    "backward": True,
    "validated_on": "gfx942",
    "validated_shape": "n=4, C=4096 (DeepSeek-V4-Flash) and C=7168",
    "launch_configs": "one config per op for every M; the bits do not depend on it",
    "backward_implementation": (
        "Triton, exact derivative of the forward from saved forward values, written "
        "reduction order; parameter gradients through the canonical backward session"
    ),
}

BACKWARD_SUM_OVER_C = (
    "per stream: element-wise fma over C tiles of 128, then pairwise over the 128 columns"
)
BACKWARD_PARAMETER_REDUCTION = (
    "canonical backward session, logical-key order: fn by the fixed-order fp32 det_gemm, "
    "scale and base by a serial fp32 fold"
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
}

_ARITHMETIC = {
    "mhc_pre": {
        "rms_form": RMS_FORM,
        "sum_of_squares": SUM_OF_SQUARES,
        "projection": PROJECTION,
        "reference_projection": REFERENCE_PROJECTION,
        "pre_weights": "sigmoid(.) + hc_pre_eps",
        "post_weights": "hc_post_mult_value * sigmoid(.)",
        "sinkhorn": SINKHORN,
        "stream_sum": STREAM_SUM,
        "downcast_points": ["layer_input"],
        "arithmetic_constants": dict(ARITHMETIC_CONSTANTS),
        "backward": {
            "saved": "sigmoid(pre), sigmoid(post), raw comb logits, acc, r per token",
            "sum_over_c": BACKWARD_SUM_OVER_C,
            "sinkhorn": "steps replayed from the saved logits, walked back with pairwise sums",
            "projection_transpose": "one fma per fn row, rows 0..23 in order",
            "residual_gradient": "pre_i * g, then the 24 fn rows, then the RMS term; one bf16 cast",
            "parameter_reduction": BACKWARD_PARAMETER_REDUCTION,
        },
    },
    "mhc_post": {
        "formula": "out[j] = post[j] * y + sum_h comb[h, j] * residual[h]",
        "accumulation": (
            "fp32: post[j] * y, then fma(comb[h, j], residual[h], .) for h = 0 .. n-1 in order"
        ),
        "residual_terms": "once, through comb",
        "downcast_points": ["out"],
        "backward": {
            "dy_and_dresidual": "one fma per destination stream in order; one bf16 cast",
            "dpost_and_dcomb": BACKWARD_SUM_OVER_C,
        },
    },
}

OPS = tuple(_SHAPES)


class MHCContractError(ValueError):
    """Raised for an mHC operator name outside this contract."""


def pre_fn_rows(n: int = HC_MULT) -> int:
    """Rows of mHC pre's ``fn`` and ``base``: n pre, n post and n * n comb logits."""
    return 2 * n + n * n


def describe(op: str) -> dict[str, object]:
    """Provenance fields of one mHC operator: ``"mhc_pre"`` or ``"mhc_post"``."""
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
    }


__all__ = [
    "AITER_COMMIT",
    "ARITHMETIC_CONSTANTS",
    "BACKWARD_ARITHMETIC_CONSTANTS",
    "BATCH_INVARIANCE",
    "HC_MULT",
    "HC_POST_MULT_VALUE",
    "HC_PRE_EPS",
    "HC_SINKHORN_EPS",
    "MIX_DTYPE",
    "MHCContractError",
    "OPS",
    "OUTPUT_DTYPE",
    "PARAM_DTYPE",
    "PROFILE",
    "RESIDUAL_DTYPE",
    "RMS_EPS",
    "SINKHORN_REPEAT",
    "describe",
    "pre_fn_rows",
]
