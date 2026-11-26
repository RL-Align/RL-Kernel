# SPDX-License-Identifier: Apache-2.0
"""Rollout prefill indexer logits on the paged decode kernel, plus the det weights_proj GEMM.

The flydsl prefill kernel and the paged decode kernel round differently (k-scale per head
before ReLU vs after the head sum). The actor uses the paged kernel for the whole sequence,
so rollout prefill must too: ``rocm_fp8_mqa_logits`` is wrapped to return the paged-kernel
logits in the original absolute-column layout ([M, N], row i valid at [ks_i, ke_i)).
Decode already runs this kernel (N_SPLITS does not change bits: tiles are disjoint pages).
"""

import importlib
import sys
from types import MethodType

import torch

from rl_engine.backends.rocm.attention.compressor_projection import compressor_projection
from rl_engine.backends.rocm.attention.indexer_paged_logits import (
    HEAD_DIM,
    paged_c4_logits_contiguous,
)

_SPARSE = "vllm.v1.attention.ops.rocm_aiter_mla_sparse"
_NAME = "rocm_fp8_mqa_logits"
_STATE = {"executed": False}


def _make_wrapper(original):
    def rocm_fp8_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke):
        k_fp8, k_scale = kv
        if (
            q.ndim != 3
            or q.shape[2] != HEAD_DIM
            or k_fp8.ndim != 2
            or k_fp8.shape[1] != HEAD_DIM
            or k_scale.dtype != torch.float32
            or k_scale.numel() != k_fp8.shape[0]
            or weights.dtype != torch.float32
            or weights.shape != q.shape[:2]
        ):
            raise ValueError(
                "unexpected indexer prefill operands "
                f"{tuple(q.shape)} {q.dtype} {tuple(k_fp8.shape)} {k_fp8.dtype} "
                f"{k_scale.dtype} {tuple(weights.shape)} {weights.dtype}"
            )
        logits = paged_c4_logits_contiguous(
            q,
            weights,
            k_fp8,
            k_scale,
            cu_seqlen_ks,
            cu_seqlen_ke,
            absolute_columns=True,
            num_columns=k_fp8.shape[0],
        )
        if not _STATE["executed"]:
            _STATE["executed"] = True
            print("RLK_ROLLOUT_PAGED_INDEXER_LOGITS_EXECUTED", tuple(q.shape), tuple(logits.shape), flush=True)
        return logits

    rocm_fp8_mqa_logits._rlk_paged = True
    rocm_fp8_mqa_logits._rlk_original = original
    return rocm_fp8_mqa_logits


def bind_rollout_paged_indexer_logits():
    """Idempotent, process-wide. Must run before the model's first forward."""
    sparse = importlib.import_module(_SPARSE)
    current = getattr(sparse, _NAME)
    if getattr(current, "_rlk_paged", False):
        wrapper, original = current, current._rlk_original
    else:
        original = current
        wrapper = _make_wrapper(original)
        setattr(sparse, _NAME, wrapper)
    # Rebind modules that did `from ... import rocm_fp8_mqa_logits`.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, _NAME, None) is original:
            setattr(module, _NAME, wrapper)


def bind_rollout_weights_proj(attn):
    """Indexer ``weights_proj`` on the shared fixed-order det GEMM (M-invariant), BF16 out."""
    indexer = attn.indexer
    if indexer is None:
        return
    layer = indexer.weights_proj
    if getattr(layer, "bias", None) is not None or layer.weight.dtype != torch.bfloat16:
        raise ValueError("expected bias-free BF16 indexer weights_proj")

    def forward(self, x):
        y = compressor_projection(x, self.weight).to(torch.bfloat16)
        if not getattr(self, "_rlk_executed", False):
            print("RLK_ROLLOUT_INDEXER_WEIGHTS_PROJ_DET_EXECUTED", tuple(x.shape), flush=True)
            self._rlk_executed = True
        return y, None

    layer.forward = MethodType(forward, layer)
