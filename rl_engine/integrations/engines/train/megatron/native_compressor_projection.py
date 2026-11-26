# SPDX-License-Identifier: Apache-2.0
"""Preserve the real compressor FP32 GEMM output while retaining its Parameters."""

import torch

from rl_engine.backends.rocm.attention.compressor_projection import compressor_projection


class CompressorProjectionFP32(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        bias = getattr(original, "bias", None)
        # The pinned TE represents bias=False with an empty Tensor.
        has_bias = bias is not None and (not isinstance(bias, torch.Tensor) or bias.numel() != 0)
        if original.weight.dtype != torch.bfloat16 or has_bias:
            raise ValueError("expected original bias-free BF16 compressor projection")
        self.weight = original.weight

    def forward(self, x, **kwargs):
        if kwargs or x.dtype != torch.bfloat16:
            raise ValueError("compressor GEMM requires original BF16 operands")
        shape = x.shape
        y = compressor_projection(x.reshape(-1, shape[-1]).contiguous(), self.weight)
        if not hasattr(self, "_executed"):
            print(
                "RLK_NATIVE_COMPRESSOR_FP32_EXECUTED",
                __file__,
                "fixed no-split FP32 GEMM; original BF16 operands",
                flush=True,
            )
            self._executed = True
        return y.reshape(*shape[:-1], self.weight.shape[0]), None

    def backward_dw(self):
        pass


def bind_compressor_projection(compressor):
    for name in ("linear_wkv", "linear_wgate"):
        original = getattr(compressor, name)
        replacement = CompressorProjectionFP32(original)
        assert replacement.weight is original.weight
        setattr(compressor, name, replacement)
    return compressor


class IndexerWeightsProjDet(CompressorProjectionFP32):
    """Indexer ``weights_proj`` on the fixed-order det GEMM (same as rollout), BF16 output."""

    def forward(self, x, **kwargs):
        y, bias = super().forward(x, **kwargs)
        return y.to(torch.bfloat16), bias


def bind_indexer_weights_proj(indexer):
    original = indexer.linear_weights_proj
    replacement = IndexerWeightsProjDet(original)
    assert replacement.weight is original.weight
    indexer.linear_weights_proj = replacement
    return indexer
