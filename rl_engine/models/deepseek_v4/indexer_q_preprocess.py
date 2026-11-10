# SPDX-License-Identifier: Apache-2.0
"""Item 15: Indexer Q RoPE -> Hadamard -> MXFP4; identity STE on the decoded value."""

from dataclasses import dataclass

import torch

from rl_engine.contracts.operators.indexer_q_preprocess import (
    IndexerQPreprocessContract,
    require,
)
from rl_engine.models.deepseek_v4.partial_rope import check_backend, select_angles


@dataclass(frozen=True)
class IndexerQResult:
    values: torch.Tensor
    packed: torch.Tensor
    scales: torch.Tensor
    implementation_fingerprint: str


class _IndexerQPreprocess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cos, sin, backend, contract):
        ctx.save_for_backward(cos, sin)
        ctx.backend = backend
        values, packed, scales = backend.forward(q, cos, sin, contract)
        ctx.mark_non_differentiable(packed, scales)
        return values, packed, scales

    @staticmethod
    def backward(ctx, grad, _packed, _scales):
        require(not torch.is_grad_enabled(), "higher-order Q gradients are not supported")
        cos, sin = ctx.saved_tensors
        grad = grad.to(torch.bfloat16).contiguous()
        return ctx.backend.vjp(grad, cos, sin), None, None, None, None


class IndexerQPreprocess:
    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(
            isinstance(contract, IndexerQPreprocessContract),
            "typed Indexer Q preprocess contract required",
        )
        if backend is None:
            from rl_engine.backends.rocm.attention.partial_rope import AiterIndexerQBackend

            backend = AiterIndexerQBackend()
        self.contract, self.backend = contract, backend
        self.implementation_fingerprint = check_backend(
            contract, backend, expected_implementation, __file__
        )

    def __call__(self, q, positions, cos_cache, sin_cache):
        """q: BF16 [T,H,128]. Returns BF16 values, uint8 [T,H,64] codes, E8M0 [T,H,4]."""
        masked, cos, sin = select_angles(
            q,
            positions,
            cos_cache,
            sin_cache,
            contract=self.contract.rope,
            backend=self.backend,
            heads=self.contract.num_heads,
        )
        require(bool(torch.isfinite(masked).all()), "non-finite Indexer Q")
        values, packed, scales = _IndexerQPreprocess.apply(
            masked, cos, sin, self.backend, self.contract
        )
        return IndexerQResult(values, packed, scales, self.implementation_fingerprint)


indexer_q_rope_hadamard_mxfp4 = IndexerQPreprocess

__all__ = ["IndexerQPreprocess", "IndexerQResult", "indexer_q_rope_hadamard_mxfp4"]
