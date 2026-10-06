# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


def _require_mhc_extension() -> None:
    required = ("mhc_pre_h_aggregate", "mhc_pre_h_aggregate_backward")
    if not _EXT_AVAILABLE or _C is None or not all(hasattr(_C, name) for name in required):
        raise RuntimeError("MHC H Aggregate CUDA symbols are unavailable; rebuild rl_engine._C")


class MHCPreHAggregateCudaOp:
    """Explicit forward/backward CUDA MHC weighted collapse.

    ``backward_fp32`` exposes the aggregate-only composite boundary without
    downcasting ``dR_from_aggregate`` before the controller-gradient merge.
    This operator intentionally has no standalone autograd path.
    """

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "cuda_fixed_tree_mhc_pre_h_aggregate_backward"

    def __init__(self) -> None:
        _require_mhc_extension()

    def __call__(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.forward(residual, pre)

    def forward(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and (residual.requires_grad or pre.requires_grad):
            raise RuntimeError(
                "MHC H Aggregate does not expose standalone autograd; "
                "use the explicit FP32 composite backward"
            )
        return _C.mhc_pre_h_aggregate(residual, pre)

    def backward_fp32(
        self,
        grad_output: torch.Tensor,
        residual: torch.Tensor,
        pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(_C.mhc_pre_h_aggregate_backward(grad_output, residual, pre))
