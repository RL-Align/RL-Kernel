# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch

MHC_PRE_HC_MULT = 4
MHC_PRE_HIDDEN_SIZE = 4096
MHC_PRE_H_AGGREGATE_BACKWARD_THREADS = 256
MHC_WARP_SIZE = 32


class NativeMHCPreHAggregateOp:
    """PyTorch reference for the four-stream MHC weighted collapse."""

    op_class = "reduction"

    def __call__(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.forward(residual, pre)

    def forward(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self._compute(residual, pre).to(residual.dtype)

    def forward_fp32(self, residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self._compute(residual, pre)

    def backward_fp32(
        self,
        grad_output: torch.Tensor,
        residual: torch.Tensor,
        pre: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit FP32 backward oracle matching the CUDA reduction tree."""

        self._validate_inputs(residual, pre)
        if grad_output.shape != (residual.shape[0], MHC_PRE_HIDDEN_SIZE):
            raise ValueError("grad_output must have shape [num_tokens, 4096]")
        if grad_output.dtype is not torch.bfloat16:
            raise TypeError("grad_output must be bfloat16")
        if grad_output.device != residual.device:
            raise RuntimeError("grad_output, residual, and pre must be on the same device")
        if not grad_output.is_contiguous():
            raise ValueError("grad_output, residual, and pre must be contiguous")

        grad_output_fp32 = grad_output.float()
        grad_residual = grad_output_fp32[:, None, :] * pre[:, :, None]

        products = grad_output_fp32[:, None, :] * residual.float()
        values_per_thread = MHC_PRE_HIDDEN_SIZE // MHC_PRE_H_AGGREGATE_BACKWARD_THREADS
        thread_products = products.reshape(
            residual.shape[0],
            MHC_PRE_HC_MULT,
            values_per_thread,
            MHC_PRE_H_AGGREGATE_BACKWARD_THREADS,
        )
        thread_sums = torch.zeros_like(thread_products[:, :, 0])
        for index in range(values_per_thread):
            thread_sums = thread_sums + thread_products[:, :, index]

        num_warps = MHC_PRE_H_AGGREGATE_BACKWARD_THREADS // MHC_WARP_SIZE
        warp_lanes = thread_sums.reshape(
            residual.shape[0], MHC_PRE_HC_MULT, num_warps, MHC_WARP_SIZE
        )
        warp_sums = self._warp_lane_zero(warp_lanes)
        zero_lanes = torch.zeros(
            (*warp_sums.shape[:-1], MHC_WARP_SIZE - num_warps),
            dtype=torch.float32,
            device=residual.device,
        )
        block_lanes = torch.cat((warp_sums, zero_lanes), dim=-1)
        grad_pre = self._warp_lane_zero(block_lanes)
        return grad_residual, grad_pre

    @staticmethod
    def _compute(residual: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        NativeMHCPreHAggregateOp._validate_inputs(residual, pre)

        residual_fp32 = residual.float()
        product_0 = pre[:, 0, None] * residual_fp32[:, 0]
        product_1 = pre[:, 1, None] * residual_fp32[:, 1]
        product_2 = pre[:, 2, None] * residual_fp32[:, 2]
        product_3 = pre[:, 3, None] * residual_fp32[:, 3]
        left = product_0 + product_1
        right = product_2 + product_3
        return left + right

    @staticmethod
    def _validate_inputs(residual: torch.Tensor, pre: torch.Tensor) -> None:
        if residual.dim() != 3 or residual.shape[1:] != (
            MHC_PRE_HC_MULT,
            MHC_PRE_HIDDEN_SIZE,
        ):
            raise ValueError("residual must have shape [num_tokens, 4, 4096]")
        if pre.shape != (residual.shape[0], MHC_PRE_HC_MULT):
            raise ValueError("pre must have shape [num_tokens, 4]")
        if residual.dtype is not torch.bfloat16:
            raise TypeError("residual must be bfloat16")
        if pre.dtype is not torch.float32:
            raise TypeError("pre must be float32")
        if residual.device != pre.device:
            raise RuntimeError("residual and pre must be on the same device")
        if not residual.is_contiguous() or not pre.is_contiguous():
            raise ValueError("residual and pre must be contiguous")

    @staticmethod
    def _warp_lane_zero(values: torch.Tensor) -> torch.Tensor:
        """Return lane 0 after the CUDA offset=16,8,4,2,1 shuffle tree."""

        if values.shape[-1] != MHC_WARP_SIZE:
            raise ValueError("fixed warp reduction requires exactly 32 lanes")
        reduced = values
        offset = MHC_WARP_SIZE // 2
        while offset:
            reduced = reduced[..., :offset] + reduced[..., offset : 2 * offset]
            offset //= 2
        return reduced.squeeze(-1)
