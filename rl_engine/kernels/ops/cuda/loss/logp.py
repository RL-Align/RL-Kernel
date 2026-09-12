# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


def _grad_requested(logits: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and logits.requires_grad


_BACKWARD_EXT_SYMBOLS = (
    "fused_logp_forward_with_lse",
    "fused_logp_forward_indexed_with_lse",
    "fused_logp_forward_online_with_lse",
    "fused_logp_forward_online_indexed_with_lse",
    "fused_logp_backward",
    "fused_logp_backward_indexed",
)

_SM90_BACKWARD_EXT_SYMBOLS = (
    "fused_logp_sm90_with_lse",
    "fused_logp_backward",
)


class _FusedLogpSM90Function(torch.autograd.Function):
    """SM90 TMA fused forward with the generic CUDA backward."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels_i32: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        logp, row_max, log_sum = _C.fused_logp_sm90_with_lse(logits, labels_i32)
        ctx.save_for_backward(logits, token_ids, row_max, log_sum)
        return logp

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        logits, token_ids, row_max, log_sum = ctx.saved_tensors
        grad_logits = _C.fused_logp_backward(grad_out, logits, token_ids, row_max, log_sum)
        return grad_logits, None, None


class FusedLogpSM90Op:
    """TMA-accelerated fused logp for supported SM90+ GPUs."""

    is_fused_logp = True

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp_sm90"):
            raise RuntimeError(
                "The TMA fused logp kernel is unavailable in this build or on this GPU. "
                "Rebuild on a supported GPU with "
                "'KERNEL_ALIGN_FORCE_SM90=1 pip install -e .'"
            )
        self.op = _C.fused_logp_sm90
        self._fallback = None
        logger.info("Successfully linked to precompiled _C.fused_logp_sm90 kernel.")

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.apply(logits, labels)

    def _fallback_op(self):
        if self._fallback is None:
            self._fallback = FusedLogpGenericOp()
        return self._fallback

    def _can_use_sm90(self, logits: torch.Tensor) -> bool:
        return (
            os.getenv("RL_KERNEL_ENABLE_EXPERIMENTAL_SM90_LOGP") == "1"
            and logits.dim() == 2
            and logits.dtype == torch.bfloat16
            and logits.is_contiguous()
            and logits.size(1) % 8 == 0
        )

    def _check_backward_symbols(self) -> None:
        missing = [name for name in _SM90_BACKWARD_EXT_SYMBOLS if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                "Differentiable SM90 fused logp requires kernels missing from the compiled "
                f"extension: {', '.join(missing)}. "
                "Rebuild with 'KERNEL_ALIGN_FORCE_SM90=1 pip install -e .'"
            )

    def apply(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self._can_use_sm90(logits):
            return self._fallback_op().apply(logits, labels)
        labels_fused = labels.to(device=logits.device, dtype=torch.int32).contiguous()
        if _grad_requested(logits):
            self._check_backward_symbols()
            token_ids = labels.to(device=logits.device, dtype=torch.long).contiguous()
            return _FusedLogpSM90Function.apply(logits, labels_fused, token_ids)
        return self.op(logits, labels_fused)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self._fallback_op().apply_fp32(logits, token_ids)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self._fallback_op().out(logits, token_ids, output)

    def indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._fallback_op().indexed_out(logits, token_ids, row_indices, output)

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        return self._fallback_op().indexed_fp32(logits, token_ids, row_indices)

    def online_out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self._fallback_op().online_out(logits, token_ids, output)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self._fallback_op().online_fp32(logits, token_ids)

    def online_indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self._fallback_op().online_indexed_out(logits, token_ids, row_indices, output)

    def online_indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        return self._fallback_op().online_indexed_fp32(logits, token_ids, row_indices)


class _FusedLogpFunction(torch.autograd.Function):
    """Fused logp variants with CUDA backward from saved softmax statistics."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor | None = None,
        online: bool = False,
    ) -> torch.Tensor:
        # Make logits contiguous once so backward reuses the saved buffer instead
        # of materializing another [T, V] copy.
        logits = logits.contiguous()
        if row_indices is None:
            forward = (
                _C.fused_logp_forward_online_with_lse if online else _C.fused_logp_forward_with_lse
            )
            logp, row_max, log_sum = forward(logits, token_ids)
            ctx.save_for_backward(logits, token_ids, row_max, log_sum)
        else:
            forward = (
                _C.fused_logp_forward_online_indexed_with_lse
                if online
                else _C.fused_logp_forward_indexed_with_lse
            )
            logp, row_max, log_sum = forward(logits, token_ids, row_indices)
            ctx.save_for_backward(logits, token_ids, row_max, log_sum, row_indices)
        return logp

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if len(ctx.saved_tensors) == 5:
            logits, token_ids, row_max, log_sum, row_indices = ctx.saved_tensors
            grad_logits = _C.fused_logp_backward_indexed(
                grad_out, logits, token_ids, row_max, log_sum, row_indices
            )
        else:
            logits, token_ids, row_max, log_sum = ctx.saved_tensors
            grad_logits = _C.fused_logp_backward(grad_out, logits, token_ids, row_max, log_sum)
        return grad_logits, None, None, None


class FusedLogpGenericOp:
    """Generic custom CUDA fallback Fused LogP with RL variants."""

    is_fused_logp = True

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp"):
            raise RuntimeError("Base custom kernel 'fused_logp' is unavailable.")
        self._backend = _C
        self.op = self._backend.fused_logp
        logger.info("Successfully linked to precompiled _C.fused_logp fallback kernel.")

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply(logits, token_ids)

    def _prepare_inputs(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        orig_shape = logits.shape[:-1]
        logits_2d = logits.reshape(-1, logits.size(-1))
        token_ids_1d = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        return logits_2d, token_ids_1d, orig_shape

    def _prepare_output(self, output: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        if output.shape != orig_shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match logits leading shape "
                f"{tuple(orig_shape)}"
            )
        return output.view(-1)

    def _prepare_indices(self, row_indices: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return row_indices.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()

    def _needs_grad(self, logits: torch.Tensor) -> bool:
        if not _grad_requested(logits):
            return False
        missing = [name for name in _BACKWARD_EXT_SYMBOLS if not hasattr(_C, name)]
        if missing:
            raise RuntimeError(
                "Differentiable fused logp requires kernels missing from the compiled "
                f"extension: {', '.join(missing)}. "
                "Rebuild the extension with 'pip install -e .'"
            )
        return True

    def _reject_grad(self, logits: torch.Tensor, variant: str) -> None:
        if _grad_requested(logits):
            raise RuntimeError(
                f"{type(self).__name__}.{variant} is forward-only and does not propagate "
                "gradients to logits; call it under torch.no_grad() or detach logits first."
            )

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        if self._needs_grad(logits_2d):
            results = _FusedLogpFunction.apply(logits_2d, token_ids_1d).to(logits_2d.dtype)
        else:
            results = self.op(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        if self._needs_grad(logits_2d):
            results = _FusedLogpFunction.apply(logits_2d, token_ids_1d)
        else:
            results = self._backend.fused_logp_forward_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self._reject_grad(logits, "out")
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    def indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._reject_grad(logits, "indexed_out")
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_indexed_out(
            logits_2d, token_ids_1d, row_indices_1d, output_1d
        )
        return results.view(orig_shape)

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        if self._needs_grad(logits_2d):
            results = _FusedLogpFunction.apply(logits_2d, token_ids_1d, row_indices_1d)
        else:
            results = self._backend.fused_logp_forward_indexed_fp32(
                logits_2d, token_ids_1d, row_indices_1d
            )
        return results.view(orig_shape)

    def online_out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self._reject_grad(logits, "online_out")
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_online_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        if self._needs_grad(logits_2d):
            results = _FusedLogpFunction.apply(logits_2d, token_ids_1d, None, True)
        else:
            results = self._backend.fused_logp_forward_online_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def online_indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._reject_grad(logits, "online_indexed_out")
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_online_indexed_out(
            logits_2d, token_ids_1d, row_indices_1d, output_1d
        )
        return results.view(orig_shape)

    def online_indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        if self._needs_grad(logits_2d):
            results = _FusedLogpFunction.apply(logits_2d, token_ids_1d, row_indices_1d, True)
        else:
            results = self._backend.fused_logp_forward_online_indexed_fp32(
                logits_2d, token_ids_1d, row_indices_1d
            )
        return results.view(orig_shape)


class DeterministicLogpCUDAOp(FusedLogpGenericOp):
    """Batch-invariant deterministic CUDA LogP.

    The default call path returns float32 output so tests and downstream KL
    code observe the fixed reduction result before any lower-precision cast.
    """

    is_batch_invariant = True

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "deterministic_logp_forward_fp32"):
            raise RuntimeError("Deterministic CUDA logp kernel is unavailable.")
        self._backend = _C
        self.op = self._backend.deterministic_logp_forward_fp32
        logger.info("Successfully linked to precompiled _C.deterministic_logp kernel.")

    def __call__(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_fp32(logits, token_ids)

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_fp32(logits, token_ids)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.deterministic_logp_forward_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.deterministic_logp_forward_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    def indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.deterministic_logp_forward_indexed_out(
            logits_2d, token_ids_1d, row_indices_1d, output_1d
        )
        return results.view(orig_shape)

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        results = self._backend.deterministic_logp_forward_indexed_fp32(
            logits_2d, token_ids_1d, row_indices_1d
        )
        return results.view(orig_shape)

    def online_out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        return self.out(logits, token_ids, output)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.apply_fp32(logits, token_ids)

    def online_indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self.indexed_out(logits, token_ids, row_indices, output)

    def online_indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.indexed_fp32(logits, token_ids, row_indices)
