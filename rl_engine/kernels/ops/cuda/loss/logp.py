# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger


class _FusedLogpAutograd(torch.autograd.Function):
    """Autograd bridge for the generic CUDA selected-logprob forward.

    The VJP is row-local: ``dlogits = grad * (one_hot(target) - softmax)``.
    It runs in FP32 on CUDA and casts only the final input VJP to the BF16
    execution dtype.  There is no cross-token reduction or borrowed Triton
    candidate, so Batch/Chunk layout cannot change the result.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, token_ids: torch.Tensor, backend):
        logits_2d = logits.reshape(-1, logits.size(-1)).contiguous()
        labels = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        output = backend.fused_logp(logits_2d, labels)
        ctx.save_for_backward(logits_2d, labels)
        ctx.backend = backend
        ctx.input_shape = tuple(logits.shape)
        ctx.input_dtype = logits.dtype
        return output.reshape(logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, labels = ctx.saved_tensors
        if (
            logits.device.type == "musa"
            and hasattr(ctx.backend, "fused_logp_backward")
        ):
            grad = ctx.backend.fused_logp_backward(
                logits,
                labels,
                grad_output.reshape(-1).contiguous(),
            )
            return grad.reshape(ctx.input_shape), None, None

        probs = torch.softmax(logits.float(), dim=-1)
        rows = torch.arange(logits.size(0), device=logits.device)
        probs[rows, labels] -= 1.0
        grad = -grad_output.reshape(-1, 1).float() * probs
        return grad.to(ctx.input_dtype).reshape(ctx.input_shape), None, None


class FusedLogpSM90Op:
    """TMA-accelerated Fused LogP for SM90+ cards."""

    is_fused_logp = True

    def __init__(self):
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_logp_sm90"):
            raise RuntimeError(
                "TMA Fused LogP kernel is not compiled or unsupported on this card architecture. "
                "Please rebuild extension using 'pip install -e .'"
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
        )

    def apply(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self._can_use_sm90(logits):
            return self._fallback_op().apply(logits, labels)
        labels_fused = labels.to(device=logits.device, dtype=torch.int32).contiguous()
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

    def apply(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return _FusedLogpAutograd.apply(logits, token_ids, self._backend)

    def apply_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.fused_logp_forward_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
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
        results = self._backend.fused_logp_forward_indexed_fp32(
            logits_2d, token_ids_1d, row_indices_1d
        )
        return results.view(orig_shape)

    def online_out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        output_1d = self._prepare_output(output, orig_shape)
        results = self._backend.fused_logp_forward_online_out(logits_2d, token_ids_1d, output_1d)
        return results.view(orig_shape)

    def online_fp32(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.fused_logp_forward_online_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def online_indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
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
