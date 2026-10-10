# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
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
        if logits.device.type == "musa" and hasattr(ctx.backend, "fused_logp_backward"):
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


def _require_deterministic_symbols(backend, *names):
    if any(not hasattr(backend, name) for name in names):
        raise RuntimeError("Rebuild rl_engine._C to enable deterministic LogP autograd/FP64")


class _RejectDeterministicLogpThirdOrder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dx, dg, logits, upstream, h):
        return dx, dg

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("Deterministic LogP third-order gradients are unsupported")


class _DeterministicLogpBackward(torch.autograd.Function):
    """Native VJP with an explicit native derivative, up to order two."""

    @staticmethod
    def forward(ctx, logits, labels, lse, upstream, backend):
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        coef = upstream.reshape(-1).to(dtype).contiguous()
        input_2d = logits.reshape(-1, logits.size(-1))
        active = torch.isfinite(lse) & (labels >= 0) & (labels < logits.size(-1))
        coef = coef.masked_fill(~active, 0)
        targets = labels.masked_fill(~active, -1)
        with torch.cuda.device(logits.device):
            if dtype == torch.float64:
                grad = backend.deterministic_logp_backward_fp64(input_2d, lse, coef, targets)
            else:
                grad = backend.deterministic_logp_backward(
                    input_2d,
                    lse,
                    coef,
                    torch.zeros_like(coef),
                    targets,
                    0,
                    logits.size(-1),
                    False,
                )
        ctx.save_for_backward(logits, targets, lse, coef, upstream)
        ctx.upstream_shape = upstream.shape
        ctx.upstream_dtype = upstream.dtype
        ctx.backend = backend
        return grad.reshape(logits.shape)

    @staticmethod
    def backward(ctx, h):
        dx, _, _, dg, _ = _DeterministicLogpBackward._native_backward(ctx, h)
        if torch.is_grad_enabled():
            # A constant h bypasses once_differentiable's error edge. Attach an
            # explicit edge to all differentiable inputs so compound losses
            # cannot silently return a partial third derivative.
            logits, _, _, _, upstream = ctx.saved_tensors
            dx, dg = _RejectDeterministicLogpThirdOrder.apply(dx, dg, logits, upstream, h)
        return dx, None, None, dg, None

    @staticmethod
    @once_differentiable
    def _native_backward(ctx, h):
        logits, labels, lse, coef, _ = ctx.saved_tensors
        _require_deterministic_symbols(ctx.backend, "deterministic_logp_double_backward")
        dx, dg = ctx.backend.deterministic_logp_double_backward(
            logits.reshape(-1, logits.size(-1)),
            lse,
            coef,
            labels,
            h.reshape(-1, logits.size(-1)).contiguous(),
        )
        return (
            dx.reshape(logits.shape),
            None,
            None,
            dg.to(ctx.upstream_dtype).reshape(ctx.upstream_shape),
            None,
        )


class _DeterministicLogpAutograd(torch.autograd.Function):
    """Save the original input so native double backward reconnects to its graph."""

    @staticmethod
    def forward(ctx, logits, token_ids, row_indices, backend, fp64):
        if fp64:
            if logits.dtype != torch.float64:
                raise TypeError("FP64 LogP requires float64 logits")
            _require_deterministic_symbols(
                backend,
                "deterministic_logp_forward_fp64",
                "deterministic_logp_backward_fp64",
            )
        else:
            if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError(
                    "FP32 LogP autograd supports float32, float16, and bfloat16; "
                    "use apply_fp64 or indexed_fp64 for float64 gradients"
                )
            _require_deterministic_symbols(
                backend,
                "deterministic_logp_forward_with_lse",
                "deterministic_logp_backward",
            )
        logits_2d = logits.reshape(-1, logits.size(-1))
        labels = token_ids.reshape(-1).to(device=logits.device, dtype=torch.long).contiguous()
        with torch.cuda.device(logits.device):
            if fp64:
                output, lse = backend.deterministic_logp_forward_fp64(
                    logits_2d, labels, row_indices, True
                )
            else:
                output, lse = backend.deterministic_logp_forward_with_lse(
                    logits_2d, labels, row_indices
                )
        ctx.save_for_backward(logits, labels, lse)
        ctx.backend = backend
        return output.reshape(logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, lse = ctx.saved_tensors
        grad = _DeterministicLogpBackward.apply(logits, labels, lse, grad_output, ctx.backend)
        return grad, None, None, None, None


class DeterministicLogpCUDAOp(FusedLogpGenericOp):
    """Batch-invariant deterministic CUDA LogP.

    The default call path returns float32 output so tests and downstream KL
    code observe the fixed reduction result before any lower-precision cast.
    Dense, indexed and out calls support native first- and second-order gradients.
    Explicit FP64 methods use double-precision arithmetic throughout.
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
        if torch.is_grad_enabled() and logits.requires_grad:
            return _DeterministicLogpAutograd.apply(logits, token_ids, None, self._backend, False)
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.deterministic_logp_forward_fp32(logits_2d, token_ids_1d)
        return results.view(orig_shape)

    def _validate_out(self, logits, token_ids, row_indices, output, fp64=False):
        self._prepare_output(output, logits.shape[:-1])
        if output.device != logits.device or not output.is_contiguous():
            raise RuntimeError("output must be contiguous and on the logits device")
        allowed = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        if output.dtype not in allowed or (fp64 and output.dtype != torch.float64):
            raise RuntimeError("output dtype must be floating point (float64 for FP64 interfaces)")
        if torch.is_grad_enabled():
            if output.requires_grad:
                raise RuntimeError(
                    "out buffer must not require gradients; detach it after the previous backward"
                )
            for tensor in (logits, token_ids, row_indices):
                if tensor is not None and torch._C._overlaps(output, tensor):
                    raise RuntimeError("out buffer must not share storage with inputs")

    def _copy_out(self, logits, token_ids, row_indices, output, fp64=False):
        # copy_ supplies the standard autograd mutation/version tracking. In
        # indexed mode unselected buffer values are constants, not zero-filled.
        if row_indices is None:
            result = (
                self.apply_fp64(logits, token_ids) if fp64 else self.apply_fp32(logits, token_ids)
            )
        else:
            result = (
                self.indexed_fp64(logits, token_ids, row_indices)
                if fp64
                else self.indexed_fp32(logits, token_ids, row_indices)
            )
            rows = self._prepare_indices(row_indices, logits)
            rows = rows[(rows >= 0) & (rows < result.numel())].unique()
            mask = torch.zeros(result.numel(), dtype=torch.bool, device=logits.device)
            mask[rows] = True
            result = torch.where(mask.reshape(result.shape), result, output)
        output.copy_(result)
        return output

    def out(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self._validate_out(logits, token_ids, None, output)
        if torch.is_grad_enabled() and logits.requires_grad:
            return self._copy_out(logits, token_ids, None, output)
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.deterministic_logp_forward_out(
            logits_2d, token_ids_1d, self._prepare_output(output, orig_shape)
        )
        return results.view(orig_shape)

    def indexed_out(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_out(logits, token_ids, row_indices, output)
        if torch.is_grad_enabled() and logits.requires_grad:
            return self._copy_out(logits, token_ids, row_indices, output)
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        results = self._backend.deterministic_logp_forward_indexed_out(
            logits_2d,
            token_ids_1d,
            self._prepare_indices(row_indices, logits),
            self._prepare_output(output, orig_shape),
        )
        return results.view(orig_shape)

    def apply_fp64(self, logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self._fp64(logits, token_ids, None)

    def indexed_fp64(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        return self._fp64(logits, token_ids, self._prepare_indices(row_indices, logits))

    def _fp64(self, logits, token_ids, row_indices):
        if logits.dtype != torch.float64:
            raise TypeError("FP64 LogP requires float64 logits")
        _require_deterministic_symbols(self._backend, "deterministic_logp_forward_fp64")
        if torch.is_grad_enabled() and logits.requires_grad:
            return _DeterministicLogpAutograd.apply(
                logits, token_ids, row_indices, self._backend, True
            )
        x, labels, shape = self._prepare_inputs(logits, token_ids)
        result, _ = self._backend.deterministic_logp_forward_fp64(x, labels, row_indices, False)
        return result.reshape(shape)

    def out_fp64(
        self, logits: torch.Tensor, token_ids: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self._validate_out(logits, token_ids, None, output, fp64=True)
        return self._copy_out(logits, token_ids, None, output, fp64=True)

    def indexed_out_fp64(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        row_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_out(logits, token_ids, row_indices, output, fp64=True)
        return self._copy_out(logits, token_ids, row_indices, output, fp64=True)

    online_fp64 = apply_fp64
    online_indexed_fp64 = indexed_fp64
    online_out_fp64 = out_fp64
    online_indexed_out_fp64 = indexed_out_fp64

    def indexed_fp32(
        self, logits: torch.Tensor, token_ids: torch.Tensor, row_indices: torch.Tensor
    ) -> torch.Tensor:
        logits_2d, token_ids_1d, orig_shape = self._prepare_inputs(logits, token_ids)
        row_indices_1d = self._prepare_indices(row_indices, logits)
        if torch.is_grad_enabled() and logits.requires_grad:
            return _DeterministicLogpAutograd.apply(
                logits, token_ids, row_indices_1d, self._backend, False
            )
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
