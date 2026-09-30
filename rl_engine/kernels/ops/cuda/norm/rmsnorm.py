import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32, rmsnorm_dweight_rows_fp32


def _require_cuda_symbols(what: str, *names: str) -> None:
    """Raise when the compiled kernels backing ``what`` are missing.

    The registry treats a backend whose construction raises as unavailable and
    falls through to the next candidate, so calling this from ``__init__`` is
    what lets a CUDA-first priority list degrade to the PyTorch reference on a
    build without the extension. Mirrors ``_require_cuda_activation`` in the
    activation ops.
    """
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError(f"{what} requires the compiled rl_engine._C extension.")
    missing = [name for name in names if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            f"{what} symbols ({', '.join(missing)}) are not compiled into _C. "
            "Rebuild the extension with csrc/cuda/rmsnorm.cu."
        )


class RMSNormCuda(torch.autograd.Function):
    """
    PyTorch autograd wrapper for CUDA RMSNorm.
    """

    @staticmethod
    def forward(ctx, x, weight, mask=None, eps=1e-6, weight_offset=0.0):
        """
        Forward:
          y = x * rsqrt(mean(x^2) + eps) * (weight_offset + weight)

        Input:
          x:      [T, H], fp16/bf16/fp32 CUDA tensor
          weight: [H],    fp16/bf16/fp32 CUDA tensor
          mask:   [T],    bool CUDA tensor
          eps:    float
          weight_offset: float, added to weight in fp32 inside the kernel.
                  1.0 selects the zero-centred (1 + w) convention.

        Output:
          y: [T, H]
        """
        assert x.is_cuda, "x must be CUDA tensor"
        assert weight.is_cuda, "weight must be CUDA tensor"
        assert x.is_contiguous(), "x must be contiguous"
        assert weight.is_contiguous(), "weight must be contiguous"
        assert x.dim() == 2, "x must be [T, H]"
        assert weight.dim() == 1, "weight must be [H]"
        assert x.shape[1] == weight.shape[0], "hidden size mismatch"
        assert _EXT_AVAILABLE and hasattr(
            _C, "rmsnorm_forward"
        ), "RMSNorm CUDA extension is unavailable. Please rebuild with rmsnorm.cu."

        if mask is None:
            mask = torch.ones((x.shape[0],), device=x.device, dtype=torch.bool)
        else:
            assert mask.is_cuda, "mask must be CUDA tensor"
            assert mask.is_contiguous(), "mask must be contiguous"
            assert mask.dtype == torch.bool, "mask must be bool"
            assert mask.dim() == 1, "mask must be [T]"
            assert mask.shape[0] == x.shape[0], "mask length mismatch"

        y, rstd = _C.rmsnorm_forward(x, weight, float(eps), float(weight_offset))

        ctx.save_for_backward(x, weight, rstd, mask)
        ctx.eps = eps
        ctx.weight_offset = float(weight_offset)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward:
          dx = CUDA row-wise deterministic kernel
          dw = FP32 row contributions followed by an ascending-row left fold
        """
        x, weight, rstd, mask = ctx.saved_tensors
        dy = grad_out.contiguous()

        dx = _C.rmsnorm_backward_dx(dy, x, weight, rstd, ctx.weight_offset)

        # The shape-independent FP32 left fold preserves the C2 Batch/Chunk
        # reduction order while the CUDA reducer executes it in one launch.
        rows = rmsnorm_dweight_rows_fp32(x, dy, rstd=rstd)
        # Multiplication is part of the pre-existing mask contract, including
        # IEEE propagation for non-finite inactive contributions.
        rows = rows * mask.to(dtype=rows.dtype).unsqueeze(-1)
        dw = reduce_rows_fp32(rows).to(weight.dtype)
        record_backward(
            "rms_norm",
            kernel_id=(
                "rl_engine._C.rmsnorm_backward_dx"
                "+rl_engine.kernels.ops.vjp_fp32.rmsnorm_dweight_rows_fp32"
                "+rl_engine.kernels.ops.vjp_fp32.reduce_rows_fp32"
            ),
            impl="cuda_rmsnorm_dx_declared_fp32_rowfold_dw",
            family="cuda",
        )

        # dw is unchanged by the offset: d/dw (offset + w) == d/dw w.
        return dx, dw, None, None, None


def rmsnorm_cuda(x, weight, eps=1e-6, mask=None, weight_offset=0.0):
    """
    use:
        y = rmsnorm_cuda(x, weight)
        y = rmsnorm_cuda(x, weight, mask=mask)
        y = rmsnorm_cuda(x, weight, weight_offset=1.0)   # zero-centred weight
    """
    return RMSNormCuda.apply(x, weight, mask, eps, weight_offset)


class RMSNormCudaOp:
    """CUDA RMSNorm wrapper compatible with the shared operator harness."""

    backward_impl = "cuda_rmsnorm_dx_declared_fp32_rowfold_dw"

    def __init__(self) -> None:
        _require_cuda_symbols(
            "CUDA RMSNorm",
            "rmsnorm_forward",
            "rmsnorm_backward_dx",
        )
    #: Added to the weight in fp32 inside the kernel. Subclasses override it;
    #: 0.0 is the plain convention.
    weight_offset = 0.0

    def __call__(self, x, weight, *, eps=1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps=1e-6):
        hidden = x.shape[-1]
        x_2d = x.contiguous().view(-1, hidden)
        y_2d = rmsnorm_cuda(
            x_2d, weight.contiguous(), eps=eps, weight_offset=self.weight_offset
        )
        return y_2d.view_as(x)

    def parameter_vjp_contributions_fp32(self, *, x, weight, grad_output, eps=1e-6):
        del weight
        x32 = x.float()
        rstd = torch.rsqrt(x32.square().mean(dim=-1) + float(eps))
        rows = grad_output.float() * x32 * rstd.unsqueeze(-1)
        return {"weight": rows}


class Qwen3NextRMSNormCudaOp(RMSNormCudaOp):
    """Zero-centred CUDA RMSNorm: ``y = x * rstd * (1 + weight)``.

    The decoder and final norms of Qwen3-Next (and Gemma) store a zero-centred
    weight. The ``+1`` is applied inside the kernel after the fp32 upcast, so it
    is never rounded through the low-precision weight dtype.
    """

    weight_offset = 1.0
