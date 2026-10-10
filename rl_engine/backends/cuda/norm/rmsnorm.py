import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32, rmsnorm_dweight_rows_fp32


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
            "Rebuild the extension with csrc/cuda/norm/rmsnorm.cu."
        )


class RMSNormCuda(torch.autograd.Function):
    """
    PyTorch autograd wrapper for CUDA RMSNorm.
    """

    @staticmethod
    def forward(ctx, x, weight, mask=None, eps=1e-6):
        """
        Forward:
          y = x * rsqrt(mean(x^2) + eps) * weight

        Input:
          x:      [T, H], fp16/bf16/fp32 CUDA tensor
          weight: [H],    fp16/bf16/fp32 CUDA tensor
          mask:   [T],    bool CUDA tensor
          eps:    float

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

        y, rstd = _C.rmsnorm_forward(x, weight, float(eps))

        ctx.save_for_backward(x, weight, rstd, mask)
        ctx.eps = eps

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

        dx = _C.rmsnorm_backward_dx(dy, x, weight, rstd)

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
                "+rl_engine.ops.autograd.vjp_fp32.rmsnorm_dweight_rows_fp32"
                "+rl_engine.ops.autograd.vjp_fp32.reduce_rows_fp32"
            ),
            impl="cuda_rmsnorm_dx_declared_fp32_rowfold_dw",
            family="cuda",
        )

        return dx, dw, None, None


def rmsnorm_cuda(x, weight, eps=1e-6, mask=None):
    """
    use:
        y = rmsnorm_cuda(x, weight)
        y = rmsnorm_cuda(x, weight, mask=mask)
    """
    return RMSNormCuda.apply(x, weight, mask, eps)


class RMSNormCudaOp:
    """CUDA RMSNorm wrapper compatible with the shared operator harness."""

    backward_impl = "cuda_rmsnorm_dx_declared_fp32_rowfold_dw"

    def __init__(self) -> None:
        _require_cuda_symbols(
            "CUDA RMSNorm",
            "rmsnorm_forward",
            "rmsnorm_backward_dx",
        )

    def __call__(self, x, weight, *, eps=1e-6):
        return self.forward(x, weight, eps=eps)

    def forward(self, x, weight, *, eps=1e-6):
        hidden = x.shape[-1]
        x_2d = x.contiguous().view(-1, hidden)
        y_2d = rmsnorm_cuda(x_2d, weight.contiguous(), eps=eps)
        return y_2d.view_as(x)

    def parameter_vjp_contributions_fp32(self, *, x, weight, grad_output, eps=1e-6):
        del weight
        x32 = x.float()
        rstd = torch.rsqrt(x32.square().mean(dim=-1) + float(eps))
        rows = grad_output.float() * x32 * rstd.unsqueeze(-1)
        return {"weight": rows}
