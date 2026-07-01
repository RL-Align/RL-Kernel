import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE


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
          dw = CUDA two-pass deterministic kernel
        """
        x, weight, rstd, mask = ctx.saved_tensors
        dy = grad_out.contiguous()

        dx = _C.rmsnorm_backward_dx(dy, x, weight, rstd)

        dw = _C.rmsnorm_backward_dw(dy, x, rstd, mask)

        return dx, dw, None, None


def rmsnorm_cuda(x, weight, eps=1e-6, mask=None):
    """
    use:
        y = rmsnorm_cuda(x, weight)
        y = rmsnorm_cuda(x, weight, mask=mask)
    """
    return RMSNormCuda.apply(x, weight, mask, eps)
