import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch RMSNorm reference.

    x:      [T, H], fp16/bf16/fp32 CUDA tensor
    weight: [H],    fp16/bf16/fp32 CUDA tensor
    return: [T, H]

    Accumulation is done in FP32 for numerical stability.
    """
    x_fp32 = x.float()
    w_fp32 = weight.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    y = x_fp32 * rstd * w_fp32
    return y.to(x.dtype)


class RMSNormRef(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps: float = 1e-6):
        x_fp32 = x.float()
        w_fp32 = weight.float()

        #   rstd[t] = 1 / sqrt(mean_h(x[t, h]^2) + eps)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        y = x_fp32 * rstd * w_fp32

        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        x_fp32 = x.float()
        w_fp32 = weight.float()
        go_fp32 = grad_out.float()
        rstd_fp32 = rstd.float()
        H = x.shape[-1]

        gw = go_fp32 * w_fp32
        dot = (gw * x_fp32).sum(dim=-1, keepdim=True)
        dx = rstd_fp32 * gw - x_fp32 * (rstd_fp32**3) * dot / H

        # dw_i = sum_t(g_ti * x_ti * r_t)
        dw = (go_fp32 * x_fp32 * rstd_fp32).sum(dim=0)
        return dx.to(x.dtype), dw.to(weight.dtype), None


def rmsnorm_ref_custom(x, weight, eps: float = 1e-6):
    return RMSNormRef.apply(x, weight, eps)
