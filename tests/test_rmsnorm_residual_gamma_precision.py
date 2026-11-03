"""Guard FP32 gain storage and unchanged explicit-gradient precision."""

import pytest
import torch

from rl_engine.mhc import oracle


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("d", [128, 4096])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_fp32_gamma_is_not_rounded_to_bf16(backend, d):
    if backend == "cuda":
        from rl_engine.kernels.ops.cuda.norm.rmsnorm_residual import (
            cuda_rmsnorm_residual_fwd as fwd,
            cuda_rmsnorm_residual_bwd as bwd,
            rmsnorm_residual,
        )
    else:
        from rl_engine.kernels.ops.triton.rmsnorm_residual_triton import (
            triton_rmsnorm_residual_fwd as fwd,
            triton_rmsnorm_residual_bwd as bwd,
            rmsnorm_residual,
        )

    generator = torch.Generator().manual_seed(53)
    x = torch.randn(7, d, generator=generator).to("cuda", torch.bfloat16)
    gamma = torch.full((d,), 1.0 + 2.0**-10, device="cuda", dtype=torch.float32)
    rounded_gamma = gamma.bfloat16().float()
    dy = torch.zeros_like(x)
    dy[:, 0] = 1
    dr = torch.zeros_like(x)
    want_y, want_residual, want_saved = oracle.rmsnorm_residual_fwd(x, gamma, 1e-6)
    want_dx, want_dg = oracle.rmsnorm_residual_bwd(dy, dr, gamma, want_saved)
    rounded_y, _, rounded_saved = oracle.rmsnorm_residual_fwd(x, rounded_gamma, 1e-6)
    rounded_dx, _ = oracle.rmsnorm_residual_bwd(dy, dr, rounded_gamma, rounded_saved)
    assert not torch.equal(want_y, rounded_y), "fixture must expose forward gain rounding"
    assert not torch.equal(want_dx, rounded_dx), "fixture must expose backward gain rounding"
    assert not torch.equal(want_dx, want_dx.bfloat16().float())

    y, residual, saved = fwd(x, gamma)
    dx, dg = bwd(dy, dr, x, gamma, saved)
    for got, want in (
        (y, want_y), (residual, want_residual),
        (saved["r"], want_saved["r"]), (dx, want_dx), (dg, want_dg),
    ):
        assert got.dtype == want.dtype
        assert torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
    assert dx.dtype == dg.dtype == torch.float32
    assert y.dtype == residual.dtype == torch.bfloat16
    for bad_gamma in (gamma.bfloat16(), gamma.half(), gamma.double()):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            bwd(dy, dr, x, bad_gamma, saved)

    x.requires_grad_(True)
    gamma.requires_grad_(True)
    y, residual = rmsnorm_residual(x, gamma)
    torch.autograd.backward((y, residual), (dy, dr))
    assert x.grad.dtype == torch.bfloat16
    assert gamma.grad.dtype == torch.float32
    assert torch.equal(x.grad, want_dx.bfloat16())
    assert torch.equal(gamma.grad, want_dg)
