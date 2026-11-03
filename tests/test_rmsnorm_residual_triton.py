import pytest
import torch
from torch.version import hip as torch_hip

import rl_engine.kernels.ops.triton.rmsnorm_residual_triton as api
from rl_engine.mhc import fixtures, oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch_hip is not None,
    reason="requires NVIDIA CUDA",
)


def assert_bytes(got, want, name):
    assert got.shape == want.shape, f"{name}: shape mismatch"
    assert got.dtype == want.dtype, f"{name}: dtype mismatch"
    got_bytes = got.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    want_bytes = want.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    assert torch.equal(got_bytes, want_bytes), f"{name}: raw bytes differ"


def inputs(t, d):
    generator = torch.Generator().manual_seed(42)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator).to("cuda", dtype)

    return rand(t, d), rand(d, dtype=torch.float32), rand(t, d), rand(t, d)


def compare(x, gamma, dy, d_residual):
    want_y, want_residual, want_saved = oracle.rmsnorm_residual_fwd(x, gamma, 1.0e-6)
    got_y, got_residual, got_saved = api.triton_rmsnorm_residual_fwd(x, gamma, 1.0e-6)
    assert_bytes(got_saved["r"], want_saved["r"], "r")
    assert_bytes(got_y, want_y, "y")
    assert_bytes(got_residual, want_residual, "residual")
    assert got_residual.data_ptr() != x.data_ptr(), "residual must not alias x"

    want_dx, want_dgamma = oracle.rmsnorm_residual_bwd(
        dy, d_residual, gamma, want_saved
    )
    got_dx, got_dgamma = api.triton_rmsnorm_residual_bwd(
        dy, d_residual, x, gamma, got_saved
    )
    assert_bytes(got_dx, want_dx, "dx")
    assert_bytes(got_dgamma, want_dgamma, "dgamma")
    for value in (got_y, got_saved["r"], got_dx, got_dgamma):
        assert torch.isfinite(value).all()
    return got_y, got_residual, got_saved["r"], got_dx, got_dgamma


@pytest.mark.parametrize("t", [1, 7, 16])
@pytest.mark.parametrize("d", [128, 4096])
def test_exact(t, d):
    compare(*inputs(t, d))


def test_edges():
    x = fixtures.make_rms_edge_inputs().cuda()
    _, gamma, dy, d_residual = inputs(*x.shape)
    compare(x, gamma, dy, d_residual)
    compare(x, gamma, dy, torch.zeros_like(d_residual))


def strided_copy(value):
    storage = torch.empty(
        (*value.shape[:-1], value.shape[-1] * 2),
        dtype=value.dtype,
        device=value.device,
    )
    view = storage[..., ::2]
    view.copy_(value)
    assert not view.is_contiguous()
    return view


@pytest.mark.parametrize("d", [128, 4096])
def test_layout_and_batch(d):
    x, gamma, dy, d_residual = inputs(7, d)
    full = compare(x, gamma, dy, d_residual)
    repeat = compare(x, gamma, dy, d_residual)
    strided = compare(
        *(strided_copy(value) for value in (x, gamma, dy, d_residual)),
    )
    for index in range(5):
        assert_bytes(repeat[index], full[index], f"repeat[{index}]")
        assert_bytes(strided[index], full[index], f"stride[{index}]")

    for row in (0, 3, 6):
        one = compare(
            x[row : row + 1],
            gamma,
            dy[row : row + 1],
            d_residual[row : row + 1],
        )
        for index in range(4):
            assert_bytes(one[index], full[index][row : row + 1], f"row[{index}]")

    def pad(value):
        zero = torch.zeros_like(value[:1])
        return torch.cat((zero, value, zero), dim=0)

    padded = compare(pad(x), gamma, pad(dy), pad(d_residual))
    for index in range(4):
        assert_bytes(padded[index][1:-1], full[index], f"padding[{index}]")


@pytest.mark.parametrize("branch", ["y", "residual", "both"])
def test_autograd(branch):
    x, gamma, dy, d_residual = inputs(7, 128)
    x.requires_grad_(True)
    gamma.requires_grad_(True)
    candidate = api.RMSNormResidualTritonOp()
    y, residual = candidate.forward(x=x, gamma=gamma, eps=1.0e-6)
    if branch == "y":
        y.backward(dy)
        d_residual = torch.zeros_like(d_residual)
    elif branch == "residual":
        residual.backward(d_residual)
        dy = torch.zeros_like(dy)
    else:
        torch.autograd.backward((y, residual), (dy, d_residual))

    x_ref = x.detach()
    gamma_ref = gamma.detach()
    _, _, saved = oracle.rmsnorm_residual_fwd(x_ref, gamma_ref, 1.0e-6)
    dx, dgamma = oracle.rmsnorm_residual_bwd(dy, d_residual, gamma_ref, saved)
    assert_bytes(x.grad, dx.to(x.dtype), "autograd.dx")
    assert_bytes(gamma.grad, dgamma.to(gamma.dtype), "autograd.dgamma")


def test_invalid_inputs():
    x, gamma, dy, d_residual = inputs(1, 128)
    errors = (ValueError, TypeError, RuntimeError)
    for bad_x in (x.float(), x.cpu(), x[:, :64], x[:0], x.flatten()):
        with pytest.raises(errors):
            api.triton_rmsnorm_residual_fwd(bad_x, gamma)
    for bad_gamma in (gamma.bfloat16(), gamma.half(), gamma.double(), gamma.cpu(), gamma[:-1]):
        with pytest.raises(errors):
            api.triton_rmsnorm_residual_fwd(x, bad_gamma)
    with pytest.raises(errors):
        api.triton_rmsnorm_residual_fwd(x, gamma, 1.0e-5)

    _, _, saved = api.triton_rmsnorm_residual_fwd(x, gamma)
    with pytest.raises(errors):
        api.triton_rmsnorm_residual_bwd(dy.float(), d_residual, x, gamma, saved)
    with pytest.raises(errors):
        api.triton_rmsnorm_residual_bwd(dy, d_residual, x, gamma, {})
    for bad_r in (saved["r"].bfloat16(), saved["r"].cpu(), saved["r"][:0]):
        with pytest.raises(errors):
            api.triton_rmsnorm_residual_bwd(
                dy,
                d_residual,
                x,
                gamma,
                {"r": bad_r, "d": x.shape[1]},
            )
