import pytest
import torch

import rl_engine.kernels.ops.cuda.norm.rmsnorm_residual as api
from rl_engine.mhc import fixtures, oracle

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


def assert_bytes(got, want, name):
    assert got.shape == want.shape, f"{name}: shape mismatch"
    assert got.dtype == want.dtype, f"{name}: dtype mismatch"
    a = got.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    b = want.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    assert torch.equal(a, b), f"{name}: raw bytes differ"


def inputs(t, d):
    gen = torch.Generator().manual_seed(42)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen).to("cuda", dtype)

    return rand(t, d), rand(d, dtype=torch.float32), rand(t, d), rand(t, d)


def compare(x, gamma, dy, dr):
    wy, wr, ws = oracle.rmsnorm_residual_fwd(x, gamma, 1e-6)
    gy, gr, gs = api.cuda_rmsnorm_residual_fwd(x, gamma, 1e-6)
    assert_bytes(gs["r"], ws["r"], "r")
    assert_bytes(gy, wy, "y")
    assert_bytes(gr, wr, "residual")
    assert gr.data_ptr() != x.data_ptr(), "residual must not alias x"

    wdx, wdg = oracle.rmsnorm_residual_bwd(dy, dr, gamma, ws)
    gdx, gdg = api.cuda_rmsnorm_residual_bwd(dy, dr, x, gamma, gs)
    assert_bytes(gdx, wdx, "dx")
    assert_bytes(gdg, wdg, "dgamma")
    for value in (gy, gs["r"], gdx, gdg):
        assert torch.isfinite(value).all()
    return gy, gr, gs["r"], gdx, gdg


@pytest.mark.parametrize("t", [1, 7, 16])
@pytest.mark.parametrize("d", [128, 4096])
def test_exact(t, d):
    compare(*inputs(t, d))


def test_edges():
    x = fixtures.make_rms_edge_inputs().cuda()
    _, gamma, dy, dr = inputs(*x.shape)
    compare(x, gamma, dy, dr)
    compare(x, gamma, dy, torch.zeros_like(dr))


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
    x, gamma, dy, dr = inputs(7, d)
    full = compare(x, gamma, dy, dr)
    repeat = compare(x, gamma, dy, dr)
    strided = compare(*(strided_copy(v) for v in (x, gamma, dy, dr)))
    for i in range(5):
        assert_bytes(repeat[i], full[i], f"repeat[{i}]")
        assert_bytes(strided[i], full[i], f"stride[{i}]")

    for row in (0, 3, 6):
        one = compare(x[row : row + 1], gamma, dy[row : row + 1], dr[row : row + 1])
        for i in range(4):
            assert_bytes(one[i], full[i][row : row + 1], f"row[{i}]")

    def pad(value):
        zero = torch.zeros_like(value[:1])
        return torch.cat((zero, value, zero), dim=0)

    padded = compare(pad(x), gamma, pad(dy), pad(dr))
    for i in range(4):
        assert_bytes(padded[i][1:-1], full[i], f"padding[{i}]")


@pytest.mark.parametrize("branch", ["y", "residual", "both"])
def test_autograd(branch):
    x, gamma, dy, dr = inputs(7, 128)
    x.requires_grad_(True)
    gamma.requires_grad_(True)
    candidate = api.RMSNormResidualCudaOp()
    y, residual = candidate.forward(x=x, gamma=gamma, eps=1e-6)
    if branch == "y":
        y.backward(dy)
        dr = torch.zeros_like(dr)
    elif branch == "residual":
        residual.backward(dr)
        dy = torch.zeros_like(dy)
    else:
        torch.autograd.backward((y, residual), (dy, dr))

    x0, gamma0 = x.detach(), gamma.detach()
    _, _, saved = oracle.rmsnorm_residual_fwd(x0, gamma0, 1e-6)
    dx, dg = oracle.rmsnorm_residual_bwd(dy, dr, gamma0, saved)
    assert_bytes(x.grad, dx.to(x.dtype), "autograd.dx")
    assert_bytes(gamma.grad, dg.to(gamma.dtype), "autograd.dgamma")


def test_invalid_inputs():
    x, gamma, dy, dr = inputs(1, 128)
    errors = (ValueError, TypeError, RuntimeError)
    for bad in (x.float(), x.cpu(), x[:, :64], x[:0], x.flatten()):
        with pytest.raises(errors):
            api.cuda_rmsnorm_residual_fwd(bad, gamma)
    for bad_gamma in (gamma.bfloat16(), gamma.half(), gamma.double(), gamma.cpu(), gamma[:-1]):
        with pytest.raises(errors):
            api.cuda_rmsnorm_residual_fwd(x, bad_gamma)
    with pytest.raises(errors):
        api.cuda_rmsnorm_residual_fwd(x, gamma, 1e-5)

    _, _, saved = api.cuda_rmsnorm_residual_fwd(x, gamma)
    with pytest.raises(errors):
        api.cuda_rmsnorm_residual_bwd(dy.float(), dr, x, gamma, saved)
    with pytest.raises(errors):
        api.cuda_rmsnorm_residual_bwd(dy, dr, x, gamma, {})
    for bad_r in (saved["r"].bfloat16(), saved["r"].cpu(), saved["r"][:0]):
        with pytest.raises(errors):
            api.cuda_rmsnorm_residual_bwd(
                dy, dr, x, gamma, {"r": bad_r, "d": x.shape[1]}
            )


@pytest.mark.parametrize("suffix", ["forward", "backward"])
def test_missing_symbol(monkeypatch, suffix):
    ext = api._extension()
    x, gamma, dy, dr = inputs(1, 128)
    if suffix == "forward":
        monkeypatch.delattr(ext, "mhc_rmsnorm_residual_forward")
        with pytest.raises(RuntimeError, match="mhc_rmsnorm_residual_forward"):
            api.cuda_rmsnorm_residual_fwd(x, gamma)

    else:
        _, _, saved = api.cuda_rmsnorm_residual_fwd(x, gamma)
        monkeypatch.delattr(ext, "mhc_rmsnorm_residual_backward")
        with pytest.raises(RuntimeError, match="mhc_rmsnorm_residual_backward"):
            api.cuda_rmsnorm_residual_bwd(dy, dr, x, gamma, saved)
