# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1-2 fp32_gemm_rms: bit-wise alignment tests against the pinned FP32 oracle.

Every comparison in this file is on raw bytes (``tensor_bytes`` equality),
never on tolerances: the P1 contract is byte equality with the oracle
executed on the same device. CPU-scope tests cover the reference backend and
the fixed-K deliverables; CUDA-scope tests cover both kernel backends on the
fixture K and the production K = 16384 / N = 24 geometry.
"""

from __future__ import annotations

import math

import pytest
import torch

from rl_engine.mhc import oracle
from rl_engine.mhc.contract import tensor_bytes
from rl_engine.mhc.fp32_gemm_rms import (
    CudaGemmRMSProvider,
    TritonGemmRMSProvider,
    fixed_k_gemm_bitequal_harness,
    fixed_k_gemm_reference,
    fp32_gemm_rms,
)

EPS = 1e-6
PROD_K = 16384
PROD_N = 24

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def _triton_ready() -> bool:
    if not torch.cuda.is_available():
        return False
    from rl_engine.mhc.fp32_gemm_rms_triton import triton_available

    return triton_available()


def _backends() -> list[str]:
    marks = []
    marks.append(pytest.param("cuda", marks=cuda_only))
    marks.append(
        pytest.param(
            "triton",
            marks=pytest.mark.skipif(
                not _triton_ready(), reason="CUDA GPU + triton inline-asm support required"
            ),
        )
    )
    return marks


def _provider(backend: str):
    return {"cuda": CudaGemmRMSProvider, "triton": TritonGemmRMSProvider}[backend]()


def _make_inputs(tokens: int, k: int, device: str, seed: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(tokens, k, generator=gen, dtype=torch.float32).to(device)
    w = torch.randn(PROD_N, k, generator=gen, dtype=torch.float32).to(device)
    dp = torch.randn(tokens, PROD_N, generator=gen, dtype=torch.float32).to(device)
    dr = torch.randn(tokens, generator=gen, dtype=torch.float32).to(device)
    return x, w, dp, dr


def _assert_bytes_equal(got: torch.Tensor, want: torch.Tensor, label: str) -> None:
    assert got.shape == want.shape, f"{label} shape {got.shape} != {want.shape}"
    assert got.dtype == want.dtype, f"{label} dtype {got.dtype} != {want.dtype}"
    if tensor_bytes(got) != tensor_bytes(want):
        diff = (got.float() - want.float()).abs()
        raise AssertionError(
            f"{label} bytes diverged: max_abs={diff.max().item()} "
            f"mismatches={(got != want).sum().item()}/{want.numel()}"
        )


def _assert_pair_matches_oracle(backend: str, tokens: int, k: int, seed: int = 0) -> None:
    x, w, dp, dr = _make_inputs(tokens, k, "cuda", seed=seed)
    provider = _provider(backend)

    p, r, saved = provider.fp32_gemm_rms_fwd(x, w, EPS)
    gp, gr, gsaved = oracle.fp32_gemm_rms_fwd(x, w, EPS)
    _assert_bytes_equal(p, gp, f"{backend} P[T={tokens},K={k}]")
    _assert_bytes_equal(r, gr, f"{backend} r[T={tokens},K={k}]")
    for key in ("s", "norm", "q"):
        _assert_bytes_equal(saved[key], gsaved[key], f"{backend} saved.{key}")

    dx, dw = provider.fp32_gemm_rms_bwd(dp, dr, x, w, saved)
    gdx, gdw = oracle.fp32_gemm_rms_bwd(dp, dr, x, w, gsaved)
    _assert_bytes_equal(dx, gdx, f"{backend} dX[T={tokens},K={k}]")
    _assert_bytes_equal(dw, gdw, f"{backend} dW[T={tokens},K={k}]")


# ---------------------------------------------------------------------------
# CPU scope: reference backend, controller-RMS semantics, fixed-K deliverables
# ---------------------------------------------------------------------------


def test_reference_backend_matches_oracle_bytes_cpu():
    x, w, dp, dr = _make_inputs(5, 96, "cpu")
    x = x.requires_grad_(True)
    w = w.requires_grad_(True)
    p, r = fp32_gemm_rms(x, w, EPS, backend="reference")
    gp, gr, gsaved = oracle.fp32_gemm_rms_fwd(x.detach(), w.detach(), EPS)
    _assert_bytes_equal(p.detach(), gp, "reference P")
    _assert_bytes_equal(r.detach(), gr, "reference r")

    torch.autograd.backward([p, r], [dp, dr])
    gdx, gdw = oracle.fp32_gemm_rms_bwd(dp, dr, x.detach(), w.detach(), gsaved)
    _assert_bytes_equal(x.grad, gdx, "reference dX")
    _assert_bytes_equal(w.grad, gdw, "reference dW")


def test_controller_rms_is_not_rmsnorm():
    """The task pins r = 1/(sqrt(mean(X^2)) + eps); rsqrt(mean + eps) is the
    other operator's formula and must differ at the byte level."""
    x, w, _, _ = _make_inputs(8, 96, "cpu", seed=3)
    _, r, saved = oracle.fp32_gemm_rms_fwd(x, w, EPS)
    mean = saved["s"] / float(x.shape[1])
    rmsnorm_style = torch.rsqrt(mean + EPS)
    assert tensor_bytes(r) != tensor_bytes(rmsnorm_style)
    # The distinction is the epsilon placement, not a large numeric gap.
    assert torch.allclose(r, rmsnorm_style, rtol=1e-4)
    # And r matches its own formula exactly.
    manual = 1.0 / (torch.sqrt(saved["s"]) / math.sqrt(float(x.shape[1])) + EPS)
    _assert_bytes_equal(r, manual, "controller r formula")


def test_fixed_k_gemm_reference_and_harness_cpu():
    x, w, _, _ = _make_inputs(4, 64, "cpu", seed=1)
    want = oracle.fixed_k_gemm_fwd(x, w)
    _assert_bytes_equal(fixed_k_gemm_reference(x, w), want, "fixed_k reference")

    ok = fixed_k_gemm_bitequal_harness(fixed_k_gemm_reference, x, w)
    assert ok["bitwise_equal"] and ok["mismatch_count"] == 0

    def perturbed(a, b):
        out = fixed_k_gemm_reference(a, b).clone()
        out.view(-1)[0] = torch.nextafter(out.view(-1)[0], torch.tensor(float("inf")))
        return out

    bad = fixed_k_gemm_bitequal_harness(perturbed, x, w)
    assert not bad["bitwise_equal"] and bad["mismatch_count"] == 1


def test_fail_closed_validation():
    x, w, _, _ = _make_inputs(2, 32, "cpu")
    provider = CudaGemmRMSProvider()
    with pytest.raises(RuntimeError, match="fail-closed"):
        provider.fp32_gemm_rms_fwd(x, w, EPS)  # CPU tensors
    with pytest.raises(ValueError):
        provider.fp32_gemm_rms_fwd(x, w[:, :16], EPS)  # K mismatch
    with pytest.raises(ValueError):
        provider.fp32_gemm_rms_fwd(x, w, 0.0)  # non-positive eps
    with pytest.raises(TypeError):
        provider.fp32_gemm_rms_fwd(x.to(torch.bfloat16), w.to(torch.bfloat16), EPS)
    with pytest.raises(ValueError):
        fp32_gemm_rms(x, w, EPS, backend="nope")


# ---------------------------------------------------------------------------
# CUDA scope: byte equality with the oracle on the same device
# ---------------------------------------------------------------------------


# Token counts cross every CUDA dispatch regime: solo-split (T <= 32),
# two-tokens-per-block (T <= 288), eight-tokens-per-block (T > 288), and the
# dual-stream backward (T >= 64).
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("tokens", [1, 7, 16, 70, 300])
def test_fixture_k_matches_oracle(backend, tokens):
    _assert_pair_matches_oracle(backend, tokens, 512, seed=tokens)


@cuda_only
def test_dispatch_regimes_agree_bytewise():
    """The same row must produce identical bytes through every kernel the
    launcher can pick (solo-split vs G2 vs G8), on top of oracle equality."""
    x, w, _, _ = _make_inputs(300, 512, "cuda", seed=23)
    provider = CudaGemmRMSProvider()
    p_g8, r_g8, _ = provider.fp32_gemm_rms_fwd(x, w, EPS)  # T=300 -> G8
    p_g2, r_g2, _ = provider.fp32_gemm_rms_fwd(x[:100].contiguous(), w, EPS)  # G2
    p_solo, r_solo, _ = provider.fp32_gemm_rms_fwd(x[:8].contiguous(), w, EPS)  # solo
    _assert_bytes_equal(p_g2, p_g8[:100], "G2 vs G8 P")
    _assert_bytes_equal(r_g2, r_g8[:100], "G2 vs G8 r")
    _assert_bytes_equal(p_solo, p_g8[:8], "solo vs G8 P")
    _assert_bytes_equal(r_solo, r_g8[:8], "solo vs G8 r")


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("tokens", [2, 33])
def test_production_k_matches_oracle(backend, tokens):
    _assert_pair_matches_oracle(backend, tokens, PROD_K, seed=tokens)


@pytest.mark.parametrize("backend", _backends())
def test_batch_invariance(backend):
    x, w, _, _ = _make_inputs(16, 512, "cuda", seed=9)
    provider = _provider(backend)
    p_full, r_full, _ = provider.fp32_gemm_rms_fwd(x, w, EPS)
    for row in range(16):
        p_one, r_one, _ = provider.fp32_gemm_rms_fwd(x[row : row + 1].contiguous(), w, EPS)
        _assert_bytes_equal(p_one[0], p_full[row], f"{backend} batch-invariance P row{row}")
        _assert_bytes_equal(r_one[0], r_full[row], f"{backend} batch-invariance r row{row}")


@pytest.mark.parametrize("backend", _backends())
def test_autograd_entry_matches_oracle(backend):
    x, w, dp, dr = _make_inputs(6, 512, "cuda", seed=11)
    x = x.requires_grad_(True)
    w = w.requires_grad_(True)
    p, r = fp32_gemm_rms(x, w, EPS, backend=backend)
    gp, gr, gsaved = oracle.fp32_gemm_rms_fwd(x.detach(), w.detach(), EPS)
    _assert_bytes_equal(p.detach(), gp, f"{backend} autograd P")
    _assert_bytes_equal(r.detach(), gr, f"{backend} autograd r")
    torch.autograd.backward([p, r], [dp, dr])
    gdx, gdw = oracle.fp32_gemm_rms_bwd(dp, dr, x.detach(), w.detach(), gsaved)
    _assert_bytes_equal(x.grad, gdx, f"{backend} autograd dX")
    _assert_bytes_equal(w.grad, gdw, f"{backend} autograd dW")


@pytest.mark.parametrize("backend", _backends())
def test_zero_row_and_subnormal_inputs(backend):
    # Zero rows exercise r = 1/eps and the oracle's NaN-producing RMS
    # derivative; subnormal magnitudes exercise non-FTZ arithmetic.
    x, w, dp, dr = _make_inputs(4, 512, "cuda", seed=13)
    x[0].zero_()
    x[1].mul_(1e-42)  # subnormal products
    provider = _provider(backend)
    p, r, saved = provider.fp32_gemm_rms_fwd(x, w, EPS)
    gp, gr, gsaved = oracle.fp32_gemm_rms_fwd(x, w, EPS)
    _assert_bytes_equal(p, gp, f"{backend} P zero/subnormal")
    _assert_bytes_equal(r, gr, f"{backend} r zero/subnormal")
    dx, dw = provider.fp32_gemm_rms_bwd(dp, dr, x, w, saved)
    gdx, gdw = oracle.fp32_gemm_rms_bwd(dp, dr, x, w, gsaved)
    _assert_bytes_equal(dx, gdx, f"{backend} dX zero/subnormal")
    _assert_bytes_equal(dw, gdw, f"{backend} dW zero/subnormal")


@pytest.mark.parametrize("backend", _backends())
def test_empty_batch(backend):
    x, w, dp, dr = _make_inputs(0, 512, "cuda")
    provider = _provider(backend)
    p, r, saved = provider.fp32_gemm_rms_fwd(x, w, EPS)
    assert p.shape == (0, PROD_N) and r.shape == (0,)
    dx, dw = provider.fp32_gemm_rms_bwd(dp, dr, x, w, saved)
    gdx, gdw = oracle.fp32_gemm_rms_bwd(dp, dr, x, w, {"q": saved["q"], "r": saved["r"], "k": 512})
    _assert_bytes_equal(dx, gdx, f"{backend} empty dX")
    _assert_bytes_equal(dw, gdw, f"{backend} empty dW")


@pytest.mark.parametrize("backend", _backends())
def test_non_contiguous_inputs(backend):
    x2, w, dp, dr = _make_inputs(12, 512, "cuda", seed=17)
    x_nc = x2.t().contiguous().t()  # same values, non-contiguous layout
    assert not x_nc.is_contiguous()
    provider = _provider(backend)
    p, r, _ = provider.fp32_gemm_rms_fwd(x_nc.contiguous(), w, EPS)
    gp, gr, _ = oracle.fp32_gemm_rms_fwd(x2, w, EPS)
    _assert_bytes_equal(p, gp, f"{backend} non-contiguous P")
    _assert_bytes_equal(r, gr, f"{backend} non-contiguous r")


@pytest.mark.parametrize("backend", _backends())
def test_fixed_k_gemm_backend_bitequal(backend):
    x, w, dp, _ = _make_inputs(5, 512, "cuda", seed=19)
    provider = _provider(backend)
    report = fixed_k_gemm_bitequal_harness(provider.fixed_k_gemm_fwd, x, w)
    assert report["bitwise_equal"], report

    dx, dw = provider.fixed_k_gemm_bwd(dp, x, w)
    gdx, gdw = oracle.fixed_k_gemm_bwd(dp, x, w)
    _assert_bytes_equal(dx, gdx, f"{backend} fixed_k dX")
    _assert_bytes_equal(dw, gdw, f"{backend} fixed_k dW")


@cuda_only
def test_cuda_rejects_unsupported_n():
    x = torch.randn(2, 64, device="cuda", dtype=torch.float32)
    w = torch.randn(40, 64, device="cuda", dtype=torch.float32)
    provider = CudaGemmRMSProvider()
    with pytest.raises(RuntimeError, match="fail-closed"):
        provider.fp32_gemm_rms_fwd(x, w, EPS)
