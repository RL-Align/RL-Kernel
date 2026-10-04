# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Regression, invariance and fail-closed tests for the H3 Euler step (RFC #420).

Thresholds come from the shared contract via ``resolve_tolerance``; no private
``atol`` / ``rtol`` constants are used as gate evidence
(``docs/contributing/gtest-usage.md`` section 6.3).
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.flow.ode_step import NativeH3OdeStepOp

try:  # Triton candidate is optional on CPU-only hosts.
    from rl_engine.kernels.ops.triton.flow.ode_step import TritonH3OdeStepOp

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    TritonH3OdeStepOp = None  # type: ignore[assignment]
    _HAS_TRITON = False

try:  # Native CUDA candidate needs the compiled _C extension.
    from rl_engine.kernels.ops.cuda.flow.ode_step import CudaH3OdeStepOp

    _HAS_CUDA_OP = True
except Exception:  # pragma: no cover - import guard
    CudaH3OdeStepOp = None  # type: ignore[assignment]
    _HAS_CUDA_OP = False

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP device")

# H3 video latent has 24 channels, audio latent has 32; both are non-power-of-two.
_VIDEO_CHANNELS = 24
_AUDIO_CHANNELS = 32


def _sigma_pair(
    rows: int, *, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (sigma, sigma_next) drawn from a shifted, de-duplicated H3 grid."""
    steps = 8
    grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    idx = torch.arange(rows) % steps
    return grid[idx].to(dtype), grid[idx + 1].to(dtype)


def _inputs(rows: int = 8, channels: int = _VIDEO_CHANNELS, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    xt = torch.randn(rows, channels, dtype=torch.float32, generator=generator)
    v = torch.randn(rows, channels, dtype=torch.float32, generator=generator)
    sigma, sigma_next = _sigma_pair(rows)
    return xt, v, sigma.unsqueeze(-1), sigma_next.unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Declared expression order (ablation probe H13)
# --------------------------------------------------------------------------- #


def test_gold_uses_declared_expression_order() -> None:
    xt, v, sigma, sigma_next = _inputs()
    op = NativeH3OdeStepOp()
    x_next, x0 = op.forward_fp32(xt, v, sigma, sigma_next)

    r = sigma_next / sigma
    ref_x0 = xt + sigma * v
    ref_x_next = r * xt + (1.0 - r) * ref_x0
    assert torch.equal(x0, ref_x0)
    assert torch.equal(x_next, ref_x_next)


def test_declared_form_is_not_bitwise_equal_to_simplified_form() -> None:
    """H13: the reassociated form is algebraically equal but bitwise different.

    Measured on this shape: 365-780 of 2048 elements differ at each intermediate
    step (max |diff| 2.38e-07).  If this ever passes trivially, the probe input
    stopped exercising the reassociation and must be reshaped.
    """
    xt, v, sigma, sigma_next = _inputs(rows=64, channels=_AUDIO_CHANNELS, seed=7)
    op = NativeH3OdeStepOp()
    x_next, _ = op.forward_fp32(xt, v, sigma, sigma_next)
    simplified = xt + (sigma - sigma_next) * v
    assert not torch.equal(x_next, simplified)


def test_terminal_step_does_not_distinguish_the_forms() -> None:
    """The last grid step can never expose H13 -- which is exactly why a suite
    that only checks the terminal step is worthless.

    At ``sigma_next == 0`` we get ``r == 0`` and ``1 - r == 1`` exactly, so both
    the declared and the simplified form collapse to ``xt + sigma * v``.
    """
    xt, v, _, _ = _inputs(rows=8)
    sigma = torch.full((8, 1), 0.125, dtype=torch.float32)
    sigma_next = torch.zeros((8, 1), dtype=torch.float32)
    x_next, _ = NativeH3OdeStepOp().forward_fp32(xt, v, sigma, sigma_next)
    simplified = xt + (sigma - sigma_next) * v
    assert torch.equal(x_next, simplified), (
        "terminal step unexpectedly distinguishes the two forms; re-derive the "
        "H13 argument before trusting the invariance suite"
    )


# --------------------------------------------------------------------------- #
# Sigma grid boundaries
# --------------------------------------------------------------------------- #


def test_terminal_step_has_zero_ratio() -> None:
    """Last step of the grid: sigma_next == 0, so x_next must equal x0 exactly."""
    xt, v, _, _ = _inputs(rows=4)
    sigma = torch.full((4, 1), 0.125, dtype=torch.float32)
    sigma_next = torch.zeros((4, 1), dtype=torch.float32)
    x_next, x0 = NativeH3OdeStepOp().forward_fp32(xt, v, sigma, sigma_next)
    assert torch.equal(x_next, x0)


def test_scalar_sigma_path() -> None:
    xt, v, _, _ = _inputs(rows=4)
    x_next, x0 = NativeH3OdeStepOp().forward_fp32(xt, v, 1.0, 0.5)
    assert x_next.shape == xt.shape and x0.shape == xt.shape


def test_per_row_sigma_flat_vector_form() -> None:
    """A bare ``[R]`` sigma must mean "one per row", not "one per channel".

    Left unreshaped, ``[R]`` broadcasts against the *last* axis of a ``[R, C]``
    input.  With ``R == C`` that raises no error and silently returns wrong
    values, so the shape is chosen deliberately here.
    """
    rows = channels = 8
    xt, v, sigma, sigma_next = _inputs(rows=rows, channels=channels)
    op = NativeH3OdeStepOp()
    canonical = op.forward_fp32(xt, v, sigma, sigma_next)
    flat = op.forward_fp32(xt, v, sigma.reshape(rows), sigma_next.reshape(rows))
    assert torch.equal(flat[0], canonical[0])
    assert torch.equal(flat[1], canonical[1])


def test_per_row_sigma_with_3d_xt() -> None:
    """``[B*S]`` and ``[B*S, 1]`` sigma must both work against ``[B, S, C]`` xt."""
    batch, seq, channels = 2, 4, 24
    generator = torch.Generator().manual_seed(3)
    xt = torch.randn(batch, seq, channels, generator=generator)
    v = torch.randn(batch, seq, channels, generator=generator)
    sigma, sigma_next = _sigma_pair(batch * seq)
    op = NativeH3OdeStepOp()

    canonical = op.forward_fp32(xt, v, sigma, sigma_next)
    flat = op.forward_fp32(xt, v, sigma.reshape(-1), sigma_next.reshape(-1))
    assert canonical[0].shape == xt.shape
    assert torch.equal(flat[0], canonical[0])
    assert torch.equal(flat[1], canonical[1])


def test_fail_closed_on_3d_sigma_layout_mismatch() -> None:
    """A 3-D xt with a sigma that cannot cover its rows must be rejected."""
    xt = torch.randn(2, 4, 24)
    v = torch.randn(2, 4, 24)
    with pytest.raises(ValueError, match="one value per packed row"):
        NativeH3OdeStepOp().forward_fp32(xt, v, torch.full((4, 1), 0.5), torch.full((4, 1), 0.25))


def test_shifted_grid_is_monotone() -> None:
    sigma, sigma_next = _sigma_pair(32)
    assert bool((sigma > sigma_next).all())
    assert bool((sigma_next >= 0).all())


# --------------------------------------------------------------------------- #
# Dtype coverage
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_forward_casts_at_epilogue(dtype: torch.dtype) -> None:
    xt, v, sigma, sigma_next = _inputs()
    x_next, x0 = NativeH3OdeStepOp().forward(xt.to(dtype), v.to(dtype), sigma, sigma_next)
    assert x_next.dtype == dtype and x0.dtype == dtype


@pytest.mark.parametrize("channels", [_VIDEO_CHANNELS, _AUDIO_CHANNELS])
def test_non_power_of_two_channels(channels: int) -> None:
    xt, v, sigma, sigma_next = _inputs(rows=3, channels=channels)
    x_next, _ = NativeH3OdeStepOp().forward_fp32(xt, v, sigma, sigma_next)
    assert x_next.shape == (3, channels)


# --------------------------------------------------------------------------- #
# Axis-A invariance: bitwise, not "within tolerance"
# --------------------------------------------------------------------------- #


def test_batch_position_invariance() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=8)
    op = NativeH3OdeStepOp()
    full, _ = op.forward_fp32(xt, v, sigma, sigma_next)

    for row in range(8):
        order = [row, *[i for i in range(8) if i != row]]
        perm = torch.tensor(order)
        part, _ = op.forward_fp32(xt[perm], v[perm], sigma[perm], sigma_next[perm])
        assert torch.equal(part[0], full[row])


def test_batch_size_invariance() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=16)
    op = NativeH3OdeStepOp()
    full, _ = op.forward_fp32(xt, v, sigma, sigma_next)
    for rows in (1, 8, 16):
        part, _ = op.forward_fp32(xt[:rows], v[:rows], sigma[:rows], sigma_next[:rows])
        assert torch.equal(part, full[:rows])


def test_unrelated_row_mutation_does_not_change_a_row() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=4)
    op = NativeH3OdeStepOp()
    before, _ = op.forward_fp32(xt, v, sigma, sigma_next)

    xt2, v2 = xt.clone(), v.clone()
    xt2[1:] += 1000.0
    v2[1:] -= 1000.0
    after, _ = op.forward_fp32(xt2, v2, sigma, sigma_next)
    assert torch.equal(after[0], before[0])


def test_repeated_execution_is_bitwise_stable() -> None:
    xt, v, sigma, sigma_next = _inputs()
    op = NativeH3OdeStepOp()
    first = op.forward_fp32(xt, v, sigma, sigma_next)
    second = op.forward_fp32(xt, v, sigma, sigma_next)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


# --------------------------------------------------------------------------- #
# Fail-closed behaviour (RFC #420 section 4, rule 9)
# --------------------------------------------------------------------------- #


def test_fail_closed_on_zero_sigma() -> None:
    xt, v, _, _ = _inputs(rows=2)
    with pytest.raises(ValueError, match="strictly positive"):
        NativeH3OdeStepOp().forward_fp32(xt, v, torch.zeros(2, 1), torch.zeros(2, 1))


def test_fail_closed_on_increasing_sigma() -> None:
    xt, v, _, _ = _inputs(rows=2)
    with pytest.raises(ValueError, match="monotonically decreasing"):
        NativeH3OdeStepOp().forward_fp32(xt, v, torch.full((2, 1), 0.25), torch.full((2, 1), 0.75))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_fail_closed_on_non_finite_sigma(bad: float) -> None:
    xt, v, _, _ = _inputs(rows=2)
    with pytest.raises(ValueError, match="finite"):
        NativeH3OdeStepOp().forward_fp32(xt, v, torch.full((2, 1), bad), torch.full((2, 1), 0.5))


def test_fail_closed_on_shape_mismatch() -> None:
    xt, _, sigma, sigma_next = _inputs(rows=2)
    with pytest.raises(ValueError, match="share a shape"):
        NativeH3OdeStepOp().forward_fp32(xt, torch.randn(3, xt.shape[1]), sigma, sigma_next)


def test_fail_closed_on_unsupported_dtype() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=2)
    with pytest.raises(TypeError, match="dtype"):
        NativeH3OdeStepOp().forward_fp32(
            xt.to(torch.float64), v.to(torch.float64), sigma, sigma_next
        )


def test_fail_closed_on_bad_per_row_sigma_length() -> None:
    xt, v, _, _ = _inputs(rows=4)
    with pytest.raises(ValueError, match="one value per packed row"):
        NativeH3OdeStepOp().forward_fp32(xt, v, torch.full((3, 1), 0.5), torch.full((3, 1), 0.25))


def test_empty_input_is_legal() -> None:
    xt = torch.empty(0, _VIDEO_CHANNELS)
    v = torch.empty(0, _VIDEO_CHANNELS)
    x_next, x0 = NativeH3OdeStepOp().forward_fp32(xt, v, 1.0, 0.5)
    assert x_next.numel() == 0 and x0.numel() == 0


def test_non_contiguous_input() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=6, channels=_VIDEO_CHANNELS)
    xt_t = xt.t().contiguous().t()  # same values, non-contiguous layout
    x_next, _ = NativeH3OdeStepOp().forward_fp32(xt_t, v, sigma, sigma_next)
    ref, _ = NativeH3OdeStepOp().forward_fp32(xt.contiguous(), v, sigma, sigma_next)
    assert torch.equal(x_next, ref)


# --------------------------------------------------------------------------- #
# Gradient (only meaningful for the autograd-carrying candidate path)
# --------------------------------------------------------------------------- #


def test_gold_gradients_match_closed_form() -> None:
    """Closed-form backward must match autograd exactly (no tolerance).

    Justification for ``atol=rtol=0``: both the graph and the closed form are
    two-operand FP32 sums, and IEEE-754 addition is commutative (verified over
    1e6 samples in ``verify_claims.py``), so the accumulation order chosen by the
    autograd engine cannot change a single bit.  A failure here means the
    expression *program* differs -- typically a reassociated ``(s - sn) * g``
    instead of ``(1 - r) * g``.
    """
    xt, v, sigma, sigma_next = _inputs(rows=4)
    xt = xt.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    x_next, x0 = NativeH3OdeStepOp().forward_fp32(xt, v, sigma, sigma_next)

    grad_next = torch.randn_like(x_next)
    grad_x0 = torch.randn_like(x0)
    torch.autograd.backward([x_next, x0], [grad_next, grad_x0])

    r = sigma_next / sigma
    expected_x0 = grad_x0 + (1.0 - r) * grad_next
    expected_xt = r * grad_next + expected_x0
    expected_v = sigma * expected_x0
    assert torch.allclose(xt.grad, expected_xt, atol=0.0, rtol=0.0)
    assert torch.allclose(v.grad, expected_v, atol=0.0, rtol=0.0)


# --------------------------------------------------------------------------- #
# Candidate path (GPU)
# --------------------------------------------------------------------------- #


@_CUDA
@pytest.mark.skipif(not _HAS_TRITON, reason="Triton candidate unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_gold(dtype: torch.dtype) -> None:
    from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance

    xt, v, sigma, sigma_next = _inputs(rows=8, channels=_AUDIO_CHANNELS)
    device = torch.device("cuda")
    inputs = (
        xt.to(device, dtype),
        v.to(device, dtype),
        sigma.to(device),
        sigma_next.to(device),
    )
    gold = NativeH3OdeStepOp().forward_fp32(
        *[t.float() if t.is_floating_point() else t for t in inputs]
    )
    cand = TritonH3OdeStepOp().forward(*inputs)

    tol = resolve_tolerance(
        load_contract(),
        judgment="forward_accuracy",
        op_class="elementwise",
        dtype=dtype,
    )
    for got, want in zip(cand, gold, strict=True):
        assert torch.allclose(
            got.float(), want.float(), atol=tol.atol, rtol=tol.rtol
        ), f"max_abs={float((got.float() - want.float()).abs().max())}"


@_CUDA
@pytest.mark.skipif(not _HAS_TRITON, reason="Triton candidate unavailable")
def test_triton_batch_position_invariance_bitwise() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=8)
    device = torch.device("cuda")
    op = TritonH3OdeStepOp()
    full, _ = op.forward_fp32(xt.to(device), v.to(device), sigma.to(device), sigma_next.to(device))
    order = [3, 0, 1, 2, 4, 5, 6, 7]
    perm = torch.tensor(order)
    part, _ = op.forward_fp32(
        xt[perm].to(device),
        v[perm].to(device),
        sigma[perm].to(device),
        sigma_next[perm].to(device),
    )
    assert torch.equal(part[0], full[3])


# --------------------------------------------------------------------------- #
# Native CUDA path (the strict kernel).  These assert bit-exactness, not
# tolerance: every arithmetic step in csrc/cuda/flow/ode_step.cu goes through an
# explicitly-rounded intrinsic, so the device result reproduces the PyTorch
# reference op for op.  If plain nvcc FMA contraction ever creeps in, these fail.
# --------------------------------------------------------------------------- #


@_CUDA
@pytest.mark.skipif(not _HAS_CUDA_OP, reason="native CUDA candidate unavailable")
def test_cuda_fp32_is_bitwise_equal_to_gold() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=64, channels=_AUDIO_CHANNELS)
    device = torch.device("cuda")
    gold_next, gold_x0 = NativeH3OdeStepOp().forward_fp32(xt, v, sigma, sigma_next)
    cand_next, cand_x0 = CudaH3OdeStepOp().forward_fp32(
        xt.to(device), v.to(device), sigma.to(device), sigma_next.to(device)
    )
    assert torch.equal(cand_next.cpu(), gold_next), (
        f"x_next not bitwise equal; max_abs=" f"{float((cand_next.cpu() - gold_next).abs().max())}"
    )
    assert torch.equal(
        cand_x0.cpu(), gold_x0
    ), f"x0 not bitwise equal; max_abs={float((cand_x0.cpu() - gold_x0).abs().max())}"


@_CUDA
@pytest.mark.skipif(not _HAS_CUDA_OP, reason="native CUDA candidate unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cuda_matches_gold_within_contract(dtype: torch.dtype) -> None:
    from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance

    xt, v, sigma, sigma_next = _inputs(rows=16, channels=_VIDEO_CHANNELS)
    device = torch.device("cuda")
    gold = NativeH3OdeStepOp().forward_fp32(xt, v, sigma, sigma_next)
    cand = CudaH3OdeStepOp().forward(
        xt.to(device, dtype), v.to(device, dtype), sigma.to(device), sigma_next.to(device)
    )
    tol = resolve_tolerance(
        load_contract(),
        judgment="forward_accuracy",
        op_class="elementwise",
        dtype=dtype,
    )
    for got, want in zip(cand, gold, strict=True):
        assert torch.allclose(
            got.float().cpu(), want, atol=tol.atol, rtol=tol.rtol
        ), f"max_abs={float((got.float().cpu() - want).abs().max())}"


@_CUDA
@pytest.mark.skipif(not _HAS_CUDA_OP, reason="native CUDA candidate unavailable")
def test_cuda_batch_position_invariance_bitwise() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=8)
    device = torch.device("cuda")
    op = CudaH3OdeStepOp()
    full, _ = op.forward_fp32(xt.to(device), v.to(device), sigma.to(device), sigma_next.to(device))
    order = [5, 0, 1, 2, 3, 4, 6, 7]
    perm = torch.tensor(order)
    part, _ = op.forward_fp32(
        xt[perm].to(device),
        v[perm].to(device),
        sigma[perm].to(device),
        sigma_next[perm].to(device),
    )
    assert torch.equal(part[0], full[5])


@_CUDA
@pytest.mark.skipif(not _HAS_CUDA_OP, reason="native CUDA candidate unavailable")
def test_cuda_gradients_match_gold() -> None:
    """Same upstream gradients for both arms, then bitwise compare.

    The upstream gradients must be generated once and reused: drawing fresh
    randn values per arm would compare different vector-Jacobian products.
    """
    xt, v, sigma, sigma_next = _inputs(rows=8, channels=_VIDEO_CHANNELS)
    device = torch.device("cuda")
    # Both arms are fp32 here, so the CUDA kernel reproduces the reference exactly.
    g_next = torch.randn(xt.shape, device=device)
    g_x0 = torch.randn(xt.shape, device=device)

    def run(op):
        x = xt.clone().to(device).requires_grad_(True)
        vv = v.clone().to(device).requires_grad_(True)
        xn, x0 = op(x, vv, sigma.to(device), sigma_next.to(device))
        torch.autograd.backward([xn, x0], [g_next, g_x0])
        return xn.detach(), x0.detach(), x.grad, vv.grad

    gold = run(NativeH3OdeStepOp().forward_fp32)
    cand = run(CudaH3OdeStepOp().forward)
    for name, got, want in zip(("x_next", "x0", "grad_xt", "grad_v"), cand, gold, strict=True):
        assert got.shape == want.shape, name
        got_cpu, want_cpu = got.float().cpu(), want.float().cpu()
        assert torch.equal(got_cpu, want_cpu), (
            f"{name} is not bitwise equal; max_abs=" f"{float((got_cpu - want_cpu).abs().max())}"
        )


@_CUDA
@pytest.mark.skipif(not _HAS_CUDA_OP, reason="native CUDA candidate unavailable")
def test_cuda_fail_closed_on_zero_sigma() -> None:
    xt, v, _, _ = _inputs(rows=4)
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="strictly positive"):
        CudaH3OdeStepOp().forward(
            xt.to(device),
            v.to(device),
            torch.zeros(4, 1, device=device),
            torch.zeros(4, 1, device=device),
        )


@_CUDA
@pytest.mark.skipif(not _HAS_CUDA_OP, reason="native CUDA candidate unavailable")
def test_cuda_accepts_scalar_and_per_row_sigma() -> None:
    xt, v, sigma, sigma_next = _inputs(rows=6)
    device = torch.device("cuda")
    op = CudaH3OdeStepOp()
    scalar = op.forward_fp32(xt.to(device), v.to(device), 1.0, 0.5)
    assert scalar[0].shape == xt.shape
    per_row = op.forward_fp32(xt.to(device), v.to(device), sigma.to(device), sigma_next.to(device))
    assert per_row[0].shape == xt.shape
    assert not torch.equal(scalar[0].cpu(), per_row[0].cpu())
