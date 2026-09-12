# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""SiLU / SwiGLU tests: native gold + CUDA / Triton / Ascend C candidates.

Covers:
- Native correctness (fp32 formula, dtype path, shape guard)
- Axis A batch invariance (slice + padding, forward + backward)
- CUDA / Triton forward+backward vs NativeSiLUOp / NativeSwiGLUOp (issue #108 harness)
- Ascend C SwiGLU integration, forward/backward accuracy and NPU acceptance
- Registry dispatch + OP_SPECS candidate paths
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.gtest.op_checks import run_operator_suite
from rl_engine.kernels.gtest.operator_specs import (
    OP_SPECS,
    make_candidate,
    make_operator_case,
    operator_names,
)
from rl_engine.kernels.ops.ascend.activation import swiglu as ascend
from rl_engine.kernels.ops.pytorch.activation.swiglu import NativeSiLUOp, NativeSwiGLUOp
from rl_engine.kernels.registry import KernelRegistry, OpBackend, kernel_registry
from rl_engine.platforms.device import _npu_available, device_ctx

try:
    from rl_engine.kernels.ops.triton.activation.swiglu import TritonSiLUOp, TritonSwiGLUOp

    _HAS_TRITON_ACTIVATION = True
except ImportError:  # pragma: no cover - triton may be missing in CPU-only builds.
    _HAS_TRITON_ACTIVATION = False
    TritonSiLUOp = None  # type: ignore[misc, assignment]
    TritonSwiGLUOp = None  # type: ignore[misc, assignment]

try:
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    from rl_engine.kernels.ops.cuda.activation.swiglu import SiLUCudaOp, SwiGLUCudaOp

    _HAS_CUDA_ACTIVATION = (
        _EXT_AVAILABLE and hasattr(_C, "silu_forward") and hasattr(_C, "swiglu_forward")
    )
except ImportError:  # pragma: no cover - extension may be missing in CPU-only builds.
    _HAS_CUDA_ACTIVATION = False
    SiLUCudaOp = None  # type: ignore[misc, assignment]
    SwiGLUCudaOp = None  # type: ignore[misc, assignment]

# Qwen3-8B SwiGLU intermediate dim (gate/up_proj output width).
_INTERMEDIATE = 12288
_ASCEND_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


# Shared helper
def _rand(shape, *, seed, dtype=torch.float32, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(*shape, generator=gen, dtype=torch.float32)
    return t.to(device=device, dtype=dtype)


def _dtype_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    # Matches elementwise row of tolerance_contract.json (issue #108).
    if dtype is torch.float32:
        return 1e-5, 1e-5
    if dtype is torch.float16:
        return 1e-3, 1e-3
    if dtype is torch.bfloat16:
        return 2e-2, 1.6e-2
    raise ValueError(f"unsupported dtype: {dtype}")


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_npu = pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
requires_cuda_activation = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_CUDA_ACTIVATION),
    reason="CUDA SiLU/SwiGLU extension is not available",
)
requires_triton_activation = pytest.mark.skipif(
    not _HAS_TRITON_ACTIVATION,
    reason="Triton SiLU/SwiGLU is not available",
)
requires_nvidia_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)


# ---------------------------------------------------------------------------
# Native gold (PyTorch reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_silu_matches_fp32_reference(dtype: torch.dtype):
    x = torch.linspace(-6.0, 6.0, 33, dtype=dtype).reshape(3, 11)

    fp32_reference = x.float() * torch.sigmoid(x.float())
    result = NativeSiLUOp().forward(x)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSiLUOp().forward_fp32(x), fp32_reference)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_native_swiglu_matches_fp32_reference(dtype: torch.dtype):
    gate = torch.linspace(-4.0, 4.0, 48, dtype=dtype).reshape(2, 3, 8)
    up = torch.linspace(0.5, 2.0, 48, dtype=dtype).reshape(2, 3, 8)

    fp32_reference = gate.float() * torch.sigmoid(gate.float()) * up.float()
    result = NativeSwiGLUOp().forward(gate, up)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSwiGLUOp().forward_fp32(gate, up), fp32_reference)


def test_native_swiglu_rejects_mismatched_shape():
    gate = torch.randn(2, 3)
    up = torch.randn(2, 4)

    with pytest.raises(ValueError, match="share shape"):
        NativeSwiGLUOp().forward(gate, up)


def test_native_activation_rejects_invalid_dtypes():
    with pytest.raises(TypeError, match="fp16, bf16, or fp32"):
        NativeSiLUOp().forward(torch.ones(8, dtype=torch.int32))

    with pytest.raises(TypeError, match="share dtype"):
        NativeSwiGLUOp().forward(
            torch.ones(8, dtype=torch.float16),
            torch.ones(8, dtype=torch.bfloat16),
        )


# Axis A -- batch invariance, bitwise (the WS1 "aligned" property).
# A row's output must not depend on how many rows share the batch.
def test_silu_batch_invariance_slice():
    op = NativeSiLUOp()
    x = _rand((8, 32, _INTERMEDIATE), seed=2)
    full = op.forward_fp32(x)  # compute on full batch...
    assert torch.equal(op.forward_fp32(x[:1]), full[:1])  # ...then slice
    assert torch.equal(op.forward_fp32(x[3:5]), full[3:5])


def test_swiglu_batch_invariance_slice():
    op = NativeSwiGLUOp()
    gate = _rand((8, 32, _INTERMEDIATE), seed=3)
    up = _rand((8, 32, _INTERMEDIATE), seed=4)
    full = op.forward_fp32(gate, up)
    assert torch.equal(op.forward_fp32(gate[:1], up[:1]), full[:1])
    assert torch.equal(op.forward_fp32(gate[3:5], up[3:5]), full[3:5])


def test_silu_batch_invariance_with_padding():
    """Padding extra rows must not perturb the real rows (bitwise)."""
    op = NativeSiLUOp()
    x = _rand((4, _INTERMEDIATE), seed=5)
    padded = torch.cat([x, _rand((6, _INTERMEDIATE), seed=99)], dim=0)
    assert torch.equal(op.forward_fp32(padded)[:4], op.forward_fp32(x))


def test_swiglu_batch_invariance_with_padding():
    op = NativeSwiGLUOp()
    gate = _rand((4, _INTERMEDIATE), seed=6)
    up = _rand((4, _INTERMEDIATE), seed=7)
    pad_gate = torch.cat([gate, _rand((6, _INTERMEDIATE), seed=98)], dim=0)
    pad_up = torch.cat([up, _rand((6, _INTERMEDIATE), seed=97)], dim=0)
    assert torch.equal(op.forward_fp32(pad_gate, pad_up)[:4], op.forward_fp32(gate, up))


# Purity -- inputs not mutated in-place
def test_silu_inputs_not_mutated():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=8)
    xc = x.clone()
    op.forward(x)
    op.forward_fp32(x)
    assert torch.equal(x, xc)


def test_swiglu_inputs_not_mutated():
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=9)
    up = _rand((2, _INTERMEDIATE), seed=10)
    gc, uc = gate.clone(), up.clone()
    op.forward(gate, up)
    op.forward_fp32(gate, up)
    assert torch.equal(gate, gc) and torch.equal(up, uc)


# Gradient flows (fp32 autograd = backward golden source)
def test_silu_gradient_flows():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=11).requires_grad_(True)
    op.forward_fp32(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_swiglu_gradient_flows():
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=12).requires_grad_(True)
    up = _rand((2, _INTERMEDIATE), seed=13).requires_grad_(True)
    op.forward_fp32(gate, up).sum().backward()
    assert torch.isfinite(gate.grad).all() and torch.isfinite(up.grad).all()


def test_silu_backward_batch_invariance_slice():
    """Axis A: Gradients must be bitwise identical regardless of batch size."""
    op = NativeSiLUOp()

    x_full = _rand((8, 32, _INTERMEDIATE), seed=1).requires_grad_(True)
    out_full = op.forward_fp32(x_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_full_sliced = x_full.grad[:1].clone()

    x_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(x_slice)
    out_slice.backward(dy_full[:1])

    assert torch.equal(x_slice.grad, grad_full_sliced)


def test_swiglu_backward_batch_invariance_slice():
    """Axis A: Gradients must be bitwise identical regardless of batch size."""
    op = NativeSwiGLUOp()

    gate_full = _rand((8, 32, _INTERMEDIATE), seed=1).requires_grad_(True)
    up_full = _rand((8, 32, _INTERMEDIATE), seed=2).requires_grad_(True)
    out_full = op.forward_fp32(gate_full, up_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_gate_full_sliced = gate_full.grad[:1].clone()
    grad_up_full_sliced = up_full.grad[:1].clone()

    gate_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    up_slice = _rand((8, 32, _INTERMEDIATE), seed=2)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(gate_slice, up_slice)

    out_slice.backward(dy_full[:1])

    assert torch.equal(gate_slice.grad, grad_gate_full_sliced)
    assert torch.equal(up_slice.grad, grad_up_full_sliced)


def test_registry_dispatches_native_activation_ops_on_cpu():
    assert isinstance(kernel_registry.get_op("silu", device="cpu"), NativeSiLUOp)
    assert isinstance(kernel_registry.get_op("swiglu", device="cpu"), NativeSwiGLUOp)


# ---------------------------------------------------------------------------
# CUDA / Triton candidates vs native gold (RMSNorm-style)
# ---------------------------------------------------------------------------


def _silu_impls():
    return [
        pytest.param("triton", marks=requires_triton_activation, id="triton"),
        pytest.param("cuda", marks=requires_cuda_activation, id="cuda"),
    ]


@requires_nvidia_cuda
def test_cuda_activation_symbols_are_built_on_cuda_host():
    if not _HAS_CUDA_ACTIVATION:
        pytest.skip(
            "CUDA is available but SiLU/SwiGLU symbols are missing from rl_engine._C; "
            "rebuild the extension from the current source tree to exercise the CUDA path"
        )


def _make_silu_op(impl: str):
    if impl == "cuda":
        return SiLUCudaOp()
    if impl == "triton":
        return TritonSiLUOp()
    raise ValueError(impl)


def _make_swiglu_op(impl: str):
    if impl == "cuda":
        return SwiGLUCudaOp()
    if impl == "triton":
        return TritonSwiGLUOp()
    raise ValueError(impl)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64),
        (8, 256),
        (4, 32, 512),
        (2, 8, _INTERMEDIATE),  # Qwen3-8B intermediate width
    ],
)
def test_cuda_triton_silu_matches_native_forward_and_backward(impl, dtype, shape):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SiLU extension is not available")

    native = NativeSiLUOp()
    cand = _make_silu_op(impl)

    x_cpu = _rand(shape, seed=0, dtype=torch.float32)
    dy_cpu = _rand(shape, seed=1, dtype=torch.float32)

    x_ref = x_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_cpu.to(dtype).float()
    y_ref = native.forward_fp32(x_ref)
    y_ref.backward(dy_ref)

    x_gpu = x_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    y_gpu = cand.forward(x_gpu)
    y_gpu.backward(dy_gpu)

    atol, rtol = _dtype_tolerance(dtype)
    torch.testing.assert_close(y_gpu.detach().cpu().float(), y_ref.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_gpu.grad.detach().cpu().float(), x_ref.grad, atol=atol, rtol=rtol)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 64),
        (8, 256),
        (4, 32, 512),
        (2, 8, _INTERMEDIATE),
    ],
)
def test_cuda_triton_swiglu_matches_native_forward_and_backward(impl, dtype, shape):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    native = NativeSwiGLUOp()
    cand = _make_swiglu_op(impl)

    gate_cpu = _rand(shape, seed=2, dtype=torch.float32)
    up_cpu = _rand(shape, seed=3, dtype=torch.float32)
    dy_cpu = _rand(shape, seed=4, dtype=torch.float32)

    gate_ref = gate_cpu.to(dtype).float().detach().requires_grad_(True)
    up_ref = up_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_cpu.to(dtype).float()
    y_ref = native.forward_fp32(gate_ref, up_ref)
    y_ref.backward(dy_ref)

    gate_gpu = gate_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    up_gpu = up_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    y_gpu = cand.forward(gate_gpu, up_gpu)
    y_gpu.backward(dy_gpu)

    atol, rtol = _dtype_tolerance(dtype)
    torch.testing.assert_close(y_gpu.detach().cpu().float(), y_ref.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        gate_gpu.grad.detach().cpu().float(), gate_ref.grad, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        up_gpu.grad.detach().cpu().float(), up_ref.grad, atol=atol, rtol=rtol
    )


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_silu_batch_invariance_bitwise(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SiLU extension is not available")

    op = _make_silu_op(impl)
    x = _rand((8, 32, 256), seed=5, dtype=torch.bfloat16, device="cuda")
    full = op.forward(x)
    assert torch.equal(op.forward(x[:1]), full[:1])
    assert torch.equal(op.forward(x[3:5]), full[3:5])


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_batch_invariance_bitwise(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    op = _make_swiglu_op(impl)
    gate = _rand((8, 32, 256), seed=6, dtype=torch.bfloat16, device="cuda")
    up = _rand((8, 32, 256), seed=7, dtype=torch.bfloat16, device="cuda")
    full = op.forward(gate, up)
    assert torch.equal(op.forward(gate[:1], up[:1]), full[:1])
    assert torch.equal(op.forward(gate[3:5], up[3:5]), full[3:5])


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_silu_deterministic_repeat(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SiLU extension is not available")

    op = _make_silu_op(impl)
    x = _rand((32, 1024), seed=8, dtype=torch.bfloat16, device="cuda")
    dy = _rand((32, 1024), seed=9, dtype=torch.bfloat16, device="cuda")

    def _run():
        x_r = x.detach().clone().requires_grad_(True)
        y = op.forward(x_r)
        y.backward(dy)
        return y.detach(), x_r.grad.detach()

    y0, dx0 = _run()
    torch.cuda.synchronize()
    for _ in range(5):
        y, dx = _run()
        torch.cuda.synchronize()
        assert torch.equal(y0, y)
        assert torch.equal(dx0, dx)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_rejects_mismatched_shape(impl):
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    op = _make_swiglu_op(impl)
    gate = torch.randn(2, 3, device="cuda")
    up = torch.randn(2, 4, device="cuda")
    with pytest.raises(ValueError, match="share shape"):
        op.forward(gate, up)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_silu_backward_batch_invariance_bitwise(impl):
    op = _make_silu_op(impl)
    x = _rand((6, 4, 64), seed=10, dtype=torch.bfloat16, device="cuda")
    dy = _rand(x.shape, seed=11, dtype=torch.bfloat16, device="cuda")

    x_full = x.detach().clone().requires_grad_(True)
    op.forward(x_full).backward(dy)
    full_grad = x_full.grad[2:4].clone()

    x_slice = x[2:4].detach().clone().requires_grad_(True)
    op.forward(x_slice).backward(dy[2:4])
    assert torch.equal(x_slice.grad, full_grad)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_backward_batch_invariance_bitwise(impl):
    op = _make_swiglu_op(impl)
    gate = _rand((6, 4, 64), seed=12, dtype=torch.bfloat16, device="cuda")
    up = _rand((6, 4, 64), seed=13, dtype=torch.bfloat16, device="cuda")
    dy = _rand(gate.shape, seed=14, dtype=torch.bfloat16, device="cuda")

    gate_full = gate.detach().clone().requires_grad_(True)
    up_full = up.detach().clone().requires_grad_(True)
    op.forward(gate_full, up_full).backward(dy)
    full_d_gate = gate_full.grad[2:4].clone()
    full_d_up = up_full.grad[2:4].clone()

    gate_slice = gate[2:4].detach().clone().requires_grad_(True)
    up_slice = up[2:4].detach().clone().requires_grad_(True)
    op.forward(gate_slice, up_slice).backward(dy[2:4])
    assert torch.equal(gate_slice.grad, full_d_gate)
    assert torch.equal(up_slice.grad, full_d_up)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_silu_padding_invariance_forward_and_backward(impl):
    op = _make_silu_op(impl)
    x = _rand((4, 64), seed=15, dtype=torch.bfloat16, device="cuda")
    dy = _rand(x.shape, seed=16, dtype=torch.bfloat16, device="cuda")
    x_padded = torch.cat(
        [x, _rand((3, 64), seed=17, dtype=torch.bfloat16, device="cuda")], dim=0
    ).requires_grad_(True)
    dy_padded = torch.cat([dy, _rand((3, 64), seed=18, dtype=torch.bfloat16, device="cuda")], dim=0)

    y_padded = op.forward(x_padded)
    y_padded.backward(dy_padded)

    x_real = x.detach().clone().requires_grad_(True)
    y_real = op.forward(x_real)
    y_real.backward(dy)
    assert torch.equal(y_padded[:4], y_real)
    assert torch.equal(x_padded.grad[:4], x_real.grad)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_padding_invariance_forward_and_backward(impl):
    op = _make_swiglu_op(impl)
    gate = _rand((4, 64), seed=19, dtype=torch.bfloat16, device="cuda")
    up = _rand((4, 64), seed=20, dtype=torch.bfloat16, device="cuda")
    dy = _rand(gate.shape, seed=21, dtype=torch.bfloat16, device="cuda")
    gate_padded = torch.cat(
        [gate, _rand((3, 64), seed=22, dtype=torch.bfloat16, device="cuda")], dim=0
    ).requires_grad_(True)
    up_padded = torch.cat(
        [up, _rand((3, 64), seed=23, dtype=torch.bfloat16, device="cuda")], dim=0
    ).requires_grad_(True)
    dy_padded = torch.cat([dy, _rand((3, 64), seed=24, dtype=torch.bfloat16, device="cuda")], dim=0)

    y_padded = op.forward(gate_padded, up_padded)
    y_padded.backward(dy_padded)

    gate_real = gate.detach().clone().requires_grad_(True)
    up_real = up.detach().clone().requires_grad_(True)
    y_real = op.forward(gate_real, up_real)
    y_real.backward(dy)
    assert torch.equal(y_padded[:4], y_real)
    assert torch.equal(gate_padded.grad[:4], gate_real.grad)
    assert torch.equal(up_padded.grad[:4], up_real.grad)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_deterministic_repeat(impl):
    op = _make_swiglu_op(impl)
    gate = _rand((16, 256), seed=25, dtype=torch.bfloat16, device="cuda")
    up = _rand((16, 256), seed=26, dtype=torch.bfloat16, device="cuda")
    dy = _rand(gate.shape, seed=27, dtype=torch.bfloat16, device="cuda")

    def _run():
        gate_r = gate.detach().clone().requires_grad_(True)
        up_r = up.detach().clone().requires_grad_(True)
        y = op.forward(gate_r, up_r)
        y.backward(dy)
        return y.detach(), gate_r.grad.detach(), up_r.grad.detach()

    expected = _run()
    for _ in range(5):
        actual = _run()
        torch.cuda.synchronize()
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(expected, actual, strict=True))


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_activation_handles_noncontiguous_and_empty_inputs(impl):
    silu = _make_silu_op(impl)
    swiglu = _make_swiglu_op(impl)

    x = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16).T.requires_grad_(True)
    assert not x.is_contiguous()
    silu.forward(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    gate = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16).T.requires_grad_(True)
    up = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16).T.requires_grad_(True)
    assert not gate.is_contiguous() and not up.is_contiguous()
    swiglu.forward(gate, up).sum().backward()
    assert gate.grad is not None and up.grad is not None

    empty = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    silu_empty = silu.forward(empty)
    silu_empty.sum().backward()
    assert silu_empty.shape == empty.shape and empty.grad.shape == empty.shape

    empty_gate = empty.detach().clone().requires_grad_(True)
    empty_up = empty.detach().clone().requires_grad_(True)
    swiglu_empty = swiglu.forward(empty_gate, empty_up)
    swiglu_empty.sum().backward()
    assert swiglu_empty.shape == empty_gate.shape
    assert empty_gate.grad.shape == empty_gate.shape and empty_up.grad.shape == empty_up.shape


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_activation_rejects_invalid_dtypes(impl):
    silu = _make_silu_op(impl)
    swiglu = _make_swiglu_op(impl)
    with pytest.raises(TypeError, match="fp16, bf16, or fp32"):
        silu.forward(torch.ones(8, device="cuda", dtype=torch.int32))
    with pytest.raises(TypeError, match="share dtype"):
        swiglu.forward(
            torch.ones(8, device="cuda", dtype=torch.float16),
            torch.ones(8, device="cuda", dtype=torch.bfloat16),
        )


@requires_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_rejects_cross_device_inputs(impl):
    op = _make_swiglu_op(impl)
    gate = torch.ones(8, device="cuda:0")
    up = torch.ones(8, device="cuda:1")
    with pytest.raises(RuntimeError, match=r"same .*device"):
        op.forward(gate, up)


# ---------------------------------------------------------------------------
# Issue #108 ground-truth harness (OP_SPECS + check_operator path)
# ---------------------------------------------------------------------------


def _spec_args(op: str, **overrides) -> argparse.Namespace:
    values = dict(
        op=op,
        candidate="pytorch",
        arch_key=None,
        batch=2,
        seq=4,
        vocab=17,
        seed=123,
        input_mode="random",
        constant_value=0.5,
        token_value=3,
        normalized_dim=8,
        k_dim=8,
        n_dim=8,
        theta=1.0e6,
        eps=1.0e-6,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_silu_swiglu_registered_in_op_specs():
    assert "silu" in operator_names()
    assert "swiglu" in operator_names()


def test_silu_pytorch_candidate_suite_passes_issue_108_helper():
    args = _spec_args("silu", candidate="pytorch")
    report = run_operator_suite(
        "silu",
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, torch.float32, torch.device("cpu"))],
        check_grad=True,
    )
    assert report.passed


def test_swiglu_pytorch_candidate_suite_passes_issue_108_helper():
    args = _spec_args("swiglu", candidate="pytorch")
    report = run_operator_suite(
        "swiglu",
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, torch.float32, torch.device("cpu"))],
        check_grad=True,
    )
    assert report.passed


@requires_cuda
@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param("triton", marks=requires_triton_activation, id="triton"),
        pytest.param("cuda", marks=requires_cuda_activation, id="cuda"),
    ],
)
@pytest.mark.parametrize("op_name", ["silu", "swiglu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_silu_swiglu_cuda_triton_issue_108_harness(candidate, op_name, dtype):
    if candidate == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA activation extension is not available")
    if candidate == "triton" and not _HAS_TRITON_ACTIVATION:
        pytest.skip("Triton SiLU/SwiGLU is not available")

    args = _spec_args(op_name, candidate=candidate, batch=2, seq=8)
    device = torch.device("cuda")
    report = run_operator_suite(
        op_name,
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, dtype, device)],
        check_grad=True,
    )
    assert report.passed, (
        f"{op_name}/{candidate}/{dtype} failed against gold: "
        f"{report.candidates[0].cases[0].outputs}"
    )


# ---------------------------------------------------------------------------
# Ascend C SwiGLU integration and on-device acceptance
# ---------------------------------------------------------------------------
# NPU tests skip only when no NPU is available. On NPU hosts, a missing
# extension fails these tests instead of silently exercising PyTorch fallback.


@pytest.mark.parametrize("symbols", [(), ("swiglu_forward",), ("swiglu_backward",)])
def test_missing_extension_symbols_raise_actionable_error(monkeypatch, symbols):
    monkeypatch.setattr(ascend, "_C_npu", SimpleNamespace(**dict.fromkeys(symbols)))
    with pytest.raises(RuntimeError, match="KERNEL_ALIGN_FORCE_ASCEND=1"):
        ascend.SwiGLUAscendOp()


def test_npu_registry_selects_ascend_and_falls_back_without_extension(monkeypatch):
    monkeypatch.setattr(device_ctx, "device_type", "npu")
    symbols = SimpleNamespace(swiglu_forward=object(), swiglu_backward=object())
    monkeypatch.setattr(ascend, "_C_npu", symbols)
    assert isinstance(KernelRegistry().get_op("swiglu"), ascend.SwiGLUAscendOp)
    assert isinstance(KernelRegistry().get_op("swiglu", device="cpu"), NativeSwiGLUOp)
    monkeypatch.setattr(ascend, "_C_npu", None)
    assert isinstance(KernelRegistry().get_op("swiglu"), NativeSwiGLUOp)


def test_ascend_candidate_is_exposed_to_accuracy_harness():
    assert OP_SPECS["swiglu"].candidate_paths["ascend"] == OpBackend.ASCEND_SWIGLU.value


@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
def test_ascend_wrapper_rejects_cpu_inputs(monkeypatch, method):
    monkeypatch.setattr(
        ascend, "_C_npu", SimpleNamespace(swiglu_forward=object(), swiglu_backward=object())
    )
    with pytest.raises(RuntimeError, match="requires NPU tensors"):
        getattr(ascend.SwiGLUAscendOp(), method)(torch.ones(3), torch.ones(3))


@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_autograd_wrapper_contiguity_and_gradient_routing(monkeypatch, needs_grad, fp32_output):
    """Exercise the Python autograd boundary; this does not emulate Ascend C."""
    calls = []

    def forward(gate, up):
        assert gate.is_contiguous() and up.is_contiguous()
        calls.append(("forward", gate.dtype))
        return NativeSwiGLUOp()(gate, up)

    def backward(grad_out, gate, up):
        assert all(x.is_contiguous() for x in (grad_out, gate, up))
        calls.append(("backward", grad_out.dtype))
        g, u, dy = gate.float(), up.float(), grad_out.float()
        s = torch.sigmoid(g)
        return ((dy * u) * (s * (1 + g * (1 - s)))).to(gate.dtype), (dy * (g * s)).to(up.dtype)

    monkeypatch.setattr(
        ascend, "_C_npu", SimpleNamespace(swiglu_forward=forward, swiglu_backward=backward)
    )
    # CPU-only test of the wrapper; real device guards are tested separately.
    monkeypatch.setattr(ascend, "_validate_inputs", lambda gate, up: None)
    gate = torch.randn(7, 5, dtype=torch.bfloat16).t().requires_grad_(needs_grad[0])
    up = torch.randn(7, 5, dtype=torch.bfloat16).t().requires_grad_(needs_grad[1])
    grad_out = torch.randn(7, 5).t()
    method = "forward_fp32" if fp32_output else "forward"
    result = getattr(ascend.SwiGLUAscendOp(), method)(gate, up)
    result.backward(grad_out.to(result.dtype))

    ref_gate = gate.detach().clone().requires_grad_(needs_grad[0])
    ref_up = up.detach().clone().requires_grad_(needs_grad[1])
    ref = getattr(NativeSwiGLUOp(), method)(ref_gate, ref_up)
    ref.backward(grad_out.to(ref.dtype))
    torch.testing.assert_close(result, ref, rtol=0, atol=0)
    for actual, expected in ((gate.grad, ref_gate.grad), (up.grad, ref_up.grad)):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    expected_dtype = torch.float32 if fp32_output else torch.bfloat16
    assert calls == [("forward", expected_dtype), ("backward", expected_dtype)]


@pytest.fixture
def npu_op():
    return ascend.SwiGLUAscendOp()


def _ascend_dtype_tolerance(dtype):
    return {
        torch.float32: (1e-5, 1e-5),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (2e-2, 1.6e-2),
    }[dtype]


@requires_npu
@pytest.mark.parametrize("dtype", _ASCEND_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0, 17),
        (1,),
        (7,),
        (15,),
        (31,),
        (33,),
        (2047,),
        (2048,),
        (2049,),
        (2, 3, 65),
        (2, 12288),
        (65539,),
    ],
)
def test_npu_forward_backward_and_fp32_path(npu_op, dtype, shape):
    generator = torch.Generator().manual_seed(29)
    gate_cpu = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    up_cpu = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    grad_cpu = torch.randn(shape, generator=generator)
    gate = gate_cpu.detach().to("npu").requires_grad_()
    up = up_cpu.detach().to("npu").requires_grad_()
    for method in ("forward", "forward_fp32"):
        gate.grad = up.grad = gate_cpu.grad = up_cpu.grad = None
        out = getattr(npu_op, method)(gate, up)
        ref = getattr(NativeSwiGLUOp(), method)(gate_cpu, up_cpu)
        assert out.device == gate.device and out.shape == gate.shape
        assert out.dtype == ref.dtype
        out.backward(grad_cpu.to(device="npu", dtype=out.dtype))
        ref.backward(grad_cpu.to(ref.dtype))
        rtol, atol = _ascend_dtype_tolerance(out.dtype)
        torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)
        rtol, atol = _ascend_dtype_tolerance(dtype)
        torch.testing.assert_close(gate.grad.cpu(), gate_cpu.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(up.grad.cpu(), up_cpu.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(gate.detach().cpu(), gate_cpu.detach(), rtol=0, atol=0)
    torch.testing.assert_close(up.detach().cpu(), up_cpu.detach(), rtol=0, atol=0)


@requires_npu
@pytest.mark.parametrize("dtype", _ASCEND_DTYPES)
def test_npu_strided_inputs_and_upstream_gradient(npu_op, dtype):
    # Chunked gate/up projections and a transposed upstream gradient.
    packed = torch.randn(5, 66, dtype=dtype, device="npu", requires_grad=True)
    gate, up = packed.chunk(2, dim=-1)
    assert not gate.is_contiguous() and not up.is_contiguous()
    dy = torch.randn(33, 5, dtype=dtype, device="npu").t()
    result = npu_op(gate, up)
    result.backward(dy)
    ref_packed = packed.detach().cpu().requires_grad_()
    ref = NativeSwiGLUOp()(*ref_packed.chunk(2, dim=-1))
    ref.backward(dy.cpu())
    rtol, atol = _ascend_dtype_tolerance(dtype)
    torch.testing.assert_close(result.cpu(), ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(packed.grad.cpu(), ref_packed.grad, rtol=rtol, atol=atol)


@requires_npu
@pytest.mark.parametrize("dtype", _ASCEND_DTYPES)
def test_npu_batch_position_and_repeat_invariance(npu_op, dtype):
    gate = torch.randn(33, dtype=dtype, device="npu")
    up = torch.randn_like(gate)
    dy = torch.randn_like(gate)

    def evaluate(g, u, grad):
        g = g.detach().requires_grad_()
        u = u.detach().requires_grad_()
        out = npu_op(g, u)
        return (out.detach(), *torch.autograd.grad(out, (g, u), grad))

    expected = evaluate(gate, up, dy)
    for rows, position in ((1, 0), (7, 3), (65, 64), (65, 64)):
        g = torch.randn(rows, 33, dtype=dtype, device="npu")
        u, grad = torch.randn_like(g), torch.randn_like(g)
        g[position], u[position], grad[position] = gate, up, dy
        actual = evaluate(g, u, grad)
        for a, e in zip(actual, expected):
            assert torch.equal(a[position], e)


@requires_npu
def test_npu_input_validation_and_native_boundary(npu_op):
    x = torch.ones(7, device="npu")
    with pytest.raises(ValueError, match="share shape"):
        npu_op(x, x[:3])
    with pytest.raises(TypeError, match="share dtype"):
        npu_op(x, x.half())
    with pytest.raises(TypeError, match="fp16, bf16, or fp32"):
        npu_op(x.int(), x.int())
    with pytest.raises(RuntimeError, match="contiguous"):
        ascend._C_npu.swiglu_forward(x[::2], x[::2])
    with pytest.raises(RuntimeError, match="share shape"):
        ascend._C_npu.swiglu_backward(x[:3], x, x)
    with pytest.raises(RuntimeError, match="share dtype"):
        ascend._C_npu.swiglu_backward(x.half(), x, x)


@requires_npu
def test_npu_current_stream_ordering(npu_op):
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        gate = torch.randn(4099, device="npu").mul_(2).requires_grad_()
        up = torch.randn_like(gate).requires_grad_()
        dy = torch.randn_like(gate)
        actual = npu_op(gate, up)
        grads = torch.autograd.grad(actual, (gate, up), dy)
        expected = NativeSwiGLUOp()(gate, up)
        ref_grads = torch.autograd.grad(expected, (gate, up), dy)
    stream.synchronize()
    for actual_tensor, expected_tensor in zip((actual, *grads), (expected, *ref_grads)):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)


@requires_npu
def test_npu_device_guard_and_cross_device_rejection(npu_op):
    if torch.npu.device_count() < 2:
        pytest.skip("Two NPUs required")
    with torch.npu.device(0):
        gate = torch.randn(33, device="npu:1", requires_grad=True)
        up = torch.randn_like(gate, requires_grad=True)
        out = npu_op(gate, up)
        out.sum().backward()
        assert torch.npu.current_device() == 0
        assert out.device == gate.device == gate.grad.device == up.grad.device
        torch.testing.assert_close(out, NativeSwiGLUOp()(gate, up), rtol=1e-5, atol=1e-5)
        with pytest.raises(RuntimeError, match="same NPU device"):
            npu_op(gate, up.to("npu:0"))
        with pytest.raises(RuntimeError, match="same NPU device"):
            ascend._C_npu.swiglu_forward(gate, up.to("npu:0"))
