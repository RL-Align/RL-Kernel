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

from rl_engine.backends.ascend.activation import swiglu as ascend
from rl_engine.contracts.operators.activation import SwiGLUPackedOrder
from rl_engine.platforms.device import _npu_available, device_ctx
from rl_engine.reference.activation.swiglu import NativeSiLUOp, NativeSwiGLUOp
from rl_engine.runtime.registry import KernelRegistry, OpBackend, kernel_registry
from rl_engine.validation.operators.op_checks import run_operator_suite
from rl_engine.validation.operators.operator_specs import (
    OP_SPECS,
    make_candidate,
    make_operator_case,
    operator_names,
)

try:
    from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp, TritonSwiGLUOp

    _HAS_TRITON_ACTIVATION = True
except ImportError:  # pragma: no cover - triton may be missing in CPU-only builds.
    _HAS_TRITON_ACTIVATION = False
    TritonSiLUOp = None  # type: ignore[misc, assignment]
    TritonSwiGLUOp = None  # type: ignore[misc, assignment]

try:
    from rl_engine.backends.cuda.activation.swiglu import SiLUCudaOp, SwiGLUCudaOp
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

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
    """Check FP32 arithmetic and the requested output cast against an explicit formula."""
    gate = torch.linspace(-4.0, 4.0, 48, dtype=dtype).reshape(2, 3, 8)
    up = torch.linspace(0.5, 2.0, 48, dtype=dtype).reshape(2, 3, 8)

    fp32_reference = gate.float() * torch.sigmoid(gate.float()) * up.float()
    result = NativeSwiGLUOp().forward(gate=gate, up=up)

    assert result.dtype == dtype
    assert torch.equal(result, fp32_reference.to(dtype))
    assert torch.equal(NativeSwiGLUOp().forward_fp32(gate=gate, up=up), fp32_reference)


@pytest.mark.parametrize(
    "op_class,device",
    [
        pytest.param(NativeSwiGLUOp, "cpu", id="pytorch"),
        pytest.param(
            TritonSwiGLUOp,
            "cuda",
            id="triton",
            marks=[requires_cuda, requires_triton_activation],
        ),
    ],
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("method", ("__call__", "forward", "forward_fp32"))
def test_swiglu_requires_named_operands(op_class, device, dtype, method):
    """Names determine gate/up roles; keyword order is harmless and positional calls fail."""
    gate = _rand((2, 3, 17), seed=41, dtype=dtype, device=device).requires_grad_(True)
    up = _rand((2, 3, 17), seed=42, dtype=dtype, device=device).requires_grad_(True)
    fn = getattr(op_class(), method)
    expected = fn(gate=gate, up=up)
    upstream = _rand(expected.shape, seed=43, dtype=expected.dtype, device=device)
    expected_grads = torch.autograd.grad(expected, (gate, up), upstream)

    actual = fn(up=up, gate=gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(actual, (gate, up), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    with pytest.raises(TypeError, match="positional"):
        fn(gate, up)
    with pytest.raises(TypeError, match="positional"):
        fn(up, gate)
    with pytest.raises(TypeError, match="positional"):
        fn(gate, up=up)


@pytest.mark.parametrize("method", ("__call__", "forward", "forward_fp32"))
def test_swiglu_rejects_conflicting_or_incomplete_keyword_inputs(method):
    """Require one complete input form and reject the superseded packed names."""
    fn = getattr(NativeSwiGLUOp(), method)
    x = torch.ones(2, 8)
    for old_name in ("input", "projected"):
        with pytest.raises(TypeError, match=old_name):
            fn(**{old_name: x}, packed_order=SwiGLUPackedOrder.GATE_UP)
    with pytest.raises(TypeError, match="packed_input cannot"):
        fn(packed_input=x, gate=x, up=x, packed_order=SwiGLUPackedOrder.GATE_UP)
    with pytest.raises(TypeError, match="both gate and up"):
        fn(up=x)
    with pytest.raises(TypeError, match="both gate and up"):
        fn(gate=x, packed_order=SwiGLUPackedOrder.GATE_UP)
    with pytest.raises(ValueError, match="packed_order"):
        fn(gate=x, up=x, packed_order=SwiGLUPackedOrder.GATE_UP)


def test_native_swiglu_rejects_mismatched_shape():
    """Reject separate operands whose shapes differ."""
    gate = torch.randn(2, 3)
    up = torch.randn(2, 4)

    with pytest.raises(ValueError, match="share shape"):
        NativeSwiGLUOp().forward(gate=gate, up=up)


def test_native_activation_rejects_invalid_dtypes():
    """Reject integer SiLU inputs and mismatched SwiGLU operand dtypes."""
    with pytest.raises(TypeError, match="dtype"):
        NativeSiLUOp().forward(torch.ones(8, dtype=torch.int32))

    with pytest.raises(TypeError, match="share dtype"):
        NativeSwiGLUOp().forward(
            gate=torch.ones(8, dtype=torch.float16),
            up=torch.ones(8, dtype=torch.bfloat16),
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
    """Require FP32 outputs to remain bitwise equal when evaluated as batch slices."""
    op = NativeSwiGLUOp()
    gate = _rand((8, 32, _INTERMEDIATE), seed=3)
    up = _rand((8, 32, _INTERMEDIATE), seed=4)
    full = op.forward_fp32(gate=gate, up=up)
    assert torch.equal(op.forward_fp32(gate=gate[:1], up=up[:1]), full[:1])
    assert torch.equal(op.forward_fp32(gate=gate[3:5], up=up[3:5]), full[3:5])


def test_silu_batch_invariance_with_padding():
    """Padding extra rows must not perturb the real rows (bitwise)."""
    op = NativeSiLUOp()
    x = _rand((4, _INTERMEDIATE), seed=5)
    padded = torch.cat([x, _rand((6, _INTERMEDIATE), seed=99)], dim=0)
    assert torch.equal(op.forward_fp32(padded)[:4], op.forward_fp32(x))


def test_swiglu_batch_invariance_with_padding():
    """Adding unrelated rows must preserve each original row's FP32 output bits."""
    op = NativeSwiGLUOp()
    gate = _rand((4, _INTERMEDIATE), seed=6)
    up = _rand((4, _INTERMEDIATE), seed=7)
    pad_gate = torch.cat([gate, _rand((6, _INTERMEDIATE), seed=98)], dim=0)
    pad_up = torch.cat([up, _rand((6, _INTERMEDIATE), seed=97)], dim=0)
    assert torch.equal(
        op.forward_fp32(gate=pad_gate, up=pad_up)[:4], op.forward_fp32(gate=gate, up=up)
    )


# Purity -- inputs not mutated in-place
def test_silu_inputs_not_mutated():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=8)
    xc = x.clone()
    op.forward(x)
    op.forward_fp32(x)
    assert torch.equal(x, xc)


def test_swiglu_inputs_not_mutated():
    """Both public output modes must leave the gate and up tensors unchanged."""
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=9)
    up = _rand((2, _INTERMEDIATE), seed=10)
    gc, uc = gate.clone(), up.clone()
    op.forward(gate=gate, up=up)
    op.forward_fp32(gate=gate, up=up)
    assert torch.equal(gate, gc) and torch.equal(up, uc)


# Gradient flows (fp32 autograd = backward golden source)
def test_silu_gradient_flows():
    op = NativeSiLUOp()
    x = _rand((2, _INTERMEDIATE), seed=11).requires_grad_(True)
    op.forward_fp32(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_swiglu_gradient_flows():
    """Require finite gradients for both separate operands through the FP32 path."""
    op = NativeSwiGLUOp()
    gate = _rand((2, _INTERMEDIATE), seed=12).requires_grad_(True)
    up = _rand((2, _INTERMEDIATE), seed=13).requires_grad_(True)
    op.forward_fp32(gate=gate, up=up).sum().backward()
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
    out_full = op.forward_fp32(gate=gate_full, up=up_full)

    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)

    grad_gate_full_sliced = gate_full.grad[:1].clone()
    grad_up_full_sliced = up_full.grad[:1].clone()

    gate_slice = _rand((8, 32, _INTERMEDIATE), seed=1)[:1].detach().requires_grad_(True)
    up_slice = _rand((8, 32, _INTERMEDIATE), seed=2)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(gate=gate_slice, up=up_slice)

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
    """Compare GPU outputs and both gradients with the independent FP32 reference."""
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
    y_ref = native.forward_fp32(gate=gate_ref, up=up_ref)
    y_ref.backward(dy_ref)

    gate_gpu = gate_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    up_gpu = up_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    y_gpu = cand.forward(gate=gate_gpu, up=up_gpu)
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
    """Require GPU output bits to be unchanged when evaluating batch slices."""
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    op = _make_swiglu_op(impl)
    gate = _rand((8, 32, 256), seed=6, dtype=torch.bfloat16, device="cuda")
    up = _rand((8, 32, 256), seed=7, dtype=torch.bfloat16, device="cuda")
    full = op.forward(gate=gate, up=up)
    assert torch.equal(op.forward(gate=gate[:1], up=up[:1]), full[:1])
    assert torch.equal(op.forward(gate=gate[3:5], up=up[3:5]), full[3:5])


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
    """Reject differently shaped gate/up tensors before launching the GPU kernel."""
    if impl == "cuda" and not _HAS_CUDA_ACTIVATION:
        pytest.skip("CUDA SwiGLU extension is not available")

    op = _make_swiglu_op(impl)
    gate = torch.randn(2, 3, device="cuda")
    up = torch.randn(2, 4, device="cuda")
    with pytest.raises(ValueError, match="share shape"):
        op.forward(gate=gate, up=up)


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
    """Require both operand gradients to match bitwise across full and sliced batches."""
    op = _make_swiglu_op(impl)
    gate = _rand((6, 4, 64), seed=12, dtype=torch.bfloat16, device="cuda")
    up = _rand((6, 4, 64), seed=13, dtype=torch.bfloat16, device="cuda")
    dy = _rand(gate.shape, seed=14, dtype=torch.bfloat16, device="cuda")

    gate_full = gate.detach().clone().requires_grad_(True)
    up_full = up.detach().clone().requires_grad_(True)
    op.forward(gate=gate_full, up=up_full).backward(dy)
    full_d_gate = gate_full.grad[2:4].clone()
    full_d_up = up_full.grad[2:4].clone()

    gate_slice = gate[2:4].detach().clone().requires_grad_(True)
    up_slice = up[2:4].detach().clone().requires_grad_(True)
    op.forward(gate=gate_slice, up=up_slice).backward(dy[2:4])
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
    """Unrelated padded rows must not change real rows' outputs or either gradient."""
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

    y_padded = op.forward(gate=gate_padded, up=up_padded)
    y_padded.backward(dy_padded)

    gate_real = gate.detach().clone().requires_grad_(True)
    up_real = up.detach().clone().requires_grad_(True)
    y_real = op.forward(gate=gate_real, up=up_real)
    y_real.backward(dy)
    assert torch.equal(y_padded[:4], y_real)
    assert torch.equal(gate_padded.grad[:4], gate_real.grad)
    assert torch.equal(up_padded.grad[:4], up_real.grad)


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_deterministic_repeat(impl):
    """Repeated GPU evaluations must reproduce output and gradient bits exactly."""
    op = _make_swiglu_op(impl)
    gate = _rand((16, 256), seed=25, dtype=torch.bfloat16, device="cuda")
    up = _rand((16, 256), seed=26, dtype=torch.bfloat16, device="cuda")
    dy = _rand(gate.shape, seed=27, dtype=torch.bfloat16, device="cuda")

    def _run():
        """Evaluate one fresh autograd graph and return detached output and gradients."""
        gate_r = gate.detach().clone().requires_grad_(True)
        up_r = up.detach().clone().requires_grad_(True)
        y = op.forward(gate=gate_r, up=up_r)
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
    """Exercise forward and backward on transposed tensors and empty batches."""
    silu = _make_silu_op(impl)
    swiglu = _make_swiglu_op(impl)

    x = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16).T.requires_grad_(True)
    assert not x.is_contiguous()
    silu.forward(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    gate = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16).T.requires_grad_(True)
    up = torch.randn(5, 3, device="cuda", dtype=torch.bfloat16).T.requires_grad_(True)
    assert not gate.is_contiguous() and not up.is_contiguous()
    swiglu.forward(gate=gate, up=up).sum().backward()
    assert gate.grad is not None and up.grad is not None

    empty = torch.empty((0, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    silu_empty = silu.forward(empty)
    silu_empty.sum().backward()
    assert silu_empty.shape == empty.shape and empty.grad.shape == empty.shape

    empty_gate = empty.detach().clone().requires_grad_(True)
    empty_up = empty.detach().clone().requires_grad_(True)
    swiglu_empty = swiglu.forward(gate=empty_gate, up=empty_up)
    swiglu_empty.sum().backward()
    assert swiglu_empty.shape == empty_gate.shape
    assert empty_gate.grad.shape == empty_gate.shape and empty_up.grad.shape == empty_up.shape


@requires_cuda
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_activation_rejects_invalid_dtypes(impl):
    """Reject unsupported SiLU dtypes and mismatched GPU SwiGLU operand dtypes."""
    silu = _make_silu_op(impl)
    swiglu = _make_swiglu_op(impl)
    with pytest.raises(TypeError, match="dtype"):
        silu.forward(torch.ones(8, device="cuda", dtype=torch.int32))
    with pytest.raises(TypeError, match="share dtype"):
        swiglu.forward(
            gate=torch.ones(8, device="cuda", dtype=torch.float16),
            up=torch.ones(8, device="cuda", dtype=torch.bfloat16),
        )


@requires_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("impl", _silu_impls())
def test_cuda_triton_swiglu_rejects_cross_device_inputs(impl):
    """Require gate and up to share a device even when both devices support CUDA."""
    op = _make_swiglu_op(impl)
    gate = torch.ones(8, device="cuda:0")
    up = torch.ones(8, device="cuda:1")
    with pytest.raises(RuntimeError, match=r"same .*device"):
        op.forward(gate=gate, up=up)


@requires_cuda
@requires_triton_activation
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("method", ("forward", "forward_fp32"))
@pytest.mark.parametrize("strided", (False, True), ids=("contiguous", "strided"))
def test_triton_swiglu_noncurrent_device(dtype, method, strided):
    """Issue #664: public train/infer paths preserve the caller's device and results."""
    device = torch.device("cuda:1")
    stream = torch.cuda.Stream(device=device)
    op = getattr(TritonSwiGLUOp(), method)
    with torch.cuda.device(0), torch.cuda.stream(stream):
        gate = _rand((3, 1025), seed=664, dtype=dtype, device=device)
        up = _rand(gate.shape, seed=665, dtype=dtype, device=device)
        if strided:
            gate, up = gate.T, up.T
        gate.requires_grad_(True)
        up.requires_grad_(True)
        upstream = _rand(gate.shape, seed=666, dtype=dtype, device=device)
        if method == "forward_fp32":
            upstream = upstream.float()

        # Establish the same operator's result with the input device current.
        expected = op(gate=gate, up=up)
        expected_grads = torch.autograd.grad(expected, (gate, up), upstream)
        gate_ref = gate.detach().float().requires_grad_(True)
        up_ref = up.detach().float().requires_grad_(True)
        native = NativeSwiGLUOp().forward_fp32(gate=gate_ref, up=up_ref)
        native_grads = torch.autograd.grad(native, (gate_ref, up_ref), upstream.float())

        torch.cuda.set_device(0)
        actual = op(gate=gate, up=up)
        assert torch.cuda.current_device() == 0
        actual_grads = torch.autograd.grad(actual, (gate, up), upstream)
        assert torch.cuda.current_device() == 0
        with torch.inference_mode():
            inference = op(gate=gate, up=up)
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(device) == stream
    stream.synchronize()

    assert torch.equal(actual, inference)
    atol, rtol = _dtype_tolerance(dtype)
    for result, matched_device, reference in zip(
        (actual, *actual_grads), (expected, *expected_grads), (native, *native_grads), strict=True
    ):
        assert result.device == device
        assert torch.equal(result, matched_device)
        torch.testing.assert_close(result.float(), reference, atol=atol, rtol=rtol)


@requires_cuda
@requires_triton_activation
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("target", (0, 1))
def test_triton_swiglu_launchers_use_input_device_and_stream(monkeypatch, dtype, target):
    """Direct backward must also be guarded, without relying on autograd's device switch."""
    from rl_engine.backends.shared.triton.activation import swiglu

    device = torch.device("cuda", target)
    caller = 1 - target
    stream = torch.cuda.Stream(device=device)
    launches = []

    def record_launch(kernel):
        original_run = kernel.run

        def run(*args, **kwargs):
            launches.append((torch.cuda.current_device(), torch.cuda.current_stream()))
            return original_run(*args, **kwargs)

        monkeypatch.setattr(kernel, "run", run)

    for kernel in (
        swiglu._swiglu_fwd_kernel,
        swiglu._swiglu_bwd_kernel,
        swiglu._swiglu_dgate_kernel,
    ):
        record_launch(kernel)

    with torch.cuda.device(caller), torch.cuda.stream(stream):
        gate = _rand((3, 1025), seed=664, dtype=dtype, device=device)
        up = _rand(gate.shape, seed=665, dtype=dtype, device=device)
        dy = _rand(gate.shape, seed=666, dtype=dtype, device=device)
        expected = swiglu._launch_swiglu_fwd(gate, up)
        expected_grads = swiglu._launch_swiglu_bwd(dy, gate, up)
        launches.clear()

        torch.cuda.set_device(caller)
        actual = swiglu._launch_swiglu_fwd(gate, up)
        assert torch.cuda.current_device() == caller
        gradients = swiglu._launch_swiglu_bwd(dy, gate, up)
        assert torch.cuda.current_device() == caller
        assert torch.cuda.current_stream(device) == stream
    stream.synchronize()

    assert launches == [(target, stream)] * 3
    for result, reference in zip((actual, *gradients), (expected, *expected_grads), strict=True):
        assert result.device == device
        assert torch.equal(result, reference)


@requires_cuda
@requires_triton_activation
def test_triton_activation_kernels_index_past_int32_numel():
    """Issue #640: an int32 pid * BLOCK wrapped negative past 2**31 elements,
    passed the mask and read/wrote outside the buffers."""
    from rl_engine.backends.shared.triton.activation import swiglu as triton_swiglu

    block = triton_swiglu._BLOCK
    # A full block starting at element 2**31 plus a one-element partial block.
    numel = 2**31 + block + 1
    # Peak is the backward: shared bf16 input, bf16 d_gate/d_up and fp32 silu_grad.
    peak_bytes = numel * (2 + 2 + 2 + 4)
    free, _ = torch.cuda.mem_get_info()
    if free < peak_bytes * 1.1:
        pytest.skip(f"needs ~{peak_bytes * 1.1 / 1e9:.0f} GB of free GPU memory")

    gen = torch.Generator(device="cuda").manual_seed(640)
    # The kernels are elementwise, so one buffer can feed every input role.
    x = torch.randn(numel, generator=gen, device="cuda", dtype=torch.bfloat16)
    tail = slice(2**31 - block, None)
    x_tail = x[tail].clone()
    launches = {
        "silu_fwd": lambda t: (triton_swiglu._launch_silu_fwd(t),),
        "silu_bwd": lambda t: (triton_swiglu._launch_silu_bwd(t, t),),
        "swiglu_fwd": lambda t: (triton_swiglu._launch_swiglu_fwd(t, t),),
        "swiglu_bwd": lambda t: triton_swiglu._launch_swiglu_bwd(t, t, t),
    }
    for name, launch in launches.items():
        full = launch(x)
        for actual, expected in zip(full, launch(x_tail), strict=True):
            assert torch.equal(actual[tail], expected), name
        # `actual` outlives the loop and would keep a full-size output alive.
        del full, actual, expected


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


@pytest.mark.parametrize("packed_order", (None, "gate_up", "up_gate"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_swiglu_pytorch_candidate_suite_passes_issue_108_helper(packed_order, dtype):
    """Check reference accuracy and reported gradient names for both input forms."""
    args = _spec_args("swiglu", candidate="pytorch", packed_order=packed_order)
    report = run_operator_suite(
        "swiglu",
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, dtype, torch.device("cpu"))],
        check_grad=True,
    )
    assert report.passed
    gradient_names = ("gate", "up") if packed_order is None else ("packed_input",)
    checks = report.candidates[0].cases[0].outputs
    assert [check.message for check in checks[1:]] == [
        f"gradient:{name}" for name in gradient_names
    ]


@requires_cuda
@requires_triton_activation
@pytest.mark.parametrize("order", tuple(SwiGLUPackedOrder))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_swiglu_packed_triton_issue_108_harness(order, dtype):
    """Validate packed Triton outputs and input gradients through the shared harness."""
    args = _spec_args("swiglu", candidate="triton", packed_order=order.value)
    report = run_operator_suite(
        "swiglu",
        candidates=[make_candidate(args)],
        cases=[make_operator_case(args, dtype, torch.device("cuda"))],
        check_grad=True,
    )
    assert report.passed, report.candidates[0].cases[0].outputs


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
    """An available Ascend extension must still reject CPU operands in both output modes."""
    monkeypatch.setattr(
        ascend, "_C_npu", SimpleNamespace(swiglu_forward=object(), swiglu_backward=object())
    )
    with pytest.raises(RuntimeError, match="requires NPU tensors"):
        getattr(ascend.SwiGLUAscendOp(), method)(gate=torch.ones(3), up=torch.ones(3))


@pytest.mark.parametrize("needs_grad", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_autograd_wrapper_contiguity_and_gradient_routing(monkeypatch, needs_grad, fp32_output):
    """Exercise the Python autograd boundary; this does not emulate Ascend C."""
    calls = []

    def forward(gate, up):
        """Record the wrapper's dtype and verify contiguous inputs at the mocked boundary."""
        assert gate.is_contiguous() and up.is_contiguous()
        calls.append(("forward", gate.dtype))
        return NativeSwiGLUOp()(gate=gate, up=up)

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
    result = getattr(ascend.SwiGLUAscendOp(), method)(gate=gate, up=up)
    result.backward(grad_out.to(result.dtype))

    # The wrapper hands the kernel contiguous copies, so the reference must see the
    # same memory layout: CPU sigmoid's SIMD body and scalar tail differ by 1 ulp,
    # and a transposed view routes different elements through the tail.
    ref_gate = gate.detach().contiguous().requires_grad_(needs_grad[0])
    ref_up = up.detach().contiguous().requires_grad_(needs_grad[1])
    ref = getattr(NativeSwiGLUOp(), method)(gate=ref_gate, up=ref_up)
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
    """Check both NPU output modes and operand gradients against the native reference."""
    generator = torch.Generator().manual_seed(29)
    gate_cpu = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    up_cpu = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    grad_cpu = torch.randn(shape, generator=generator)
    gate = gate_cpu.detach().to("npu").requires_grad_()
    up = up_cpu.detach().to("npu").requires_grad_()
    for method in ("forward", "forward_fp32"):
        gate.grad = up.grad = gate_cpu.grad = up_cpu.grad = None
        out = getattr(npu_op, method)(gate=gate, up=up)
        ref = getattr(NativeSwiGLUOp(), method)(gate=gate_cpu, up=up_cpu)
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
    """Check chunked operands and a strided upstream gradient against native autograd."""
    # Chunked gate/up projections and a transposed upstream gradient.
    packed = torch.randn(5, 66, dtype=dtype, device="npu", requires_grad=True)
    gate, up = packed.chunk(2, dim=-1)
    assert not gate.is_contiguous() and not up.is_contiguous()
    dy = torch.randn(33, 5, dtype=dtype, device="npu").t()
    result = npu_op(gate=gate, up=up)
    result.backward(dy)
    ref_packed = packed.detach().cpu().requires_grad_()
    ref_gate, ref_up = ref_packed.chunk(2, dim=-1)
    ref = NativeSwiGLUOp()(gate=ref_gate, up=ref_up)
    ref.backward(dy.cpu())
    rtol, atol = _ascend_dtype_tolerance(dtype)
    torch.testing.assert_close(result.cpu(), ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(packed.grad.cpu(), ref_packed.grad, rtol=rtol, atol=atol)


@requires_npu
@pytest.mark.parametrize("dtype", _ASCEND_DTYPES)
def test_npu_batch_position_and_repeat_invariance(npu_op, dtype):
    """Moving a row within larger batches must preserve its NPU output and gradient bits."""
    gate = torch.randn(33, dtype=dtype, device="npu")
    up = torch.randn_like(gate)
    dy = torch.randn_like(gate)

    def evaluate(g, u, grad):
        """Evaluate a fresh graph and return its output and both operand gradients."""
        g = g.detach().requires_grad_()
        u = u.detach().requires_grad_()
        out = npu_op(gate=g, up=u)
        return (out.detach(), *torch.autograd.grad(out, (g, u), grad))

    expected = evaluate(gate, up, dy)
    for rows, position in ((1, 0), (7, 3), (65, 64), (65, 64)):
        g = torch.randn(rows, 33, dtype=dtype, device="npu")
        u, grad = torch.randn_like(g), torch.randn_like(g)
        g[position], u[position], grad[position] = gate, up, dy
        actual = evaluate(g, u, grad)
        for a, e in zip(actual, expected, strict=True):
            assert torch.equal(a[position], e)


@requires_npu
def test_npu_input_validation_and_native_boundary(npu_op):
    """Exercise shape/dtype validation and the Ascend extension's contiguous-input contract."""
    x = torch.ones(7, device="npu")
    with pytest.raises(ValueError, match="share shape"):
        npu_op(gate=x, up=x[:3])
    with pytest.raises(TypeError, match="share dtype"):
        npu_op(gate=x, up=x.half())
    with pytest.raises(TypeError, match="fp16, bf16, or fp32"):
        npu_op(gate=x.int(), up=x.int())
    with pytest.raises(RuntimeError, match="contiguous"):
        ascend._C_npu.swiglu_forward(x[::2], x[::2])
    with pytest.raises(RuntimeError, match="share shape"):
        ascend._C_npu.swiglu_backward(x[:3], x, x)
    with pytest.raises(RuntimeError, match="share dtype"):
        ascend._C_npu.swiglu_backward(x.half(), x, x)


@requires_npu
def test_npu_current_stream_ordering(npu_op):
    """Compare outputs and gradients produced on a nondefault NPU stream with native."""
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        gate = torch.randn(4099, device="npu").mul_(2).requires_grad_()
        up = torch.randn_like(gate).requires_grad_()
        dy = torch.randn_like(gate)
        actual = npu_op(gate=gate, up=up)
        grads = torch.autograd.grad(actual, (gate, up), dy)
        expected = NativeSwiGLUOp()(gate=gate, up=up)
        ref_grads = torch.autograd.grad(expected, (gate, up), dy)
    stream.synchronize()
    for actual_tensor, expected_tensor in zip(
        (actual, *grads), (expected, *ref_grads), strict=True
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)


@requires_npu
def test_npu_device_guard_and_cross_device_rejection(npu_op):
    """Run on a noncurrent NPU, restore the caller's device, and reject mixed devices."""
    if torch.npu.device_count() < 2:
        pytest.skip("Two NPUs required")
    with torch.npu.device(0):
        gate = torch.randn(33, device="npu:1", requires_grad=True)
        up = torch.randn_like(gate, requires_grad=True)
        out = npu_op(gate=gate, up=up)
        out.sum().backward()
        assert torch.npu.current_device() == 0
        assert out.device == gate.device == gate.grad.device == up.grad.device
        torch.testing.assert_close(out, NativeSwiGLUOp()(gate=gate, up=up), rtol=1e-5, atol=1e-5)
        with pytest.raises(RuntimeError, match="same NPU device"):
            npu_op(gate=gate, up=up.to("npu:0"))
        with pytest.raises(RuntimeError, match="same NPU device"):
            ascend._C_npu.swiglu_forward(gate, up.to("npu:0"))
