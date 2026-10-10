# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU contract/orchestration checks and separate full-width SM90 checks."""

from contextlib import nullcontext

import pytest
import torch

import rl_engine.kernels.ops.h3_ffn_down as down
from rl_engine.kernels.ops.canonical_backward import canonical_backward_session
from rl_engine.kernels.ops.h3_down_validation import (
    comparison,
    logical_keys,
    raw_equal,
    reference_projection,
    training_projection,
)


def test_dimensions_and_checkpoint_are_pinned():
    assert (down.H3_FFN_DIM, down.H3_HIDDEN_DIM, down.H3_MAX_ROWS) == (14336, 5376, 32768)
    assert down.H3_CHECKPOINT_REVISION == "42ed227ee7df40d41602854ae760620d6eb651fe"


@pytest.mark.parametrize("shape", [(14336,), (2, 7), (1, 1, 1, 14336), (0, 14336), (32769, 14336)])
def test_rejects_unsupported_shapes_before_dispatch(shape):
    with pytest.raises(ValueError):
        down.h3_ffn_down_gemm(
            torch.empty(shape, dtype=torch.bfloat16, device="meta"),
            torch.empty((5376, 14336), dtype=torch.bfloat16, device="meta"),
        )


def test_native_weight_layout_and_dtype_are_required():
    x = torch.empty((1, 14336), dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="native shape"):
        down.h3_ffn_down_gemm(x, torch.empty((14336, 5376), dtype=torch.bfloat16, device="meta"))
    with pytest.raises(TypeError, match="bfloat16"):
        down.h3_ffn_down_gemm(
            x.float(), torch.empty((5376, 14336), dtype=torch.bfloat16, device="meta")
        )


def test_cpu_cannot_silently_execute_cuda_profile():
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        down.h3_ffn_down_gemm(
            torch.empty((1, 14336), dtype=torch.bfloat16),
            torch.empty((5376, 14336), dtype=torch.bfloat16),
        )


def test_missing_extension_and_wrong_hardware_fail_closed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(RuntimeError, match="SM90"):
        down._require_native(torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(down.base, "_EXT_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        down._require_native(torch.device("cuda"))


def test_fp32_backend_missing_triton_and_wrong_hardware_fail_closed(monkeypatch):
    from rl_engine.kernels.ops.triton.matmul import det_gemm

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(RuntimeError, match="SM90"):
        down._require_backend(torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(det_gemm, "_TRITON_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        down._require_backend(torch.device("cuda"))


def test_raw_bytes_and_nonfinite_policy():
    positive = torch.tensor([0.0], dtype=torch.bfloat16)
    negative = torch.tensor([-0.0], dtype=torch.bfloat16)
    assert torch.equal(positive, negative)
    assert not raw_equal(positive, negative)
    assert not comparison(positive, negative, "forward_invariance")["passed"]
    nan = torch.tensor([float("nan")], dtype=torch.bfloat16)
    assert raw_equal(nan, nan)
    assert not comparison(nan, nan, "forward_invariance")["passed"]


def test_fp32_reference_keeps_quantized_inputs_and_gradients():
    x = torch.tensor([[0.25, -0.5], [1.0, 0.125]], dtype=torch.bfloat16)
    w = torch.tensor([[0.5, 2.0]], dtype=torch.bfloat16)
    dy = torch.tensor([[0.5], [-0.25]], dtype=torch.bfloat16)
    y, dx, dw = reference_projection(x, w, dy)
    assert y.dtype == dx.dtype == dw.dtype == torch.float32
    torch.testing.assert_close(y.double(), x.double() @ w.double().t(), atol=0, rtol=0)
    torch.testing.assert_close(dx.double(), dy.double() @ w.double(), atol=0, rtol=0)
    torch.testing.assert_close(dw.double(), dy.double().t() @ x.double(), atol=0, rtol=0)


class _CPUOrchestrationOnlyExtension:
    """Surrogate for testing session flow; does NOT emulate CUDA arithmetic."""

    def __init__(self):
        self.weight_reduction_inputs = []

    def det_gemm_fwd_rhs_transposed(self, x, w):
        return (x.float() @ w.float().t()).bfloat16()

    def det_gemm_fwd(self, x, w):
        return (x.float() @ w.float()).bfloat16()

    def det_gemm_db_transposed(self, x, dy):
        self.weight_reduction_inputs.append((x.clone(), dy.clone()))
        return (dy.float().t() @ x.float()).bfloat16()


@pytest.fixture
def cpu_orchestration(monkeypatch):
    extension = _CPUOrchestrationOnlyExtension()
    monkeypatch.setattr(down, "H3_FFN_DIM", 4)
    monkeypatch.setattr(down, "H3_HIDDEN_DIM", 3)
    monkeypatch.setattr(down, "_validate_inputs", lambda x, w: None)
    monkeypatch.setattr(down, "_require_backend", lambda device, backend: extension)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return extension


def test_canonical_chunk_permutation_and_padding_orchestration(cpu_orchestration):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(7, 4, generator=generator).bfloat16()
    w = torch.randn(3, 4, generator=generator).bfloat16()
    dy = torch.randn(7, 3, generator=generator).bfloat16()
    keys = logical_keys(7, "cpu")
    op = down.H3FFNDownGemmOp()
    canonical, _ = training_projection(op, x, w, dy, keys)
    chunked, traces = training_projection(op, x, w, dy, keys, chunk_size=2)
    assert all(raw_equal(a, b) for a, b in zip(canonical, chunked, strict=True))
    assert sum(trace["weight_gradient_executed"] for trace in traces) == 1
    order = torch.tensor([6, 1, 3, 0, 5, 2, 4])
    permuted, _ = training_projection(op, x[order], w, dy[order], keys[order])
    assert raw_equal(canonical[2], permuted[2])
    torch.testing.assert_close(permuted[0], canonical[0][order], atol=0, rtol=0)
    padded_x = torch.cat((x, torch.ones(2, 4).bfloat16()))
    padded_dy = torch.cat((dy, torch.zeros(2, 3).bfloat16()))
    padded_keys = torch.cat((keys, torch.full((2, 2), -1, dtype=torch.int64)))
    padded, _ = training_projection(op, padded_x, w, padded_dy, padded_keys)
    assert raw_equal(padded[2], canonical[2])
    for rows, grads in cpu_orchestration.weight_reduction_inputs:
        assert rows.shape[0] == grads.shape[0] == 32
        assert raw_equal(rows[:7], x)
        assert torch.count_nonzero(rows[7:]) == torch.count_nonzero(grads[7:]) == 0


def test_training_requires_session_and_unique_stable_keys(cpu_orchestration):
    x, w = torch.ones(2, 4).bfloat16(), torch.ones(3, 4).bfloat16().requires_grad_(True)
    op = down.H3FFNDownGemmOp()
    with pytest.raises(RuntimeError, match="active canonical"):
        op(x, w)
    with canonical_backward_session():
        keys = logical_keys(2, "cpu")
        op(x, w, logical_keys=keys)
        with pytest.raises(ValueError, match="unique"):
            op(x, w, logical_keys=keys)
        with pytest.raises(ValueError, match="another weight"):
            op(x, w.detach().clone().requires_grad_(True), logical_keys=keys + 10)
        with pytest.raises(ValueError, match="one parameter_id"):
            op(x, w, logical_keys=keys + 10, parameter_id="other")


def test_keys_are_snapshotted_and_incomplete_backward_is_detected(cpu_orchestration):
    x = torch.ones(2, 4).bfloat16().requires_grad_(True)
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    keys = logical_keys(2, "cpu")
    with canonical_backward_session() as session:
        y = down.h3_ffn_down_gemm(x, w, logical_keys=keys)
        keys.fill_(-1)
        with pytest.raises(RuntimeError, match="incomplete"):
            session.validate_complete()
        y.backward(torch.ones_like(y))
        session.validate_complete()
    assert w.grad is not None


def test_padding_only_first_chunk_and_3d_gradient_shape(cpu_orchestration):
    x = torch.ones(2, 4).bfloat16().requires_grad_(True)
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    op = down.H3FFNDownGemmOp()
    with canonical_backward_session() as session:
        padding = op(x[:1], w, logical_keys=torch.full((1, 2), -1, dtype=torch.int64))
        active = op(x[1:].reshape(1, 1, 4), w, logical_keys=logical_keys(1, "cpu"))
        torch.autograd.backward(
            (padding, active), (torch.zeros_like(padding), torch.ones_like(active))
        )
        session.validate_complete()
    assert x.grad.shape == (2, 4)
    assert torch.count_nonzero(x.grad[0]) == 0
    assert raw_equal(w.grad, torch.ones_like(w))


@pytest.mark.parametrize("mutated", ["x", "weight"])
def test_strided_inputs_keep_autograd_version_checks(cpu_orchestration, mutated):
    x = torch.ones(4, 2).bfloat16().t().requires_grad_(True)
    w = torch.ones(4, 3).bfloat16().t().requires_grad_(True)
    assert not x.is_contiguous() and not w.is_contiguous()
    with canonical_backward_session():
        y = down.h3_ffn_down_gemm(x, w, logical_keys=logical_keys(2, "cpu"))
        with torch.no_grad():
            (x if mutated == "x" else w).add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace"):
            y.backward(torch.ones_like(y))


def test_candidate_registration_cannot_enter_default_strict_resolution():
    from rl_engine.kernels.registry import _default_semantic_descriptors
    from rl_engine.kernels.semantic_registry import (
        OperatorRequirements,
        OperatorResolutionError,
        OperatorResolutionPolicy,
        SemanticOperatorCatalog,
    )

    catalog = SemanticOperatorCatalog(_default_semantic_descriptors())
    request = dict(
        semantic_op="h3_ffn_down_gemm",
        requested_backend=down.H3_DOWN_BACKEND,
        target="training",
        requirements=OperatorRequirements(
            device="cuda", dtype="bfloat16", topology={"world_size": 1, "tensor_parallel_size": 1}
        ),
    )
    with pytest.raises(OperatorResolutionError):
        catalog.session().resolve(**request)
    session = catalog.session(OperatorResolutionPolicy(allow_test_backends=True))
    resolved = session.resolve(**request)
    instance = session.instantiate(resolved)
    assert isinstance(instance, down.H3FFNDownGemmOp)
    assert instance.last_execution == {}
    assert not resolved.descriptor.determinism_or_alignment_properties["gpu_qualified"]


def test_nonzero_padding_gradient_is_rejected(cpu_orchestration):
    x, w = torch.ones(2, 4).bfloat16(), torch.ones(3, 4).bfloat16().requires_grad_(True)
    keys = torch.tensor([[0, 0], [-1, -1]])
    with canonical_backward_session():
        y = down.h3_ffn_down_gemm(x, w, logical_keys=keys)
        with pytest.raises(ValueError, match="zero upstream"):
            y.backward(torch.ones_like(y))


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90; a missing requested backend on SM90 is a failure",
)


@pytest.mark.cuda_only
@requires_sm90
@pytest.mark.parametrize("rows", [1, 33, 129])
def test_full_h3_accuracy_and_four_judgments_on_gpu(rows):
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(420)
    x = torch.randn(rows, 14336, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(5376, 14336, device="cuda", generator=generator) / 14336**0.5).bfloat16()
    dy = (torch.randn(rows, 5376, device="cuda", generator=generator) * 0.1).bfloat16()
    keys = logical_keys(rows, "cuda")
    op = down.H3FFNDownGemmOp()
    actual, _ = training_projection(op, x, w, dy, keys)
    expected = reference_projection(x, w, dy)
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        judgment = "forward_accuracy" if index == 0 else "gradient_accuracy"
        result = comparison(left, right, judgment)
        assert result["passed"], result
    chunked, _ = training_projection(op, x, w, dy, keys, chunk_size=31)
    for index, (left, right) in enumerate(zip(chunked, actual, strict=True)):
        judgment = "forward_invariance" if index == 0 else "gradient_invariance"
        result = comparison(left, right, judgment)
        assert result["passed"], result
