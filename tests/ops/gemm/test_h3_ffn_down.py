# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU contract/orchestration checks and separate full-width SM90 checks."""

from contextlib import nullcontext

import pytest
import torch

import rl_engine.ops.gemm.h3_ffn_down as down
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.validation.operators.h3_down_validation import (
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
    from rl_engine.backends.shared.triton.gemm import det_gemm

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
        self.forward_calls = 0
        self.input_gradient_calls = 0

    def det_gemm_fwd_rhs_transposed(self, x, w):
        self.forward_calls += 1
        return (x.float() @ w.float().t()).bfloat16()

    def det_gemm_fwd(self, x, w):
        self.input_gradient_calls += 1
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
    assert sum(trace["weight_gradient_kernel"] is not None for trace in traces) == 1
    assert all(trace["input_gradient_kernel"] is not None for trace in traces)
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
    from rl_engine.runtime.registry import _default_semantic_descriptors
    from rl_engine.runtime.semantic_registry import (
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


def test_backend_is_resolved_once_per_forward_and_retained_for_backward(
    cpu_orchestration, monkeypatch
):
    resolutions = []

    def resolve(device, backend):
        extension = _CPUOrchestrationOnlyExtension()
        resolutions.append(extension)
        return extension

    monkeypatch.setattr(down, "_require_backend", resolve)
    op = down.H3FFNDownGemmOp()
    x = torch.ones(2, 4).bfloat16().requires_grad_(True)
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    with canonical_backward_session() as session:
        first = op(x[:1], w, logical_keys=logical_keys(1, "cpu"))
        second = op(x[1:], w, logical_keys=logical_keys(1, "cpu") + 1)
        assert len(resolutions) == 2
        torch.autograd.backward((first, second), (torch.ones_like(first), torch.ones_like(second)))
        session.validate_complete()
    assert len(resolutions) == 2
    assert sum(len(ext.weight_reduction_inputs) for ext in resolutions) == 1


def test_unpadded_backward_needs_no_padding_scalar_readback(cpu_orchestration, monkeypatch):
    x = torch.ones(2, 4).bfloat16().requires_grad_(True)
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    with canonical_backward_session() as session:
        y = down.h3_ffn_down_gemm(x, w, logical_keys=logical_keys(2, "cpu"))

        def unexpected_item(self, *args, **kwargs):
            raise AssertionError("unpadded backward must not read a validation scalar")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", unexpected_item)
            y.backward(torch.ones_like(y))
            session.validate_complete()
    assert x.grad is not None and w.grad is not None


@pytest.mark.parametrize("padding_gradient", [1.0, float("nan"), float("inf")])
def test_padding_validation_uses_forward_snapshot(cpu_orchestration, padding_gradient):
    x = torch.ones(2, 4).bfloat16()
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    keys = torch.tensor([[0, 0], [-1, -1]])
    with canonical_backward_session():
        y = down.h3_ffn_down_gemm(x, w, logical_keys=keys)
        keys[1] = torch.tensor([1, 1])
        gradient = torch.ones_like(y)
        gradient[1] = padding_gradient
        with pytest.raises(ValueError, match="zero upstream"):
            y.backward(gradient)
    assert not cpu_orchestration.weight_reduction_inputs


@pytest.mark.parametrize("input_grad,weight_grad", [(True, False), (False, True), (True, True)])
def test_gradient_kernel_roles_and_runtime_ids(
    cpu_orchestration, monkeypatch, input_grad, weight_grad
):
    kernel_id = "test.shared_gemm_kernel"
    monkeypatch.setattr(cpu_orchestration, "h3_kernel_id", kernel_id, raising=False)
    events = []
    monkeypatch.setattr(down, "record_backward", lambda kind, **event: events.append(event))
    x = torch.ones(2, 4).bfloat16().requires_grad_(input_grad)
    w = torch.ones(3, 4).bfloat16().requires_grad_(weight_grad)
    op = down.H3FFNDownGemmOp()
    with canonical_backward_session() as session:
        y = op(x, w, logical_keys=logical_keys(2, "cpu"))
        y.backward(torch.ones_like(y))
        session.validate_complete()
    assert op.last_execution["input_gradient_kernel"] == (kernel_id if input_grad else None)
    assert op.last_execution["weight_gradient_kernel"] == (kernel_id if weight_grad else None)
    assert len(events) == 1 and events[0]["kernel_id"] == kernel_id
    assert cpu_orchestration.forward_calls == 1
    assert cpu_orchestration.input_gradient_calls == int(input_grad)
    assert len(cpu_orchestration.weight_reduction_inputs) == int(weight_grad)
    if input_grad:
        torch.testing.assert_close(x.grad, torch.full_like(x, 3), atol=0, rtol=0)
    if weight_grad:
        torch.testing.assert_close(w.grad, torch.full_like(w, 2), atol=0, rtol=0)


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90; a missing requested backend on SM90 is a failure",
)


@pytest.mark.cuda_only
@requires_sm90
@pytest.mark.parametrize("rows", [1, 33, 129, 4097])
def test_full_h3_accuracy_and_four_judgments_on_gpu(rows, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    generator = torch.Generator(device="cuda").manual_seed(420)
    x = torch.randn(rows, 14336, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(5376, 14336, device="cuda", generator=generator) / 14336**0.5).bfloat16()
    dy = (torch.randn(rows, 5376, device="cuda", generator=generator) * 0.1).bfloat16()
    keys = logical_keys(rows, "cuda")
    op = down.H3FFNDownGemmOp()
    actual, traces = training_projection(op, x, w, dy, keys)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["actual_backend"] == down.H3_DOWN_BACKEND
    assert trace["forward_executed"] and trace["weight_gradient_executed"]
    assert trace["input_gradient_kernel"] == trace["weight_gradient_kernel"] == trace["kernel_id"]
    assert trace["forward_launch"]["cubin_sha256"]
    expected = reference_projection(x, w, dy)
    # The projection contract is exactly (forward, input gradient, weight gradient).
    assert len(actual) == len(expected) == 3
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        judgment = "forward_accuracy" if index == 0 else "gradient_accuracy"
        result = comparison(left, right, judgment)
        assert result["passed"], result
    chunked, _ = training_projection(op, x, w, dy, keys, chunk_size=31 if rows < 4096 else 2048)
    assert len(chunked) == 3
    for index, (left, right) in enumerate(zip(chunked, actual, strict=True)):
        judgment = "forward_invariance" if index == 0 else "gradient_invariance"
        result = comparison(left, right, judgment)
        assert result["passed"], result
    if rows >= 4096:
        # Exercise both the CUDA canonical-order reuse and the sorting/filtering
        # paths after the shared reducer changed. Keep the logical row set fixed.
        order = torch.randperm(rows, device="cuda", generator=generator)
        permuted, _ = training_projection(op, x[order], w, dy[order], keys[order])
        assert raw_equal(permuted[0], actual[0][order])
        assert raw_equal(permuted[1], actual[1][order])
        assert raw_equal(permuted[2], actual[2])
        padded_x = torch.cat((x, torch.ones_like(x[:3])))
        padded_dy = torch.cat((dy, torch.zeros_like(dy[:3])))
        padded_keys = torch.cat((keys, torch.full_like(keys[:3], -1)))
        padded, _ = training_projection(op, padded_x, w, padded_dy, padded_keys)
        assert raw_equal(padded[0][:rows], actual[0])
        assert raw_equal(padded[1][:rows], actual[1])
        assert torch.count_nonzero(padded[1][rows:]) == 0
        assert raw_equal(padded[2], actual[2])
