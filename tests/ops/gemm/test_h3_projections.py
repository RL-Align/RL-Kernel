# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU contract/orchestration checks and separate full-width SM90 checks."""

from contextlib import nullcontext

import pytest
import torch

import rl_engine.ops.gemm.h3_projections as down
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.validation.operators.h3_projection_validation import (
    comparison,
    logical_keys,
    raw_equal,
    reference_projection,
    training_projection,
)


@pytest.fixture(params=tuple(down.SPECS))
def kind(request):
    return request.param


def test_contract_and_rejections(kind):
    k, n = down.SPECS[kind]
    assert (k, n) == ((5376, 28672) if kind == "h3_ffn_gate_up_gemm" else (7168, 5376))
    op = down.H3ProjectionOp(kind)
    w = torch.empty((n, k), dtype=torch.bfloat16, device="meta")
    for shape in [(k,), (2, 7), (1, 1, 1, k), (0, k), (down.H3_MAX_ROWS + 1, k)]:
        with pytest.raises(ValueError):
            op(torch.empty(shape, dtype=torch.bfloat16, device="meta"), w)
    x = torch.empty((1, k), dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="native shape"):
        op(x, w.t())
    with pytest.raises(TypeError, match="bfloat16"):
        op(x.float(), w)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        op(x, w)


def test_hardware_and_backend_fail_closed(monkeypatch, kind):
    from rl_engine.backends.shared.triton.gemm import det_gemm

    backend = down.H3ProjectionOp(kind).backend
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(RuntimeError, match="SM90"):
        down._require_backend(torch.device("cuda"), backend)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(RuntimeError, match="TF32"):
        down._require_backend(torch.device("cuda"), backend)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(det_gemm, "_TRITON_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        down._require_backend(torch.device("cuda"), backend)
    with pytest.raises(ValueError, match="unsupported"):
        down.H3ProjectionOp("unknown")


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
def cpu_orchestration(monkeypatch, kind):
    extension = _CPUOrchestrationOnlyExtension()
    monkeypatch.setitem(down.SPECS, kind, (4, 3))
    monkeypatch.setattr(down, "_validate_inputs", lambda x, w, kind: None)
    monkeypatch.setattr(down, "_require_backend", lambda device, backend: extension)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return extension


def test_canonical_chunk_permutation_and_padding_orchestration(cpu_orchestration, kind):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(7, 4, generator=generator).bfloat16()
    w = torch.randn(3, 4, generator=generator).bfloat16()
    dy = torch.randn(7, 3, generator=generator).bfloat16()
    keys = logical_keys(7, "cpu")
    op = down.H3ProjectionOp(kind)
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


def test_training_requires_session_and_unique_stable_keys(cpu_orchestration, kind):
    x, w = torch.ones(2, 4).bfloat16(), torch.ones(3, 4).bfloat16().requires_grad_(True)
    op = down.H3ProjectionOp(kind)
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


def test_keys_are_snapshotted_and_incomplete_backward_is_detected(cpu_orchestration, kind):
    x = torch.ones(2, 4).bfloat16().requires_grad_(True)
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    keys = logical_keys(2, "cpu")
    with canonical_backward_session() as session:
        y = down.H3ProjectionOp(kind)(x, w, logical_keys=keys)
        keys.fill_(-1)
        with pytest.raises(RuntimeError, match="incomplete"):
            session.validate_complete()
        y.backward(torch.ones_like(y))
        session.validate_complete()
    assert w.grad is not None


def test_padding_only_first_chunk_and_3d_gradient_shape(cpu_orchestration, kind):
    x = torch.ones(2, 4).bfloat16().requires_grad_(True)
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    op = down.H3ProjectionOp(kind)
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
def test_strided_inputs_keep_autograd_version_checks(cpu_orchestration, mutated, kind):
    x = torch.ones(4, 2).bfloat16().t().requires_grad_(True)
    w = torch.ones(4, 3).bfloat16().t().requires_grad_(True)
    assert not x.is_contiguous() and not w.is_contiguous()
    with canonical_backward_session():
        y = down.H3ProjectionOp(kind)(x, w, logical_keys=logical_keys(2, "cpu"))
        with torch.no_grad():
            (x if mutated == "x" else w).add_(1)
        with pytest.raises(RuntimeError, match="modified by an inplace"):
            y.backward(torch.ones_like(y))


def test_candidate_registration_cannot_enter_default_strict_resolution(kind):
    from rl_engine.runtime.registry import _default_semantic_descriptors
    from rl_engine.runtime.semantic_registry import (
        OperatorRequirements,
        OperatorResolutionError,
        OperatorResolutionPolicy,
        SemanticOperatorCatalog,
    )

    catalog = SemanticOperatorCatalog(_default_semantic_descriptors())
    request = dict(
        semantic_op=kind,
        requested_backend=down.H3ProjectionOp(kind).backend,
        target="training",
        requirements=OperatorRequirements(
            device="cuda",
            dtype="bfloat16",
            topology={"world_size": 1, "tensor_parallel_size": 1},
        ),
    )
    with pytest.raises(OperatorResolutionError):
        catalog.session().resolve(**request)
    session = catalog.session(OperatorResolutionPolicy(allow_test_backends=True))
    resolved = session.resolve(**request)
    instance = session.instantiate(resolved)
    assert isinstance(instance, down.H3ProjectionOp)
    assert instance.last_execution == {}
    assert not resolved.descriptor.determinism_or_alignment_properties["gpu_qualified"]


def test_nonzero_padding_gradient_is_rejected(cpu_orchestration, kind):
    x, w = torch.ones(2, 4).bfloat16(), torch.ones(3, 4).bfloat16().requires_grad_(True)
    keys = torch.tensor([[0, 0], [-1, -1]])
    with canonical_backward_session():
        y = down.H3ProjectionOp(kind)(x, w, logical_keys=keys)
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
def test_full_h3_accuracy_and_four_judgments_on_gpu(rows, kind):
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = torch.Generator(device="cuda").manual_seed(420)
    input_dim, output_dim = down.SPECS[kind]
    x = torch.randn(rows, input_dim, device="cuda", generator=generator).bfloat16()
    w = (
        torch.randn(output_dim, input_dim, device="cuda", generator=generator) / input_dim**0.5
    ).bfloat16()
    dy = (torch.randn(rows, output_dim, device="cuda", generator=generator) * 0.1).bfloat16()
    keys = logical_keys(rows, "cuda")
    op = down.H3ProjectionOp(kind)
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


@pytest.mark.parametrize("train_x,train_w", [(True, False), (False, True), (True, True)])
def test_partial_gradients_and_native_output_order(cpu_orchestration, kind, train_x, train_w):
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16).requires_grad_(train_x)
    w = torch.tensor(
        [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]], dtype=torch.bfloat16
    ).requires_grad_(train_w)
    with canonical_backward_session() as session:
        y = down.H3ProjectionOp(kind)(x, w, logical_keys=logical_keys(1, "cpu"))
        assert raw_equal(y, torch.tensor([[1, 4, 9]], dtype=torch.bfloat16))
        y.backward(torch.ones_like(y))
        session.validate_complete()
    if train_x:
        assert raw_equal(x.grad, torch.tensor([[1, 2, 3, 0]], dtype=torch.bfloat16))
    else:
        assert x.grad is None
    if train_w:
        assert raw_equal(w.grad, x.detach().expand(3, 4))
    else:
        assert w.grad is None


@pytest.mark.parametrize("operand", ["x", "weight"])
def test_mixed_device_inputs_fail_before_launch(kind, operand):
    k, n = down.SPECS[kind]
    x = torch.empty((1, k), dtype=torch.bfloat16, device="meta")
    w = torch.empty((n, k), dtype=torch.bfloat16, device="meta")
    # Expanded CPU storage avoids allocating the full weight for a rejection test.
    if operand == "x":
        x = torch.zeros((), dtype=torch.bfloat16).expand(1, k)
    else:
        w = torch.zeros((), dtype=torch.bfloat16).expand(n, k)
    with pytest.raises(ValueError, match="share one device"):
        down.H3ProjectionOp(kind)(x, w)


@pytest.mark.parametrize("invalid", ["dtype", "rank", "count", "width", "device", "duplicate"])
def test_invalid_logical_keys(cpu_orchestration, kind, invalid):
    x = torch.ones(2, 4).bfloat16()
    w = torch.ones(3, 4).bfloat16().requires_grad_(True)
    keys = logical_keys(2, "cpu")
    variants = {
        "dtype": keys.int(),
        "rank": keys.flatten(),
        "count": keys[:1],
        "width": keys[:, :1],
        "device": keys.to("meta"),
        "duplicate": keys[:1].expand(2, 2),
    }
    with canonical_backward_session():
        with pytest.raises((TypeError, ValueError)):
            down.H3ProjectionOp(kind)(x, w, logical_keys=variants[invalid])


@pytest.mark.parametrize(
    "topology",
    [
        {"world_size": 2, "tensor_parallel_size": 1},
        {"world_size": 2, "tensor_parallel_size": 2},
        {"world_size": 1, "tensor_parallel_size": 2},
    ],
)
def test_unsupported_topology_is_rejected_by_registry(kind, topology):
    from rl_engine.runtime.registry import _default_semantic_descriptors
    from rl_engine.runtime.semantic_registry import (
        OperatorRequirements,
        OperatorResolutionError,
        OperatorResolutionPolicy,
        SemanticOperatorCatalog,
    )

    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session(
        OperatorResolutionPolicy(allow_test_backends=True)
    )
    with pytest.raises(OperatorResolutionError):
        session.resolve(
            semantic_op=kind,
            requested_backend=down.H3ProjectionOp(kind).backend,
            target="training",
            requirements=OperatorRequirements(device="cuda", dtype="bfloat16", topology=topology),
        )


def test_expanded_row_boundary_and_session_limit(monkeypatch, kind):
    k, n = down.SPECS[kind]
    w = torch.empty((n, k), dtype=torch.bfloat16, device="meta")
    for rows in (37296, 106238, down.H3_MAX_ROWS):
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            down._validate_inputs(
                torch.empty((rows, k), dtype=torch.bfloat16, device="meta"), w, kind
            )
    monkeypatch.setattr(down, "H3_MAX_ROWS", 3)
    with canonical_backward_session() as session:
        weight = torch.ones(1)
        down._register_rows(session, weight, logical_keys(2, "cpu"), 2, "weight")
        with pytest.raises(ValueError, match="row limit"):
            down._register_rows(session, weight, logical_keys(2, "cpu") + 10, 2, "weight")


def test_byte_mismatch_count_crosses_comparison_chunk_boundary():
    lhs = torch.zeros((1 << 19) + 2, dtype=torch.bfloat16)
    rhs = lhs.clone()
    rhs[-1] = -0.0
    result = comparison(lhs, rhs, "forward_invariance")
    assert not result["passed"] and result["mismatched_bytes"] == 1
