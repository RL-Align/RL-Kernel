# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""AdaLN dispatch policy and observations at the public resolver seam."""

import os

import pytest
import torch

from argparse import Namespace

from rl_engine.validation.operators import CandidateSpec, run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case
from rl_engine.validation.operators.operator_specs import GtestAdaLNReference
from rl_engine.contracts.numerical import (
    BackendProvenance,
    ContractResolveError,
    load_contract,
    resolve_tolerance,
    validate_backend_provenance,
)
from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.runtime.registry import OpBackend
from rl_engine.runtime.registry import KernelRegistry, kernel_registry


@pytest.fixture
def cuda_op(request):
    try:
        return CudaAdaLNModulationOp(threads=getattr(request, "param", 256))
    except RuntimeError:
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")


def test_adaln_registry_trace_reports_selected_backend():
    op, trace = kernel_registry.get_adaln_modulation_op(device="cpu", hidden=4)
    assert isinstance(op.implementation, NativeAdaLNModulationOp)
    assert trace["selected_backend"] == OpBackend.PYTORCH_ADALN_MODULATION.name
    assert trace["fallback"] is False
    assert trace["accumulator_dtype"] == "fp32"
    assert trace["split_k"] is False and trace["stream_k"] is False


def test_adaln_modulate_index_uses_explicit_native_fallback():
    x = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]], requires_grad=True)
    # First batch row is choice 0, second batch row is choice 1.
    modulation = torch.tensor(
        [
            [0.5, -0.5, 1.0, 0.0, 2.0, 3.0],
            [-1.0, 1.0, 0.0, 1.0, 4.0, 5.0],
        ],
        requires_grad=True,
    )
    modulate_index = torch.tensor([[0, 1]])

    op, trace = kernel_registry.get_adaln_modulation_op(
        device=x.device,
        hidden=x.shape[-1],
        modulate_index=modulate_index,
    )
    y, gate = op(x, modulation, modulate_index=modulate_index, eps=0.0)

    torch.testing.assert_close(y, torch.tensor([[[-1.5, 0.5], [-2.0, 3.0]]]))
    torch.testing.assert_close(gate, torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]))
    (y.sum() + gate.sum()).backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(
        modulation.grad,
        torch.tensor([[1.0, 1.0, -1.0, 1.0, 1.0, 1.0]]).repeat(2, 1),
    )
    assert isinstance(op.implementation, NativeAdaLNModulationOp)
    assert trace["selected_backend"] == OpBackend.PYTORCH_ADALN_MODULATION.name
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == "modulate_index_requires_select01"


def test_adaln_cuda_registry_reports_real_fallback():
    try:
        torch.cuda.init()
    except RuntimeError:
        pytest.skip("CUDA required")
    op, trace = kernel_registry.get_adaln_modulation_op(device="cuda", hidden=32)
    assert type(op.implementation).__name__ in {"CudaAdaLNModulationOp", "TritonAdaLNModulationOp"}
    if type(op.implementation).__name__ == "TritonAdaLNModulationOp":
        assert trace["fallback"] is True
        assert "CUDA_ADALN_MODULATION" in trace["rejected_backends"]
    else:
        assert trace["fallback"] is False


def test_adaln_registry_skips_cuda_for_large_hidden():
    pytest.importorskip("triton")
    op, trace = kernel_registry.get_adaln_modulation_op(device="cuda", hidden=5000)
    assert type(op.implementation).__name__ == "TritonAdaLNModulationOp"
    assert trace["fallback"] is True
    assert "CUDA_ADALN_MODULATION: H > 4096" in trace["rejected_backends"]


@pytest.mark.parametrize("hidden", [4095, 4096, 4097])
def test_cuda_shared_registry_respects_hidden_limit(cuda_op, hidden):
    _, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=hidden)
    if hidden <= 4096:
        assert trace["selected_backend"] == OpBackend.CUDA_ADALN_MODULATION.name
        assert not trace["fallback"]
    else:
        assert trace["selected_backend"] != OpBackend.CUDA_ADALN_MODULATION.name
        assert trace["fallback"]
        assert "CUDA_ADALN_MODULATION: H > 4096" in trace["rejected_backends"]


def test_strict_cpu_call_records_execution_without_claiming_hardware_qualification():
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, strict=True)
    assert trace["actual_backend"] is None
    x = torch.tensor([[[-3.0, 3.0]]], requires_grad=True)
    m = torch.zeros(1, 6, requires_grad=True)
    y, gate = op(x, m, eps=7.0)
    dx, dm = torch.autograd.grad(
        (y, gate), (x, m), (torch.tensor([[[1.0, 0.0]]]), torch.ones_like(gate))
    )
    torch.testing.assert_close(y, torch.tensor([[[-0.75, 0.75]]]))
    torch.testing.assert_close(dx, torch.tensor([[[7 / 128, -7 / 128]]]))
    torch.testing.assert_close(dm, torch.tensor([[1.0, 0.0, -0.75, 0.0, 1.0, 1.0]]))
    assert trace["actual_backend"] == "pytorch"
    assert trace["execution_dtype"] == "float32"
    assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
    assert trace["observation"] == "successful_forward"
    assert trace["binary_sha256"] is None
    assert trace["hardware_qualified"] is None


def test_gtest_records_cpu_calls_without_inventing_a_bf16_hardware_profile():
    args = Namespace(
        op="adaln_modulation",
        candidate="pytorch",
        arch_key=None,
        batch=2,
        seq=3,
        normalized_dim=7,
        seed=386,
    )
    candidate = make_candidate(args, observe_backend=True)
    case = make_operator_case(args, torch.bfloat16, torch.device("cpu"))
    report = run_operator_suite(
        "adaln_modulation", candidates=[candidate], cases=[case], check_grad=True
    )
    assert report.passed
    result = report.candidates[0]
    assert result.backend_provenance is None
    trace = result.cases[0].backend_trace
    assert trace["actual_backend"] == "pytorch"
    assert trace["execution_dtype"] == "bfloat16"
    assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
    assert trace["profiler_observed"] is False


def test_common_provenance_rejects_cuda_to_triton_fallback():
    proof = BackendProvenance(
        backend_profile="triton_cuda_bf16",
        requested_backend="cuda",
        actual_backend="triton",
        execution_dtype="bfloat16",
        accumulation_dtype="float32",
        output_dtype="bfloat16",
        reference_dtype="float32",
        candidate_tf32_enabled=False,
        reference_tf32_enabled=False,
    )
    with pytest.raises(ContractResolveError, match="requested_backend"):
        validate_backend_provenance(load_contract(), proof)


def test_strict_rejects_indexed_backward():
    registry = KernelRegistry()
    with pytest.raises(RuntimeError, match="select01"):
        registry.get_adaln_modulation_op(
            "cpu", hidden=2, strict=True, modulate_index=torch.tensor([[0]])
        )
    op, _ = registry.get_adaln_modulation_op("cpu", hidden=2, strict=True)
    with pytest.raises(RuntimeError, match="select01"):
        op(torch.zeros(1, 1, 2), torch.zeros(2, 6), modulate_index=torch.tensor([[0]]))


def test_strict_call_rechecks_resolved_hidden_and_observations_are_not_stale():
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, strict=True)
    op(torch.zeros(1, 1, 2), torch.zeros(1, 6))
    with pytest.raises(ValueError, match="hidden"):
        op(torch.zeros(1, 1, 4), torch.zeros(1, 12))
    assert trace["actual_backend"] is None


def test_ordinary_select01_keeps_explicit_fallback_and_separate_profile():
    index = torch.tensor([[0]])
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, modulate_index=index)
    op(torch.zeros(1, 1, 2), torch.zeros(2, 6), modulate_index=index)
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == "modulate_index_requires_select01"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    assert trace["actual_backend"] == "pytorch"


def test_strict_refuses_unqualified_fallback_when_optimized_backends_are_unavailable(monkeypatch):
    registry = KernelRegistry()
    load = registry._get_or_create_backend

    def unavailable(backend):
        if backend in {OpBackend.CUDA_ADALN_MODULATION, OpBackend.TRITON_ADALN_MODULATION}:
            return None
        return load(backend)

    monkeypatch.setattr(registry, "_get_or_create_backend", unavailable)
    op, trace = registry.get_adaln_modulation_op("cuda", hidden=7)
    assert trace["fallback"] and trace["actual_backend"] is None
    assert isinstance(op.implementation, NativeAdaLNModulationOp)
    with pytest.raises(RuntimeError, match="no declared strict replay profile"):
        registry.get_adaln_modulation_op("cuda", hidden=7, strict=True)
    cpu_op, cpu_trace = registry.get_adaln_modulation_op("cpu", hidden=7, strict=True)
    cpu_op(torch.zeros(1, 1, 7), torch.zeros(1, 21))
    assert cpu_trace["fallback"] is False
    assert trace["actual_backend"] is None  # Independent resolution, no cache trace leakage.


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required; run explicitly on target GPU",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("target", ["pytorch", "triton"])
def test_gpu_fallback_forward_backward_accuracy_and_declared_replay(monkeypatch, dtype, target):
    registry = KernelRegistry()
    load = registry._get_or_create_backend

    def unavailable(backend):
        if backend == OpBackend.CUDA_ADALN_MODULATION:
            return None
        if target == "pytorch" and backend == OpBackend.TRITON_ADALN_MODULATION:
            return None
        return load(backend)

    monkeypatch.setattr(registry, "_get_or_create_backend", unavailable)
    if target == "triton":
        pytest.importorskip("triton")
    op, trace = registry.get_adaln_modulation_op("cuda", hidden=3072, strict=target == "triton")
    torch.manual_seed(386)
    x = torch.randn(2, 3, 3072, device="cuda", dtype=dtype, requires_grad=True)
    m = torch.randn(2, 9216, device="cuda", dtype=dtype, requires_grad=True)
    dy, dg = torch.randn_like(x), torch.randn(2, 1, 3072, device="cuda", dtype=dtype)
    y, gate = op(x, m)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    assert trace["actual_backend"] == target and trace["fallback"] is True
    assert trace["fallback_reason"] and trace["profiler_observed"] is False
    xx, mm = x.detach().cpu().float().requires_grad_(), m.detach().cpu().float().requires_grad_()
    gold_y, gold_gate = GtestAdaLNReference().forward_fp32(xx, mm)
    gold_dx, gold_dm = torch.autograd.grad(
        (gold_y, gold_gate), (xx, mm), (dy.cpu().float(), dg.cpu().float())
    )
    for actual, gold, judgment in (
        (y, gold_y, "forward_accuracy"),
        (dx, gold_dx, "gradient_accuracy"),
        (dm, gold_dm, "gradient_accuracy"),
    ):
        tol = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="reduction", dtype=dtype
        )
        torch.testing.assert_close(actual.cpu().float(), gold, atol=tol.atol, rtol=tol.rtol)
    torch.testing.assert_close(gate.cpu(), gold_gate.to(dtype), atol=0, rtol=0)
    # Triton shares the frozen replay contract; Native GPU has only an accuracy claim.
    if target == "triton":
        xx, mm = x.detach().cpu().requires_grad_(), m.detach().cpu().requires_grad_()
        fixed_y, fixed_gate = NativeAdaLNModulationOp()(xx, mm)
        fixed_dx, fixed_dm = torch.autograd.grad(
            (fixed_y, fixed_gate), (xx, mm), (dy.cpu(), dg.cpu())
        )
        for actual, expected in zip(
            (y, gate, dx, dm), (fixed_y, fixed_gate, fixed_dx, fixed_dm), strict=True
        ):
            assert torch.equal(
                actual.detach().cpu().contiguous().view(torch.uint8),
                expected.detach().contiguous().view(torch.uint8),
            )
    else:
        with pytest.raises(RuntimeError, match="no declared strict replay profile"):
            registry.get_adaln_modulation_op("cuda", hidden=3072, strict=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required; run explicitly on target GPU",
)
def test_gpu_gtest_persists_observed_bf16_fallback_trace(monkeypatch):
    pytest.importorskip("triton")
    args = Namespace(
        op="adaln_modulation",
        candidate="triton",
        arch_key=None,
        batch=2,
        seq=3,
        normalized_dim=7,
        seed=386,
    )
    case = make_operator_case(args, torch.bfloat16, torch.device("cuda"))
    registry = KernelRegistry()
    load = registry._get_or_create_backend
    monkeypatch.setattr(
        registry,
        "_get_or_create_backend",
        lambda backend: None if backend == OpBackend.CUDA_ADALN_MODULATION else load(backend),
    )
    op, _ = registry.get_adaln_modulation_op("cuda", hidden=7, strict=True)
    report = run_operator_suite(
        "adaln_modulation",
        candidates=[CandidateSpec(name="triton-adaln", backend="triton", fn=op)],
        cases=[case],
        check_grad=True,
    )
    assert report.passed
    result = report.candidates[0]
    assert result.backend_provenance is None
    payload = report.to_dict()["candidates"][0]["cases"][0]
    assert "backend_provenance" not in payload
    trace = payload["backend_trace"]
    assert trace["actual_backend"] == "triton"
    assert trace["requested_backend"] == "cuda" and trace["fallback"] is True
    assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
    assert trace["observation"] == "successful_forward" and trace["binary_sha256"] is None
