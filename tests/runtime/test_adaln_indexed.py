# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""L3: public indexed AdaLN resolution, fallback and execution observations."""

import pytest
import torch

from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.runtime.registry import KernelRegistry, kernel_registry
from tests.helpers.adaln_indexed import assert_bytes, check_small_case, require_cuda, small_case


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_indexed_resolution_reports_backend_policy_without_execution(device):
    if device == "cuda":
        if torch.version.hip is not None:
            pytest.skip("NVIDIA resolution policy")
        from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

        try:
            cuda = CudaAdaLNModulationOp()
        except RuntimeError:
            pytest.skip("compiled CUDA AdaLN extension required")
        if not cuda.indexed_available:
            pytest.skip("indexed CUDA extension required")
    index = torch.zeros(2, 8, dtype=torch.int32)
    _, trace = KernelRegistry().get_adaln_modulation_op(device, hidden=2, modulate_index=index)
    accelerated = device == "cuda"
    assert trace["selected_backend"] == (
        "CUDA_ADALN_MODULATION" if accelerated else "PYTORCH_ADALN_MODULATION"
    )
    assert trace["fallback"] is (not accelerated)
    assert trace["fallback_reason"] == (None if accelerated else "modulate_index_requires_select01")
    assert trace["arithmetic_profile"] == (
        "adaln-indexed-fp32-tree-v1" if accelerated else "adaln-select01-framework-v1"
    )
    assert trace["requested_backend"] == ("cuda" if accelerated else "pytorch")
    assert trace["kernel_fingerprint"] == (
        "adaln_indexed_cuda_v1" if accelerated else "adaln_modulation_select01_native_v1"
    )
    assert trace["reduction_order"] == (
        "pairwise_lower_upper_H; ascending_selected_token_S_per_bank"
        if accelerated
        else "pairwise_lower_upper_H; framework_select01_backward"
    )
    assert trace["accumulator_dtype"] == "fp32"
    assert not any(trace[key] for key in ("split_k", "stream_k", "tf32", "rejected_backends"))
    assert trace["actual_backend"] is None and trace["observation"] is None


def test_indexed_small_cpu_public_dispatch_retains_explicit_fallback():
    x, m, index, dy, dg = small_case("cpu")
    op, trace = kernel_registry.get_adaln_modulation_op("cpu", hidden=2, modulate_index=index)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    assert torch.isfinite(dx).all() and torch.count_nonzero(dm)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    assert trace["fallback"]


@pytest.mark.cuda_only
def test_cuda_indexed_small_fp32_public_dispatch():
    require_cuda()
    index = small_case("cuda")[2]
    op, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    check_small_case(op, "cuda", trace)


@pytest.mark.cuda_only
@pytest.mark.parametrize("unsupported", ["eps", "eps_underflow", "geometry"])
def test_cuda_indexed_execution_rechecks_support_and_clears_failed_observation(unsupported):
    require_cuda()
    x, m, index, _, _ = small_case("cuda")
    op, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    op(x, m, modulate_index=index, eps=7.0)
    assert trace["actual_backend"] == "cuda"

    fallback_x = x[:1].expand(65536, -1, -1) if unsupported == "geometry" else x
    fallback_index = index[:1] if unsupported == "geometry" else index
    fallback_m = m[:1].expand(131072, -1) if unsupported == "geometry" else m
    eps = {"eps": 1e300, "eps_underflow": 2**-150, "geometry": 7.0}[unsupported]
    actual = op(fallback_x, fallback_m, modulate_index=fallback_index, eps=eps)
    expected = NativeAdaLNModulationOp()(
        fallback_x, fallback_m, modulate_index=fallback_index, eps=eps
    )
    for a, e in zip(actual, expected, strict=True):
        assert_bytes(a, e)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    assert trace["fallback_reason"] == "indexed_cuda_domain_unsupported"

    invalid_index = torch.full_like(index, 2)
    with pytest.raises(ValueError, match="values must be 0 or 1"):
        op(x, m, modulate_index=invalid_index, eps=7.0)
    assert trace["actual_backend"] is None
    assert trace["observation"] is None


@pytest.mark.cuda_only
def test_cuda_indexed_resolution_can_execute_shared_without_stale_profile():
    require_cuda()
    x, m, index, dy, dg = small_case("cuda")
    op, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    m_shared = m.detach()[:2].requires_grad_()
    y, gate = op(x, m_shared, eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m_shared), (dy, dg[:, :1]))
    rx = x.detach().cpu().requires_grad_()
    rm = m_shared.detach().cpu().requires_grad_()
    ry, rg = NativeAdaLNModulationOp()(rx, rm, eps=7.0)
    rdx, rdm = torch.autograd.grad((ry, rg), (rx, rm), (dy.cpu(), dg[:, :1].cpu()))
    for a, e in ((y, ry), (gate, rg), (dx, rdx), (dm, rdm)):
        assert_bytes(a, e)
    assert trace["actual_backend"] == "cuda"
    assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
    assert trace["kernel_fingerprint"] == "adaln_modulation_v1:CUDA_ADALN_MODULATION"
    check_small_case(op, "cuda", trace)


@pytest.mark.cuda_only
@pytest.mark.parametrize("explicit_zero", [False, True])
def test_cuda_indexed_public_dispatch_preserves_absent_vs_zero_dy(explicit_zero):
    require_cuda()
    from tests.helpers.adaln_indexed import check_absent_vs_zero_dy

    index = torch.zeros(2, 8, dtype=torch.int32, device="cuda")
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    check_absent_vs_zero_dy(op, "cuda", explicit_zero)
    assert trace["actual_backend"] == "cuda"
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    assert trace["kernel_fingerprint"] == "adaln_indexed_cuda_v1"
    assert not trace["fallback"]


@pytest.mark.parametrize("hidden", [1, 3, 127, 128, 129, 4096])
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_cuda_indexed_resolution_full_domain_metadata(hidden, dtype):
    from rl_engine.backends.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

    try:
        cuda = CudaAdaLNModulationOp()
    except RuntimeError:
        pytest.skip("compiled CUDA extension required for resolution")
    if not cuda.indexed_available or torch.version.hip is not None:
        pytest.skip("NVIDIA indexed symbols required")
    index = torch.zeros(1, 3, dtype=dtype)
    _, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=hidden, modulate_index=index)
    assert trace["selected_backend"] == "CUDA_ADALN_MODULATION"
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    assert trace["fallback_reason"] is None
    assert trace["actual_backend"] is None


@pytest.mark.parametrize(
    "invalid,exception,message",
    [
        ("value", ValueError, "values must be 0 or 1"),
        ("negative_value", ValueError, "values must be 0 or 1"),
        ("shape", ValueError, "must have shape"),
        ("float_index", TypeError, "boolean or integer"),
        ("index_device", ValueError, "share device"),
        ("modulation_device", ValueError, "share device"),
        ("modulation_dtype", TypeError, "share FP32 or BF16"),
        ("modulation_shape", ValueError, "shape"),
        ("empty", ValueError, "nonempty"),
        ("negative_eps", ValueError, "finite and nonnegative"),
        ("nan_eps", ValueError, "finite and nonnegative"),
        ("inf_eps", ValueError, "finite and nonnegative"),
    ],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda_only)])
def test_indexed_invalid_inputs_clear_successful_trace(invalid, exception, message, device):
    if device == "cuda":
        require_cuda()
    x, m, index, _, _ = small_case(device)
    op, trace = KernelRegistry().get_adaln_modulation_op(device, hidden=2, modulate_index=index)
    op(x, m, modulate_index=index, eps=7.0)
    assert trace["actual_backend"] == ("cuda" if device == "cuda" else "pytorch")
    eps = 7.0
    if invalid in {"value", "negative_value"}:
        index = torch.full_like(index, 2 if invalid == "value" else -1)
    elif invalid == "shape":
        index = index[:, :-1]
    elif invalid == "float_index":
        index = index.float()
    elif invalid == "index_device":
        index = index.to("meta")
    elif invalid == "modulation_device":
        m = m.to("meta")
    elif invalid == "modulation_dtype":
        m = m.double()
    elif invalid == "modulation_shape":
        m = m[:, :-1]
    elif invalid == "empty":
        x = x[:, :0]
    else:
        eps = {"negative_eps": -1.0, "nan_eps": float("nan"), "inf_eps": float("inf")}[invalid]
    with pytest.raises(exception, match=message):
        op(x, m, modulate_index=index, eps=eps)
    assert trace["actual_backend"] is None and trace["observation"] is None


@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize(
    "index_dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_cuda_indexed_full_domain_actual_execution_trace(dtype, broadcast, index_dtype):
    require_cuda()
    from tests.helpers.adaln_indexed import check_domain_routes

    index = torch.zeros(1 if broadcast else 2, 2, dtype=index_dtype)
    # Resolve using incomplete CPU metadata, execute with real CUDA inputs.
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    check_domain_routes(op, "cuda", dtype, index_dtype, broadcast)
    assert trace["actual_backend"] == "cuda"
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    assert trace["fallback_reason"] is None and not trace["fallback"]


@pytest.mark.parametrize("eps", [0.0, 2**-149, 2**-150, 1e300])
def test_indexed_strict_stays_closed_at_resolution_and_execution(eps):
    x, m, index, _, _ = small_case("cpu")
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, modulate_index=index, strict=True)
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, strict=True)
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        op(x, m, modulate_index=index, eps=eps)
    assert trace["actual_backend"] is None


@pytest.mark.cuda_only
@pytest.mark.parametrize("unsupported", ["H", "B", "eps_overflow", "eps_underflow"])
def test_cuda_indexed_execution_fallback_from_shared_resolution(unsupported):
    require_cuda()
    b, h = (65536, 2) if unsupported == "B" else (1, 4097 if unsupported == "H" else 3)
    x = torch.linspace(-1, 1, h, device="cuda").repeat(b, 1, 1).requires_grad_()
    m = torch.zeros(2 * b, 3 * h, device="cuda", requires_grad=True)
    index = torch.ones(1, 1, dtype=torch.int64, device="cuda")
    eps = {"eps_overflow": 1e300, "eps_underflow": 2**-150}.get(unsupported, 7.0)
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=3)
    actual = op(x, m, modulate_index=index, eps=eps)
    expected = NativeAdaLNModulationOp()(x, m, modulate_index=index, eps=eps)
    for a, e in zip(actual, expected, strict=True):
        assert_bytes(a, e)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    assert trace["fallback"]
    assert trace["fallback_reason"] == "modulate_index_requires_select01"
