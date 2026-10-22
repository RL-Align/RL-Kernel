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
    assert trace["fallback_reason"] == (
        None if accelerated else "modulate_index_requires_select01"
    )
    assert trace["arithmetic_profile"] == (
        "adaln-indexed-fp32-tree-v1" if accelerated else "adaln-select01-framework-v1"
    )
    assert trace["requested_backend"] == ("cuda" if accelerated else "pytorch")
    assert trace["kernel_fingerprint"] == (
        "adaln_indexed_cuda_v1" if accelerated else "adaln_modulation_select01_native_v1"
    )
    assert trace["reduction_order"] == (
        "pairwise_lower_upper_H; ascending_selected_token_S_per_bank"
        if accelerated else "pairwise_lower_upper_H; framework_select01_backward"
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

    fallback_x = x[:, :7] if unsupported == "geometry" else x
    fallback_index = index[:, :7] if unsupported == "geometry" else index
    eps = {"eps": 1e300, "eps_underflow": 2**-150, "geometry": 7.0}[unsupported]
    actual = op(fallback_x, m, modulate_index=fallback_index, eps=eps)
    expected = NativeAdaLNModulationOp()(fallback_x, m, modulate_index=fallback_index, eps=eps)
    for a, e in zip(actual, expected, strict=True):
        assert_bytes(a, e)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    assert trace["fallback_reason"] == "indexed_cuda_slice_unsupported"

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
