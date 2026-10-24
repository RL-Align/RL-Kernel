# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""AdaLN-local strict activation at the public resolver/callable seam."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.runtime.registry import KernelRegistry
from tests.helpers.adaln_indexed import small_case


def test_rocm_default_device_resolution_preserves_shared_and_indexed_fallback():
    root = Path(__file__).resolve().parents[2]
    # Simulate only the external platform APIs before real device discovery.
    script = """
import torch
torch.version.hip = "test-ROCm-runtime"
torch.cuda.is_available = lambda: True
from rl_engine.runtime.registry import KernelRegistry
from tests.helpers.adaln_indexed import check_domain_routes, assert_bytes

for device in (None, "cuda"):
    for indexed in (False, True):
        op, trace = KernelRegistry().get_adaln_modulation_op(
            device, hidden=2,
            modulate_index=torch.zeros(1, 2, dtype=torch.int32) if indexed else None)
        assert trace["selected_backend"] == "PYTORCH_ADALN_MODULATION"
        if indexed:
            check_domain_routes(op, "cpu", torch.float32, torch.int32, False)
            assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
            assert trace["fallback"] is True
        else:
            x = torch.tensor([[[-3.,3.]]],requires_grad=True)
            m = torch.zeros(1,6,requires_grad=True)
            y,g = op(x,m,eps=7.)
            dx,dm = torch.autograd.grad((y,g),(x,m),
                (torch.tensor([[[1.,0.]]]),torch.ones_like(g)))
            assert_bytes(y,torch.tensor([[[-.75,.75]]]))
            assert_bytes(dx,torch.tensor([[[7/128,-7/128]]]))
            assert_bytes(dm,torch.tensor([[1.,0.,-.75,0.,1.,1.]]))
            assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
        assert trace["actual_backend"] == "pytorch"
        try:
            KernelRegistry().get_adaln_modulation_op(
                device, hidden=2, strict=True,
                modulate_index=torch.zeros(1,2,dtype=torch.int32) if indexed else None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("ROCm framework accepted as strict GPU replay")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_ordinary_indexed_dispatch_adds_no_route_scan_or_scalar_read():
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
    from tests.helpers.adaln_indexed import assert_bytes

    x, m, index, _, _ = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, modulate_index=index)
    observed = []
    outputs = []
    for callable_op in (NativeAdaLNModulationOp(), op):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True
        ) as profile:
            outputs.append(callable_op(x, m, modulate_index=index, eps=7.0))
        observed.append({event.key: event.count for event in profile.key_averages()})
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        assert_bytes(actual, expected)
    for event in ("aten::all", "aten::item"):
        assert observed[0][event] == 1
        assert observed[1][event] == observed[0][event]
    assert trace["actual_backend"] == "pytorch" and trace["fallback"] is True
    assert trace["profiler_observed"] is False


def test_default_indexed_strict_rejection_clears_previous_shared_success():
    x, m, index, _, _ = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, strict=True)
    op(x, m[:2], eps=7.0)
    assert trace["actual_backend"] == "pytorch"
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        op(x, m, modulate_index=index, eps=7.0)
    for key in (
        "actual_backend",
        "observation",
        "execution_dtype",
        "output_dtype",
        "execution_device",
        "binary_path",
        "binary_sha256",
        "hardware_qualified",
    ):
        assert trace[key] is None
    assert trace["profiler_observed"] is False


def test_explicit_indexed_activation_resolves_without_enabling_cpu_select01():
    from rl_engine.runtime.adaln import IndexedAdaLNQualification

    scope = IndexedAdaLNQualification(
        backend="cuda",
        device="cuda:0",
        device_name="reviewed target",
        capability=(8, 6),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    x, m, index, _, _ = small_case("cpu")
    # Resolution is metadata-only; hardware is checked again before execution.
    op, trace = KernelRegistry().get_adaln_modulation_op(
        "cuda:0",
        hidden=2,
        modulate_index=index,
        strict=True,
        indexed_qualification=scope,
    )
    assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        op(x, m, modulate_index=index, eps=7.0)
    assert trace["actual_backend"] is None


@pytest.mark.parametrize("eps", [7.0, 1e300, 2**-150])
def test_strict_invalid_index_is_an_input_error_before_qualification_rejection(eps):
    x, m, index, _, _ = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, strict=True)
    op(x, m[:2], eps=7.0)
    with pytest.raises(ValueError, match="values must be 0 or 1"):
        op(x, m, modulate_index=index + 2, eps=eps)
    assert trace["actual_backend"] is None and trace["observation"] is None


def test_indexed_resolution_then_shared_execution_reports_actual_shared_contract():
    x, m, index, _, _ = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op("cpu", hidden=2, modulate_index=index)
    op(x, m, modulate_index=index, eps=7.0)
    assert trace["fallback"] is True
    y, gate = op(x, m[:2], eps=7.0)
    assert y.shape == x.shape and gate.shape == (2, 1, 2)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
    assert trace["fallback"] is False and trace["fallback_reason"] is None
    assert trace["kernel_fingerprint"] == "adaln_modulation_v1:PYTORCH_ADALN_MODULATION"


@pytest.mark.parametrize(
    "eps,reason",
    [(7.0, "NVIDIA CUDA tensors"), (1e300, "finite FP32 eps"), (2**-150, "finite FP32 eps")],
)
def test_activated_strict_rechecks_actual_domain_before_execution(eps, reason):
    from rl_engine.runtime.adaln import IndexedAdaLNQualification

    scope = IndexedAdaLNQualification(
        backend="cuda",
        device="cuda:0",
        device_name="reviewed target",
        capability=(8, 6),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    x, m, index, _, _ = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op(
        "cuda:0",
        hidden=2,
        modulate_index=index,
        strict=True,
        indexed_qualification=scope,
    )
    with pytest.raises(RuntimeError, match=reason):
        op(x, m, modulate_index=index, eps=eps)
    assert trace["actual_backend"] is None


def test_ordinary_gpu_resolution_rechecks_shared_cpu_device_with_real_fallback():
    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp
    from tests.helpers.adaln_indexed import assert_bytes

    x, m, _, dy, dg = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2)
    y, gate = op(x, m[:2], eps=7.0)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg[:, :1]))
    rx, rm = x.detach().requires_grad_(), m.detach().requires_grad_()
    ry, rg = NativeAdaLNModulationOp()(rx, rm[:2], eps=7.0)
    rdx, rdm = torch.autograd.grad((ry, rg), (rx, rm), (dy, dg[:, :1]))
    for a, e in zip((y, gate, dx, dm), (ry, rg, rdx, rdm), strict=True):
        assert_bytes(a, e)
    assert trace["actual_backend"] == "pytorch"
    assert trace["arithmetic_profile"] == "adaln-shared-fp32-tree-v1"
    assert trace["fallback"] and trace["fallback_reason"] == "shared_cuda_domain_unsupported"


@pytest.mark.parametrize(
    "index,error",
    [
        (torch.zeros(8, dtype=torch.int32), ValueError),
        (torch.zeros(1, 8), TypeError),
        (torch.zeros(1, 0, dtype=torch.int32), ValueError),
    ],
)
@pytest.mark.parametrize("strict", [False, True])
def test_resolver_rejects_illegal_route_metadata_before_policy(index, error, strict):
    with pytest.raises(error, match="modulate_index"):
        KernelRegistry().get_adaln_modulation_op(
            "cpu", hidden=2, modulate_index=index, strict=strict
        )


@pytest.mark.parametrize("mismatch", ["backend", "torch", "cuda", "device", "cpu", "hidden"])
def test_activation_scope_does_not_qualify_another_backend_or_runtime(mismatch):
    from rl_engine.runtime.adaln import IndexedAdaLNQualification

    scope = IndexedAdaLNQualification(
        backend="triton" if mismatch == "backend" else "cuda",
        device="cuda:1" if mismatch == "device" else "cuda:0",
        device_name="reviewed target",
        capability=(8, 6),
        torch_version="unqualified" if mismatch == "torch" else torch.__version__,
        cuda_version="unqualified" if mismatch == "cuda" else torch.version.cuda,
        triton_version="unqualified" if mismatch == "backend" else None,
    )
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        KernelRegistry().get_adaln_modulation_op(
            "cpu" if mismatch == "cpu" else "cuda:0",
            hidden=4097 if mismatch == "hidden" else 2,
            modulate_index=torch.zeros(1, 8, dtype=torch.int32),
            strict=True,
            indexed_qualification=scope,
        )


def test_cuda_device_string_on_hip_cannot_open_indexed_qualification(monkeypatch):
    # torch.version is the external platform boundary, never a registry collaborator.
    monkeypatch.setattr(torch.version, "hip", "test-ROCm-runtime")
    index = torch.zeros(1, 8, dtype=torch.int32)
    op, trace = KernelRegistry().get_adaln_modulation_op("cuda", hidden=2, modulate_index=index)
    assert trace["selected_backend"] == "PYTORCH_ADALN_MODULATION"
    assert trace["arithmetic_profile"] == "adaln-select01-framework-v1"
    with pytest.raises(RuntimeError, match="strict AdaLN rejects"):
        KernelRegistry().get_adaln_modulation_op(
            "cuda", hidden=2, modulate_index=index, strict=True
        )
    x, m, index, _, _ = small_case("cpu")
    op(x, m, modulate_index=index, eps=7.0)
    assert trace["actual_backend"] == "pytorch" and trace["fallback"]


@pytest.mark.parametrize("resolution", ["indexed", "shared"])
@pytest.mark.parametrize("hidden", [2, 3])
def test_ordinary_cpu_rechecks_mode_and_hidden_without_stale_trace(resolution, hidden):
    x, m, index, _, _ = small_case("cpu")
    op, trace = KernelRegistry().get_adaln_modulation_op(
        "cpu",
        hidden=hidden,
        modulate_index=index if resolution == "indexed" else None,
    )
    for indexed in (True, False, True):
        op(x, m if indexed else m[:2], modulate_index=index if indexed else None, eps=7.0)
        assert trace["arithmetic_profile"] == (
            "adaln-select01-framework-v1" if indexed else "adaln-shared-fp32-tree-v1"
        )
        assert trace["fallback"] is indexed
        assert trace["fallback_reason"] == ("modulate_index_requires_select01" if indexed else None)
