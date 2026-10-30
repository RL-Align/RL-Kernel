# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Check strict backend selection on CPU with instrumented native exports."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.dsv4.attention import cuda_runtime
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import _det_gemm_linear
from rl_engine.kernels.ops import base
from rl_engine.kernels.ops.cuda.matmul import det_gemm


@pytest.fixture
def native_runtime(monkeypatch):
    calls = []
    active_devices = []
    device_entries = []

    @contextmanager
    def device_context(device):
        active_devices.append(device)
        device_entries.append(device)
        try:
            yield
        finally:
            active_devices.pop()

    monkeypatch.setattr(torch.cuda, "device", device_context)

    def forward(x, weight):
        assert active_devices[-1] == x.device
        calls.append("forward")
        return (x.float() @ weight.float().T).bfloat16()

    def input_gradient(grad, weight):
        assert active_devices[-1] == grad.device
        calls.append("input_gradient")
        return (grad.float() @ weight.float()).bfloat16()

    def weight_gradient(x, grad):
        assert active_devices[-1] == x.device
        calls.append("weight_gradient")
        return (grad.float().T @ x.float()).bfloat16()

    native = SimpleNamespace(
        det_gemm_fwd_rhs_transposed=forward,
        det_gemm_fwd=input_gradient,
        det_gemm_db_transposed=weight_gradient,
        det_gemm_sm90_compiled=lambda: True,
        active_devices=active_devices,
        device_entries=device_entries,
    )
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(det_gemm, "_C", None)
    monkeypatch.setattr(det_gemm, "_EXT_AVAILABLE", False)
    monkeypatch.setattr(det_gemm, "_CUBLASLT_CONFIGURED", False)
    monkeypatch.setattr(det_gemm, "_ROUTE_REPORTED", False)
    monkeypatch.setattr(cuda_runtime, "ensure_native_kernels", lambda: "jit")
    return native, calls


@pytest.mark.parametrize("backend", ["sm90", "cublaslt_nosplitk"])
def test_selected_backend_handles_forward_and_both_gradients(native_runtime, monkeypatch, backend):
    native, calls = native_runtime
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", backend)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    configurations = []
    monkeypatch.setattr(
        det_gemm, "_configure_cublaslt_nosplitk", lambda *args: configurations.append(True)
    )
    linear = _det_gemm_linear()
    assert det_gemm._C is native and det_gemm._EXT_AVAILABLE
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16, requires_grad=True)
    weight = torch.tensor([[1, 0, -1], [0, 2, 1]], dtype=torch.bfloat16, requires_grad=True)
    y = linear(x, weight)
    y.backward(torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16))
    assert torch.equal(y, torch.tensor([[-2, 7], [-2, 16]], dtype=torch.bfloat16))
    assert torch.equal(x.grad, torch.tensor([[1, 4, 1], [3, 8, 1]], dtype=torch.bfloat16))
    assert torch.equal(
        weight.grad, torch.tensor([[13, 17, 21], [18, 24, 30]], dtype=torch.bfloat16)
    )
    assert native.device_entries == [x.device, x.device]
    assert not native.active_devices
    if backend == "sm90":
        assert calls == ["forward", "input_gradient", "weight_gradient"]
        assert not configurations
    else:
        assert not calls
        assert len(configurations) == 4


def test_backend_constructed_on_input_device(native_runtime, monkeypatch):
    from rl_engine.kernels.dsv4.attention.o_proj import o_proj_grouped

    native, _ = native_runtime
    x = torch.ones(1, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.Tensor, "device", property(lambda self: torch.device("cuda:1")))

    def construct():
        assert native.active_devices == [torch.device("cuda:1")]
        return lambda x, weight: x

    monkeypatch.setattr(o_proj_grouped, "_det_gemm_linear", construct)
    o_proj_grouped.OProjGroupedOp(backend="auto")._linear(x)
    assert not native.active_devices


def test_mixed_linear_devices_rejected_before_launch(native_runtime, monkeypatch):
    _, calls = native_runtime
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "sm90")
    linear = _det_gemm_linear()
    with pytest.raises(DSv4FailClosedError, match="same device") as exc:
        linear(torch.empty(1), torch.empty(1, device="meta"))
    assert exc.value.status is DSv4Status.SCHEMA_MISMATCH
    assert not calls


@pytest.mark.parametrize("backend", ["sm90", "auto"])
def test_uncompiled_sm90_rejected_before_native_call(native_runtime, monkeypatch, backend):
    native, calls = native_runtime
    native.det_gemm_sm90_compiled = lambda: False
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", backend)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("CUBLASLT_WORKSPACE_SIZE", raising=False)
    with pytest.raises(DSv4FailClosedError, match="refusing naive fallback") as exc:
        _det_gemm_linear()
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY
    assert not calls


def test_cublaslt_requires_startup_configuration(native_runtime, monkeypatch):
    _, calls = native_runtime
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublaslt_nosplitk")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    with pytest.raises(DSv4FailClosedError, match="CUBLAS_WORKSPACE_CONFIG=:16:8") as exc:
        _det_gemm_linear()
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY
    assert not calls
