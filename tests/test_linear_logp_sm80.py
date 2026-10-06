# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.cuda.loss import linear_logp_sm80 as sm80
from rl_engine.kernels import registry as registry_module
from rl_engine.kernels.registry import KernelRegistry, OpBackend


def _a100_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
        and sm80.sm80_linear_logp_available()
    )


requires_a100 = pytest.mark.skipif(
    not _a100_available(), reason="requires compiled SM80 kernel on A100"
)


@pytest.fixture(scope="module")
def a100_weight():
    torch.manual_seed(1234)
    return torch.randn(
        sm80.SM80_LINEAR_LOGP_V,
        sm80.SM80_LINEAR_LOGP_D,
        device="cuda",
        dtype=torch.bfloat16,
    )


def _reference(hidden, weight, target):
    logits = F.linear(hidden.float(), weight.float())
    return logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)


@requires_a100
@pytest.mark.parametrize("n", [1, 16, 31, 32, 33, 128, 256, 512, 1024])
def test_sm80_basic_correctness(n, a100_weight):
    torch.manual_seed(1000 + n)
    hidden = torch.randn(
        n, sm80.SM80_LINEAR_LOGP_D, device="cuda", dtype=torch.bfloat16
    ) / sm80.SM80_LINEAR_LOGP_D**0.5
    target = torch.randint(sm80.SM80_LINEAR_LOGP_V, (n,), device="cuda")
    out = sm80.FusedLinearLogpSM80Op()(hidden, a100_weight, target)
    ref = _reference(hidden, a100_weight, target)
    err = (out - ref).abs()
    assert torch.isfinite(out).all()
    assert err.max().item() <= 5e-5
    assert err.mean().item() <= 1.5e-5


def _boundary_targets(split_v: int):
    tiles = sm80.SM80_LINEAR_LOGP_V // 128
    split_start = (tiles // split_v) * 128
    split_end = (2 * tiles // split_v) * 128
    return [
        0,
        sm80.SM80_LINEAR_LOGP_V - 1,
        split_start - 1,
        split_start,
        split_end - 1,
        split_end,
    ]


@requires_a100
@pytest.mark.parametrize("split_v,n", [(32, 33), (16, 513)])
def test_sm80_target_ownership_boundaries(split_v, n, a100_weight):
    torch.manual_seed(2000 + split_v)
    hidden = torch.randn(
        n, sm80.SM80_LINEAR_LOGP_D, device="cuda", dtype=torch.bfloat16
    ) / sm80.SM80_LINEAR_LOGP_D**0.5
    values = _boundary_targets(split_v)
    target = torch.tensor(
        [values[i % len(values)] for i in range(n)], device="cuda"
    )
    out = sm80.FusedLinearLogpSM80Op()(hidden, a100_weight, target)
    ref = _reference(hidden, a100_weight, target)
    assert torch.isfinite(out).all()
    assert (out - ref).abs().max().item() <= 5e-5


@requires_a100
def test_sm80_uses_current_cuda_stream(a100_weight):
    n = 33
    hidden = torch.randn(
        n, sm80.SM80_LINEAR_LOGP_D, device="cuda", dtype=torch.bfloat16
    ) / sm80.SM80_LINEAR_LOGP_D**0.5
    target = torch.randint(sm80.SM80_LINEAR_LOGP_V, (n,), device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        out = sm80.FusedLinearLogpSM80Op()(hidden, a100_weight, target)
    stream.synchronize()
    ref = _reference(hidden, a100_weight, target)
    assert (out - ref).abs().max().item() <= 5e-5


def test_sm80_runtime_dispatch_policy(monkeypatch):
    op = object.__new__(sm80.FusedLinearLogpSM80Op)
    native = mock.Mock(return_value=torch.tensor([1.0]))
    fallback = mock.Mock(return_value=torch.tensor([2.0]))
    monkeypatch.setattr(sm80, "_C", mock.Mock(fused_linear_logp_sm80=native))
    monkeypatch.setattr(sm80, "_fallback_op", lambda _hidden: fallback)

    hidden = torch.empty(1)
    weight = torch.empty(1)
    target = torch.zeros(1, dtype=torch.long)
    monkeypatch.setattr(sm80, "_is_native_supported", lambda *a, **k: True)
    assert op(hidden, weight, target).item() == 1.0
    native.assert_called_once()

    monkeypatch.setattr(sm80, "_is_native_supported", lambda *a, **k: False)
    assert op(hidden, weight, target).item() == 2.0
    fallback.assert_called_once()


def test_registry_prioritizes_sm80_only_on_sm80(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        fused_linear_logp_sm80 = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda, "get_device_capability", lambda: (8, 0)
    )
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())
    registry = KernelRegistry()
    assert registry._priority_map["cuda"]["linear_logp"][0] is (
        OpBackend.CUDA_FUSED_LINEAR_LOGP_SM80
    )


def test_registry_sm80_does_not_preempt_sm90(monkeypatch):
    from rl_engine.kernels.ops import base as base_module

    class FakeExtension:
        fused_linear_logp_sm80 = object()
        fused_linear_logp_sm90 = object()

    monkeypatch.setattr(registry_module.device_ctx, "device_type", "cuda")
    monkeypatch.setattr(
        registry_module.torch.cuda, "get_device_capability", lambda: (9, 0)
    )
    monkeypatch.setattr(base_module, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(base_module, "_C", FakeExtension())
    registry = KernelRegistry()
    assert registry._priority_map["cuda"]["linear_logp"][0] is (
        OpBackend.CUDA_FUSED_LINEAR_LOGP_SM90
    )
    assert OpBackend.CUDA_FUSED_LINEAR_LOGP_SM80 not in registry._priority_map[
        "cuda"
    ]["linear_logp"]


@requires_a100
@pytest.mark.parametrize("n,expected", [(128, "native_sm80"), (512, "native_sm80"), (1024, "native_sm80"), (2048, "fallback")])
def test_sm80_selected_backend(n, expected, a100_weight):
    hidden = torch.empty(
        n, sm80.SM80_LINEAR_LOGP_D, device="cuda", dtype=torch.bfloat16
    )
    target = torch.zeros(n, device="cuda", dtype=torch.long)
    assert sm80.FusedLinearLogpSM80Op().selected_backend(
        hidden, a100_weight, target
    ) == expected


@requires_a100
def test_sm80_unsupported_configurations_fallback(a100_weight):
    op = sm80.FusedLinearLogpSM80Op()
    target = torch.zeros(1, device="cuda", dtype=torch.long)
    bf16 = torch.empty(1, sm80.SM80_LINEAR_LOGP_D, device="cuda", dtype=torch.bfloat16)
    assert op.selected_backend(bf16.float(), a100_weight, target) == "fallback"
    assert op.selected_backend(bf16, a100_weight[:, :2048], target) == "fallback"
    assert op.selected_backend(bf16.requires_grad_(), a100_weight, target) == "fallback"
