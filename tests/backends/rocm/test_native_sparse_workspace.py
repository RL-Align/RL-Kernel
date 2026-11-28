# SPDX-License-Identifier: Apache-2.0
"""Budget routing and the target geometry can be checked without allocating GPU memory."""

import types

import pytest
import torch

from rl_engine.backends.rocm.attention.native_sparse import (
    NativeSparseAttention,
    sparse_workspace_budget,
)
from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend
from rl_engine.contracts.operators.joint_attention import (
    JointAttentionContract,
    JointAttentionError,
)


def test_explicit_sparse_budget_keeps_default_and_validation(monkeypatch):
    monkeypatch.delenv("RLK_NATIVE_SPARSE_MAX_WORKSPACE_BYTES", raising=False)
    monkeypatch.delenv("RLK_NATIVE_SPARSE_WORKSPACE_BYTES", raising=False)
    assert sparse_workspace_budget() == 1 << 30
    monkeypatch.setenv("RLK_NATIVE_SPARSE_WORKSPACE_BYTES", str(16 << 30))
    assert sparse_workspace_budget() == 16 << 30
    assert sparse_workspace_budget(2 << 30) == 2 << 30
    for value in (0, -1, True, 1.5, "large"):
        with pytest.raises(ValueError):
            sparse_workspace_budget(value)
    for value in ("invalid", "0", "-1"):
        monkeypatch.setenv("RLK_NATIVE_SPARSE_WORKSPACE_BYTES", value)
        with pytest.raises(ValueError):
            sparse_workspace_budget()


def test_primary_sparse_budget_takes_precedence_over_legacy_alias(monkeypatch):
    monkeypatch.setenv("RLK_NATIVE_SPARSE_WORKSPACE_BYTES", str(2 << 30))
    monkeypatch.setenv("RLK_NATIVE_SPARSE_MAX_WORKSPACE_BYTES", str(16 << 30))
    assert sparse_workspace_budget() == 16 << 30
    assert sparse_workspace_budget(4 << 30) == 4 << 30
    for invalid in ("invalid", "", "0", "-1"):
        monkeypatch.setenv("RLK_NATIVE_SPARSE_MAX_WORKSPACE_BYTES", invalid)
        with pytest.raises(ValueError):
            sparse_workspace_budget()


def test_target_geometry_remains_checked_against_explicit_budget():
    with pytest.raises(JointAttentionError, match="workspace budget exceeded"):
        JointAttentionContract(64, 512, 8192, 8192, 0, 640, graph=False)
    contract = JointAttentionContract(
        64, 512, 8192, 8192, 0, 640, graph=False, max_workspace_bytes=16 << 30
    )
    assert 1 << 30 < contract.workspace_bytes < 16 << 30
    assert 1 << 30 < AiterJointBackwardBackend.workspace_bytes(8192, 64, 512, 640, 8192) < 16 << 30


def test_forward_and_backward_use_the_same_budget_without_changing_operands():
    seen = []
    operator = NativeSparseAttention.__new__(NativeSparseAttention)
    operator.max_workspace_bytes = 16 << 30
    operator.executed = True

    def forward(q, kv, sink, indices, out, lse, contract):
        assert contract.max_workspace_bytes == operator.max_workspace_bytes
        seen.append("forward")
        out.copy_(q)
        lse.zero_()

    def allocate(**kwargs):
        assert kwargs["max_bytes"] == operator.max_workspace_bytes
        seen.append("backward")
        return types.SimpleNamespace(
            dq=torch.empty_like(q), dkv=torch.empty_like(kv), dsink=torch.empty_like(sink)
        )

    operator.forward_backend = types.SimpleNamespace(forward_into=forward)
    operator.backward_backend = types.SimpleNamespace(
        build_csr=lambda i, n: (None, None), allocate=allocate, backward_into=lambda *args: None
    )
    q = torch.empty(8192, 64, 512, device="meta", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.empty(8192, 512, device="meta", dtype=torch.bfloat16, requires_grad=True)
    sink = torch.empty(64, device="meta", requires_grad=True)
    indices = torch.empty(8192, 640, device="meta", dtype=torch.int32)
    operator(q, kv, sink, indices, 512**-0.5).sum().backward()
    assert seen == ["forward", "backward"]


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
def test_budget_does_not_change_forward_or_input_gradient_bits():
    torch.manual_seed(7)
    q0 = torch.randn(8, 4, 512, dtype=torch.bfloat16, device="cuda")
    kv0 = torch.randn(16, 512, dtype=torch.bfloat16, device="cuda")
    sink0 = torch.randn(4, dtype=torch.float32, device="cuda")
    indices = torch.full((8, 640), -1, dtype=torch.int32, device="cuda")
    indices[:, :4] = torch.arange(4, dtype=torch.int32, device="cuda")
    dy = torch.randn_like(q0)
    results = []
    for budget in (1 << 30, 16 << 30):
        q, kv, sink = [t.detach().clone().requires_grad_() for t in (q0, kv0, sink0)]
        operator = NativeSparseAttention(max_workspace_bytes=budget)
        out = operator(q, kv, sink, indices, 512**-0.5)
        out.backward(dy)
        tensors = [out.detach(), q.grad, kv.grad, sink.grad]
        assert all(torch.isfinite(t).all() for t in tensors)
        results.append([t.contiguous().view(torch.uint8).cpu() for t in tensors])
    assert all(torch.equal(a, b) for a, b in zip(*results, strict=True))
