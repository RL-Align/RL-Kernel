# SPDX-License-Identifier: Apache-2.0
"""The native adapter honors an explicit memory budget without changing math."""

import pytest
import torch
from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention
from rl_engine.contracts.operators.joint_attention import JointAttentionError


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="ROCm required"
)
def test_explicit_budget_is_shared_by_forward_and_backward(monkeypatch):
    monkeypatch.setenv("RLK_NATIVE_SPARSE_MAX_WORKSPACE_BYTES", "34359738368")
    op = NativeSparseAttention()
    assert op.max_workspace_bytes == 32 << 30
    q = torch.randn(
        (3, 32, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    kv = torch.randn((4, 512), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    sink = torch.zeros(32, device="cuda", requires_grad=True)
    indices = torch.full((3, 640), -1, device="cuda", dtype=torch.int32)
    indices[:, :4] = torch.arange(4, device="cuda", dtype=torch.int32)
    op.max_workspace_bytes = 1
    with pytest.raises(JointAttentionError, match="workspace budget exceeded"):
        op(q, kv, sink, indices, 512**-0.5)
    op.max_workspace_bytes = 32 << 30
    output = op(q, kv, sink, indices, 512**-0.5)
    reference = NativeSparseAttention(max_workspace_bytes=1 << 30)(
        q, kv, sink, indices, 512**-0.5
    )
    assert torch.equal(output.view(torch.int16), reference.view(torch.int16))
    op.max_workspace_bytes = 1
    with pytest.raises(JointAttentionError, match="backward workspace budget exceeded"):
        output.sum().backward()
    op.max_workspace_bytes = 32 << 30
    op(q, kv, sink, indices, 512**-0.5).sum().backward()
    for tensor in (q, kv, sink):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget_fails_before_backend_initialization(value):
    with pytest.raises(ValueError, match="positive native sparse workspace budget"):
        NativeSparseAttention(max_workspace_bytes=value)
