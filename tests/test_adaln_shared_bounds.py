# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Public AdaLN preflight checks; enormous shapes use metadata, not allocations."""

import os

import pytest
import torch

from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
from rl_engine.kernels.registry import OpBackend, kernel_registry


@pytest.fixture
def cuda_op(request):
    try:
        return CudaAdaLNModulationOp(threads=getattr(request, "param", 256))
    except RuntimeError:
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")


@pytest.mark.parametrize("cuda_op", [128, 256, 512], indirect=True)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_shared_rejects_batch_above_grid_limit(cuda_op, dtype):
    device = "cuda" if torch.cuda.is_available() else "meta"
    x = torch.zeros(65536, 1, 1, device=device, dtype=dtype)
    m = torch.zeros(65536, 3, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="B <= 65535"):
        cuda_op(x, m)


@pytest.mark.parametrize("shape", [(1, 524289, 4096), (1, 2**31, 1)])
def test_cuda_shared_rejects_int32_address_overflow_without_allocating(cuda_op, shape):
    x = torch.empty(shape, device="meta")
    m = torch.empty(shape[0], 3 * shape[-1], device="meta")
    with pytest.raises(ValueError, match="int32 indexing"):
        cuda_op(x, m)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_shared_rejects_hidden_4097(cuda_op, dtype):
    device = "cuda" if torch.cuda.is_available() else "meta"
    x = torch.empty(1, 1, 4097, device=device, dtype=dtype)
    m = torch.empty(1, 3 * 4097, device=device, dtype=dtype)
    with pytest.raises(RuntimeError, match="H <= 4096"):
        cuda_op(x, m)


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


@pytest.mark.parametrize("cuda_op", [128, 256, 512], indirect=True)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_shared_maximum_batch_forward_backward(cuda_op, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.tensor([-3.0, 3.0], device="cuda", dtype=dtype).repeat(65535, 1, 1)
    x.requires_grad_()
    m = torch.zeros(65535, 6, device="cuda", dtype=dtype)
    # Exact small integer tags expose gate batch addressing, including the last row.
    tag = (torch.arange(65535, device="cuda") % 17).to(dtype)
    m[:, 4], m[:, 5] = tag, -tag
    m.requires_grad_()
    y, gate = cuda_op(x, m, eps=7.0)
    dy = torch.tensor([1.0, 0.0], device="cuda", dtype=dtype).expand_as(y)
    dg = torch.tensor([2.0, -3.0], device="cuda", dtype=dtype).expand_as(gate)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    # variance+eps=16, norm=[-3/4,3/4]; the VJP is [7/128,-7/128].
    assert torch.equal(
        y, torch.tensor([-0.75, 0.75], device="cuda", dtype=dtype).expand_as(y)
    )
    assert torch.equal(gate[:, 0], torch.stack((tag, -tag), dim=1))
    expected_dx = torch.tensor([7 / 128, -7 / 128], device="cuda", dtype=dtype)
    assert torch.equal(dx, expected_dx.expand_as(dx))
    expected = torch.zeros_like(m)
    expected[:, 0], expected[:, 2] = 1.0, -0.75
    expected[:, 4], expected[:, 5] = 2.0, -3.0
    assert torch.equal(dm, expected)
