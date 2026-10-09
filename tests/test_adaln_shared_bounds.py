# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Public AdaLN preflight checks; enormous shapes use metadata, not allocations."""

import os

import pytest
import torch

from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
from rl_engine.kernels.registry import OpBackend, kernel_registry


@pytest.fixture
def cuda_op():
    try:
        return CudaAdaLNModulationOp()
    except RuntimeError:
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")


def test_cuda_shared_rejects_batch_above_grid_limit(cuda_op):
    device = "cuda" if torch.cuda.is_available() else "meta"
    x = torch.zeros(65536, 1, 1, device=device)
    m = torch.zeros(65536, 3, device=device)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_shared_maximum_batch_forward_backward(cuda_op, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.zeros(65535, 1, 1, device="cuda", dtype=dtype, requires_grad=True)
    m = torch.zeros(65535, 3, device="cuda", dtype=dtype, requires_grad=True)
    y, gate = cuda_op(x, m)
    dx, dm = torch.autograd.grad(gate, (x, m), torch.ones_like(gate))
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(gate, torch.zeros_like(gate))
    assert torch.equal(dx, torch.zeros_like(dx))
    expected = torch.zeros_like(m)
    expected[:, 2] = 1
    assert torch.equal(dm, expected)
