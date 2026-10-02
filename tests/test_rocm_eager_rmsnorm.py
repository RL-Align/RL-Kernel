# SPDX-License-Identifier: Apache-2.0
"""Bitwise parity against eager PyTorch, including strided Q/K and graph replay."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm"
)


@pytest.mark.parametrize("width", [4, 64, 128, 256, 1024, 4096, 8192])
@pytest.mark.parametrize("eps", [1e-6, 1e-5, 0.01])
@pytest.mark.parametrize("add", [False, True])
def test_bitwise_eager_and_changing_graph_inputs(width, eps, add):
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import strict_add_rms_norm, strict_rms_norm
    from rl_engine.kernels.ops.rocm import rmsnorm

    backing = torch.empty((4, 3, 8, width), device="cuda", dtype=torch.bfloat16)
    x = backing[:, 1]  # Q/K heads are strided slices of the QKV projection.
    residual = torch.empty_like(x) if add else None
    weight = torch.randn(width, device="cuda", dtype=x.dtype)
    if not rmsnorm.supports(x, weight, residual):
        pytest.skip("requires the pinned gfx942/PyTorch RMSNorm contract")

    def call():
        return (
            strict_add_rms_norm(x, residual, weight, eps=eps)
            if add
            else strict_rms_norm(x, weight, eps=eps)
        )

    x.normal_()
    if add:
        residual.normal_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    for seed, scale in enumerate([0.0, 0.001, 0.1, 1.0, 100.0]):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x) * scale)
        x[..., 0] = -0.0
        if add:
            residual.copy_(torch.randn_like(residual) * scale)
        updated = x + residual if add else x
        reference = torch.nn.functional.rms_norm(updated, (width,), weight, eps)
        eager = call()
        graph.replay()
        for result in (eager, captured):
            actual = result[0] if add else result
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
            if add:
                assert torch.equal(result[1].view(torch.int16), updated.view(torch.int16))


@pytest.mark.parametrize("dtype,width", [(torch.float16, 128), (torch.bfloat16, 96)])
def test_other_contracts_keep_eager_path(dtype, width):
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import strict_rms_norm
    from rl_engine.kernels.ops.rocm import rmsnorm

    x = torch.randn(4, width, device="cuda", dtype=dtype)
    weight = torch.randn(width, device="cuda", dtype=dtype)
    assert not rmsnorm.supports(x, weight, None)
    actual = strict_rms_norm(x, weight, eps=1e-6)
    expected = torch.nn.functional.rms_norm(x, (width,), weight, 1e-6)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
