# SPDX-License-Identifier: Apache-2.0
"""Check the intermediate BF16 rounding and changing graph inputs of fused FFN."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm"
)


@pytest.mark.parametrize(
    "rows,n,k",
    [(1, 3072, 4096), (4, 6144, 4096), (8, 12288, 4096), (32, 24576, 4096), (4, 3074, 1088)],
)
def test_fused_swiglu_matches_separate_hip_bits(rows, n, k):
    from rl_engine import _C
    from rl_engine.backends.rocm.gemm.mfma_gemm import mfma_linear
    from rl_engine.backends.rocm.gemm.swiglu_gemm import forward

    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=a.dtype) * 0.01
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward(a, weight)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward(a, weight)
    for seed, scale in enumerate([0.0, 0.001, 0.1, 1.0, 10.0, 100.0]):
        torch.manual_seed(seed)
        a.copy_(torch.randn_like(a) * scale)
        weight.copy_(torch.randn_like(weight) * 0.01)
        reference = _C.swiglu_packed_forward(mfma_linear(a, weight))
        eager = forward(a, weight)
        graph.replay()
        for actual in (eager, captured):
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
