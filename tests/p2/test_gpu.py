# SPDX-License-Identifier: Apache-2.0
"""Opt-in hardware infrastructure gates, NOT certification of T02--T07 kernels."""

import os

import pytest
import torch

from rl_engine.p2 import oracle
from rl_engine.p2.fixtures import values

RUN_GPU = os.environ.get("P2_RUN_GPU") == "1"
pytestmark = pytest.mark.skipif(not RUN_GPU, reason="set P2_RUN_GPU=1 for CUDA hardware gates")
DEVICES = (list(range(torch.cuda.device_count())) or [0]) if RUN_GPU else [0]


@pytest.mark.parametrize("device_index", DEVICES)
def test_gpu_fp4_and_fp32_reference_bytes(device_index):
    assert torch.cuda.is_available(), "requested hardware gate must not silently skip"
    device = torch.device("cuda", device_index)
    x = values((2, 64, 128), 3)
    for function in (oracle.scale, oracle.fixed_sum, oracle.hadamard):
        cpu = function(x)
        gpu = function(x.to(device)).cpu()
        assert torch.equal(cpu.view(torch.uint8), gpu.view(torch.uint8))
    p, s = oracle.pack_mxfp4(x)
    gp, gs = oracle.pack_mxfp4(x.to(device))
    assert torch.equal(p, gp.cpu()) and torch.equal(s, gs.cpu())


@pytest.mark.parametrize("device_index", DEVICES)
@pytest.mark.parametrize("mode", ["training", "prefill", "eager_decode", "graph_decode"])
def test_per_token_cuda_transport_lifecycle(device_index, mode):
    """The mock's opaque state bytes survive functional/eager/chunk/real graph writes."""
    assert torch.cuda.is_available(), "requested hardware gate must not silently skip"
    with torch.cuda.device(device_index):
        source = torch.zeros(1, 16, dtype=torch.uint8, device="cuda")
        slot = torch.zeros(1, dtype=torch.int64, device="cuda")
        cache = torch.zeros(128, 16, dtype=torch.uint8, device="cuda")
        expected = torch.zeros(128, 16, dtype=torch.uint8)
        addresses = source.data_ptr(), slot.data_ptr(), cache.data_ptr()
        graph = None
        if mode == "graph_decode":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    cache.index_copy_(0, slot, source)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            # Explicit per-device stream: the default graph context can reuse
            # a cached stream from the first GPU in a multi-device process.
            with torch.cuda.graph(graph, stream=stream):
                cache.index_copy_(0, slot, source)
            cache.zero_()
        chunks = [7] * 36 + [5] if mode == "prefill" else [257]
        position = 0
        for chunk in chunks:
            for _ in range(chunk):
                row = ((torch.arange(16) + position) % 256).to(torch.uint8)
                expected[position % 128] = row
                source.copy_(row.unsqueeze(0))
                slot.fill_(position % 128)
                if mode == "training":
                    cache = cache.index_copy(0, slot, source)
                elif graph is not None and position >= 65:
                    graph.replay()
                    assert addresses == (source.data_ptr(), slot.data_ptr(), cache.data_ptr())
                else:
                    cache.index_copy_(0, slot, source)
                assert torch.equal(cache.cpu(), expected), (mode, position)
                position += 1
        assert position == 257
