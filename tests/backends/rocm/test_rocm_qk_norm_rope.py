# SPDX-License-Identifier: Apache-2.0
"""The fused decode path must preserve normalization rounding and RoPE bits."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm"
)


@pytest.mark.parametrize("heads", [(4, 1), (8, 2), (16, 4), (32, 8)])
@pytest.mark.parametrize("batch", [1, 4, 32])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_changed_inputs_and_positions_in_graph_are_bitwise(heads, batch, eps):
    from rl_engine.backends.rocm.rope.qk_norm_rope import strict_qk_norm_rope
    from rl_engine.backends.rocm.rope.rope import RocmDeterministicRoPEOp

    hq, hk = heads
    backing = torch.empty(batch, hq + 2 * hk, 128, device="cuda", dtype=torch.bfloat16)
    query, key = backing[:, :hq], backing[:, hq : hq + hk]
    weights = [torch.randn(128, device="cuda", dtype=backing.dtype) for _ in range(2)]
    positions = torch.zeros(batch, device="cuda", dtype=torch.int64)
    op = RocmDeterministicRoPEOp()
    cosine, sine = op.build_position_table(8192, 128, device=query.device, theta=1e6)

    def call():
        return strict_qk_norm_rope(query, key, *weights, positions, cosine, sine, eps, eps * 2)

    backing.normal_()
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
        backing.copy_(torch.randn_like(backing) * scale)
        positions.random_(0, 8192)
        reference = []
        for value, weight, epsilon in zip((query, key), weights, (eps, eps * 2), strict=True):
            norm = torch.nn.functional.rms_norm(value, (128,), weight, epsilon)
            reference.append(
                op.forward_token_major(
                    norm.reshape(batch, -1),
                    positions,
                    cosine,
                    sine,
                    head_dim=128,
                ).reshape(value.shape)
            )
        eager = call()
        graph.replay()
        for result in (eager, captured):
            for actual, expected in zip(result, reference, strict=True):
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_large_prefill_keeps_eager_norm_contract():
    from rl_engine.backends.rocm.rope.qk_norm_rope import strict_qk_norm_rope
    from rl_engine.backends.rocm.rope.rope import RocmDeterministicRoPEOp

    q = torch.randn(160, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(160, 2, 128, device="cuda", dtype=q.dtype)
    wq = torch.randn(128, device="cuda", dtype=q.dtype)
    wk = torch.randn_like(wq)
    positions = torch.arange(160, device="cuda")
    op = RocmDeterministicRoPEOp()
    cos, sin = op.build_position_table(160, 128, device=q.device, theta=1e6)
    actual = strict_qk_norm_rope(q, k, wq, wk, positions, cos, sin, 1e-6, 1e-6)
    for value, weight, output in zip((q, k), (wq, wk), actual, strict=True):
        norm = torch.nn.functional.rms_norm(value, (128,), weight, 1e-6)
        expected = op.forward_token_major(norm.reshape(160, -1), positions, cos, sin, head_dim=128)
        assert torch.equal(
            output.reshape_as(expected).view(torch.int16), expected.view(torch.int16)
        )
