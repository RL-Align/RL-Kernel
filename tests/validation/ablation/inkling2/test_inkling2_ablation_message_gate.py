# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.inkling2.message_gates import GateWeights, STREAMS, blend_loop_streams
from rl_engine.validation.ablation.inkling2.message_gate import message_gate_probe
from rl_engine.validation.ablation.inkling2.overlay import compare_boundaries


@pytest.mark.parametrize("platform", ["cpu", "cuda", "rocm"])
def test_gate_fault_is_stream_local(platform):
    if platform != "cpu" and (
        not torch.cuda.is_available() or (torch.version.hip is not None) != (platform == "rocm")
    ):
        pytest.skip(f"{platform} required")
    device = "cpu" if platform == "cpu" else "cuda"
    dst = {s: torch.ones(2, 4, device=device) for s in STREAMS}
    msg = {s: torch.full((2, 4), float(i + 1), device=device) for i, s in enumerate(STREAMS)}
    weights = {
        s: GateWeights(
            torch.full((64, 8), 0.125, device=device),
            torch.full((4, 64), 0.015625, device=device),
            torch.zeros(4, 64, device=device),
        )
        for s in STREAMS
    }
    base = blend_loop_streams(dst, msg, weights)
    for fault in ("bypass", "misroute"):
        for stream in STREAMS:
            changed = message_gate_probe(dst, msg, weights, stream=stream, fault=fault)
            result = compare_boundaries(base, changed)
            assert result["first_drift"] == stream
            assert all(n == 0 for s, n in result["mismatches"].items() if s != stream)
