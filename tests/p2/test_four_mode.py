# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise actual recorded attention/RoPE/DetGemm execution and graph replay."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.p2.cuda_runtime import ensure_t06_cuda_kernel
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.fixtures.catalog import make_attn_case, make_oproj_case
from rl_engine.kernels.p2.four_mode import verify_recorded_four_modes
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp


@pytest.fixture(scope="module")
def cuda_runtime():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    # GPU callers hold /tmp/rl-kernel-t06-gpu.lock around the pytest process.
    ensure_t06_cuda_kernel()
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _inputs(layer_type, *, device="cuda", sink_mode="normal", tokens=2):
    case = make_attn_case(
        f"four-mode-{layer_type}",
        layer_type=layer_type,
        tokens=tokens,
        n_compressed={"C0": 0, "C4": 4, "C128": 3}[layer_type],
        n_recent=4,
        invalid_prefix=1,
        invalid_recent=1,
        sink_mode=sink_mode,
        device=device,
        seed=19,
    )
    proj = make_oproj_case("four-mode", tokens=tokens, device=device, seed=23)
    return (
        case.q,
        case.k,
        case.v,
        case.sink,
        case.plan,
        proj.w_a.to(torch.bfloat16),
        proj.w_b.to(torch.bfloat16),
        proj.cos,
        proj.sin,
        case.state_gate,
    )


def test_recorded_four_modes_rejects_cpu_graph():
    case = make_attn_case(
        "cpu-graph", layer_type="C0", tokens=2, n_compressed=0, n_recent=1
    )
    unused = torch.empty(0)
    with pytest.raises(P2FailClosedError) as exc:
        verify_recorded_four_modes(
            case.q, case.k, case.v, case.sink, case.plan,
            unused, unused, unused, unused, case.state_gate,
        )
    assert exc.value.status is P2Status.UNSUPPORTED_CAPABILITY


@pytest.mark.parametrize(
    "layer_type,sink_mode", [("C0", "normal"), ("C4", "shared"), ("C128", "normal")]
)
def test_recorded_four_modes_full_chain_cuda(cuda_runtime, monkeypatch, layer_type, sink_mode):
    # Observe real uncaptured full-sequence outputs so a second replay with
    # unchanged inputs cannot satisfy the changed-input requirement by label.
    eager_outputs = []
    captured_outputs = []
    original_forward = OProjGroupedOp.forward

    def observe_forward(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            captured_outputs.append(result.y)
        elif result.y.shape[0] == 2:
            eager_outputs.append(result.y.detach().clone())
        return result

    monkeypatch.setattr(OProjGroupedOp, "forward", observe_forward)
    report = verify_recorded_four_modes(*_inputs(layer_type, sink_mode=sink_mode))
    assert report["status"] == P2Status.PASS.value, report["checks"]
    assert report["four_mode_equal"] is True
    assert report["graph_includes_backward"] is True
    assert report["replays_with_changed_inputs"] == 2
    assert report["tokens"] == 2
    assert report["layer_type"] == layer_type
    assert "frozen" in report["candidate_state"]
    assert "inverse RoPE" in report["scope"]
    assert "DetGemm" in report["scope"]
    expected = {
        f"replay{replay}.{mode}.{tensor}"
        for replay in range(2)
        for mode in ("prefill", "eager_decode", "graph_decode", "graph_training")
        for tensor in ("O", "Y")
    }
    expected.update(
        f"replay{replay}.graph_training.{gradient}"
        for replay in range(2)
        for gradient in ("dQ", "dK", "dV", "dsink", "dW_a", "dW_b")
    )
    assert set(report["checks"]) == expected
    assert all(report["checks"].values())
    assert {result.shape[0] for result in captured_outputs} == {1, 2}
    assert len(eager_outputs) >= 4
    assert not torch.equal(eager_outputs[0], eager_outputs[-1])


def test_graph_output_corruption_fails_four_mode_equality(cuda_runtime, monkeypatch):
    captured_outputs = []
    original_forward = OProjGroupedOp.forward
    original_replay = torch.cuda.CUDAGraph.replay

    def observe_capture(self, *args, **kwargs):
        result = original_forward(self, *args, **kwargs)
        if torch.cuda.is_current_stream_capturing() and result.y.shape[0] == 1:
            captured_outputs.append(result.y)
        return result

    def corrupt_decode_output(graph):
        original_replay(graph)
        # Alter the real captured result after replay, before its comparison.
        with torch.no_grad():
            for output in captured_outputs:
                output.add_(1.0)

    monkeypatch.setattr(OProjGroupedOp, "forward", observe_capture)
    monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", corrupt_decode_output)
    report = verify_recorded_four_modes(*_inputs("C0"))
    assert captured_outputs
    assert report["four_mode_equal"] is False
    assert report["status"] == P2Status.BYTE_MISMATCH.value
    for replay in range(2):
        assert report["checks"][f"replay{replay}.graph_decode.Y"] is False
        assert report["checks"][f"replay{replay}.graph_decode.O"] is True
        assert report["checks"][f"replay{replay}.prefill.Y"] is True
        assert report["checks"][f"replay{replay}.graph_training.Y"] is True
