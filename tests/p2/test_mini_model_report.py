# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU-only checks for mini-model report lifecycle and final verdicts."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def mini_model():
    path = Path(__file__).resolve().parents[2] / "examples/p2_t06_min_model.py"
    spec = importlib.util.spec_from_file_location("t06_mini_model_report_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure_stage", ["initialization", "training"])
def test_main_replaces_stale_pass_before_run_and_persists_failure(
    mini_model, monkeypatch, tmp_path, failure_stage
):
    report_path = tmp_path / "model.json"
    report_path.write_text(json.dumps({"status": "PASS", "four_mode_equal": True}))
    monkeypatch.setattr(sys, "argv", ["mini-model", "--json-out", str(report_path)])
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", previous_tf32)
    failure = RuntimeError(f"injected {failure_stage} failure")
    observed = []

    def fail(*_args):
        # This executes inside the actual main() boundary, before it handles
        # the failure, so retaining the previous PASS file cannot pass.
        pending = json.loads(report_path.read_text())
        observed.append(pending)
        assert pending["status"] == "RUNNING"
        assert pending["four_mode_equal"] is False
        assert pending["cuda_graph_status"] == "NOT_RUN"
        raise failure

    if failure_stage == "initialization":
        # Exercise the real _run until CUDA initialization; no GPU call occurs.
        monkeypatch.setattr(mini_model, "ensure_t06_cuda_kernel", fail)
    else:
        # Any exception escaping the training body has the same main boundary.
        monkeypatch.setattr(mini_model, "_run", fail)
    with pytest.raises(RuntimeError) as exc:
        mini_model.main()
    assert exc.value is failure
    assert len(observed) == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "FAIL"
    assert report["four_mode_equal"] is False
    assert report["cuda_graph_status"] != "PASS"
    assert report["error"] == f"RuntimeError: injected {failure_stage} failure"


class _CpuLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.5))
        self.sink = self.w_a = self.w_b = torch.zeros(1)
        self.state_gate = object()
        self.forward_calls = 0

    def recorded_inputs(self, hidden, _cos, _sin):
        return hidden, hidden, hidden, object()

    def forward(self, hidden, _cos, _sin):
        self.forward_calls += 1
        return hidden + self.bias


class _CpuModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_CpuLayer(), _CpuLayer()])

    def forward(self, hidden, cos, sin):
        for layer in self.layers:
            hidden = layer(hidden, cos, sin)
        return hidden


@pytest.mark.parametrize(
    "mode_result, update_weights, expected_rc, expected_status, expected_graph",
    [
        ("PASS", True, 0, "PASS", "PASS"),
        ("BYTE_MISMATCH", True, 2, "FAIL", "FAIL"),
        ("exception", True, 2, "FAIL", "FAIL"),
        ("PASS", False, 2, "FAIL", "PASS"),
    ],
)
def test_run_aggregates_real_training_and_mode_verdicts_on_cpu(
    mini_model, monkeypatch, tmp_path,
    mode_result, update_weights, expected_rc, expected_status, expected_graph,
):
    # Keep the real _run training loop, optimizer, loss checks, layer iteration,
    # JSON writer and exit-code logic. Replace only the large GPU model and
    # four-mode kernel measurement with small CPU boundary fixtures.
    model = _CpuModel()
    monkeypatch.setattr(mini_model, "P2MiniModel", lambda: model)
    monkeypatch.setattr(mini_model, "HIDDEN_SIZE", 2)
    monkeypatch.setattr(mini_model, "LAYER_TABLE", ("C0", "C4"))
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", previous_tf32)
    observed_layers = []

    def measured_modes(*_args):
        observed_layers.append(len(observed_layers))
        if mode_result == "exception":
            raise RuntimeError("injected capture failure")
        # Fail only the second layer to ensure all layer results count.
        status = "PASS" if len(observed_layers) == 1 else mode_result
        return {"status": status, "four_mode_equal": status == "PASS"}

    monkeypatch.setattr(mini_model, "verify_recorded_four_modes", measured_modes)
    if not update_weights:
        monkeypatch.setattr(torch.optim.Adam, "step", lambda self: None)
    args = SimpleNamespace(
        device="cpu", tokens=2, steps=2, json_out=tmp_path / "model.json"
    )
    assert mini_model._run(args) == expected_rc
    report = json.loads(args.json_out.read_text())
    assert report["status"] == expected_status
    assert report["cuda_graph_status"] == expected_graph
    assert report["four_mode_equal"] is (mode_result == "PASS")
    assert report["loss_dropped"] is update_weights
    assert len(report["losses"]) == 2
    assert len(observed_layers) == (1 if mode_result == "exception" else 2)
    assert len(report["four_mode_layers"]) == len(observed_layers)
    assert [layer.forward_calls for layer in model.layers] == [
        args.steps + int(mode_result != "exception"), args.steps
    ]
    if mode_result == "exception":
        assert "injected capture failure" in report["four_mode_layers"][0]["error"]
