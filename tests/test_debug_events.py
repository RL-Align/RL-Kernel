# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Portable CPU regressions; the same native observer also runs on ROCm/CUDA."""

import json

import pytest
import torch
from torch import nn

from rl_engine.alignment.debug import (
    AutoCapture,
    EventCapture,
    LogicalTensor,
    ModuleBoundary,
    prepare_auto,
)
from rl_engine.alignment.debug.entry import analyze, main


def identity():
    return dict(weights="frozen", tokens="fixed", positions=[0, 1], mask=[1, 1], sampling={})


def capture(root, side, model, x, **kwargs):
    with AutoCapture(root, side, identity(), model, positions=[0, 1], **kwargs):
        return model(x)


class Reverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.late = nn.Identity()
        self.early = nn.Identity()

    def forward(self, x):
        return self.late(self.early(x))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_execution_not_registration_order(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("device smoke requires CUDA or ROCm")
    model = Reverse().to(device)
    x = torch.ones(2, 3, device=device)
    prepare_auto(tmp_path, model, positions=[0, 1], scope="native eager order")
    capture(tmp_path, "training", model, x)
    handle = model.early.register_forward_hook(lambda m, a, out: out + 0.125)
    try:
        capture(tmp_path, "rollout", model, x)
    finally:
        handle.remove()
    report = analyze(tmp_path)
    first = report["first_divergence"]
    assert report["status"] == "diverged"
    assert first["boundary"] == "auto.early.output"
    assert first["input_evidence"] == "equal"
    assert first["last_equal_boundary"]["boundary"] == "auto.early.input"


class Repeated(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Identity()
        self.count = 2

    def forward(self, x):
        for _ in range(self.count):
            x = self.shared(x)
        return x


def test_second_invocation_is_not_dropped(tmp_path):
    model, x = Repeated(), torch.ones(2, 3)
    prepare_auto(tmp_path, model, positions=[0, 1], scope="reused module")
    capture(tmp_path, "training", model, x)
    calls = []

    def inject(m, a, out):
        calls.append(1)
        return out + 0.125 if len(calls) == 2 else out

    handle = model.shared.register_forward_hook(inject)
    try:
        actual = capture(tmp_path, "rollout", model, x)
    finally:
        handle.remove()
    assert not torch.equal(actual, x)
    report = analyze(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "auto.shared.output"
    assert report["first_divergence"]["invocation"] == 1


def test_different_call_count_is_not_comparable(tmp_path):
    model, x = Repeated(), torch.ones(2, 3)
    prepare_auto(tmp_path, model, positions=[0, 1], scope="different trace")
    capture(tmp_path, "training", model, x)
    model.count = 3
    capture(tmp_path, "rollout", model, x)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert report["first_divergence"] is None
    assert any("order/count" in e for e in report["errors"])


def test_all_output_leaves_are_compared(tmp_path):
    class Multi(nn.Module):
        def forward(self, x):
            return {"hidden": x, "state": (None, x.clone())}

    model, x = nn.Sequential(Multi()), torch.ones(2, 3)
    prepare_auto(tmp_path, model, positions=[0, 1], scope="structured output")
    capture(tmp_path, "training", model, x)
    handle = model[0].register_forward_hook(
        lambda m, a, out: {"hidden": out["hidden"], "state": (None, out["state"][1] + 1)}
    )
    try:
        capture(tmp_path, "rollout", model, x)
    finally:
        handle.remove()
    first = analyze(tmp_path)["first_divergence"]
    assert first["boundary"] == "auto.0.output"
    assert first["tensor_path"] == "$['state'][1]"


def test_keyword_inputs_and_multiple_forwards(tmp_path):
    class Keyword(nn.Module):
        def forward(self, *, x, scale=1):
            return x * scale

    model, x = Keyword(), torch.ones(2, 3)
    prepare_auto(tmp_path, model, positions=[0, 1], scope="two forwards with kwargs")
    for side in ("training", "rollout"):
        with AutoCapture(tmp_path, side, identity(), model, positions=[0, 1]):
            model(x=x, scale=1)
            model(x=x, scale=2 if side == "rollout" else 1)
    first = analyze(tmp_path)["first_divergence"]
    assert first["frame"] == "forward:1"
    assert first["boundary"] == "model.input"
    assert first["tensor_path"] == "$['kwargs']['scale']"


def test_different_module_graphs_explicit_binding(tmp_path):
    training = nn.Sequential(nn.Linear(3, 3, bias=False))
    rollout = nn.ModuleDict({"projection": nn.Linear(3, 3, bias=False)})

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.engine = rollout

        def forward(self, x):
            return self.engine["projection"](x)

    other = Other()
    other.engine["projection"].load_state_dict(training[0].state_dict())
    train_bindings = {"layer.0.projection": ModuleBoundary("0", layer=0)}
    roll_bindings = {"layer.0.projection": ModuleBoundary("engine.projection", layer=0)}
    prepare_auto(
        tmp_path,
        training,
        positions=[0, 1],
        scope="mapped models",
        bindings=train_bindings,
    )
    x = torch.ones(2, 3)
    capture(
        tmp_path,
        "training",
        training,
        x,
        bindings=train_bindings,
        weights=lambda: {"projection.weight": training[0].weight},
    )
    capture(
        tmp_path,
        "rollout",
        other,
        x,
        bindings=roll_bindings,
        weights=lambda: {"projection.weight": other.engine["projection"].weight},
    )
    assert analyze(tmp_path)["status"] == "equal"


def test_actual_weights_checked_even_if_caller_copies_identity(tmp_path):
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    prepare_auto(tmp_path, model, positions=[0, 1], scope="actual weights")
    capture(tmp_path, "training", model, torch.ones(2, 3))
    with torch.no_grad():
        model[0].weight.add_(1)
    capture(tmp_path, "rollout", model, torch.ones(2, 3))
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("identities differ" in e for e in report["errors"])


def test_failed_forward_and_cleanup(tmp_path):
    model = Repeated()
    prepare_auto(tmp_path, model, positions=[0, 1], scope="failed execution")
    capture(tmp_path, "training", model, torch.ones(2, 3))
    with pytest.raises(RuntimeError, match="forward failure"):
        with AutoCapture(tmp_path, "rollout", identity(), model, positions=[0, 1]):
            model(torch.ones(2, 3))
            raise RuntimeError("forward failure")
    assert not model._forward_hooks
    assert not model.shared._forward_hooks
    assert not model.shared._forward_pre_hooks
    assert analyze(tmp_path)["status"] == "not_comparable"


@pytest.mark.parametrize("mutation", ["missing_event", "missing_session", "open", "empty"])
def test_incomplete_evidence_cannot_pass(tmp_path, mutation):
    model = Repeated()
    prepare_auto(tmp_path, model, positions=[0, 1], scope="completion gate")
    for side in ("training", "rollout"):
        capture(tmp_path, side, model, torch.ones(2, 3))
    folder = tmp_path / "events/rollout-rank0"
    if mutation == "missing_event":
        (folder / "00000002.pt").unlink()
    elif mutation == "missing_session":
        (folder / "session.pt").unlink()
    else:
        session = torch.load(folder / "session.pt", weights_only=True)
        if mutation == "open":
            session["state"] = "open"
        else:
            session["count"] = 0
        torch.save(session, folder / "session.pt")
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_identity_frozen_and_unobserved_boundary_rejected(tmp_path):
    model = Reverse()
    model.unused = nn.Identity()
    prepare_auto(tmp_path, model, positions=[0, 1], scope="declared coverage")
    for side in ("training", "rollout"):
        capture(tmp_path, side, model, torch.ones(2, 3))
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("unobserved boundaries" in e for e in report["errors"])


def test_logical_rows_and_different_tp_layouts(tmp_path):
    EventCapture.prepare(
        tmp_path,
        [{"id": "layer.output", "stage": "output"}],
        scope="logical TP comparison",
        ranks={"training": [0], "rollout": [0, 1]},
    )
    value = torch.arange(12).float().reshape(3, 4)
    rows = [(0, 0), (1, 0), (1, 1)]
    with EventCapture(tmp_path, "training", identity()) as observer:
        observer.record("layer.output", LogicalTensor(value, rows))
    for rank in range(2):
        with EventCapture(tmp_path, "rollout", identity(), rank=rank) as observer:
            observer.record(
                "layer.output",
                LogicalTensor(value.flip(0)[:, rank * 2 : rank * 2 + 2], rows[::-1], rank * 2, 4),
            )
    assert analyze(tmp_path)["status"] == "equal"
    path = tmp_path / "events/rollout-rank1/00000000.pt"
    record = torch.load(path, weights_only=True)
    record["tree"]["value"][1, 1] += 0.25
    torch.save(record, path)
    first = analyze(tmp_path)["first_divergence"]
    assert (first["sample_id"], first["position"], first["feature"]) == (1, 0, 3)


@pytest.mark.parametrize("value", [torch.tensor([]), torch.tensor([float("nan")])])
def test_invalid_tensors_never_pass(tmp_path, value):
    EventCapture.prepare(tmp_path, [{"id": "out", "stage": "output"}], scope="invalid values")
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, identity()) as observer:
            observer.record("out", value)
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_unsupported_state_is_not_silently_skipped(tmp_path):
    EventCapture.prepare(tmp_path, [{"id": "out", "stage": "output"}], scope="unknown state")
    with EventCapture(tmp_path, "training", identity()) as observer:
        with pytest.raises(TypeError, match="explicit semantic selector"):
            observer.record("out", object())
        observer.record("out", torch.ones(1))
    with EventCapture(tmp_path, "rollout", identity()) as observer:
        observer.record("out", torch.ones(1))
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_cli_offline_and_no_replacement_required(tmp_path, capsys):
    model = Repeated()
    prepare_auto(tmp_path, model, positions=[0, 1], scope="offline")
    for side in ("training", "rollout"):
        capture(tmp_path, side, model, torch.ones(2, 3))
    assert main([str(tmp_path)]) == 0
    report = json.loads((tmp_path / "diagnostic-report.json").read_text())
    assert report["replacement"]["status"] == "unsupported"
    assert "not_reproduced" in capsys.readouterr().out


def test_inplace_operations_cannot_mutate_input_snapshot(tmp_path):
    model = nn.Sequential(nn.ReLU(inplace=True))
    prepare_auto(tmp_path, model, positions=[0, 1], scope="in-place snapshot")
    original = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
    for side in ("training", "rollout"):
        capture(tmp_path, side, model, original.clone())
    event = torch.load(tmp_path / "events/training-rank0/00000001.pt", weights_only=True)
    assert torch.equal(event["tree"]["items"]["args"]["items"][0]["value"], original)
    assert analyze(tmp_path)["status"] == "equal"


def test_weight_updates_during_capture_are_not_comparable(tmp_path):
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    prepare_auto(tmp_path, model, positions=[0, 1], scope="no optimizer update allowed")
    capture(tmp_path, "training", model, torch.ones(2, 3))
    with AutoCapture(tmp_path, "rollout", identity(), model, positions=[0, 1]):
        model(torch.ones(2, 3))
        with torch.no_grad():
            model[0].weight.add_(1)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    session = torch.load(tmp_path / "events/rollout-rank0/session.pt", weights_only=True)
    assert "actual weights changed during observation" in session["errors"]


def test_event_order_mismatch_is_not_guessed(tmp_path):
    specs = [{"id": role, "stage": "output"} for role in ("left", "right")]
    EventCapture.prepare(tmp_path, specs, scope="parallel branches without a DAG")
    for side, roles in (("training", ("left", "right")), ("rollout", ("right", "left"))):
        with EventCapture(tmp_path, side, identity()) as observer:
            for role in roles:
                observer.record(role, torch.ones(1))
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert report["first_divergence"] is None


@pytest.mark.parametrize("mutation", ["corrupt", "undeclared_worker", "missing_worker"])
def test_bad_worker_artifacts_fail_closed(tmp_path, mutation):
    EventCapture.prepare(tmp_path, [{"id": "out", "stage": "output"}], scope="bad artifacts")
    for side in ("training", "rollout"):
        with EventCapture(tmp_path, side, identity()) as observer:
            observer.record("out", torch.ones(1))
    if mutation == "corrupt":
        (tmp_path / "events/rollout-rank0/00000000.pt").write_bytes(b"broken snapshot")
    elif mutation == "undeclared_worker":
        (tmp_path / "events/rollout-rank9").mkdir()
    else:
        manifest = json.loads((tmp_path / "capture-manifest.json").read_text())
        manifest["ranks"]["rollout"].append(1)
        (tmp_path / "capture-manifest.json").write_text(json.dumps(manifest))
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_calculation_contract_difference_is_not_a_numerical_cause(tmp_path):
    EventCapture.prepare(
        tmp_path, [{"id": "out", "stage": "output", "contracts": ["scale"]}], scope="contracts"
    )
    for side, scale in (("training", 1), ("rollout", 2)):
        with EventCapture(tmp_path, side, identity()) as observer:
            observer.record("out", torch.ones(1), contracts={"scale": scale})
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert report["diagnosis"]["level"] == "contract_difference"


def test_evidence_does_not_overwrite_a_previous_session(tmp_path):
    EventCapture.prepare(tmp_path, [{"id": "out", "stage": "output"}], scope="no overwrite")
    with EventCapture(tmp_path, "training", identity()) as observer:
        observer.record("out", torch.ones(1))
    with pytest.raises(FileExistsError):
        with EventCapture(tmp_path, "training", identity()):
            pass


def test_snapshot_scope_no_longer_has_implicit_256_module_limit(tmp_path):
    model = nn.Sequential(*[nn.Identity() for _ in range(257)])
    selected = prepare_auto(tmp_path, model, positions=[0, 1], scope="all leaves")
    assert len(selected) == 257


def test_output_only_fused_boundary_never_invents_equal_inputs(tmp_path):
    model = nn.Sequential(nn.Identity())
    bindings = {"fused": ModuleBoundary("0", observe_inputs=False)}
    prepare_auto(
        tmp_path,
        model,
        positions=[0, 1],
        scope="opaque fused inputs",
        bindings=bindings,
        capture_root=False,
    )
    capture(tmp_path, "training", model, torch.ones(2, 3), bindings=bindings, capture_root=False)
    capture(tmp_path, "rollout", model, torch.zeros(2, 3), bindings=bindings, capture_root=False)
    report = analyze(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "fused.output"
    assert report["first_divergence"]["input_evidence"] == "unknown"


def test_actual_weight_inventory_can_stream_parameters(tmp_path):
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    prepare_auto(tmp_path, model, positions=[0, 1], scope="streaming logical weights")
    calls = []

    def weights():
        calls.append("inventory")
        yield "linear.weight", model[0].weight

    for side in ("training", "rollout"):
        capture(tmp_path, side, model, torch.ones(2, 3), weights=weights)
    assert calls == ["inventory"] * 4  # independently before/after BOTH runs
    assert analyze(tmp_path)["status"] == "equal"


def test_streaming_inventory_rejects_duplicate_names(tmp_path):
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    prepare_auto(tmp_path, model, positions=[0, 1], scope="duplicate canonical parameter")

    def weights():
        yield "weight", model[0].weight
        yield "weight", model[0].weight

    with pytest.raises(ValueError, match="duplicate canonical weight"):
        capture(tmp_path, "training", model, torch.ones(2, 3), weights=weights)


@pytest.mark.parametrize("wrong_cp_group", [False, True])
def test_tp_cp_grid_requires_every_feature_for_every_logical_row(tmp_path, wrong_cp_group):
    EventCapture.prepare(
        tmp_path,
        [{"id": "out", "stage": "output"}],
        scope="TP2 CP2 ownership",
        ranks={"training": [0, 1, 2, 3], "rollout": [0]},
    )
    rows = [(0, 0), (0, 1), (1, 0), (1, 1)]
    value = torch.arange(8).float().reshape(4, 2)
    for rank in range(4):
        tp_rank, cp_rank = rank % 2, rank // 2
        # Wrong WORLD-group partition: TP replicas see different token slices,
        # even though the UNION of their tokens still covers the complete batch.
        indices = [rank] if wrong_cp_group else list(range(cp_rank, 4, 2))
        with EventCapture(tmp_path, "training", identity(), rank=rank) as observer:
            observer.record(
                "out",
                LogicalTensor(
                    value[indices, tp_rank : tp_rank + 1],
                    [rows[index] for index in indices],
                    tp_rank,
                    2,
                ),
            )
    with EventCapture(tmp_path, "rollout", identity()) as observer:
        observer.record("out", LogicalTensor(value, rows))
    report = analyze(tmp_path)
    assert report["status"] == ("not_comparable" if wrong_cp_group else "equal")
    if wrong_cp_group:
        assert any("incomplete logical feature coverage" in error for error in report["errors"])
