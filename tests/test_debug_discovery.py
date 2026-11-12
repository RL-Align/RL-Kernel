# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Unfamiliar names/classes/return trees; no supplied semantic bindings."""

import json
import gc
import weakref

import pytest
import torch
from torch import nn

from rl_engine.alignment.debug import DiscoverCapture
from rl_engine.alignment.debug.entry import analyze


TOKENS = torch.tensor([[1, 2, 3], [4, 5, 6]])
POSITIONS = torch.arange(3).expand(2, -1)


class OrdinaryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(5)
        self.projection = nn.Linear(5, 5)

    def forward(self, hidden):
        return hidden + self.projection(self.norm(hidden)).tanh()


class ForeignBlock(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.scaler = nn.LayerNorm(5)
        self.matrix = nn.Linear(5, 5)
        self.scaler.load_state_dict(source.norm.state_dict())
        self.matrix.load_state_dict(source.projection.state_dict())

    def forward(self, *, state):
        return {"result": (state + self.matrix(self.scaler(state)).tanh(), None)}


class Training(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 5)
        self.layers = nn.ModuleList([OrdinaryBlock() for _ in range(3)])

    def forward(self, input_ids, position_ids):
        x = self.embedding(input_ids)
        for block in self.layers:
            x = block(x)
        return x


class Foreign(nn.Module):
    def __init__(self, source, *, reverse=False):
        super().__init__()
        self.dictionary = nn.Embedding(16, 5)
        self.dictionary.load_state_dict(source.embedding.state_dict())
        blocks = [ForeignBlock(block) for block in source.layers]
        self.network = nn.ModuleDict(
            {
                f"unit_{index}": block
                for index, block in enumerate(blocks[::-1] if reverse else blocks)
            }
        )
        self.reverse = reverse

    def forward(self, input_ids, position_ids):
        x = self.dictionary(input_ids)
        blocks = list(self.network.values())
        for block in blocks[::-1] if self.reverse else blocks:
            x = block(state=x)["result"][0]
        return {"last_hidden": x}


def observe(root, side, model, **kwargs):
    with (
        torch.no_grad(),
        DiscoverCapture(
            root,
            side,
            model,
            tokens=TOKENS,
            positions=POSITIONS,
            **kwargs,
        ),
    ):
        return model(input_ids=TOKENS, position_ids=POSITIONS)


def models():
    torch.manual_seed(442)
    left = Training().eval()
    return left, Foreign(left).eval()


def test_unknown_names_and_return_structures_need_no_mapping(tmp_path):
    left, right = models()
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    report = analyze(tmp_path)
    assert report["status"] == "equal", report
    plan = report["automatic_mapping"]
    assert plan["status"] == "validated"
    assert len(plan["blocks"]) == 3
    assert plan["blocks"][0]["training"] == "layers.0"
    assert plan["blocks"][0]["rollout"] == "network.unit_0"
    assert (
        plan["blocks"][0]["block_output"]["training"]
        != plan["blocks"][0]["block_output"]["rollout"]
    )


def test_middle_block_fault_is_localized_without_fitting_activations(tmp_path):
    left, right = models()
    observe(tmp_path, "training", left)
    handle = right.network["unit_1"].matrix.register_forward_hook(lambda m, a, out: out + 0.25)
    try:
        observe(tmp_path, "rollout", right)
    finally:
        handle.remove()
    report = analyze(tmp_path)
    assert report["status"] == "diverged", report
    assert report["first_divergence"]["layer"] == 1
    assert report["first_divergence"]["boundary"] == "block.1.projection.0.output"
    assert report["first_divergence"]["input_evidence"] == "equal"


def test_registration_order_does_not_define_layer_order(tmp_path):
    left = Training().eval()
    right = Foreign(left, reverse=True).eval()
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    report = analyze(tmp_path)
    assert report["status"] == "equal", report
    assert report["automatic_mapping"]["blocks"][0]["rollout"] == "network.unit_2"


def test_wrong_execution_order_is_not_hidden_by_reordering_parameter_hashes(tmp_path):
    left, right = models()
    right.reverse = True
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert report["first_divergence"] is None
    assert report["automatic_mapping"]["status"] == "needs_mapping"


def test_transposed_square_linear_weight_is_not_silently_equated(tmp_path):
    left, right = models()
    with torch.no_grad():
        right.network["unit_0"].matrix.weight.copy_(right.network["unit_0"].matrix.weight.T.clone())
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("parameter contents" in error for error in report["errors"])


def test_ambiguous_independent_stacks_require_scope_confirmation(tmp_path):
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 5)
            self.encoder = nn.ModuleList([OrdinaryBlock() for _ in range(2)])
            self.decoder = nn.ModuleList([OrdinaryBlock() for _ in range(2)])

        def forward(self, input_ids, position_ids):
            x = self.embedding(input_ids)
            for block in self.encoder:
                x = block(x)
            for block in self.decoder:
                x = block(x)
            return x

    model = Two().eval()
    for side in ("training", "rollout"):
        observe(tmp_path, side, model)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert "multiple independent" in report["errors"][0]


def test_buffer_mutation_or_failed_forward_cannot_pass(tmp_path):
    left, right = models()
    observe(tmp_path, "training", left)
    with pytest.raises(RuntimeError, match="failed"):
        with DiscoverCapture(tmp_path, "rollout", right, tokens=TOKENS, positions=POSITIONS):
            right(input_ids=TOKENS, position_ids=POSITIONS)
            raise RuntimeError("failed")
    assert not right._forward_hooks
    assert not right.network["unit_0"]._forward_hooks
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_report_only_rebuilds_from_evidence_not_an_edited_plan(tmp_path):
    left, right = models()
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    before = analyze(tmp_path)
    (tmp_path / "mapping-plan.json").write_text(json.dumps({"status": "trust_me", "blocks": []}))
    assert analyze(tmp_path) == before
    assert not list(tmp_path.glob(".discovery-*"))


def test_actual_root_tokens_are_checked_not_copied_from_declarations(tmp_path):
    model = Training().eval()
    observe(tmp_path, "training", model)
    with pytest.raises(ValueError, match="tokens/positions differ"):
        with DiscoverCapture(tmp_path, "rollout", model, tokens=TOKENS, positions=POSITIONS):
            model(input_ids=TOKENS + 1, position_ids=POSITIONS)
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_non_eval_capture_is_not_silently_changed(tmp_path):
    model = Training()
    with pytest.raises(ValueError, match="eval-mode"):
        observe(tmp_path, "training", model)
    assert model.training


def test_opaque_block_state_is_rejected(tmp_path):
    class Opaque(nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = nn.Linear(5, 5)

        def forward(self, value, state):
            return self.matrix(value)

    class Model(Training):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Opaque() for _ in range(3)])

        def forward(self, input_ids, position_ids):
            value = self.embedding(input_ids)
            state = object()
            for block in self.layers:
                value = block(value, state)
            return value

    model = Model().eval()
    for side in ("training", "rollout"):
        observe(tmp_path, side, model)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("opaque state" in row["reason"] for row in report["automatic_mapping"]["candidates"])


def test_unsupported_probe_schema_is_rejected(tmp_path):
    left, right = models()
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    (tmp_path / "discovery-manifest.json").write_text('{"schema_version":"unknown"}')
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_float_positions_are_not_truncated_into_a_match(tmp_path):
    model = Training().eval()
    observe(tmp_path, "training", model)
    with pytest.raises(ValueError, match="integer tensors"):
        with DiscoverCapture(tmp_path, "rollout", model, tokens=TOKENS, positions=POSITIONS):
            model(input_ids=TOKENS, position_ids=POSITIONS.float() + 0.5)
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_mutating_buffers_is_inconclusive(tmp_path):
    model = Training().eval()
    model.register_buffer("counter", torch.zeros(1))
    observe(tmp_path, "training", model)
    with DiscoverCapture(tmp_path, "rollout", model, tokens=TOKENS, positions=POSITIONS):
        model(input_ids=TOKENS, position_ids=POSITIONS)
        model.counter.add_(1)
    report = analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert "buffers changed" in report["errors"][0]


def test_capture_limit_does_not_silently_drop_tensors(tmp_path):
    model = Training().eval()
    with pytest.raises(ValueError, match="byte limit"):
        observe(tmp_path, "training", model, max_capture_bytes=1)
    assert not model._forward_hooks
    assert not model.layers[0]._forward_hooks
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_probe_snapshots_do_not_retain_device_activations(tmp_path):
    model = Training().eval()
    observer = DiscoverCapture(tmp_path, "training", model, tokens=TOKENS, positions=POSITIONS)
    observer.__enter__()
    value = torch.ones(2, 3, requires_grad=True)
    reference = weakref.ref(value)
    snapshot = observer._tensors(value)
    del value
    gc.collect()
    assert reference() is None
    assert snapshot["tensors"]
    observer.__exit__(None, None, None)


def test_missing_side_and_corrupt_probe_do_not_pass(tmp_path):
    model = Training().eval()
    observe(tmp_path, "training", model)
    assert analyze(tmp_path)["status"] == "not_comparable"
    (tmp_path / "discovery/training/probe.pt").write_bytes(b"corrupt")
    assert analyze(tmp_path)["status"] == "not_comparable"


def test_explicit_scope_resolves_two_stacks_without_parameter_bindings(tmp_path):
    class Model(Training):
        def __init__(self):
            super().__init__()
            self.second = nn.ModuleList([OrdinaryBlock() for _ in range(3)])

        def forward(self, input_ids, position_ids):
            value = super().forward(input_ids, position_ids)
            for block in self.second:
                value = block(value)
            return value

    model = Model().eval()
    for side in ("training", "rollout"):
        observe(tmp_path, side, model, stack="second")
    report = analyze(tmp_path)
    assert report["status"] == "equal", report
    assert report["automatic_mapping"]["containers"] == {
        "training": "second",
        "rollout": "second",
    }


def test_residual_tuple_is_not_added_by_the_observer(tmp_path):
    class Delayed(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.norm = block.norm
            self.projection = block.projection

        def forward(self, delta, residual):
            value = delta if residual is None else delta + residual
            return self.projection(self.norm(value)).tanh(), value

    class Model(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.embedding = source.embedding
            self.blocks = nn.ModuleList([Delayed(block) for block in source.layers])

        def forward(self, input_ids, position_ids):
            value, residual = self.embedding(input_ids), None
            for block in self.blocks:
                value, residual = block(value, residual)
            return value + residual

    left = Training().eval()
    right = Model(left).eval()
    observe(tmp_path, "training", left)
    observe(tmp_path, "rollout", right)
    report = analyze(tmp_path)
    assert report["status"] == "equal", report
    assert "block_output" not in report["automatic_mapping"]["blocks"][0]
    assert report["automatic_mapping"]["blocks"][0]["anchors"]


def test_fused_groups_cannot_reorder_noncontiguous_reference_events():
    from rl_engine.alignment.debug.discovery import MappingRequired
    from rl_engine.alignment.debug.discovery_report import _projection_groups

    def atom(signature, physical_call):
        return {"signature": (signature,), "call": {"start": physical_call}}

    with pytest.raises(MappingRequired, match="execution orders differ"):
        _projection_groups(
            [atom("a", 0), atom("b", 1), atom("c", 2)],
            [atom("a", 0), atom("c", 0), atom("b", 1)],
        )


def test_discovery_cli_does_not_load_a_machine_profile(tmp_path, monkeypatch):
    from rl_engine import repro
    from rl_engine.alignment.debug import discover_runtime

    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps(TOKENS.tolist()))
    monkeypatch.setattr(
        repro, "_load_profile", lambda *a: pytest.fail("unexpected machine profile")
    )
    monkeypatch.setattr(discover_runtime, "run_cli", lambda args: 1)
    assert (
        repro.main(
            ["discover", str(tmp_path), "--tokens", str(tokens), "--output", str(tmp_path / "new")]
        )
        == 1
    )
