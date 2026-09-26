# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Product-level tests independent of Qwen or GPU frameworks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from rl_engine import repro
from rl_engine.alignment.debug import AutoCapture, adapters, entry, prepare_auto, replay, session
from rl_engine.alignment.debug.core import fingerprint
from rl_engine.alignment.debug.evidence import Capture


MODEL = {
    "model_type": "qwen3",
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "torch_dtype": "bfloat16",
}


def test_small_dense_model_uses_actual_shapes_and_tied_head():
    model = dict(
        MODEL,
        num_hidden_layers=28,
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        tie_word_embeddings=True,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000,
    )
    assert adapters.Qwen3Adapter().matches(model, "rocm")
    args = adapters.megatron_model_args(model)
    assert args[args.index("--num-layers") + 1] == "28"
    assert args[args.index("--hidden-size") + 1] == "1024"
    assert "--untie-embeddings-and-output-weights" not in args
    assert not adapters.Qwen3Adapter().matches(dict(model, tie_word_embeddings=False), "rocm")


def test_tied_model_does_not_advertise_a_strict_replacement(monkeypatch):
    monkeypatch.setattr(repro, "_validate_topology_args", lambda args: None)
    model = dict(
        MODEL,
        num_hidden_layers=28,
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        tie_word_embeddings=True,
    )
    args = SimpleNamespace(
        backend="rocm",
        tp_size=4,
        rollout_tp_size=4,
        rollout_temperature=0.7,
        rollout_top_p=0.95,
        rollout_top_k=-1,
    )
    candidate = adapters.Qwen3Adapter().replacements(model, args)[0]
    assert not candidate.eligible
    assert "tied embedding/output" in candidate.reasons[0]


def portable(root, *, different=False, identity_change=False, contract_change=False):
    """A real tiny PyTorch MLP; there is no registered RL Kernel for this model."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    specs = [
        {
            "id": "x",
            "module": "embedding",
            "stage": "input",
            "layer": 0,
            "positions": [0, 1],
            "width": 2,
        },
        {
            "id": "projection",
            "module": "ffn",
            "stage": "linear_output",
            "layer": 0,
            "positions": [0, 1],
            "width": 2,
            "inputs": ["x"],
            "contracts": ["scale"],
        },
        {
            "id": "activation",
            "module": "ffn",
            "stage": "activation_output",
            "layer": 0,
            "positions": [0, 1],
            "width": 2,
            "inputs": ["projection"],
        },
    ]
    Capture.prepare(root, specs, scope="toy MLP complete forward; CPU; no RL Kernel")
    for side in ("training", "rollout"):
        identity = {
            "weights": fingerprint(model[0].weight),
            "tokens": fingerprint(x),
            "positions": [0, 1],
            "mask": [1, 1],
            "sampling": {},
        }
        if side == "rollout" and identity_change:
            identity["weights"] = "different weights"
        cap = Capture(root, side, identity)
        cap.record("x", x, positions=[0, 1])
        # The simulated production bug is an extra output modification, not
        # a diagnostic reference forward. Observers must leave results intact.
        bug = None
        if side == "rollout" and different:
            bug = model[0].register_forward_hook(lambda _m, _a, out: out + 0.125)
        try:
            scale = 2 if side == "rollout" and contract_change else 1
            with cap.observe(
                {
                    "projection": (
                        model[0],
                        lambda m, a, k, out, scale=scale: {
                            "value": out,
                            "positions": [0, 1],
                            "contracts": {"scale": scale},
                            "route": "torch.nn.Linear",
                        },
                    ),
                    "activation": (
                        model[1],
                        lambda m, a, k, out: {
                            "value": out,
                            "positions": [0, 1],
                            "route": "torch.nn.ReLU",
                        },
                    ),
                }
            ):
                result = model(x)
            assert torch.equal(result, torch.relu(x + (0.125 if bug else 0)))
        finally:
            if bug:
                bug.remove()
        assert not model[0]._forward_hooks and not model[1]._forward_hooks


def test_unknown_model_localized_without_rl_kernel_or_machine_profile(
    tmp_path, monkeypatch, capsys
):
    portable(tmp_path, different=True)
    monkeypatch.setattr(repro, "_load_profile", lambda *_: pytest.fail("offline needs no profile"))
    assert repro.main(["debug", str(tmp_path), "--matrix", "auto"]) == 1
    report = json.loads((tmp_path / "diagnostic-report.json").read_text())
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "projection"
    assert report["first_divergence"]["input_evidence"] == "equal"
    assert report["boundaries"][2]["input_evidence"] == "different"
    assert report["diagnosis"]["level"] == "localized"
    assert report["replacement"]["status"] == "unsupported"
    out = capsys.readouterr().out
    assert "module=ffn layer=0 stage=linear_output token=0" in out
    assert "[input evidence] equal" in out
    assert "no live adapter" in out
    assert (tmp_path / "first-divergence.pt").is_file()


def test_auto_capture_generates_portable_bundle_without_adapter(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)
            self.act = torch.nn.ReLU()

        def forward(self, x):
            return self.act(self.proj(x))

    model = Tiny()
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(2))
    positions = [0, 1]
    prepare_auto(tmp_path, model, positions=positions, scope="automatic leaf capture")
    identity = {
        "weights": fingerprint(model.proj.weight),
        "tokens": fingerprint(torch.tensor([[1, 2], [3, 4]])),
        "positions": positions,
        "mask": [1, 1],
        "sampling": {},
    }
    for side in ("training", "rollout"):
        with AutoCapture(tmp_path, side, identity, model, positions=positions):
            model(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
    report = entry.analyze(tmp_path)
    assert report["status"] == "equal"
    assert report["replacement"]["status"] == "unsupported"
    assert any(item["boundary"] == "auto.proj.input" for item in report["boundaries"])


def test_auto_capture_localizes_first_divergence_without_adapter(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.ReLU())
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    positions = [0, 1]
    prepare_auto(tmp_path, model, positions=positions, scope="automatic leaf capture")
    identity = {
        "weights": fingerprint(model[0].weight),
        "tokens": fingerprint(torch.tensor([[1, 2], [3, 4]])),
        "positions": positions,
        "mask": [1, 1],
        "sampling": {},
    }
    with AutoCapture(tmp_path, "training", identity, model, positions=positions):
        model(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
    bug = model[0].register_forward_hook(lambda _m, _a, out: out + 0.125)
    try:
        with AutoCapture(tmp_path, "rollout", identity, model, positions=positions):
            model(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
    finally:
        bug.remove()
    report = entry.analyze(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["boundary"] == "auto.0.output"
    assert report["replacement"]["status"] == "unsupported"


def test_equal_is_not_reproduced_not_repaired(tmp_path):
    portable(tmp_path)
    report = entry.analyze(tmp_path)
    assert report["status"] == "equal"
    assert report["diagnosis"]["level"] == "not_reproduced"


@pytest.mark.parametrize("identity_change,contract_change", [(True, False), (False, True)])
def test_contract_or_identity_difference_cannot_pass(tmp_path, identity_change, contract_change):
    portable(tmp_path, identity_change=identity_change, contract_change=contract_change)
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    if contract_change:
        assert report["diagnosis"]["level"] == "contract_difference"
        assert report["diagnosis"]["contract_differences"]["projection"]["scale"] == {
            "training": 1,
            "rollout": 2,
        }


def test_missing_earlier_boundary_never_proves_first_operator(tmp_path):
    portable(tmp_path, different=True)
    (tmp_path / "captures/rollout-rank0-call0.pt").unlink()
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert report["first_divergence"]["input_evidence"] == "unknown"
    assert report["diagnosis"]["level"] == "inconclusive"


def test_observer_removed_after_forward_failure(tmp_path):
    cap_specs = [{"id": "out", "module": "custom", "stage": "output", "positions": [0], "width": 1}]
    Capture.prepare(tmp_path, cap_specs, scope="failure cleanup")
    cap = Capture(
        tmp_path,
        "training",
        {key: "actual" for key in ("weights", "tokens", "positions", "mask", "sampling")},
    )
    module = torch.nn.Identity()
    with pytest.raises(RuntimeError):
        with cap.observe({"out": (module, lambda *a: {})}):
            raise RuntimeError("application failure")
    assert not module._forward_hooks


def test_semantic_dependencies_must_precede_outputs(tmp_path):
    with pytest.raises(ValueError, match="inputs must precede"):
        Capture.prepare(
            tmp_path,
            [
                {
                    "id": "out",
                    "module": "x",
                    "stage": "output",
                    "positions": [0],
                    "width": 1,
                    "inputs": ["later"],
                }
            ],
            scope="test",
        )


def test_portable_shards_reconstruct_different_parallel_configurations(tmp_path):
    specs = [{"id": "out", "module": "custom", "stage": "output", "positions": [2, 7], "width": 4}]
    Capture.prepare(tmp_path, specs, scope="declared logical coordinates")
    identity = {key: "same" for key in ("weights", "tokens", "positions", "mask", "sampling")}
    x = torch.arange(8).float().reshape(2, 4)
    Capture(tmp_path, "training", identity).record("out", x, positions=[2, 7])
    for rank in range(2):
        Capture(tmp_path, "rollout", identity, rank=rank).record(
            "out", x.flip(0)[:, rank * 2 : rank * 2 + 2], positions=[7, 2], offset=rank * 2
        )
    assert entry.analyze(tmp_path)["status"] == "equal"
    record = tmp_path / "captures/rollout-rank1-call0.pt"
    payload = torch.load(record, weights_only=True)
    payload["tensors"]["value"] = payload["tensors"]["value"].half()
    torch.save(payload, record)
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("shard dtypes differ" in error for error in report["errors"])


@pytest.mark.parametrize(
    "changes",
    [
        {"model_type": "qwen3_moe"},
        {"num_hidden_layers": 40},
        {"torch_dtype": "float16"},
        {"quantization_config": {"bits": 4}},
    ],
)
def test_adapter_rejects_unsupported_architecture_or_dtype(changes, monkeypatch):
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: [])
    with pytest.raises(ValueError, match="Unsupported live adapter"):
        adapters.select_adapter({**MODEL, **changes}, "rocm")


def test_registered_adapter_discovery_is_not_a_qwen_fallback(monkeypatch):
    class Toy:
        name = "toy"

        def matches(self, model, backend):
            return model.get("model_type") == "toy"

    plugin = SimpleNamespace(load=lambda: Toy)
    monkeypatch.setattr(adapters, "entry_points", lambda **kw: [plugin])
    assert adapters.select_adapter({"model_type": "toy"}, "cpu").name == "toy"


def test_replacement_candidate_is_not_a_verified_guarantee():
    args = repro.build_parser().parse_args(
        ["debug", "sample", "--tp", "2", "--cp", "4", "--temperature", "0.73", "--top-p", "0.92"]
    )
    candidate = adapters.Qwen3Adapter().replacements(MODEL, args)[0]
    assert candidate.eligible
    assert candidate.validation == "requires_same_replay_verification"
    args.rollout_top_k = 10
    assert not adapters.Qwen3Adapter().replacements(MODEL, args)[0].eligible


def test_source_baseline_precedence_and_native_default(tmp_path):
    source = tmp_path / "sample.json"
    source.write_text("{}")
    assert replay.source_baseline(source)[0] == "M000"
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_env": {
                    "env_vars": {
                        "RL_KERNEL_ATTENTION_CASE": "R/R",
                        "RL_KERNEL_FFN_CASE": "P/P",
                        "RL_KERNEL_LOGP_CASE": "R/R",
                    }
                }
            }
        )
    )
    assert replay.source_baseline(source)[0] == "M101"
    assert replay.source_baseline(source, "M011")[0] == "M011"


def test_direct_saved_step_requires_matching_updated_weights(tmp_path):
    folder = tmp_path / "rollout_data"
    folder.mkdir()
    source = folder / "24.pt"
    torch.save({"tokens": [1, 2, 3], "prompt_length": 1}, source)
    assert replay.load_sample(source, 0, 0)["source_step"] == 24


def test_report_only_preserves_job_failure_even_with_equal_tensors(tmp_path):
    root = tmp_path / "arm"
    portable(root)
    summary = {
        "groups": [{"group": "M111", "run_dir": str(root), "process_returncode": 7}],
        "baseline": "M111",
        "scope": "test",
        "planned": False,
    }
    (tmp_path / "matrix-report.json").write_text(json.dumps(summary))
    report = entry.analyze(tmp_path)
    assert report["conclusion"] == "inconclusive"
    assert "capture process exited with 7" in report["groups"][0]["report"]["errors"]
    assert "inconclusive" in (tmp_path / "matrix-report.md").read_text()


def test_report_only_rechecks_cross_arm_identity(tmp_path):
    groups = []
    for group in ("M000", "M111"):
        root = tmp_path / group
        portable(root)
        if group == "M111":
            for path in (root / "captures").glob("*.pt"):
                data = torch.load(path, weights_only=True)
                data["identity"]["tokens"] = "different replay on both sides"
                torch.save(data, path)
        groups.append({"group": group, "run_dir": str(root), "process_returncode": 0})
    (tmp_path / "matrix-report.json").write_text(
        json.dumps({"groups": groups, "baseline": "M000", "scope": "test", "planned": False})
    )
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert "cross-arm runtime_identity changed" in report["groups"][1]["report"]["errors"]


def test_capture_source_change_cannot_be_ignored_as_a_native_validation_failure(tmp_path):
    (tmp_path / "single-arm-summary.json").write_text(
        json.dumps(
            {
                "launcher_returncode": 0,
                "frozen_sources_match": False,
            }
        )
    )
    result = {
        "run_dir": str(tmp_path),
        "process_returncode": 1,
        "report": {"status": "equal", "errors": []},
    }
    session.apply_evidence_gates(result, None)
    assert result["report"]["status"] == "not_comparable"
    assert "source fingerprint changed" in result["report"]["errors"][0]


@pytest.mark.parametrize(
    "original,replacement,expected",
    [
        ("diverged", "equal", "verified_on_replay"),
        ("equal", "equal", "not_reproduced"),
        ("diverged", "diverged", "unresolved"),
        ("not_comparable", "equal", "inconclusive"),
    ],
)
def test_verification_requires_original_reproduction(tmp_path, original, replacement, expected):
    summary = {
        "groups": [
            {"group": group, "report": {"status": status}}
            for group, status in (("M000", original), ("M111", replacement))
        ],
        "baseline": "M000",
        "scope": "test",
    }
    session.finish_summary(tmp_path, summary)
    assert summary["conclusion"] == expected


@pytest.mark.parametrize("broken", [False, True])
def test_auto_cli_runs_original_before_repair_and_stops_when_resolved(
    tmp_path, monkeypatch, broken
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(MODEL))
    profile = {
        "schema_version": "rlkernel.repro.profile.v1",
        "paths": {"workspace": str(tmp_path), "model_root": str(model)},
        "modes": {"consistency": {}},
        "requirements": {"backend": "cuda"},
    }
    monkeypatch.setattr(repro, "_load_profile", lambda _: (profile, tmp_path / "profile.json"))
    source = tmp_path / "frozen.json"
    source.write_text(
        json.dumps(
            {
                "tokens": [1, 2, 3],
                "prompt_length": 1,
                "mask": [1, 1],
                "sampling": {"temperature": 0.73, "top_p": 0.92, "top_k": -1},
            }
        )
    )
    calls = []

    def execute(command, **kwargs):
        group = command[command.index("--group") + 1]
        assert "--use-rollout-logprobs" not in command
        assert command[command.index("--rollout-temperature") + 1] == "0.73"
        assert command[command.index("--rollout-top-p") + 1] == "0.92"
        run_dir = tmp_path / "runs" / f"automatic-{group}"
        portable(run_dir, different=(broken and group == "M000"))
        calls.append(group)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(session.subprocess, "run", execute)
    assert (
        repro.main(
            [
                "debug",
                str(source),
                "--matrix",
                "auto",
                "--run-id",
                "automatic",
                "--output-root",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    assert calls == (["M000", "M111"] if broken else ["M000"])
    summary = json.loads((tmp_path / "runs/automatic/matrix-report.json").read_text())
    assert summary["conclusion"] == ("verified_on_replay" if broken else "not_reproduced")
    offline = entry.analyze(tmp_path / "runs/automatic")
    assert offline["conclusion"] == summary["conclusion"]
