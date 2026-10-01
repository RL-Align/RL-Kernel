# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rl_engine.alignment.debug import qwen3_runtime as runtime
from rl_engine.alignment.debug import core, entry, qwen3_report, replay, session
from rl_engine import repro


def test_bits_not_tolerance_and_nonfinite_not_pass():
    signed_zero = core.bitdiff(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert signed_zero["bitwise_mismatches"] == 1
    assert signed_zero["max_abs_diff"] == 0
    assert signed_zero["first_index"] == [0]
    assert not core.bitdiff(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["equal"]
    assert not core.bitdiff(torch.tensor([float("inf")]), torch.tensor([float("inf")]))["equal"]
    assert not core.bitdiff(torch.ones(2), torch.ones(3))["comparable"]
    assert not core.bitdiff(torch.ones(2), torch.ones(2).half())["comparable"]
    assert not core.bitdiff(torch.empty(0), torch.empty(0))["comparable"]


def test_snapshot_is_not_an_alias():
    value = torch.ones(3)
    captured = core.snapshot({"x": value})
    value.zero_()
    assert captured["x"].tolist() == [1, 1, 1]


def test_stitch_logical_positions_and_features_not_rank_order():
    def record(positions, offset, values):
        return {
            "tensors": {"query": torch.tensor(values)},
            "metadata": {
                "stage_positions": {"query": torch.tensor(positions)},
                "feature_offsets": {"query": offset},
                "feature_sizes": {"query": 4},
            },
        }

    records = [
        record([3, 0], 2, [[7.0, 8.0], [3.0, 4.0]]),
        record([0, 3], 0, [[1.0, 2.0], [5.0, 6.0]]),
    ]
    rows, errors = core._stitch(records, "query", {0, 3})
    assert not errors
    assert rows[0].tolist() == [1, 2, 3, 4]
    assert rows[3].tolist() == [5, 6, 7, 8]
    records.append(record([0], 0, [[1.0, 9.0]]))
    assert "replica disagreement" in core._stitch(records, "query", {0})[1][0]


@pytest.mark.parametrize("group", ["auto", "full", "M111,M011", "m000"])
def test_matrix_selection(group):
    groups = session.matrix_groups(group)
    assert groups and len(groups) == len(set(groups))


def test_attribute_matrix_is_explicit_and_runs_module_arms():
    groups = session.matrix_groups("attribute")
    assert groups == ["M000", "M111", "M011", "M101", "M110"]


def test_debug_summary_contains_scope_and_next_commands(tmp_path):
    summary = {
        "conclusion": "verified_on_replay",
        "status": "equal",
        "adapter": "qwen3-dense-vime",
        "matrix": "auto",
        "replay": str(tmp_path / "frozen-replay.json"),
        "scope": "one frozen replay",
        "groups": [
            {
                "group": "M111",
                "report": {
                    "status": "equal",
                    "first_divergence": None,
                    "errors": [],
                    "endpoint": {
                        "element_count": 2,
                        "bitwise_mismatch_count": 0,
                        "max_abs_diff": 0.0,
                    },
                },
            }
        ],
    }
    (tmp_path / "frozen-replay.json").write_text("{}")
    from rl_engine.alignment.debug.ux import compact_summary

    compact_summary(tmp_path, summary)
    value = json.loads((tmp_path / "debug-summary.json").read_text())
    assert value["result"] == "verified_on_replay"
    assert any("attribute" in command for command in value["next_commands"])
    assert "performance are unverified" in (tmp_path / "debug-summary.md").read_text()


def test_debug_summary_classifies_dry_run_as_planned(tmp_path):
    from rl_engine.alignment.debug.ux import compact_summary

    compact_summary(
        tmp_path,
        {"status": "not_comparable", "conclusion": "planned", "groups": [], "scope": "dry run"},
    )
    value = json.loads((tmp_path / "debug-summary.json").read_text())
    assert value["result"] == "planned"


@pytest.mark.parametrize("group", ["", "M111,M111", "M999", "M11", "G11"])
def test_reject_invalid_matrix(group):
    with pytest.raises(ValueError):
        session.matrix_groups(group)


def test_source_freeze_preserves_tokens_mask_and_sampling(tmp_path):
    source = tmp_path / "sample.json"
    payload = {
        "tokens": [12, 13, 14, 15],
        "prompt_length": 2,
        "mask": [1, 0],
        "sampling": {"temperature": 0.73, "top_p": 0.92, "top_k": -1},
    }
    source.write_text(json.dumps(payload))
    frozen = replay.load_sample(source, 0, 0)
    assert frozen["tokens"] == payload["tokens"]
    assert frozen["mask"] == [1, 0]
    assert frozen["sampling"] == payload["sampling"]
    source.write_text(json.dumps({**payload, "mask": [1]}))
    with pytest.raises(ValueError, match="lengths"):
        replay.load_sample(source, 0, 0)


def test_cuda_train_dump_nested_payload(tmp_path):
    source = tmp_path / "0.rank0.pt"
    torch.save(
        {
            "rollout_id": 3,
            "rank": 0,
            "rollout_data": {
                "tokens": [torch.tensor([1, 2, 3])],
                "response_lengths": [2],
                "loss_masks": [torch.tensor([1, 0])],
            },
        },
        source,
    )
    value = replay.load_sample(source, 0, 0)
    assert value["source_step"] == 3
    assert value["tokens"] == [1, 2, 3]
    assert value["mask"] == [1, 0]


def test_runtime_rejects_wrong_input_and_accepts_cp_padding(monkeypatch):
    monkeypatch.setattr(runtime, "frozen_batch", lambda: {"tokens": [8, 9, 10]})
    runtime.validate_rows(torch.tensor([10, 8, 0]), torch.tensor([2, 0, -1]))
    with pytest.raises(RuntimeError, match="runtime tokens"):
        runtime.validate_rows(torch.tensor([10, 7]), torch.tensor([2, 0]))


def test_forced_sample_preserves_original_distribution(monkeypatch):
    module = ModuleType("vllm.v1.sample.sampler")

    class Sampler:
        def sample(self, logits, sampling_metadata):
            probabilities = torch.log_softmax(logits / sampling_metadata.temperature, dim=-1)
            return logits.argmax(-1), probabilities

    module.Sampler = Sampler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.sampler", None)
    monkeypatch.setattr(runtime, "frozen_batch", lambda: {"tokens": [3, 0, 1]})
    monkeypatch.setattr(runtime, "_ROWS", {"rollout": torch.tensor([0, 1])})
    original = Sampler().sample(torch.tensor([[2.0, 1.0, 0.0]]), SimpleNamespace(temperature=0.7))
    runtime.install_fixed_sampler()
    selected, logps = Sampler().sample(
        torch.tensor([[2.0, 1.0, 0.0]]), SimpleNamespace(temperature=0.7)
    )
    assert selected.tolist() == [1]
    assert core.bitdiff(logps, original[1])["equal"]
    runtime.install_fixed_sampler()  # idempotent
    runtime._ROWS.clear()
    selected, _ = Sampler().sample(
        torch.tensor([[2.0, 1.0, 0.0]]), SimpleNamespace(temperature=0.7)
    )
    assert selected.tolist() == [0]  # profile work is untouched


def test_canonical_qkv_group_layout_and_sequence_parallel(monkeypatch):
    monkeypatch.setattr(runtime, "topology", lambda side: (1, 2))
    monkeypatch.setattr(runtime, "_ROWS", {"training": torch.tensor([0, 1, 6, 7])})
    layer = SimpleNamespace(
        config=SimpleNamespace(kv_channels=2, num_query_groups=2, num_attention_heads=4),
        self_attention=SimpleNamespace(),
    )
    packed = torch.arange(32.0).reshape(4, 8)
    tensors, metadata = runtime.canonical_tensors(
        layer, "training", {"qkv": packed, "input": torch.ones(2, 1, 8)}
    )
    assert tensors["q_projection"].tolist() == packed[:, :4].tolist()
    assert tensors["k_projection"].tolist() == packed[:, 4:6].tolist()
    assert metadata["feature_offsets"]["q_projection"] == 4
    assert metadata["stage_positions"]["input"].tolist() == [6, 7]


def test_worker_sampler_normalizes_disabled_top_k_without_hiding_other_changes(monkeypatch):
    import numpy as np

    module = ModuleType("vllm.v1.sample.sampler")
    worker = ModuleType("vllm.v1.worker.gpu.sample.sampler")

    class Sampler:
        def sample(self, logits, idx_mapping_np, pos):
            return logits.argmax(-1), logits

    module.Sampler = worker.Sampler = Sampler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, worker.__name__, worker)
    batch = {"tokens": [0, 1, 2], "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": -1}}
    monkeypatch.setattr(runtime, "frozen_batch", lambda: batch)
    monkeypatch.setattr(runtime, "_ROWS", {"rollout": torch.tensor([0, 1])})
    instance = Sampler()
    instance.sampling_states = SimpleNamespace(
        vocab_size=3,
        temperature=SimpleNamespace(np=np.array([0.7], dtype=np.float32)),
        top_p=SimpleNamespace(np=np.array([0.95], dtype=np.float32)),
        top_k=SimpleNamespace(np=np.array([3], dtype=np.int32)),
    )
    runtime.install_fixed_sampler()
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    selected, processed = instance.sample(logits, np.array([0]), torch.tensor([1]))
    assert selected.tolist() == [2]
    assert processed is logits
    instance.sampling_states.top_k.np[0] = 2
    with pytest.raises(RuntimeError, match="top_k differs"):
        instance.sample(logits, np.array([0]), torch.tensor([1]))


def test_warmup_recognition_uses_framework_stack_not_input_values():
    scope = {"__name__": "vllm.v1.worker.gpu.warmup", "probe": runtime.is_framework_warmup}
    exec("def warmup_kernels(): return probe()", scope)
    assert scope["warmup_kernels"]()
    scope["__name__"] = "application.real_requests"
    assert not scope["warmup_kernels"]()


def test_training_identity_observes_actual_data_boundary(monkeypatch):
    package = ModuleType("vime.backends.megatron_utils")
    actor = ModuleType("vime.backends.megatron_utils.actor")
    package.actor = actor
    observed = []
    data = {"tokens": [[1, 2, 3]], "loss_masks": [[1, 1]]}
    actor.process_rollout_data = lambda args, ref, rank, world: data
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, actor.__name__, actor)
    monkeypatch.setattr(runtime, "write_identity", lambda *args: observed.append(args))
    runtime.install_training_identity()
    args = SimpleNamespace(rollout_temperature=0.7, rollout_top_p=0.95, rollout_top_k=-1)
    assert actor.process_rollout_data(args, None, 0, 1) is data
    assert observed == [
        ("training", [1, 2, 3], [1, 1], {"temperature": 0.7, "top_p": 0.95, "top_k": -1})
    ]


def _replay_record(root, side, layer, value, *, rank=0, expected_layers=2):
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_env": {
                    "env_vars": {
                        f"RL_KERNEL_{module}_CASE": "R/R" for module in ("ATTENTION", "FFN", "LOGP")
                    }
                }
            }
        )
    )
    readbacks = root / "readbacks"
    readbacks.mkdir(exist_ok=True)
    for framework in ("megatron", "vllm"):
        (readbacks / f"{framework}.json").write_text(
            json.dumps(
                {
                    "framework": framework,
                    "plan": {"cases": {name: {"case_id": "R/R"} for name in qwen3_report.MODULES}},
                    "operators": {
                        name: {
                            "call_count": 1,
                            "case_id": "R/R",
                            "implementation": "rl_kernel",
                            "backend_id": f"rlkernel.{name}.test",
                            "execution_mode": "eager",
                            "provenance": {
                                "runtime_platform": "cuda",
                                "actual_backend": f"rlkernel.{name}.test",
                                "fallback": False,
                            },
                        }
                        for name in qwen3_report.MODULES
                    },
                }
            )
        )
    path = root / "diagnostics/layers" / f"{side}-rank{rank}-layer{layer}-call0.pt"
    tensors = {"input": torch.ones(3, 2), "output": value.repeat(3, 1)}
    core.atomic_save(
        path,
        {
            "schema_version": "rlkernel.layer_replay.v1",
            "side": side,
            "rank": rank,
            "layer": layer,
            "call": 0,
            "positions": torch.tensor([0, 1, 2]),
            "sequence_length": 4,
            "frozen_tokens_sha256": "tokens",
            "sampling": {"temperature": 0.7, "top_p": 0.9},
            "tensors": tensors,
            "metadata": {
                "expected_layers": expected_layers,
                "contract": {
                    "scale": 0.5,
                    "q_norm_eps": 1e-6,
                    "k_norm_eps": 1e-6,
                    "attention_dropout": 0.0,
                },
                "tp_size": 1,
                "tp_rank": 0,
                "feature_offsets": {},
                "feature_sizes": {key: 2 for key in tensors},
                "stage_positions": {key: torch.tensor([0, 1, 2]) for key in tensors},
            },
        },
    )
    weights = root / "diagnostics/weights" / f"{side}-rank{rank}-layer{layer}.json"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_text(json.dumps({"weight:0:0": "same"}))
    identity = root / "diagnostics/identity" / f"{side}-rank{rank}.json"
    identity.parent.mkdir(parents=True, exist_ok=True)
    identity.write_text(
        json.dumps(
            {
                "tokens": "tokens",
                "mask": "mask",
                "side": side,
                "sampling": {"temperature": 0.7, "top_p": 0.9},
            }
        )
    )
    (weights.parent / f"{side}-rank{rank}-head.json").write_text(json.dumps({"lm_head": "same"}))
    core.atomic_save(
        root / "diagnostics/heads" / f"{side}-rank{rank}-call0.pt",
        {
            "side": side,
            "call": 0,
            "tensors": {"head_input": torch.ones(3, 2), "logits": torch.ones(3, 2)},
            "metadata": {
                "feature_offsets": {},
                "feature_sizes": {"head_input": 2, "logits": 2},
                "stage_positions": {"head_input": torch.arange(3), "logits": torch.arange(3)},
            },
        },
    )
    return path


def test_report_first_layer_and_no_silent_missing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen3_report, "_endpoint", lambda root: {"bitwise_mismatch_count": 0, "errors": []}
    )
    for layer in range(2):
        _replay_record(tmp_path, "training", layer, torch.ones(1, 2))
        _replay_record(
            tmp_path, "rollout", layer, torch.tensor([[1.0, 2.0 if layer == 1 else 1.0]])
        )
    report = entry.analyze(tmp_path)
    assert report["status"] == "diverged"
    assert report["first_divergence"]["layer"] == 1
    assert report["first_divergence"]["first_index"] == [1]
    (tmp_path / "diagnostics/weights/rollout-rank0-layer1.json").unlink()
    assert entry.analyze(tmp_path)["status"] == "not_comparable"


def test_final_logp_difference_does_not_blame_attention(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen3_report, "_endpoint", lambda root: {"bitwise_mismatch_count": 1, "errors": []}
    )
    for side in ("training", "rollout"):
        _replay_record(tmp_path, side, 0, torch.ones(1, 2), expected_layers=1)
    report = entry.analyze(tmp_path)
    assert report["first_divergence"]["stage"] == "logp"


def test_no_snapshot_is_not_zero_mismatch(tmp_path):
    assert entry.analyze(tmp_path)["status"] == "not_comparable"


def test_cli_uses_familiar_parallel_and_sampling_options():
    args = repro.build_parser().parse_args(
        [
            "debug",
            "run-dir",
            "--tp",
            "2",
            "--cp",
            "4",
            "--rollout-tp",
            "4",
            "--rollout-cp",
            "2",
            "--temperature",
            ".7",
            "--top-p",
            ".9",
        ]
    )
    assert (args.tp_size, args.cp_size, args.rollout_tp_size, args.rollout_cp_size) == (2, 4, 4, 2)
    assert args.rollout_temperature == 0.7
    assert args.rollout_top_p == 0.9


def test_report_only_needs_no_machine_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "_load_profile", lambda value: pytest.fail("profile not needed"))
    assert repro.main(["debug", str(tmp_path), "--report-only"]) == 2


def test_cli_dry_run_freezes_once_and_keeps_sampling_and_topology(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
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
        )
    )
    source = tmp_path / "sample.json"
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
    profile = {
        "schema_version": "rlkernel.repro.profile.v1",
        "paths": {"workspace": str(tmp_path), "model_root": str(model)},
        "modes": {"consistency": {}},
        "requirements": {"backend": "cuda"},
    }
    monkeypatch.setattr(repro, "_load_profile", lambda path: (profile, tmp_path / "profile.json"))
    monkeypatch.setattr(
        session.subprocess, "run", lambda *a, **k: pytest.fail("dry run launched a job")
    )
    assert (
        repro.main(
            [
                "debug",
                str(source),
                "--dry-run",
                "--run-id",
                "test-debug",
                "--output-root",
                str(tmp_path / "runs"),
                "--tp",
                "2",
                "--rollout-tp",
                "4",
            ]
        )
        == 0
    )
    root = tmp_path / "runs/test-debug"
    frozen = json.loads((root / "frozen-replay.json").read_text())
    assert frozen["sampling"]["temperature"] == 0.73
    for group in session.matrix_groups("auto"):
        command = json.loads((root / f"{group}.plan.json").read_text())["command"]
        assert command[command.index("--group") + 1] == group
        assert command[command.index("--tp-size") + 1] == "2"
        assert command[command.index("--cp-size") + 1] == "4"
        assert command[command.index("--rollout-top-p") + 1] == "0.92"
        assert "--use-rollout-logprobs" not in command


def test_missing_layer_and_weight_change_are_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen3_report, "_endpoint", lambda root: {"bitwise_mismatch_count": 0, "errors": []}
    )
    for side in ("training", "rollout"):
        _replay_record(tmp_path, side, 0, torch.ones(1, 2), expected_layers=2)
    assert entry.analyze(tmp_path)["status"] == "not_comparable"
    for side in ("training", "rollout"):
        _replay_record(tmp_path, side, 1, torch.ones(1, 2), expected_layers=2)
    assert entry.analyze(tmp_path)["status"] == "equal"
    (tmp_path / "diagnostics/weights/rollout-rank0-layer1.json").write_text(
        json.dumps({"weight:0:0": "changed"})
    )
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("weights differ" in e for e in report["errors"])


def test_parameter_tiles_compare_across_tp_splits():
    weight = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    whole = runtime.tensor_tiles("projection", weight)
    shards = {}
    for rank in range(2):
        shards.update(
            runtime.tensor_tiles(
                "projection", weight[:, rank * 64 : (rank + 1) * 64], col=rank * 64
            )
        )
    assert whole == shards
    weight[80, 80] += 1
    assert whole != runtime.tensor_tiles("projection", weight)


@pytest.mark.parametrize("cp", [1, 2])
def test_endpoint_counts_logical_tokens_not_tp_replicas(tmp_path, cp):
    from examples.vime_rocm_attention_ablation.validate_artifacts import (
        SIDECAR_SCHEMA_VERSION,
        slice_response_mask_for_cp,
    )

    root = tmp_path / "mismatch_sidecars"
    root.mkdir()
    for rank in range(4 * cp):

        def shard(value, rank=rank):
            return slice_response_mask_for_cp(
                torch.tensor(value),
                total_length=13,
                response_length=4,
                context_parallel_size=cp,
                context_parallel_rank=rank // 4,
            )

        torch.save(
            {
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "tensor_parallel_size": 4,
                "context_parallel_size": cp,
                "rank": rank,
                "call_index": 0,
                "train_log_probs": [shard([-0.0, -1.0, -2.0, -3.0])],
                "rollout_log_probs": [shard([0.0, -1.0, -4.0, -3.0])],
                "loss_masks": [torch.ones(4)],
                "total_lengths": [13],
                "response_lengths": [4],
            },
            root / f"rank{rank}.pt",
        )
    report = qwen3_report._endpoint(tmp_path)
    assert report["errors"] == []
    assert report["element_count"] == 4
    assert report["bitwise_mismatch_count"] == 2  # Includes signed zero.
    assert report["mismatch_count"] == 1


def test_head_weights_compare_across_native_and_strict_vocab_padding(tmp_path, monkeypatch):
    real_vocab = 151936
    weight = torch.arange(152064 * 4, dtype=torch.float32).reshape(-1, 4)
    monkeypatch.setenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", str(real_vocab))
    monkeypatch.setattr(runtime, "_ROWS", {"rollout": torch.tensor([0])})

    def capture(shard_size):
        hashes = {}
        for rank in range(4):
            monkeypatch.setattr(runtime, "topology", lambda side, rank=rank: (rank, 4, 0, 1))
            monkeypatch.setattr(runtime, "_HEAD_CALLS", {})
            runtime.save_head(
                "rollout",
                torch.zeros(1, 4),
                None,
                weight[rank * shard_size : (rank + 1) * shard_size],
            )
            path = next((tmp_path / "weights").glob("*-head.json"))
            hashes.update(json.loads(path.read_text()))
        return hashes

    native = capture(37984)  # TP1/TP3 start halfway through a 64-row tile.
    assert len(native) == real_vocab
    assert native == capture(38016)
    weight[real_vocab:] += 1  # Padding is outside the model's weight identity.
    assert native == capture(38016)
    weight[37985, 2] += 1
    assert native != capture(38016)


@pytest.mark.parametrize("nested", [False, True])
def test_capture_exception_survives_framework_swallowing_it(tmp_path, monkeypatch, nested):
    directory = tmp_path
    if nested:
        tmp_path = tmp_path / "arms/r-r"
        tmp_path.mkdir(parents=True)
    monkeypatch.setenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    monkeypatch.setattr(
        qwen3_report, "_endpoint", lambda root: {"bitwise_mismatch_count": 0, "errors": []}
    )
    for side in ("training", "rollout"):
        _replay_record(tmp_path, side, 0, torch.ones(1, 2), expected_layers=1)

    @runtime.capture_guard("rollout", "head")
    def broken_capture():
        raise RuntimeError("unaligned shard")

    with pytest.raises(RuntimeError, match="unaligned shard"):
        broken_capture()
    report = entry.analyze(directory)
    assert report["status"] == "not_comparable"
    assert "capture failed" in report["errors"][0]
    assert "unaligned shard" in report["errors"][0]


def test_layer_capture_observes_original_forward_without_reimplementing_it(tmp_path, monkeypatch):
    class Core(torch.nn.Module):
        layer_name = "model.layers.1.self_attn.attn"

        def forward(self, query, key, value):
            return query + key.repeat(1, 2) + value.repeat(1, 2)

    class Attention(torch.nn.Module):
        head_dim, total_num_heads, total_num_kv_heads = 2, 2, 1
        scaling = 2**-0.5

        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(4, 8, bias=False)
            self.attn = Core()
            self.q_norm, self.k_norm = torch.nn.Identity(), torch.nn.Identity()
            self.q_norm.variance_epsilon = self.k_norm.variance_epsilon = 1e-6

        def forward(self, x):
            q, k, v = self.qkv_proj(x).split([4, 2, 2], dim=-1)
            q, k = self.q_norm(q), self.k_norm(k)
            return self.attn(q, k, v)

    class Activation(torch.nn.Module):
        def forward(self, x):
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * up

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = torch.nn.Linear(4, 8, bias=False)
            self.act_fn = Activation()
            self.down_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_up_proj(x)))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attention(), MLP()
            self.input_layernorm, self.post_attention_layernorm = (
                torch.nn.Identity(),
                torch.nn.Identity(),
            )
            self.calls = 0

        def forward(self, positions, hidden_states, residual):
            self.calls += 1
            value = hidden_states if residual is None else hidden_states + residual
            value = value + self.self_attn(self.input_layernorm(value))
            return self.mlp(self.post_attention_layernorm(value)), value

    module = ModuleType("vllm.model_executor.models.qwen3")
    module.Qwen3DecoderLayer = Layer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runtime, "topology", lambda side: (0, 1))
    monkeypatch.setattr(runtime, "_ROWS", {"rollout": torch.tensor([0, 1])})
    monkeypatch.setattr(runtime, "_WEIGHTS", {})
    monkeypatch.setattr(runtime, "weight_tiles", lambda instance, side: {"test": "weights"})
    monkeypatch.setenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", str(tmp_path))
    captured = []
    monkeypatch.setattr(
        runtime,
        "save_layer",
        lambda side, layer, tensors, metadata: captured.append((layer, tensors)),
    )
    model, inputs = Layer(), torch.randn(2, 4)
    expected = model(torch.arange(2), inputs.clone(), None)
    runtime.install_layers("rollout")
    actual = model(torch.arange(2), inputs.clone(), None)
    assert model.calls == 2
    assert all(core.bitdiff(a, b)["equal"] for a, b in zip(actual, expected, strict=True))
    assert captured[0][0] == 1
    assert {"query", "key", "value", "mlp_activation", "q_projection"} <= captured[0][1].keys()
    assert core.bitdiff(captured[0][1]["output"], expected[0] + expected[1])["equal"]


def test_megatron_instrumented_variadic_forward_keeps_actual_input(tmp_path, monkeypatch):
    class Layer(torch.nn.Module):
        layer_number = 1
        config = SimpleNamespace(kv_channels=2, layernorm_epsilon=1e-6, attention_dropout=0.0)

        def __init__(self):
            super().__init__()
            self.self_attention = torch.nn.Identity()
            for name in ("linear_qkv", "q_layernorm", "k_layernorm", "core_attention"):
                setattr(self.self_attention, name, torch.nn.Identity())
            self.mlp = torch.nn.Identity()
            self.mlp.linear_fc1 = torch.nn.Identity()
            self.mlp.linear_fc2 = torch.nn.Identity()
            self.calls = 0

        def forward(self, *args, **kwargs):
            self.calls += 1
            return args[0] * 2

    module = ModuleType("megatron.core.transformer.transformer_layer")
    module.TransformerLayer = Layer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(runtime, "_ROWS", {"training": torch.tensor([0, 1])})
    monkeypatch.setattr(runtime, "_CALLS", {})
    monkeypatch.setattr(runtime, "_WEIGHTS", {})
    monkeypatch.setattr(runtime, "canonical_tensors", lambda instance, side, tensors: (tensors, {}))
    monkeypatch.setattr(runtime, "weight_tiles", lambda *args: {"weight": "hash"})
    monkeypatch.setenv("RL_KERNEL_ALIGNMENT_DIAGNOSTICS_DIR", str(tmp_path))
    captured = []
    monkeypatch.setattr(
        runtime, "save_layer", lambda side, layer, tensors, metadata: captured.append(tensors)
    )
    runtime.install_layers("training")
    model, hidden = Layer(), torch.ones(2, 1, 4)
    assert torch.equal(model(hidden), hidden * 2)
    assert model.calls == 1
    assert torch.equal(captured[0]["input"], hidden)


def test_tied_output_head_capture_uses_the_actual_forward_weight(monkeypatch):
    class Head(torch.nn.Module):
        weight = None

        def forward(self, hidden, weight=None):
            return hidden @ weight.T, None

    class Model(torch.nn.Module):
        config = SimpleNamespace(num_layers=1)

        def __init__(self):
            super().__init__()
            self.output_layer = Head()
            self.shared = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        def forward(self, input_ids):
            return self.output_layer(input_ids.float(), weight=self.shared)

    module = ModuleType("megatron.core.models.gpt.gpt_model")
    module.GPTModel = Model
    cp = ModuleType("vime.backends.megatron_utils.cp_utils")
    cp.slice_with_cp = lambda values, dim: values
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, cp.__name__, cp)
    monkeypatch.setattr(runtime, "frozen_batch", lambda: {"tokens": [1, 2]})
    monkeypatch.setattr(runtime, "validate_rows", lambda *args: None)
    monkeypatch.setattr(runtime, "_ROWS", {})
    monkeypatch.setattr(runtime, "_HEAD_CALLS", {})
    captured = []
    monkeypatch.setattr(runtime, "save_head", lambda *args: captured.append(args))
    runtime.install_root_context("training")
    model = Model()
    result = model(torch.tensor([[1, 2]]))
    assert captured[0][3] is model.shared
    assert torch.equal(result[0], captured[0][2])
    assert not model.output_layer._forward_hooks
    assert not model.output_layer._forward_pre_hooks


def test_worker_namespace_preserves_evidence_and_requires_exact_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen3_report, "_endpoint", lambda root: {"bitwise_mismatch_count": 0, "errors": []}
    )
    for side in ("training", "rollout"):
        _replay_record(tmp_path, side, 0, torch.ones(1, 2), expected_layers=1)
    root = tmp_path / "diagnostics"
    for folder in ("layers", "heads", "weights"):
        for path in list((root / folder).iterdir()):
            path.rename(path.with_name(path.name.replace("-rank0", "-pid100-rank0")))
    assert entry.analyze(tmp_path)["status"] == "equal"
    weight = root / "weights/rollout-pid100-rank0-layer0.json"
    weight.rename(weight.with_name("rollout-pid200-rank0-layer0.json"))
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("missing runtime weight" in error for error in report["errors"])


def test_tp_cp_rank_coverage_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qwen3_report, "_endpoint", lambda root: {"bitwise_mismatch_count": 0, "errors": []}
    )
    for side in ("training", "rollout"):
        path = _replay_record(tmp_path, side, 0, torch.ones(1, 2), expected_layers=1)
        record = torch.load(path, weights_only=True)
        record["metadata"].update(cp_size=2, cp_rank=0)
        torch.save(record, path)
    report = entry.analyze(tmp_path)
    assert report["status"] == "not_comparable"
    assert any("TP/CP rank coverage" in error for error in report["errors"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("implementation", "production"),
        ("call_count", 0),
        ("execution_mode", "compiled_hip_graph"),
        ("backend_id", "unidentified"),
        ("provenance", {"fallback": True}),
    ],
)
def test_route_requires_execution_not_just_case_flags(tmp_path, field, value):
    _replay_record(tmp_path, "training", 0, torch.ones(1, 2), expected_layers=1)
    path = tmp_path / "readbacks/vllm.json"
    record = json.loads(path.read_text())
    record["operators"]["ffn"][field] = value
    path.write_text(json.dumps(record))
    assert qwen3_report.validate_routes(tmp_path)


def test_route_reads_backend_inside_packed_sequence_provenance(tmp_path):
    _replay_record(tmp_path, "training", 0, torch.ones(1, 2), expected_layers=1)
    path = tmp_path / "readbacks/megatron.json"
    record = json.loads(path.read_text())
    provenance = record["operators"]["attention"]["provenance"]
    backend = provenance.pop("actual_backend")
    provenance["sequences"] = [{"operator": {"actual_backend": backend}}]
    path.write_text(json.dumps(record))
    assert qwen3_report.validate_routes(tmp_path) == []
    provenance["sequences"][0]["operator"].clear()
    path.write_text(json.dumps(record))
    assert any("backend/platform provenance" in e for e in qwen3_report.validate_routes(tmp_path))
