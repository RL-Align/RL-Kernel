# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import rl_engine.integrations.orchestrators.vime.experiments.dense.run_arm as run_arm
from rl_engine.integrations.orchestrators.vime.experiments.dense.run_arm import (
    ARMS,
    MEGATRON_ATTENTION_BACKEND,
    RL_KERNEL_LINEAR_LOGP_PROVIDER,
    _linear_logp_provider_args,
    _max_engine_decode_batch,
    _mismatch_metrics_args,
    _rollout_topology,
    _submit_ray_job,
)
from rl_engine.integrations.orchestrators.vime.experiments.dense.run_supplement_suite import specs
from rl_engine.integrations.orchestrators.vime.experiments.dense.validate_run import (
    RL_KERNEL_MISMATCH_SIDECAR_MARKER,
    VIME_NATIVE_LINEAR_LOGP_MARKER,
    _validate_pd_disaggregation,
    _validate_readbacks,
    _validate_topology,
)


def test_waiting_ray_submission_streams_and_saves_log(tmp_path: Path, capsys):
    result = _submit_ray_job(
        [sys.executable, "-c", "print('ray job output')"],
        wait=True,
        run_dir=tmp_path,
    )

    assert result.returncode == 0
    assert "ray job output" in capsys.readouterr().out
    assert (tmp_path / "run.log").read_text(encoding="utf-8") == "ray job output\n"


def test_ray_cli_address_overrides_stale_profile_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RAY_API_SERVER_ADDRESS", "http://127.0.0.1:8266")
    result = _submit_ray_job(
        [sys.executable, "-c", "import os; print(os.environ['RAY_API_SERVER_ADDRESS'])"],
        wait=True,
        run_dir=tmp_path,
        ray_address="http://127.0.0.1:8272",
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "http://127.0.0.1:8272"


def _production_operator(framework: str, target: str, module: str) -> dict:
    return {
        "framework": framework,
        "target": target,
        "module": module,
        "case_id": "P/P",
        "implementation": "production",
        "backend_id": f"{framework}.production.{module}",
        "call_count": 1,
        "provenance": {"runtime_platform": "cuda"},
    }


def _production_readbacks() -> list[dict]:
    megatron_modules = ("attention", "ffn")
    vllm_modules = ("attention", "ffn", "logp")
    return [
        {
            "framework": "megatron",
            "target": "training",
            "installed_hooks": {module: module for module in megatron_modules},
            "operators": {
                module: _production_operator("megatron", "training", module)
                for module in megatron_modules
            },
            "fallbacks": [],
        },
        {
            "framework": "vllm",
            "target": "rollout",
            "installed_hooks": {module: module for module in vllm_modules},
            "operators": {
                module: _production_operator("vllm", "rollout", module) for module in vllm_modules
            },
            "fallbacks": [],
        },
    ]


def test_tp4_formal_matrix_pins_the_vime_qwen3_attention_backend():
    assert MEGATRON_ATTENTION_BACKEND == "fused"


def test_rollout_tp_cp_derive_router_engines_and_graph_batch():
    expected = {
        (2, 1): (2, 4, 2),
        (2, 2): (4, 2, 4),
        (4, 1): (4, 2, 4),
        (4, 2): (8, 1, 8),
        (8, 1): (8, 1, 8),
    }
    for (rollout_tp, rollout_cp), (
        gpus_per_engine,
        engines,
        graph_batch,
    ) in expected.items():
        topology = _rollout_topology(rollout_tp, rollout_cp)
        assert topology["rollout_tp"] == rollout_tp
        assert topology["rollout_cp"] == rollout_cp
        assert topology["rollout_gpus_per_engine"] == gpus_per_engine
        assert topology["rollout_engines"] == engines
        assert topology["evidence_level"] == (
            "reference" if (rollout_tp, rollout_cp) == (4, 1) else "experimental"
        )
        assert (
            _max_engine_decode_batch(
                1,
                8,
                "round_robin",
                int(topology["rollout_engines"]),
            )
            == graph_batch
        )
        assert _validate_topology(topology) == []

    legacy_topology = _rollout_topology(4, 1)
    legacy_topology.pop("rollout_tp")
    legacy_topology.pop("rollout_cp")
    assert _validate_topology(legacy_topology) == []


def test_experimental_actor_topologies_are_self_consistent():
    topology = _rollout_topology(
        8,
        1,
        tensor_parallel_size=8,
        context_parallel_size=1,
    )
    assert topology["tp"] == 8
    assert topology["cp"] == 1
    assert topology["rollout_engines"] == 1
    assert topology["evidence_level"] == "experimental"
    assert _validate_topology(topology) == []


def test_1p1d_layout_uses_full_decode_batch_and_distinct_roles():
    topology = _rollout_topology(4, 1, rollout_layout="1p1d")
    assert topology["rollout_engines"] == 2
    assert topology["evidence_level"] == "experimental"
    assert _validate_topology(topology) == []
    assert _max_engine_decode_batch(1, 8, "round_robin", 2, "1p1d") == 8
    assert _max_engine_decode_batch(1, 8, "round_robin", 2) == 4


def test_1p1d_validation_requires_config_and_both_runtime_groups(tmp_path: Path):
    source = Path(__file__).parents[4] / "examples/vime_qwen3_8b_tp4_cp2_200/vllm_1p1d.yaml"
    config = tmp_path / "vllm_1p1d.yaml"
    config.write_bytes(source.read_bytes())
    groups = json.loads(config.read_text())["vllm"][0]["server_groups"]
    manifest = {
        "topology": _rollout_topology(4, 1, rollout_layout="1p1d"),
        "rollout_routing": {
            "layout": "1p1d",
            "server_groups": groups,
            "vllm_config": str(config),
            "vllm_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
        "train_command": ["train.py", "--vllm-config", str(config)],
        "batching": {"rollout_batch_size": 1, "n_samples_per_prompt": 8},
        "vllm_execution": {"capture_sizes": list(range(1, 9))},
    }
    log = "Engine group 'prefill' ready\nEngine group 'decode' ready"
    assert _validate_pd_disaggregation(manifest, log)["passed"]
    assert not _validate_pd_disaggregation(manifest, "Engine group 'decode' ready")["passed"]
    manifest["vllm_execution"]["capture_sizes"] = list(range(1, 5))
    assert not _validate_pd_disaggregation(manifest, log)["passed"]
    manifest["vllm_execution"]["capture_sizes"] = list(range(1, 9))
    config.write_text(
        config.read_text().replace('"update_weights": true', '"update_weights": false')
    )
    assert not _validate_pd_disaggregation(manifest, log)["passed"]


def test_1p1d_dry_run_records_vime_command_and_config(tmp_path: Path, monkeypatch):
    root = Path(__file__).parents[4]
    for name in ("vime", "megatron", "model", "reference"):
        (tmp_path / name).mkdir()
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        run_arm,
        "_repository_state",
        lambda path: {"revision": "test", "dirty": False, "status": [], "tracked_diff_sha256": ""},
    )
    monkeypatch.setattr(run_arm, "_gpu_inventory", lambda: [])
    assert (
        run_arm.main(
            [
                "--group",
                "consistency",
                "--num-rollout",
                "2",
                "--run-id",
                "pd-test",
                "--output-root",
                str(tmp_path),
                "--rl-kernel-root",
                str(root),
                "--vime-root",
                str(tmp_path / "vime"),
                "--megatron-root",
                str(tmp_path / "megatron"),
                "--model-root",
                str(tmp_path / "model"),
                "--ref-load",
                str(tmp_path / "reference"),
                "--prompt-data",
                str(prompt),
                "--python",
                sys.executable,
                "--ray-bin",
                sys.executable,
                "--pd",
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "pd-test/manifest.json").read_text())
    config_path = str(root / "examples/vime_qwen3_8b_tp4_cp2_200/vllm_1p1d.yaml")
    command = manifest["train_command"]
    assert command[command.index("--vllm-config") + 1] == config_path
    assert manifest["vllm_execution"]["capture_sizes"] == list(range(1, 9))
    assert manifest["topology"]["evidence_level"] == "experimental"
    log = "Engine group 'prefill' ready\nEngine group 'decode' ready"
    assert _validate_pd_disaggregation(manifest, log)["passed"]


def test_launcher_forces_cuda_graph_without_a_logp_provider():
    root = Path(__file__).parents[4]
    launcher = (
        root
        / "rl_engine/integrations/orchestrators/vime/experiments/dense/aligned_python_entrypoint.sh"
    )
    env = os.environ.copy()
    env.update(
        RL_KERNEL_REAL_PYTHON="/bin/echo",
        RL_KERNEL_ROOT=str(root),
        RL_KERNEL_VLLM_CUDAGRAPH_MAX_CAPTURE_SIZE="8",
    )

    result = subprocess.run(
        [
            str(launcher),
            "train.py",
            "--rollout-batch-size",
            "1",
            "--n-samples-per-prompt",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "--linear-logp-provider" not in result.stdout
    assert "--vllm-optimization-level 0" in result.stdout
    assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in result.stdout
    assert '"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8]' in result.stdout
    assert "required vLLM full-decode CUDA Graph capture sizes" in result.stderr


def test_production_arms_do_not_install_the_rlkernel_logp_provider():
    assert _linear_logp_provider_args(ARMS["native"]) == ()
    assert _linear_logp_provider_args(ARMS["G10"]) == ()


def test_rlkernel_arms_install_the_strict_logp_provider():
    expected = (
        "--linear-logp-provider",
        RL_KERNEL_LINEAR_LOGP_PROVIDER,
        "--linear-logp-provider-mode",
        "strict",
    )

    assert _linear_logp_provider_args(ARMS["consistency"]) == expected
    assert _linear_logp_provider_args(ARMS["G11"]) == expected


def test_module_ablation_matrix_changes_only_operator_routes():
    expected = {
        "M000": ("P/P", "P/P", "P/P"),
        "M100": ("R/R", "P/P", "P/P"),
        "M010": ("P/P", "R/R", "P/P"),
        "M001": ("P/P", "P/P", "R/R"),
        "M110": ("R/R", "R/R", "P/P"),
        "M101": ("R/R", "P/P", "R/R"),
        "M011": ("P/P", "R/R", "R/R"),
        "M111": ("R/R", "R/R", "R/R"),
    }

    for group, cases in expected.items():
        arm = ARMS[group]
        assert not arm.framework_use_rollout_logprobs
        assert (arm.attention_case, arm.ffn_case, arm.logp_case) == cases


def test_module_ablation_logp_provider_follows_training_route():
    provider_groups = {"M001", "M101", "M011", "M111"}
    expected_module_groups = {
        "M000",
        "M100",
        "M010",
        "M001",
        "M110",
        "M101",
        "M011",
        "M111",
    }
    for group in expected_module_groups:
        has_provider = bool(_linear_logp_provider_args(ARMS[group]))
        assert has_provider == (group in provider_groups)
    assert expected_module_groups == {key for key in ARMS if key.startswith("M")}


def test_module_ablation_enables_rlkernel_mismatch_sidecars():
    assert _mismatch_metrics_args() == (
        "--get-mismatch-metrics",
        "--custom-tis-function-path",
        (
            "rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.tis_me"
            "trics.metrics_only_tis"
        ),
    )


def test_supplement_suite_uses_short_module_and_three_seed_precision_designs():
    module = specs("module")
    assert len(module) == 8
    assert all(rounds == 8 and seed == 1234 for _, rounds, seed in module)

    precision = specs("precision")
    assert len(precision) == 12
    assert {group for group, _, _ in precision} == {"native", "G10", "consistency", "G11"}
    assert {seed for _, _, seed in precision} == {1234, 2345, 3456}
    assert all(rounds == 8 for _, rounds, _ in precision)


def test_validator_accepts_native_vime_logp_evidence_for_production_arm():
    report = _validate_readbacks(
        _production_readbacks(),
        asdict(ARMS["G10"]),
        VIME_NATIVE_LINEAR_LOGP_MARKER,
    )

    assert report["passed"]
    training_logp = report["frameworks"]["megatron/training"]["modules"]["logp"]
    assert training_logp["native_marker_present"]
    assert training_logp["call_count"] == 0


def test_validator_accepts_rlkernel_sidecar_evidence_for_production_logp():
    report = _validate_readbacks(
        _production_readbacks(),
        asdict(ARMS["G10"]),
        f"{RL_KERNEL_MISMATCH_SIDECAR_MARKER}P/P",
    )

    assert report["passed"]
    training_logp = report["frameworks"]["megatron/training"]["modules"]["logp"]
    assert training_logp["native_marker_present"]


def test_validator_rejects_provider_readback_on_production_megatron_logp():
    readbacks = _production_readbacks()
    contaminated = _production_operator("megatron", "training", "logp")
    contaminated["backend_id"] = "pytorch-vocab-parallel-logp-ws2"
    contaminated["provenance"] = {
        "runtime_platform": "cuda",
        "actual_backend": "rlkernel.linear_logp.bitwise.v1",
        "deterministic_linear_logp": True,
        "execution": {"strict_backend": True},
    }
    readbacks[0]["installed_hooks"]["logp"] = "rlkernel-provider"
    readbacks[0]["operators"]["logp"] = contaminated

    report = _validate_readbacks(
        readbacks,
        asdict(ARMS["G10"]),
        VIME_NATIVE_LINEAR_LOGP_MARKER,
    )

    assert not report["passed"]
    assert any("unexpectedly entered provider readback" in error for error in report["errors"])


def test_validator_rejects_production_label_over_rlkernel_actual_backend():
    readbacks = _production_readbacks()
    readbacks[1]["operators"]["logp"]["provenance"] = {
        "runtime_platform": "cuda",
        "actual_backend": "rlkernel.linear_logp.bitwise.v1",
    }

    report = _validate_readbacks(
        readbacks,
        asdict(ARMS["G10"]),
        VIME_NATIVE_LINEAR_LOGP_MARKER,
    )

    assert not report["passed"]
    assert any(
        "production route executed an RL-Kernel backend" in error for error in report["errors"]
    )
