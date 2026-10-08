# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from rl_engine.bi import runtime
from rl_engine.bi.builtin import QWEN3_8B
from rl_engine.bi.readback import validate_plan_readbacks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_ACTIVE", None)
    monkeypatch.setenv("RL_KERNEL_BI", "1")
    monkeypatch.delenv(runtime.PLAN_ENV, raising=False)
    monkeypatch.setattr(runtime, "runtime_identity", lambda repos: {"runtime": "pinned"})
    monkeypatch.setattr(
        runtime, "hardware_identity", lambda: ("cuda", ("NVIDIA H100:sm90:80GB",) * 8)
    )
    (tmp_path / "config.json").write_text(QWEN3_8B.config_match)
    return tmp_path


def test_cuda_runner_writes_selected_plan_into_ray_env(prepared_launch, monkeypatch):
    from examples.vime_qwen3_8b_tp4_cp2_200 import run_arm

    root = prepared_launch
    prompt = root / "prompts.jsonl"
    prompt.write_text('{"prompt":"hello"}\n')
    monkeypatch.setattr(
        run_arm, "_repository_state", lambda _: {"dirty": False, "revision": "test"}
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
                "test-bi",
                "--output-root",
                str(root / "out"),
                "--rl-kernel-root",
                str(ROOT),
                "--vime-root",
                str(root),
                "--megatron-root",
                str(root),
                "--model-root",
                str(root),
                "--ref-load",
                str(root),
                "--prompt-data",
                str(prompt),
                "--python",
                sys.executable,
                "--ray-bin",
                sys.executable,
                "--dry-run",
            ]
        )
        == 0
    )
    manifest = json.loads((root / "out/test-bi/manifest.json").read_text())
    environment = manifest["runtime_env"]["env_vars"]
    assert environment["RL_KERNEL_BI"] == "1"
    assert json.loads(environment[runtime.PLAN_ENV]) == manifest["bi_plan"]
    assert environment["RL_KERNEL_DET_GEMM_BACKEND"] == "cublaslt_nosplitk"
    command = manifest["ray_command"]
    ray_env = json.loads(command[command.index("--runtime-env-json") + 1])
    assert ray_env["env_vars"] == environment


def test_rocm_shell_forwards_every_selected_route(prepared_launch, monkeypatch, capsys):
    environment = {}
    runtime.prepare_vime_environment(
        environment,
        model_root=prepared_launch,
        platform="cuda",
        topology=(8, 4, 2, 4, 1),
        workload={},
        repositories={},
    )
    script = (ROOT / "examples/vime_rocm_attention_ablation/launch_arm.sh").read_text()
    snippet = script.split("RUNTIME_ENV_JSON=\"$(python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    tree = ast.parse(snippet)
    names = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "names" for target in node.targets)
    )
    for name in names:
        monkeypatch.setenv(name, environment.get(name, "test"))
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    capsys.readouterr()
    exec(compile(tree, "rocm-ray-environment", "exec"), {})
    forwarded = json.loads(capsys.readouterr().out)["env_vars"]
    assert all(forwarded[key] == value for key, value in environment.items())


@pytest.mark.parametrize("framework", ["vllm", "megatron"])
def test_framework_bootstrap_rejects_unprepared_flag(prepared_launch, framework):
    if framework == "vllm":
        from rl_engine.integrations.vllm_runtime import register_vllm_plugin as initialize
    else:
        from rl_engine.integrations.megatron_runtime import (
            initialize_from_environment as initialize,
        )
    with pytest.raises(RuntimeError, match="prepared plan"):
        initialize()


def test_legacy_readbacks_are_unchanged():
    assert validate_plan_readbacks([{"framework": "megatron"}, {"framework": "vllm"}]) == []


def test_selected_adapter_installs_both_framework_sides(prepared_launch, monkeypatch):
    from rl_engine.bi import adapters
    from rl_engine.integrations.megatron_runtime import initialize_from_environment
    from rl_engine.integrations.vllm_runtime import register_vllm_plugin

    calls = []
    original = adapters.builtin_adapters()["vime.qwen3.v1"]
    adapter = adapters.PlanAdapter(
        original.validate,
        lambda args: calls.append(("training", args)) or "training-integration",
        lambda: calls.append(("rollout", None)),
    )
    monkeypatch.setattr(adapters, "builtin_adapters", lambda: {"vime.qwen3.v1": adapter})
    environment = {}
    runtime.prepare_vime_environment(
        environment,
        model_root=prepared_launch,
        platform="cuda",
        topology=(8, 4, 2, 4, 1),
        workload={},
        repositories={},
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    register_vllm_plugin()
    assert initialize_from_environment("model-args") == "training-integration"
    assert calls == [("rollout", None), ("training", "model-args")]


@pytest.mark.parametrize("corruption", ["different", "missing", "all_missing", "one_side"])
def test_training_rollout_plan_disagreement_fails(corruption):
    identity = {key: "same" for key in ("plan_id", "plan_digest", "context_digest", "digest")}
    values = [
        {"framework": framework, "bi_plan": dict(identity)} for framework in ("megatron", "vllm")
    ]
    assert validate_plan_readbacks(values, identity) == []
    if corruption == "different":
        values[1]["bi_plan"]["context_digest"] = "other"
    elif corruption == "missing":
        values[1].pop("bi_plan")
    elif corruption == "all_missing":
        for value in values:
            value.pop("bi_plan")
    else:
        values.pop()
    assert validate_plan_readbacks(values, identity)
