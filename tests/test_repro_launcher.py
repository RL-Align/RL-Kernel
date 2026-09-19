from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import rl_engine.repro as repro

from rl_engine.repro import (
    _arm_config,
    _resolved_paths,
    _runner_command,
    Paths,
    build_parser,
    canonical_arm,
    doctor,
)


def _profile() -> dict:
    profile_path = (
        Path(__file__).parents[1]
        / "examples/vime_qwen3_8b_tp4_cp2_200/profiles/qwen3-8b-tp4-cp2.json"
    )
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _args(tmp_path: Path, mode: str):
    return build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--te-root",
            str(tmp_path / "te"),
            "--mode",
            mode,
        ]
    )


def test_user_modes_map_to_operator_arms():
    assert canonical_arm("native") == "native"
    assert canonical_arm("consistency") == "consistency"
    assert canonical_arm("G00") == "native"
    assert canonical_arm("G01") == "consistency"
    assert _arm_config(_profile(), "native")["te_root_env"] == "TE218_ROOT"
    assert _arm_config(_profile(), "consistency")["te_root_env"] == "TE218_ROOT"


def test_consistency_command_does_not_enable_rollout_logprob_reuse(tmp_path: Path):
    profile = _profile()
    args = _args(tmp_path, "consistency")
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)

    assert command[command.index("--group") + 1] == "consistency"
    assert "--use-rollout-logprobs" not in command
    assert command[command.index("--tp-size") + 1] == "4"
    assert command[command.index("--cp-size") + 1] == "2"
    assert command[command.index("--rollout-tp-size") + 1] == "4"
    assert command[command.index("--rollout-cp-size") + 1] == "1"


def test_native_command_uses_production_operator_route(tmp_path: Path):
    profile = _profile()
    args = _args(tmp_path, "native")
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)

    assert command[command.index("--group") + 1] == "native"
    assert "--use-rollout-logprobs" not in command


def test_launcher_defaults_to_active_checkout_and_runtime(tmp_path: Path):
    profile = _profile()
    args = _args(tmp_path, "native")
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)
    repo_root = Path(__file__).parents[1].resolve()

    assert paths.rl_kernel_root == repo_root
    assert paths.python == Path(sys.executable).absolute()
    assert Path(command[1]) == repo_root / "examples/vime_qwen3_8b_tp4_cp2_200/run_arm.py"
    assert command[command.index("--ray-address") + 1] == "http://127.0.0.1:8265"
    extra_paths = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--extra-pythonpath"
    ]
    assert len(extra_paths) == len(set(extra_paths))


def test_non_reference_actor_topology_runs_without_an_opt_in_flag(tmp_path: Path):
    profile = _profile()
    base = [
        "plan",
        "--workspace",
        str(tmp_path),
        "--mode",
        "consistency",
        "--tp-size",
        "8",
        "--cp-size",
        "1",
        "--rollout-tp-size",
        "8",
    ]
    args = build_parser().parse_args(base)
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)
    assert command[command.index("--tp-size") + 1] == "8"
    assert command[command.index("--cp-size") + 1] == "1"


def test_tp1_cuda_uses_memory_safe_vllm_default(tmp_path: Path):
    profile = _profile()
    args = build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--mode",
            "consistency",
            "--tp-size",
            "1",
            "--cp-size",
            "8",
        ]
    )
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)
    index = len(command) - 1 - command[::-1].index("--vllm-gpu-memory-utilization")

    assert command[index + 1] == "0.2"


def test_explicit_vllm_memory_fraction_overrides_profile(tmp_path: Path):
    profile = _profile()
    args = build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--mode",
            "consistency",
            "--vllm-gpu-memory-utilization",
            "0.31",
        ]
    )
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)
    index = len(command) - 1 - command[::-1].index("--vllm-gpu-memory-utilization")

    assert command[index + 1] == "0.31"


def test_sampling_parameters_are_forwarded(tmp_path: Path):
    profile = _profile()
    args = build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--mode",
            "consistency",
            "--rollout-temperature",
            "0.7",
            "--rollout-top-p",
            "0.95",
        ]
    )
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)

    temperature_index = len(command) - 1 - command[::-1].index("--rollout-temperature")
    top_p_index = len(command) - 1 - command[::-1].index("--rollout-top-p")
    assert command[temperature_index + 1] == "0.7"
    assert command[top_p_index + 1] == "0.95"


def test_rocm_command_uses_rocm_runner_without_cuda_runtime_options(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "plan",
            "--backend",
            "rocm",
            "--workspace",
            str(tmp_path),
            "--mode",
            "consistency",
            "--rollouts",
            "1",
            "--run-id",
            "rocm-smoke",
            "--tp-size",
            "2",
            "--cp-size",
            "4",
            "--rollout-temperature",
            "0.7",
            "--rollout-top-p",
            "0.95",
        ]
    )
    profile = _profile()
    command = _runner_command(_resolved_paths(profile, args), profile, args)
    assert command[2] == "examples.vime_rocm_attention_ablation.run_qwen3_8b"
    assert command[command.index("--rollout-temperature") + 1] == "0.7"
    assert command[command.index("--rollout-top-p") + 1] == "0.95"
    assert command[command.index("--tp-size") + 1] == "2"
    assert "--ld-library-path" not in command
    assert "--use-rollout-logprobs" not in command


def test_doctor_enforces_frozen_runtime_and_ray(tmp_path: Path, monkeypatch):
    directories = {
        name: tmp_path / name
        for name in (
            "workspace",
            "rl_kernel_root",
            "vime_root",
            "megatron_root",
            "runtime_root",
            "cuda_runtime_root",
            "runtime_site",
            "cuda_python_site",
            "te_root",
            "data_root",
            "model_root",
            "ref_load",
            "output_root",
        )
    }
    for path in directories.values():
        path.mkdir(parents=True)
    for repository in ("rl_kernel_root", "vime_root", "megatron_root"):
        (directories[repository] / ".git").mkdir()
    prompt_data = tmp_path / "prompt.jsonl"
    python = tmp_path / "python"
    ray = tmp_path / "ray"
    for path in (prompt_data, python, ray):
        path.touch()
    paths = Paths(
        **directories,
        prompt_data=prompt_data,
        python=python,
        ray=ray,
    )
    profile = _profile()
    expected_versions = {
        key: str(profile["requirements"][key])
        for key in ("python", "torch", "vllm", "ray", "transformer_engine")
    }
    monkeypatch.setattr(repro, "_gpu_names", lambda: ["NVIDIA H100 80GB HBM3"] * 8)
    monkeypatch.setattr(repro, "_python_versions", lambda _python: expected_versions)
    monkeypatch.setattr(
        repro.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="Ray jobs: 0", stderr=""),
    )

    assert doctor(paths, profile, ray_address="http://127.0.0.1:8265") == 0

    mismatched = dict(expected_versions, vllm="0.15.0")
    monkeypatch.setattr(repro, "_python_versions", lambda _python: mismatched)
    assert doctor(paths, profile, ray_address="http://127.0.0.1:8265") == 1
