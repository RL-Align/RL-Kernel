# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from examples.vime_rocm_attention_ablation.run import MatrixConfig, build_plan
from examples.vime_rocm_attention_ablation.run_pr377_workload import (
    WorkloadConfig,
    parse_args as parse_workload_args,
)
from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
)
from rl_engine.kernels.ops.rocm.attention.strict_runtime import RCCLAGRSAttentionCPCommunication

ROOT = Path(__file__).parents[1]


def _config(tmp_path: Path, **overrides) -> MatrixConfig:
    values = {
        "vime_root": tmp_path / "vime",
        "rl_kernel_root": tmp_path / "rl-kernel",
        "megatron_root": tmp_path / "megatron",
        "model_root": tmp_path / "model",
        "reference_checkpoint": tmp_path / "checkpoint",
        "prompt_data": tmp_path / "prompts.jsonl",
        "run_dir": tmp_path / "run",
        "launcher": tmp_path / "launch.sh",
    }
    values.update(overrides)
    return MatrixConfig(**values)


def test_default_topology_matches_pr377_colocated_tp4_cp2(tmp_path):
    config = _config(tmp_path)
    config.validate(require_paths=False)

    parameters = build_plan(config)["parameters"]
    assert parameters["training"] == {
        "num_gpus": 8,
        "tensor_parallel_size": 4,
        "context_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "sequence_parallel": False,
        "dtype": "bf16",
        "attention_backend": "flash",
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "make_vocab_size_divisible_by": 128,
    }
    assert parameters["rollout"]["num_gpus"] == 8
    assert parameters["rollout"]["engine_count"] == 2
    assert parameters["rollout"]["tensor_parallel_size"] == 4
    assert parameters["rollout"]["router_policy"] == "round_robin"
    assert parameters["batch"]["rollout_batch_size"] == 2
    assert parameters["batch"]["samples_per_prompt"] == 1
    assert parameters["batch"]["global_batch_size"] == 2
    assert parameters["placement"] == {
        "colocate": True,
        "offload_train": False,
        "offload_rollout": True,
    }

    environment = build_plan(config)["arms"][0]["environment"]
    assert environment["RLK_ABLATION_COLOCATE"] == "1"
    assert environment["RLK_ABLATION_ROUTER_POLICY"] == "round_robin"
    assert environment["RL_KERNEL_FFN_CASE"] == "R/R"
    assert environment["RL_KERNEL_LOGP_CASE"] == "R/R"
    assert environment["RL_KERNEL_VLLM_REAL_VOCAB_SIZE"] == "151936"
    assert environment["RL_KERNEL_VLLM_PADDED_VOCAB_SIZE"] == "152064"


def test_rocm_user_modes_do_not_enable_rollout_logprob_reuse(tmp_path):
    native = parse_workload_args(
        ["--mode", "native", "--run-dir", str(tmp_path / "native")]
    )
    consistency = parse_workload_args(
        ["--mode", "consistency", "--run-dir", str(tmp_path / "consistency")]
    )
    assert native.case == "P/P"
    assert consistency.case == "R/R"

    config = WorkloadConfig(
        vime_root=tmp_path / "vime",
        rl_kernel_root=tmp_path / "rl-kernel",
        megatron_root=tmp_path / "megatron",
        model_root=tmp_path / "model",
        reference_checkpoint=tmp_path / "checkpoint",
        prompt_data=tmp_path / "prompts.jsonl",
        run_dir=tmp_path / "run",
        launcher=tmp_path / "launch.sh",
    )
    config.case_id = native.case
    assert config.frozen_parameters()["framework_consistency"]["use_rollout_logprobs"] is False


def test_rocm_non_reference_topology_needs_no_opt_in_flag(tmp_path):
    base = [
        "--mode",
        "consistency",
        "--run-dir",
        str(tmp_path / "run"),
        "--tp-size",
        "8",
        "--cp-size",
        "1",
        "--rollout-tp-size",
        "8",
    ]
    args = parse_workload_args(base)
    assert (args.tp_size, args.cp_size, args.rollout_tp_size) == (8, 1, 8)


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), 0.0, -1.0])
def test_rocm_rejects_invalid_temperature(tmp_path, temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        _config(tmp_path, rollout_temperature=temperature).validate(require_paths=False)


def test_rocm_sampling_and_canonical_contract_reach_workers(tmp_path):
    config = _config(
        tmp_path,
        tensor_parallel_size=2,
        context_parallel_size=4,
        rollout_temperature=0.7,
        rollout_top_p=0.95,
    )
    config.validate(require_paths=False)
    plan = build_plan(config)
    env = plan["arms"][0]["environment"]
    assert plan["parameters"]["rollout"]["temperature"] == 0.7
    assert env["RL_KERNEL_STRICT_CANONICAL_TP"] == "4"
    assert env["RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE"] == "152064"
    assert env["RL_KERNEL_VLLM_TEMPERATURE"] == "0.7"
    assert env["RLK_ABLATION_ROLLOUT_TOP_P"] == "0.95"
    launcher = (ROOT / "examples/vime_rocm_attention_ablation/launch_arm.sh").read_text()
    runtime = launcher.split("RUNTIME_ENV_JSON=", 1)[1].split("ray_started=", 1)[0]
    assert '"RL_KERNEL_STRICT_CANONICAL_TP"' in runtime
    assert '"RL_KERNEL_VLLM_TEMPERATURE"' in runtime


def test_rocm_packed_cp1_rope_accepts_odd_lengths():
    from rl_engine.integrations.megatron_runtime import _strict_rocm_rope_positions

    tensor = torch.empty(8, 1, 4)
    freqs = torch.empty(5, 1, 1, 4)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    positions = _strict_rocm_rope_positions(tensor, freqs, cu_seqlens, None)
    assert positions.tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm GPU contract")
def test_rocm_top_p_replay_has_no_64_token_limit():
    from rl_engine.integrations.linear_logp import LinearLogpWrapper

    logits = torch.linspace(-4, 4, 256, device="cuda", dtype=torch.bfloat16).repeat(2, 1)
    targets = torch.tensor([120, 220], device="cuda")
    ids = torch.arange(100, 256, device="cuda").repeat(2, 1)
    ids = torch.cat([targets[:, None], ids], dim=1)
    values = torch.zeros_like(ids, dtype=torch.float32)
    op = LinearLogpWrapper()
    actual = op.from_local_logits_top_p(
        logits,
        targets,
        ids,
        values,
        tp_group=None,
        vocab_start_index=0,
        global_vocab_size=256,
        real_vocab_size=256,
        temperature=0.7,
    )
    masked = logits.clone()
    masked[:, :100] = float("-inf")
    expected = op.from_local_logits(
        masked,
        targets,
        tp_group=None,
        vocab_start_index=0,
        global_vocab_size=256,
        real_vocab_size=256,
        temperature=0.7,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm GPU contract")
@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_rocm_scalar_temperature_matches_vime_arithmetic(temperature):
    from rl_engine.integrations.linear_logp import LinearLogpWrapper

    generator = torch.Generator(device="cuda").manual_seed(1234)
    logits = torch.randn(256, 256, generator=generator, device="cuda").mul_(12).bfloat16()
    targets = torch.arange(256, device="cuda")
    op = LinearLogpWrapper()
    kwargs = dict(tp_group=None, vocab_start_index=0, global_vocab_size=256, real_vocab_size=256)
    actual = op.from_local_logits(logits, targets, temperature=temperature, **kwargs)
    # VIME applies scalar division before constructing its provider request.
    expected = op.from_local_logits(logits.float() / temperature, targets, **kwargs)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_colocated_topology_requires_training_to_cover_all_gpus(tmp_path):
    config = _config(tmp_path, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="must use all visible GPUs"):
        config.validate(require_paths=False)


def test_router_requires_at_least_one_request_per_engine(tmp_path):
    config = _config(
        tmp_path,
        rollout_batch_size=1,
        samples_per_prompt=1,
        global_batch_size=1,
    )
    with pytest.raises(ValueError, match="one request per rollout engine"):
        config.validate(require_paths=False)


def test_rocm_cp_adapter_accepts_rccl_plan_without_widening_cuda_contract(monkeypatch):
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=4,
            tp_rank=0,
            cp_world_size=2,
            cp_rank=0,
        ),
        backend="rccl_ag_rs",
        status="implemented",
    )
    communication = object.__new__(RCCLAGRSAttentionCPCommunication)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    communication._validate_cuda_plan(plan)

    assert plan.backend == "rccl_ag_rs"


def test_dashboard_cannot_overlap_ray_worker_port_range(tmp_path):
    config = _config(tmp_path, ray_dashboard_port=18265)
    with pytest.raises(ValueError, match="worker range"):
        config.validate(require_paths=False)


def test_launcher_uses_pr377_torch_dist_actor_load_without_reference_model():
    launcher = (ROOT / "examples" / "vime_rocm_attention_ablation" / "launch_arm.sh").read_text(
        encoding="utf-8"
    )

    assert '--load "${RLK_ABLATION_REFERENCE_CHECKPOINT}"' in launcher
    assert "--megatron-to-hf-mode" not in launcher
    assert 'RLK_ABLATION_USE_KL_LOSS:-0}" == "1"' in launcher
    assert "--use-kl-loss" in launcher
    assert "--kl-loss-coef" in launcher
    assert "--linear-logp-provider" in launcher
    assert "rl_engine.integrations.vime.linear_logp_provider.provider" in launcher
    assert "--linear-logp-provider-mode strict" in launcher
    assert (
        'for module_case in "${RL_KERNEL_FFN_CASE:-}" ' '"${RL_KERNEL_LOGP_CASE:-}"; do'
    ) in launcher
    assert "FFN and Logp cases must each be P/P or R/R" in launcher
