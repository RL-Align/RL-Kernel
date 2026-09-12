# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from examples.vime_rocm_attention_ablation.run import MatrixConfig, build_plan
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


def test_colocated_topology_requires_training_to_cover_all_gpus(tmp_path):
    config = _config(tmp_path, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="must use all visible GPUs"):
        config.validate(require_paths=False)


def test_router_requires_at_least_one_request_per_engine(tmp_path):
    config = _config(
        tmp_path,
        rollout_batch_size=1,
        samples_per_prompt=2,
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
