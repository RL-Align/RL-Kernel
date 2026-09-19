# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch
from types import SimpleNamespace

import examples.vime_qwen3_8b_tp4_cp2_200.validate_run as validator
import examples.vime_rocm_attention_ablation.validate_artifacts as artifacts
from examples.vime_qwen3_8b_tp4_cp2_200.run_arm import _rollout_topology
import rl_engine.repro as repro
from rl_engine.integrations.framework_operators import _MegatronCPWeightGradient
from rl_engine.repro import build_parser, _runner_command, _resolved_paths
import rl_engine.kernels.ops.pytorch.ffn.ffn as ffn


def test_short_command_preserves_sampling_and_derives_cp(tmp_path):
    args = build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--te-root",
            str(tmp_path / "te"),
            "--tp",
            "2",
            "--rollout-tp",
            "8",
            "--temperature",
            ".7",
            "--top-p",
            ".95",
            "--top-k",
            "100",
            "--steps",
            "2",
            "--lr",
            "1e-6",
            "--require-updates",
        ]
    )
    profile = {"modes": {"consistency": {}}, "requirements": {}}
    paths = _resolved_paths(profile, args)
    command = _runner_command(paths, profile, args)
    for key, value in [
        ("--cp-size", "4"),
        ("--rollout-temperature", "0.7"),
        ("--rollout-top-p", "0.95"),
        ("--rollout-top-k", "100"),
        ("--lr", "1e-06"),
        ("--num-rollout", "2"),
    ]:
        assert command[command.index(key) + 1] == value
    assert "--require-updates" in command
    assert "--use-rollout-logprobs" not in command


def _step():
    return {
        "train/train_current_rollout_logprob_mismatch_count": 0,
        "train/train_current_rollout_logprob_max_abs_diff": 0,
        "train/train_rollout_logprob_abs_diff": 0,
        "train/train_current_rollout_logprob_numel": 10,
    }


@pytest.mark.parametrize("ids", [[0], [1, 2], [0, 2]])
def test_scalar_gate_rejects_partial_or_shifted_steps(ids):
    result = validator._validate_runtime_logprobs({i: _step() for i in ids}, 2, 8, True)
    assert not result["passed"]


def test_nonzero_runtime_difference_fails_even_with_correct_step_count():
    step = _step()
    step["train/train_current_rollout_logprob_mismatch_count"] = 1
    assert not validator._validate_runtime_logprobs({0: step}, 1, 8, True)["passed"]


@pytest.mark.parametrize("column", [True, False])
def test_canonical_backward_matches_analytic_gradient(monkeypatch, column):
    monkeypatch.setenv("RL_KERNEL_STRICT_CANONICAL_TP", "4")
    monkeypatch.setattr(ffn, "_linear_da", lambda g, w, **kw: g @ w)
    monkeypatch.setattr(ffn, "_linear_dw", lambda a, g, **kw: g.T @ a)
    a = torch.arange(32, dtype=torch.float64).reshape(4, 8).requires_grad_()
    w = torch.arange(64, dtype=torch.float64).reshape(8, 8).requires_grad_()
    g = torch.arange(32, dtype=torch.float64).reshape(4, 8)
    (a @ w.T).backward(g)
    da = ffn._canonical_tp_input_gradient(
        g, w.detach(), tp_world=2, column=column, disable_split_k=True
    )
    dw = ffn._canonical_tp_weight_gradient(
        a.detach(), g, tp_world=2, column=column, disable_split_k=True
    )
    assert torch.equal(da, a.grad)
    assert torch.equal(dw, w.grad)


def test_nonfinite_loss_is_retained_and_rejected():
    record = _step()
    record["train/loss"] = float("inf")
    parsed = validator._parse_runtime_records(f"step 0: {record!r}")
    result = validator._validate_runtime_logprobs(parsed["step"], 1, 8, True)
    assert not result["passed"]
    assert any("train/loss is nonfinite" in error for error in result["errors"])


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1"])
def test_invalid_kl_coefficient_is_not_silently_disabled(tmp_path, value):
    args = build_parser().parse_args(
        [
            "plan",
            "--workspace",
            str(tmp_path),
            "--te-root",
            str(tmp_path / "te"),
            "--kl-coef",
            value,
        ]
    )
    profile = {"modes": {"consistency": {}}, "requirements": {}}
    with pytest.raises(repro.ReproError, match="finite and nonnegative"):
        _runner_command(_resolved_paths(profile, args), profile, args)


def test_rollout_cp_is_valid_when_the_engine_product_fits():
    args = build_parser().parse_args(["verify", "--rollout-tp", "2", "--rollout-cp", "2"])
    repro._validate_topology_args(args)


def test_tp1_train_and_rollout_enable_offload():
    assert _rollout_topology(1, 1, tensor_parallel_size=1, context_parallel_size=8)["offload_train"]
    assert not _rollout_topology(4, 1, tensor_parallel_size=1, context_parallel_size=8)[
        "offload_train"
    ]


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("rollout_tp", [1, 2, 4, 8])
def test_validator_accepts_derived_offload_and_rejects_inverted_flag(tp, rollout_tp):
    topology = _rollout_topology(
        rollout_tp, 1, tensor_parallel_size=tp, context_parallel_size=8 // tp
    )
    assert validator._validate_topology(topology) == []
    topology["offload_train"] = not topology["offload_train"]
    assert any("offload_train" in error for error in validator._validate_topology(topology))


def test_verify_defaults_require_real_updates():
    args = build_parser().parse_args(["verify"])
    assert args.wait and args.require_updates
    assert args.rollouts == 2 and args.max_response_len == 512
    assert args.lr > 0 and args.kl_coef > 0


def test_busy_gpu_preflight(monkeypatch):
    monkeypatch.setattr(
        repro.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="123, 2048 MiB\n")
    )
    with pytest.raises(repro.ReproError, match="123, 2048 MiB"):
        repro._require_idle_gpus()
    monkeypatch.setattr(repro.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=""))
    repro._require_idle_gpus()


def test_bitwise_gate_rejects_signed_zero_difference(tmp_path):
    payload = {
        "schema_version": artifacts.SIDECAR_SCHEMA_VERSION,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "rank": 0,
        "call_index": 0,
        "train_log_probs": [torch.tensor([0.0])],
        "rollout_log_probs": [torch.tensor([-0.0])],
        "loss_masks": [torch.tensor([1])],
        "total_lengths": [2],
        "response_lengths": [1],
    }
    torch.save(payload, tmp_path / "rank00000.call00000000.pt")
    result = artifacts.compare_train_rollout_logps(tmp_path, require_exact=True)
    assert result["torch_equal"]
    assert result["bitwise_mismatch_count"] == 1
    assert not result["passed"]


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
def test_megatron_cp_reduction_does_not_multiply_complete_ffn_gradient(cp):
    # A strict FFN weight GEMM sees the complete gathered CP token sequence.
    # Reproduce VIME's CP loss multiplier and Megatron's average reduction.
    x = torch.arange(24, dtype=torch.float64).reshape(8, 3)
    initial = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    reference = initial.clone().requires_grad_()
    (x @ reference.T).square().sum().backward()
    rank_grads = []
    for _ in range(cp):
        weight = initial.clone().requires_grad_()
        adjusted = _MegatronCPWeightGradient.apply(weight, cp)
        (cp * (x @ adjusted.T).square().sum()).backward()
        rank_grads.append(weight.grad)
    ddp_gradient = torch.stack(rank_grads).mean(0)
    assert torch.equal(ddp_gradient, reference.grad)
