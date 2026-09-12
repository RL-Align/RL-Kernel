# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json

import torch

from examples.vime_qwen3_8b_tp4_cp2_200.validate_artifacts import validate_artifacts


def _write_readback(directory, framework, target, *, triton_used=False):
    operators = {}
    hooks = {}
    for module in ("attention", "ffn", "logp"):
        backend = (
            "rlkernel.linear_logp.bitwise.v1" if module == "logp" else f"rlkernel.{module}.test"
        )
        operators[module] = {
            "module": module,
            "implementation": "rl_kernel",
            "backend_id": backend,
            "call_count": 2,
            "provenance": {
                "runtime_platform": "cuda",
                "actual_backend": f"rlkernel.cuda.{module}",
                "triton_used": triton_used,
            },
        }
        hooks[module] = f"test.{module}"
    payload = {
        "framework": framework,
        "target": target,
        "installed_hooks": hooks,
        "fallbacks": [],
        "operators": operators,
    }
    (directory / f"{framework}-{target}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_vime_artifacts_require_exact_cuda_logp_equality(tmp_path):
    readbacks = tmp_path / "readbacks"
    train_data = tmp_path / "train-data"
    readbacks.mkdir()
    train_data.mkdir()
    _write_readback(readbacks, "megatron", "training")
    _write_readback(readbacks, "vllm", "rollout")
    values = torch.tensor([-1.25, -2.5], dtype=torch.float32)
    torch.save(
        {"samples": [{"log_probs": values.clone(), "rollout_log_probs": values.clone()}]},
        train_data / "0.pt",
    )

    report = validate_artifacts(readbacks, train_data)

    assert report["passed"] is True
    assert report["train_rollout_logp"]["torch_equal"] is True
    assert report["train_rollout_logp"]["mismatch_count"] == 0
    assert report["train_rollout_logp"]["max_abs_diff"] == 0.0


def test_vime_artifacts_reject_triton_even_when_values_match(tmp_path):
    readbacks = tmp_path / "readbacks"
    train_data = tmp_path / "train-data"
    readbacks.mkdir()
    train_data.mkdir()
    _write_readback(readbacks, "megatron", "training")
    _write_readback(readbacks, "vllm", "rollout", triton_used=True)
    values = torch.tensor([-1.25], dtype=torch.float32)
    torch.save(
        {"samples": [{"log_probs": values, "rollout_log_probs": values.clone()}]},
        train_data / "0.pt",
    )

    report = validate_artifacts(readbacks, train_data)

    assert report["passed"] is False
    assert "vllm/rollout attention used Triton" in report["readbacks"]["errors"]
