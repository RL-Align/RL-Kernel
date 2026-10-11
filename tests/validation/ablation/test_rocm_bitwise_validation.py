# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.validate_artifacts import (  # noqa: E501
    SIDECAR_SCHEMA_VERSION,
    compare_train_rollout_logps,
)


@pytest.mark.parametrize("rollout_zero,passed", [(0.0, True), (-0.0, False)])
def test_strict_validator_compares_bits_including_signed_zero(tmp_path, rollout_zero, passed):
    torch.save(
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "rank": 0,
            "call_index": 0,
            "train_log_probs": [torch.tensor([0.0, -1.0])],
            "rollout_log_probs": [torch.tensor([rollout_zero, -1.0])],
            "loss_masks": [torch.ones(2)],
            "total_lengths": [3],
            "response_lengths": [2],
        },
        tmp_path / "sidecar.pt",
    )
    result = compare_train_rollout_logps(tmp_path, require_exact=True)
    assert result["passed"] is passed
    assert result["bitwise_equal"] is passed
    assert result["bitwise_mismatch_count"] == (0 if passed else 1)
    assert result["torch_equal"] is True


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize(
    "missing", [None, "strict_entrypoint", "contract_version", "lm_head_result_reused"]
)
def test_sparse_backend_requires_explicit_contract_evidence(missing, replicated):
    from rl_engine.integrations.orchestrators.vime.experiments.rocm_attention.validate_artifacts import (  # noqa: E501
        STRICT_LINEAR_LOGP_BACKEND_ID,
        _validate_strict_dense_record,
    )

    provenance = {
        "runtime_platform": "rocm",
        "fallback": False,
        "logprob_kernel_backend": "rlkernel.sparse_nucleus.hip_serial_deterministic.v12",
        "strict_entrypoint": "sparse_nucleus_logp_from_local_logits_tp",
        "contract_version": "sparse-nucleus-hip-serial-deterministic-v12",
        "lm_head_result_reused": True,
        "deterministic_linear_logp": True,
    }
    if replicated:
        provenance.update(
            strict_entrypoint="sparse_nucleus_logp_from_replicated_logits",
            replicated_logits_reused=True,
            additional_tp_collective=False,
            preparation_backend="rlkernel.sparse_nucleus.hip_replicated.v1",
        )
    if missing:
        provenance.pop(missing)
    record = {
        "case_id": "R/R",
        "implementation": "rl_kernel",
        "backend_id": STRICT_LINEAR_LOGP_BACKEND_ID,
        "call_count": 1,
        "execution_mode": "eager",
        "provenance": provenance,
    }
    errors = []
    _validate_strict_dense_record(
        record,
        module="logp",
        framework="vllm" if replicated else "megatron",
        label="test",
        errors=errors,
    )
    assert bool(errors) is (missing is not None)
