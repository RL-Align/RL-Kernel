# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Regression coverage for real training/rollout provider comparisons."""

import sys

import pytest
import torch

from benchmarks import fused_logp_provider_evidence as evidence
from rl_engine.backends.extension import _C, _EXT_AVAILABLE


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_comparison_counts_raw_bytes_and_rows_on_noncontiguous_tensors(dtype):
    reference = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=dtype).T
    candidate = torch.tensor([[-0.0, 1.0], [2.5, -3.25]], dtype=dtype).T
    assert not reference.is_contiguous()
    assert not candidate.is_contiguous()

    result = evidence._comparison(candidate, reference)

    assert result["elements_checked"] == 4
    assert result["rows_checked"] == 2
    # +0 and -0 compare equal numerically, but provider bitwise claims include
    # their sign bit. Count changed elements once, regardless of changed bytes.
    assert result["elements_differing"] == 3
    assert result["rows_differing"] == 2
    assert result["max_abs"] == 6.25


@pytest.mark.unit
def test_comparison_equal_inputs_reports_all_checked_rows():
    values = torch.tensor([-2.0, -0.0, 1.25])

    result = evidence._comparison(values.clone(), values)

    assert result["elements_checked"] == 3
    assert result["rows_checked"] == 3
    assert result["elements_differing"] == result["rows_differing"] == 0
    assert result["max_abs"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_comparison_rejects_incomparable_boundaries(mismatch):
    reference = torch.ones(2, 3, dtype=torch.float32)
    candidate = reference.reshape(3, 2) if mismatch == "shape" else reference.bfloat16()

    with pytest.raises(ValueError):
        evidence._comparison(candidate, reference)


@pytest.fixture
def passing_provider_case():
    values = {"logp": torch.tensor([-2.0, -3.0]), "lse": torch.tensor([2.0, 3.0])}

    def pair(prefix):
        return evidence._pair_comparisons(values, values, prefix)

    return {
        "configuration": {"tp_world_size": 2},
        "comparisons": pair("vime_vs_vllm"),
        "explicit_active_mask": {"comparisons": pair("train_vs_infer")},
        "provider_active_compaction": {},
        "layouts": {
            "chunks_1": {
                "vime_train": pair("layout_vs_full"),
                "vllm_rollout": pair("layout_vs_full"),
                "integration_train_vs_rollout": pair("vime_vs_vllm"),
                "flash_train": pair("layout_vs_full"),
                "flash_rollout": pair("layout_vs_full"),
                "coordinated_flash_train_vs_rollout": pair("flash"),
            }
        },
        "flash_attn": {
            "status": "executed",
            "same_provider_train_vs_no_grad": pair("flash"),
            "replicated_tp1_train_vs_rollout_control": pair("flash"),
            "coordinated_train_vs_rollout": pair("flash"),
            "changed_implementation_vs_ws2": pair("flash_vs_ws2"),
        },
        "unsupported_requests": {"vime_top_p_replay": {"status": "rejected"}},
    }


@pytest.mark.unit
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_same_bit_nonfinite_results_fail_summary_verdict(
    nonfinite, passing_provider_case
):
    values = torch.tensor([-2.0, nonfinite])
    metric = evidence._comparison(values.clone(), values)

    summary = evidence._summarize_comparisons([metric])

    assert summary["all_bitwise_equal"] is True
    assert summary["all_finite"] is False
    assert summary["nonfinite_elements"] == 1
    assert summary["passes"] is False
    passing_provider_case["comparisons"]["vime_vs_vllm_logp"] = metric
    passing_provider_case["flash_attn"]["coordinated_train_vs_rollout"][
        "flash_logp"
    ] = metric
    verdict = evidence._summary([passing_provider_case])
    assert verdict["existing_ws2_provider_baseline"]["passes"] is False
    assert (
        verdict["flash_attn_candidate"][
            "coordinated_replacement_passes_sampled_contract"
        ]
        is False
    )


@pytest.mark.unit
def test_changed_implementation_drift_is_diagnostic_for_coordinated_replacement(
    passing_provider_case,
):
    passing_provider_case["flash_attn"]["changed_implementation_vs_ws2"][
        "flash_vs_ws2_logp"
    ] = evidence._comparison(torch.tensor([-2.0, -3.25]), torch.tensor([-2.0, -3.0]))

    verdict = evidence._summary([passing_provider_case])
    candidate = verdict["flash_attn_candidate"]

    assert verdict["existing_ws2_provider_baseline"]["passes"] is True
    assert (
        candidate["changed_implementation_vs_ws2_diagnostic"]["elements_differing"] == 1
    )
    assert candidate["changed_implementation_vs_ws2_diagnostic"]["passes"] is False
    for gate in (
        "same_provider_train_vs_no_grad",
        "replicated_tp1_train_vs_rollout_control",
        "fixed_topology_layout_invariance",
        "coordinated_train_vs_rollout",
    ):
        assert candidate[gate]["passes"] is True
    assert candidate["coordinated_replacement_passes_sampled_contract"] is True


@pytest.mark.unit
def test_coordinated_tp_train_rollout_drift_fails_even_when_layouts_pass(
    passing_provider_case,
):
    passing_provider_case["flash_attn"]["coordinated_train_vs_rollout"][
        "flash_logp"
    ] = evidence._comparison(torch.tensor([-2.0, -3.25]), torch.tensor([-2.0, -3.0]))

    verdict = evidence._summary([passing_provider_case])
    candidate = verdict["flash_attn_candidate"]

    assert verdict["existing_ws2_provider_baseline"]["passes"] is True
    assert candidate["same_provider_train_vs_no_grad"]["passes"] is True
    assert candidate["replicated_tp1_train_vs_rollout_control"]["passes"] is True
    assert candidate["fixed_topology_layout_invariance"]["passes"] is True
    assert candidate["coordinated_train_vs_rollout"]["elements_differing"] == 1
    assert candidate["coordinated_train_vs_rollout"]["passes"] is False
    assert candidate["coordinated_replacement_passes_sampled_contract"] is False


@pytest.mark.cuda_only
@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip is None
        and _EXT_AVAILABLE
        and hasattr(_C, "fused_logp_backward")
    ),
    reason="compiled CUDA fused-logp extension is required",
)
@pytest.mark.parametrize(
    ("dtype", "real_vocab", "temperature", "vocab_mask"),
    [
        (torch.bfloat16, None, 1.0, False),
        (torch.float32, 233, 0.7, True),
    ],
)
def test_real_vime_and_vllm_providers_compare_the_same_logit_boundary(
    monkeypatch, dtype, real_vocab, temperature, vocab_mask
):
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    pytest.importorskip(
        "vllm.v1.sample.sampler", reason="installed vLLM sampler is required"
    )
    previous_profiler = sys.getprofile()

    result = evidence.run_case(
        rows=5,
        vocab=256,
        seed=27,
        dtype=dtype,
        real_vocab=real_vocab,
        temperature=temperature,
        vocab_mask=vocab_mask,
        layouts=False,
    )

    assert sys.getprofile() is previous_profiler
    for provider in ("vime_train", "vllm_rollout"):
        provenance = result["providers"][provider]["provenance"]
        assert provenance["actual_backend"] == "pytorch-vocab-parallel-logp-ws2"
        assert provenance["fallback"] is False
        assert set(result["providers"][provider]["readback"]) >= {"logp", "lse"}
    for boundary in (
        "vime_vs_vllm_logp",
        "vime_vs_vllm_lse",
        "vime_train_vs_no_grad_logp",
        "vime_train_vs_no_grad_lse",
        "vllm_selected_vs_observed_logp",
    ):
        comparison = result["comparisons"][boundary]
        assert comparison["rows_checked"] == 5, boundary
        assert comparison["elements_checked"] == 5, boundary
        assert comparison["rows_differing"] == 0, boundary
        assert comparison["elements_differing"] == 0, boundary
