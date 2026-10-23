# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regressions for comparing equal VJPs across benchmark output dtypes."""

import pytest
import torch

from benchmarks import fused_logp_backward_evidence as evidence

pytestmark = pytest.mark.unit


def test_effective_upstream_gradient_is_shared_by_bf16_and_fp32_outputs():
    raw = torch.tensor([1.001, -0.12345, 3.14159, 0.0])
    common = evidence._effective_upstream_gradient(raw)

    assert common.dtype == torch.float32
    assert not torch.equal(common, raw)
    assert torch.equal(common, raw.bfloat16().float())
    for output_dtype in (torch.bfloat16, torch.float32):
        assert torch.equal(common.to(output_dtype).float(), common)


def _cpu_accuracy_inputs():
    logits = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]])
    labels = torch.tensor([0, 2])
    grad = torch.tensor([1.001, -0.12345])
    return logits, labels, grad


def _mixed_dtype_candidates():
    return {
        "fp32": (evidence._torch_logp, evidence.CHUNKED),
        "bf16": (
            lambda logits, labels: evidence._torch_logp(logits, labels).bfloat16(),
            evidence.CHUNKED,
        ),
    }


def test_accuracy_compares_mixed_output_dtypes_with_one_upstream(monkeypatch):
    logits, labels, raw = _cpu_accuracy_inputs()
    common = evidence._effective_upstream_gradient(raw)
    monkeypatch.setattr(evidence, "_inputs", lambda *args: (logits, labels, common))
    monkeypatch.setattr(evidence, "CANDIDATES", _mixed_dtype_candidates())

    result = evidence.accuracy(rows=2)

    assert result["bf16"]["output_dtype"] == "torch.bfloat16"
    assert result["fp32"]["output_dtype"] == "torch.float32"
    for metric in (
        "grad_max_abs",
        "grad_max_abs_over_absmax",
        "grad_correctly_rounded_fraction",
    ):
        assert result["bf16"][metric] == result["fp32"][metric]


def test_accuracy_rejects_upstream_changed_by_output_dtype(monkeypatch):
    monkeypatch.setattr(evidence, "_inputs", lambda *args: _cpu_accuracy_inputs())
    monkeypatch.setattr(evidence, "CANDIDATES", _mixed_dtype_candidates())

    with pytest.raises(AssertionError, match="output dtype changes the common upstream gradient"):
        evidence.accuracy(rows=2)
