# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Keep providers with incompatible weight semantics out of norm evidence."""

import sys
import types

import pytest
import torch

from scripts import qwen3_next_norm_reuse_check as reuse


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("uses_effective_weight", [False, True])
def test_megatron_zero_centered_admission(monkeypatch, dtype, uses_effective_weight):
    """Reject the 52fbcbc forward behavior while admitting a corrected provider."""
    module = types.ModuleType("megatron.core.transformer.custom_layers.batch_invariant_kernels")

    class BatchInvariantRMSNormFn:
        @staticmethod
        def apply(x, weight, eps, zero_centered_gamma):
            weight_eff = weight + 1.0 if zero_centered_gamma else weight
            scale = weight_eff if uses_effective_weight else weight
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            return (normalized * scale.float()).to(x.dtype)

    module.BatchInvariantRMSNormFn = BatchInvariantRMSNormFn
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(reuse, "DEV", "cpu")
    monkeypatch.setattr(reuse, "DT", dtype)
    factory = dict(reuse._c1_candidates(32))["megatron"]
    monkeypatch.setattr(reuse, "_c1_candidates", lambda hidden: [("megatron", factory)])

    candidate = reuse.candidates("qwen3_next_rms_norm")["megatron"]
    if not uses_effective_weight:
        assert "failed the zero-weight probe" in candidate["unavailable"]
        assert "fn" not in candidate
        return

    assert candidate["backward"] == "full"
    x = torch.ones(2, 32, dtype=dtype, requires_grad=True)
    weight = torch.zeros(32, dtype=dtype, requires_grad=True)
    output = candidate["fn"](x, weight)
    assert torch.all(output != 0)
    output.sum().backward()
    assert x.grad is not None
    assert weight.grad is not None


@pytest.mark.parametrize("op", ["qwen3_next_rms_norm", "rms_norm_gated"])
def test_required_candidate_failure_writes_no_report(monkeypatch, tmp_path, op):
    """A missing primary extension must fail before creating report artifacts."""

    def unavailable():
        raise RuntimeError("CUDA extension needs rebuilding")

    factory_name = "_c1_candidates" if op == "qwen3_next_rms_norm" else "_gated_candidates"
    monkeypatch.setattr(reuse, factory_name, lambda hidden: [("rl_kernel", unavailable)])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(reuse, "plot", lambda *args: pytest.fail("must not plot a failed run"))
    report = tmp_path / "evidence" / "report.json"
    monkeypatch.setattr(sys, "argv", ["reuse", "--op", op, "--out", str(report)])

    with pytest.raises(RuntimeError, match="Required rl_kernel candidate is unavailable"):
        reuse.main()

    assert not report.parent.exists()


@pytest.mark.parametrize("op", ["qwen3_next_rms_norm", "rms_norm_gated"])
def test_optional_candidate_failure_is_reported(monkeypatch, op):
    """Optional dependencies may be absent when the primary candidate is usable."""

    def primary():
        return "rl-kernel", lambda x, w: x, "full"

    def optional():
        raise ImportError("optional provider not installed")

    factory_name = "_c1_candidates" if op == "qwen3_next_rms_norm" else "_gated_candidates"
    monkeypatch.setattr(
        reuse, factory_name, lambda hidden: [("rl_kernel", primary), ("optional", optional)]
    )
    found = reuse.candidates(op)
    assert callable(found["rl_kernel"]["fn"])
    assert found["optional"] == {"unavailable": "ImportError: optional provider not installed"}


@pytest.mark.parametrize("op", ["qwen3_next_rms_norm", "rms_norm_gated"])
@pytest.mark.parametrize("defect", [None, "forward", "gradient"])
def test_full_batch_sub_batches_visit_previously_skipped_rows(monkeypatch, op, defect):
    """Catch a singleton-only defect at an odd row missed by the old stride."""
    monkeypatch.setattr(reuse, "DEV", "cpu")
    monkeypatch.setattr(reuse, "DT", torch.float32)
    monkeypatch.setitem(reuse.SHAPES, op, (1, (8,), 1025, (1, 7, 64), 1025))

    def inputs(op, seed, n):
        tensors = {"x": torch.arange(n, dtype=torch.float32).reshape(n, 1), "w": torch.ones(1)}
        if op == "rms_norm_gated":
            tensors["z"] = torch.ones(n, 1)
        return tensors

    monkeypatch.setattr(reuse, "_inputs", inputs)

    def provider(x, w, z=None):
        output = x * w
        if z is not None:
            output = output * z
        if len(x) == 1:
            affected = (x == 1001).float()
            if defect == "forward":
                output = output + affected
            elif defect == "gradient":
                output = output + (x - x.detach()) * affected
        return output

    result = reuse.batch_invariance(op, {"fn": provider, "backward": "x-only"}, quick=False)
    check = result["full_vs_sub_batches"]
    assert check["coverage"] == "every_row_per_sub_batch_size"
    assert check["sub_batches"] == 2 * (1025 + 147 + 17)
    assert check["fwd_differ"] == (2 if defect == "forward" else 0)
    assert check["rowgrad_differ"] == (2 if defect == "gradient" else 0)
    assert result["batch_invariant"] is (defect is None)
