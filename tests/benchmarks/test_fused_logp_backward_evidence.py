# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regressions for comparing equal VJPs across benchmark output dtypes."""

import shutil
import subprocess

import pytest
import torch

from benchmarks.operators.logprob import fused_logp_backward_evidence as evidence

pytestmark = pytest.mark.unit


def test_effective_upstream_gradient_is_shared_by_bf16_and_fp32_outputs():
    """The BF16-quantized FP32 upstream gradient survives casts to BF16 and FP32 unchanged."""
    raw = torch.tensor([1.001, -0.12345, 3.14159, 0.0])
    common = evidence._effective_upstream_gradient(raw)

    assert common.dtype == torch.float32
    assert not torch.equal(common, raw)
    assert torch.equal(common, raw.bfloat16().float())
    for output_dtype in (torch.bfloat16, torch.float32):
        assert torch.equal(common.to(output_dtype).float(), common)


def _cpu_accuracy_inputs():
    """Return two-row CPU ``(logits, labels, grad)``; the FP32 ``grad`` is not BF16-exact."""
    logits = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.75, -1.0]])
    labels = torch.tensor([0, 2])
    grad = torch.tensor([1.001, -0.12345])
    return logits, labels, grad


def _mixed_dtype_candidates():
    """Return a ``CANDIDATES`` stand-in: ``_torch_logp`` with FP32 output and with BF16 output."""
    return {
        "fp32": (evidence._torch_logp, evidence.CHUNKED),
        "bf16": (
            lambda logits, labels: evidence._torch_logp(logits, labels).bfloat16(),
            evidence.CHUNKED,
        ),
    }


def test_accuracy_compares_mixed_output_dtypes_with_one_upstream(monkeypatch):
    """Given a BF16-exact upstream, BF16- and FP32-output candidates get equal gradient metrics."""
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
    """``accuracy`` raises ``AssertionError`` when a BF16 cast changes a non-BF16-exact upstream."""
    monkeypatch.setattr(evidence, "_inputs", lambda *args: _cpu_accuracy_inputs())
    monkeypatch.setattr(evidence, "CANDIDATES", _mixed_dtype_candidates())

    with pytest.raises(AssertionError, match="output dtype changes the common upstream gradient"):
        evidence.accuracy(rows=2)


def _flash_source(root, manifest_commit=None):
    """Create a minimal flash-attention tree whose ``cross_entropy_loss`` returns its input."""
    path = root / "flash_attn" / "ops" / "triton" / "cross_entropy.py"
    path.parent.mkdir(parents=True)
    path.write_text("def cross_entropy_loss(x, y):\n    return (x,)\n")
    if manifest_commit is not None:
        (root / "SOURCE.json").write_text(f'{{"commit": "{manifest_commit}"}}')
    return root


def _recorded_source(monkeypatch, source):
    """Return the provenance string ``--flash-attn-src`` records for ``source``."""
    monkeypatch.setattr(evidence, "CANDIDATES", dict(evidence.CANDIDATES))
    monkeypatch.setattr(evidence, "EXTERNAL_SOURCES", {})
    evidence._add_external_candidates(source)
    return evidence.EXTERNAL_SOURCES["flash-attn cross_entropy (verl)"]


def _init_repo(path):
    """Commit everything under ``path`` in a new git repository and return its short HEAD."""
    run = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run([*run, "-C", str(path), "add", "-A"], check=True)
    subprocess.run([*run, "-C", str(path), "commit", "-q", "-m", "x"], check=True)
    return evidence._git("rev-parse", "--short", "HEAD", cwd=path)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_flash_attn_provenance_ignores_enclosing_checkout(tmp_path, monkeypatch):
    """A vendored tree inside another repository reports its manifest, not the outer HEAD."""
    outer = tmp_path / "outer"
    source = _flash_source(outer / "vendored", manifest_commit="94e22c9")
    outer_head = _init_repo(outer)
    assert "flash-attention source 94e22c9," in _recorded_source(monkeypatch, source)
    (source / "SOURCE.json").unlink()
    recorded = _recorded_source(monkeypatch, source)
    assert "flash-attention source unknown," in recorded
    assert outer_head not in recorded


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_flash_attn_provenance_uses_own_checkout(tmp_path, monkeypatch):
    """A real clone with no manifest reports its own HEAD."""
    source = _flash_source(tmp_path / "flash-attention")
    head = _init_repo(source)
    assert f"flash-attention source {head}," in _recorded_source(monkeypatch, source)
