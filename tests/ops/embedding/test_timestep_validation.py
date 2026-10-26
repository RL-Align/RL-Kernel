# SPDX-License-Identifier: Apache-2.0
"""Regression checks for timestep evidence collection and supported boundaries."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rl_engine.ops.embedding.timestep_embed_mlp import TimestepEmbedMLPOp
from tools.validation.operators.validate_timestep_official import (
    eager_batched,
    inputs,
    kernel_families_observed,
    prepared_eager_batched,
    require_requested_backend,
    source_revision,
)


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_profile_requires_each_backend_kernel_family(backend):
    cuda = [
        "(anonymous namespace)::timestep_" + n + "(float*)"
        for n in ("mm_warp", "embedding", "silu", "dt")
    ]
    triton = ["_timestep_" + n for n in ("mm_tree", "embedding", "silu", "dt")]
    correct, wrong = (cuda, triton) if backend == "cuda" else (triton, cuda)
    assert kernel_families_observed(correct, backend)
    assert not kernel_families_observed(wrong, backend)
    assert not kernel_families_observed(correct[:-1], backend)
    assert not kernel_families_observed(["copy/path/" + n for n in correct], backend)


def test_fallback_rejected_in_optimized_python():
    root = Path(__file__).resolve().parents[3]
    code = (
        "from types import SimpleNamespace; "
        "from tools.validation.operators.validate_timestep_official "
        "import require_requested_backend; "
        "require_requested_backend(SimpleNamespace(actual_backend='pytorch', "
        "fallback_reason='unavailable'), 'cuda')"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "requested backend did not execute without fallback" in result.stderr
    require_requested_backend(SimpleNamespace(actual_backend="cuda", fallback_reason=None), "cuda")


def test_report_records_current_checkout():
    revision = source_revision()
    assert (
        revision["head_sha"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    )
    assert isinstance(revision["tracked_dirty"], bool)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_baseline_reuses_frequency_without_changing_values_or_gradients(dtype):
    values = {k: v.requires_grad_() for k, v in inputs(3, 17, dtype, "cpu", 386).items()}
    baseline = prepared_eager_batched("cpu")
    import math

    frequency = torch.exp(-math.log(10000.0) * torch.arange(128).float() / 128)
    expected = eager_batched(**values, frequency=frequency)
    with patch("torch.arange", side_effect=AssertionError("table rebuilt inside timing")):
        actual = baseline(**values)
        again = baseline(**values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)
    upstream = torch.ones_like(expected)
    for a, b in zip(
        torch.autograd.grad(actual, tuple(values.values()), upstream),
        torch.autograd.grad(expected, tuple(values.values()), upstream),
        strict=True,
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch", [0, 3])
def test_integer_timesteps_and_physical_order(dtype, batch):
    values = inputs(batch, 17, torch.float32, "cpu", 386)
    values["timestep"] = torch.arange(batch, dtype=dtype)
    parameters = [values[k].requires_grad_() for k in ("weight1", "bias1", "weight2", "bias2")]
    op = TimestepEmbedMLPOp("pytorch")
    expected = op(**values)
    ids = torch.tensor([30, 10, 20], dtype=torch.int64)[:batch]
    actual = op(**values, sample_ids=ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual, parameters, torch.ones_like(actual))
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_rejects_unsafe_indexing_before_allocation():
    from rl_engine.backends.cuda.embedding.timestep_embed_mlp import _extension

    extension = _extension()
    scalar = torch.ones(1, device="cuda")
    with pytest.raises(RuntimeError, match="matrix output exceeds CUDA int32"):
        extension.mm(scalar.expand(46341, 1), scalar.expand(1, 46341))
    with pytest.raises(RuntimeError, match="matrix reduction exceeds CUDA int32"):
        extension.mm(scalar.expand(1, 2**31 - 1), scalar.expand(2**31 - 1, 1))
    # Zero-storage dimensions also must not narrow into signed kernel arguments.
    with pytest.raises(RuntimeError, match="dimension exceeds CUDA int32"):
        extension.mm(torch.empty((0, 2**31), device="cuda"), torch.empty((2**31, 0), device="cuda"))
    with pytest.raises(RuntimeError, match="tensor exceeds CUDA int32"):
        extension.unary(scalar.expand(2**31), False)
    with pytest.raises(RuntimeError, match="embedding output exceeds CUDA int32"):
        extension.embedding(torch.empty(2**23, device="cuda"), torch.ones(128, device="cuda"))
    with pytest.raises(RuntimeError, match="tensor exceeds CUDA int32"):
        extension.timestep_grad(
            scalar.expand(2**23, 256), scalar.expand(2**23, 256), torch.ones(128, device="cuda")
        )
