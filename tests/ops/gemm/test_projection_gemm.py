# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #642 Task 12: forward accuracy and raw-bit batch/chunk qualification.

Dimensions follow Qwen/Qwen3.6-35B-A3B config at
345d467b117d1fd5eda9b879a5988de9ae5f47a7 and Transformers v5.13.1
models/qwen3_5_moe/modeling_qwen3_5_moe.py (attention, GDN, MLP,
experts, router and LM head). This is synthetic operator coverage, not model
integration or a replacement for Task 1's manifest. Grouped experts are excluded.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from rl_engine.contracts.numerical import load_contract

DEVICE = os.getenv("TASK12_DEVICE", "cuda")
if DEVICE == "musa":
    import torch_musa  # noqa: F401

DEVICE_API = getattr(torch, DEVICE)
pytestmark = pytest.mark.skipif(not DEVICE_API.is_available(), reason=f"requires real {DEVICE}")

# Unique K,N pairs. Equivalent projections share one parameter rather than
# implying distinct engine packing has been qualified.
H, Q, KV, D = 2048, 16, 2, 256
KEY, VALUE, EXPERT = 16 * 128, 32 * 128, 512
PROJECTIONS = (
    pytest.param(H, Q * D * 2, id="gated-q-and-gdn-qkv"),
    pytest.param(H, KV * D, id="kv-and-shared-gate-up-separate"),
    pytest.param(Q * D, H, id="attention-and-gdn-output"),
    pytest.param(H, VALUE, id="gdn-z"),
    pytest.param(H, 32, id="gdn-a-b"),
    pytest.param(H, 2 * EXPERT, id="expert-gate-up"),
    pytest.param(EXPERT, H, id="expert-shared-down"),
    pytest.param(H, 256, id="router-logits"),
    pytest.param(H, 1, id="shared-expert-gate"),
    pytest.param(H, 248320, id="lm-head"),
)
assert Q * D * 2 == KEY * 2 + VALUE


@pytest.fixture(params=("native", "triton"))
def backend(request):
    if request.param == "triton":
        pytest.importorskip("triton")
        from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

        return request.param, TritonDetGemmOp()
    if DEVICE == "musa":
        from rl_engine.backends.musa.gemm.det_gemm import MusaDetGemmOp

        return request.param, MusaDetGemmOp()
    from rl_engine.backends.cuda.gemm.det_gemm import DetGemmOp, det_gemm_backend

    assert det_gemm_backend() == "sm90", "pin RL_KERNEL_DET_GEMM_BACKEND=sm90"
    return request.param, DetGemmOp()


def _linear(backend, a, weight):
    name, op = backend
    if name == "native":
        return op.linear(a, weight)
    return op(a, weight.t())


def _fp64_reference(a, b):
    # MUSA qualification must not depend on device FP64 GEMM availability.
    device = "cpu" if DEVICE == "musa" else a.device
    return a.to(device=device, dtype=torch.float64) @ b.to(device=device, dtype=torch.float64)


def _raw_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    bits = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    lhs = actual.contiguous().view(bits)
    rhs = expected.contiguous().view(bits)
    different = lhs != rhs
    if different.any():
        index = different.nonzero()[0].tolist()
        at = tuple(index)
        pytest.fail(
            f"first raw-bit difference at {index}: actual={lhs[at].item()} "
            f"expected={rhs[at].item()}, dtype={actual.dtype}"
        )


def _save_failure(a, weight, context):
    directory = Path(os.getenv("TASK12_ARTIFACT_DIR", "artifacts/task12"))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "failure.pt").exists():
        return  # Keep the first failing input rather than replacing its evidence.
    torch.save({"a": a.cpu(), "weight": weight.cpu(), "context": context}, directory / "failure.pt")
    (directory / "failure.json").write_text(json.dumps(context, indent=2) + "\n")


@pytest.mark.parametrize("k,n", PROJECTIONS)
@torch.no_grad()
def test_projection_accuracy_and_chunk_bits(backend, k, n, record_property):
    torch.manual_seed(642)
    # Unit RMS hidden activations and model initializer_range=0.02 weights.
    a = torch.randn(129, k, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16) * 0.02
    full = _linear(backend, a, weight)
    assert full.shape == (129, n) and full.dtype == torch.bfloat16
    # Independent mathematical oracle: no tree implementation or downcast.
    reference = _fp64_reference(a, weight.t())
    actual_fp64 = full.to(device=reference.device, dtype=torch.float64)
    error = (actual_fp64 - reference).abs()
    record_property("max_abs_error_vs_fp64", error.max().item())
    limits = load_contract()["accuracy"]["default"]["reduction"]["bfloat16"]
    try:
        torch.testing.assert_close(actual_fp64, reference, **limits)
        for row in (0, 64, 128):
            _raw_equal(_linear(backend, a[row : row + 1], weight), full[row : row + 1])
        for chunk in (7, 31, 64, 127):
            parts = [_linear(backend, part, weight) for part in a.split(chunk)]
            _raw_equal(torch.cat(parts), full)
    except (AssertionError, pytest.fail.Exception) as exc:
        _save_failure(a, weight, {"backend": backend[0], "shape": [129, k, n], "error": str(exc)})
        raise


@pytest.mark.parametrize("m", (1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 1024, 4096))
@torch.no_grad()
def test_projection_row_placement_and_batch_boundaries(backend, m):
    torch.manual_seed(643)
    row = torch.randn(1, H, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(512, H, device=DEVICE, dtype=torch.bfloat16) * 0.02
    expected = _linear(backend, row, weight)
    a = torch.randn(m, H, device=DEVICE, dtype=torch.bfloat16)
    positions = sorted({0, m // 2, m - 1})
    a[positions] = row
    full = _linear(backend, a, weight)
    for position in positions:
        _raw_equal(full[position : position + 1], expected)
    permutation = torch.arange(m - 1, -1, -1, device=DEVICE)
    _raw_equal(_linear(backend, a[permutation], weight)[permutation], full)
    padded = torch.cat((a, torch.zeros(17, H, device=DEVICE, dtype=a.dtype)))
    _raw_equal(_linear(backend, padded, weight)[:m], full)


@pytest.mark.parametrize("k", (1, 31, 32, 33, 63, 64, 65, 96, 512, 2048, 4096))
@pytest.mark.parametrize("n", (1, 32, 64, 65))
@torch.no_grad()
def test_projection_layout_and_k_boundaries(backend, k, n):
    torch.manual_seed(644)
    # Noncontiguous and contiguous-but-unaligned operands exercise distinct
    # materialization paths, including TMA's base alignment requirement.
    a = torch.randn(7, k * 2, device=DEVICE, dtype=torch.bfloat16)[:, ::2]
    storage = torch.randn(n * k + 1, device=DEVICE, dtype=torch.bfloat16)
    weight = storage[1:].reshape(n, k)
    expected = _linear(backend, a.contiguous(), weight.clone())
    _raw_equal(_linear(backend, a, weight), expected)
    _raw_equal(backend[1](a, weight.t()), expected)
    _raw_equal(torch.cat([_linear(backend, part, weight) for part in a.split(1)]), expected)


@pytest.mark.parametrize("backend", ("native",), indirect=True)
@pytest.mark.parametrize("k,n", ((64, 64), (65, 33), *PROJECTIONS))
@torch.no_grad()
def test_native_fp32_is_tree_output_cast(backend, k, n):
    torch.manual_seed(645)
    a = torch.randn(129, k, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
    expected = backend[1](a, b).float()
    actual = backend[1].forward_fp32(a, b)
    assert actual.dtype == torch.float32
    _raw_equal(actual, expected)
    for chunk in (1, 7, 64, 127):
        _raw_equal(torch.cat([backend[1].forward_fp32(part, b) for part in a.split(chunk)]), actual)


@torch.no_grad()
def test_canonical_k_rounding_is_observable(backend):
    # Each 32-element leaf is rounded before the midpoint merge: 257 -> 256,
    # -255 -> -255, giving 1. A full-precision dot gives 2 instead.
    a = torch.ones(3, 64, device=DEVICE, dtype=torch.bfloat16)
    b = torch.zeros(64, 64, device=DEVICE, dtype=torch.bfloat16)
    b[0], b[1], b[32], b[33] = 256, 1, -256, 1
    actual = backend[1](a, b)
    _raw_equal(actual, torch.ones_like(actual))
    assert torch.equal(a.double() @ b.double(), torch.full((3, 64), 2.0, device=DEVICE))


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@torch.no_grad()
def test_fp32_accumulation_profile(backend, dtype):
    torch.manual_seed(646)
    a = torch.randn(65, 512, device=DEVICE, dtype=dtype)
    b = torch.randn(512, 32, device=DEVICE, dtype=dtype) * 0.02
    actual = backend[1].forward_accum_fp32(a, b)
    assert actual.dtype == torch.float32
    limits = load_contract()["accuracy"]["default"]["reduction"]["float32"]
    reference = _fp64_reference(a, b)
    torch.testing.assert_close(
        actual.to(device=reference.device, dtype=torch.float64), reference, **limits
    )
    for chunk in (1, 7, 31, 64):
        _raw_equal(
            torch.cat([backend[1].forward_accum_fp32(part, b) for part in a.split(chunk)]), actual
        )


@torch.no_grad()
def test_fp16_is_rejected(backend):
    a = torch.ones(2, 32, device=DEVICE, dtype=torch.float16)
    b = torch.ones(32, 64, device=DEVICE, dtype=torch.float16)
    with pytest.raises((AssertionError, TypeError, RuntimeError), match="BF16|bf16"):
        backend[1](a, b)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_raw_comparison_detects_signed_zero(dtype):
    with pytest.raises(pytest.fail.Exception, match="first raw-bit difference"):
        _raw_equal(torch.tensor([0.0], dtype=dtype), torch.tensor([-0.0], dtype=dtype))
