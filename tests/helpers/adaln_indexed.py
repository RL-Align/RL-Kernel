# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Test-only indexed AdaLN fixtures and comparisons shared by L1/L2/L3."""

import os

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.norm.adaln_indexed import IndexedAdaLNFixedReference
from rl_engine.validation.operators.operator_specs import GtestAdaLNReference


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


def small_case(device):
    generator = torch.Generator().manual_seed(386)
    # Large positive eps keeps H2 dx sensitive; samples and banks have distinct tags.
    x = (torch.randn(2, 8, 2, generator=generator) * 2 + 0.3).to(device).requires_grad_()
    m = (torch.randn(4, 6, generator=generator) * 0.5).to(device).requires_grad_()
    index = torch.tensor(
        [[0, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0, 1]], dtype=torch.int32, device=device
    )
    dy = torch.randn(2, 8, 2, generator=generator).to(device)
    dg = torch.randn(2, 8, 2, generator=generator).to(device)
    return x, m, index, dy, dg


def check_small_case(op, device, trace=None):
    x, m, index, dy, dg = small_case(device)
    y, gate = op(x, m, modulate_index=index, eps=7.0)
    if trace is not None:
        # Check routing before numerical comparisons: a correct fallback is still red.
        assert trace["actual_backend"] == "cuda"
        assert trace["arithmetic_profile"] == "adaln-indexed-fp32-tree-v1"
        assert trace["kernel_fingerprint"] == "adaln_indexed_cuda_v1"
        assert not trace["fallback"]
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    for reference in (IndexedAdaLNFixedReference(), GtestAdaLNReference().forward_fp32):
        rx = x.detach().cpu().requires_grad_()
        rm = m.detach().cpu().requires_grad_()
        ry, rg = reference(rx, rm, modulate_index=index.cpu(), eps=7.0)
        rdx, rdm = torch.autograd.grad((ry, rg), (rx, rm), (dy.cpu(), dg.cpu()))
        assert torch.isfinite(rdx).all() and torch.count_nonzero(rdx) == rdx.numel()
        for actual, expected, judgment in (
            (y, ry, "forward_accuracy"),
            (gate, rg, "forward_accuracy"),
            (dx, rdx, "gradient_accuracy"),
            (dm, rdm, "gradient_accuracy"),
        ):
            assert actual.dtype == expected.dtype == torch.float32
            if isinstance(reference, IndexedAdaLNFixedReference):
                assert_bytes(actual, expected)
            else:
                tolerance = resolve_tolerance(
                    load_contract(), judgment=judgment, op_class="reduction", dtype=actual.dtype
                )
                torch.testing.assert_close(
                    actual.cpu(), expected, atol=tolerance.atol, rtol=tolerance.rtol
                )


def require_cuda():
    if torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA-only test; ROCm is outside this profile")
    if not torch.cuda.is_available():
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            pytest.fail("NVIDIA CUDA is required for indexed acceptance")
        pytest.skip("NVIDIA CUDA required; not an indexed GPU PASS")
    from rl_engine.backends.extension import _C

    # Once CUDA is available, missing/old symbols must fail rather than hide a red test.
    assert hasattr(_C, "adaln_indexed_forward"), "fresh indexed CUDA extension required"
    assert hasattr(_C, "adaln_indexed_backward"), "fresh indexed CUDA extension required"
