# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Four-judgment comparison and training orchestration for H3 projections."""

from __future__ import annotations

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
# Preserve historical imports without duplicating reference state or arithmetic.
from rl_engine.reference.gemm.h3_projections import fp32_math, reference_projection  # noqa: F401


def raw_equal(lhs, rhs):
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return False
    return torch.equal(
        lhs.detach().contiguous().reshape(-1).view(torch.uint8),
        rhs.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def comparison(lhs, rhs, judgment, *, contract=None):
    contract = load_contract() if contract is None else contract
    spec = resolve_tolerance(
        contract,
        judgment=judgment,
        op_class="reduction",
        dtype="bfloat16",
        arch_key="sm90",
        backend_profile="cuda_bf16",
    )
    result = {"judgment": judgment, "tolerance": spec.to_dict(), "passed": False}
    if lhs.shape != rhs.shape:
        result["error"] = "shape mismatch"
        return result
    # NaN/Inf hard fail; signed-zero differences count as bitwise mismatches.
    if not torch.isfinite(lhs).all().item() or not torch.isfinite(rhs).all().item():
        result["error"] = "nonfinite tensor"
        return result
    if spec.mode == "bitwise":
        result["passed"] = raw_equal(lhs, rhs)
        if lhs.dtype == rhs.dtype:
            bits_l = lhs.contiguous().reshape(-1).view(torch.uint8)
            bits_r = rhs.contiguous().reshape(-1).view(torch.uint8)
            result["mismatched_bytes"] = sum(
                int((bits_l[lo : lo + (1 << 20)] != bits_r[lo : lo + (1 << 20)]).sum().item())
                for lo in range(0, bits_l.numel(), 1 << 20)
            )
        else:
            result["error"] = "dtype mismatch"
        return result
    # Chunk comparisons to avoid several full-size FP32 dW temporaries.
    left, right = lhs.reshape(-1), rhs.reshape(-1)
    max_abs = max_ratio = 0.0
    failed = 0
    for lo in range(0, left.numel(), 1 << 20):
        l32 = left[lo : lo + (1 << 20)].float()
        r32 = right[lo : lo + (1 << 20)].float()
        error = (l32 - r32).abs()
        limit = spec.atol + spec.rtol * r32.abs()
        max_abs = max(max_abs, error.max().item())
        max_ratio = max(max_ratio, (error / limit).max().item())
        failed += int((error > limit).sum().item())
    result.update(
        max_abs_error=max_abs,
        max_error_over_tolerance=max_ratio,
        failing_elements=failed,
        passed=failed == 0,
    )
    return result


def logical_keys(rows, device):
    tokens = torch.arange(rows, device=device, dtype=torch.int64)
    return torch.stack((tokens // 257, tokens % 257), dim=1)


def training_projection(op, x, weight, grad, keys, *, chunk_size=None):
    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    traces = []
    with canonical_backward_session() as session:
        if chunk_size is None:
            output = op(x, weight, logical_keys=keys)
            traces.append(op.last_execution)
        else:
            parts = []
            for lo in range(0, x.shape[0], chunk_size):
                parts.append(
                    op(
                        x[lo : lo + chunk_size],
                        weight,
                        logical_keys=keys[lo : lo + chunk_size],
                    )
                )
                traces.append(op.last_execution)
            output = torch.cat(parts)
        output.backward(grad)
        session.validate_complete()
    return (output.detach(), x.grad.detach(), weight.grad.detach()), traces
