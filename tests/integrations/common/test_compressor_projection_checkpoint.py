# SPDX-License-Identifier: Apache-2.0
"""MI300X regression using original checkpoint weights and the failing input."""
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.attention.compressor_projection import compressor_projection


@pytest.fixture
def operands():
    capture = os.environ.get("DSV4_COMPRESSOR_CAPTURE")
    checkpoint = os.environ.get("DSV4_ORIGINAL_CHECKPOINT")
    if not capture or not checkpoint:
        pytest.skip("requires the original checkpoint and captured failing input on MI300X")
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"
    from examples.operators.dsv4_4l.trace_store import load_capture

    root = Path(capture)
    x = load_capture(root / "policy-sync/traces/actor-rank-0/00166.json")[0][:, 0].cuda()
    original = load_capture(root / "policy-sync/traces/rollout-rank-0/00104.json")[0].cuda()
    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        names = [
            f"layers.2.attn.{p}compressor.{w}.weight"
            for p in ("", "indexer.")
            for w in ("wkv", "wgate")
        ]
        weights = [f.get_tensor(name).cuda() for name in names]
    assert x.dtype == torch.bfloat16 and all(w.dtype == torch.bfloat16 for w in weights)
    return x, weights, original, names


def test_original_checkpoint_fusion_and_token_invariance(operands):
    x, weights, original, names = operands
    fused_weight = torch.cat(weights, 0)
    native_split = torch.cat([torch.mm(x, w.T, out_dtype=torch.float32) for w in weights[:2]], -1)
    native_fused = torch.mm(x, fused_weight.T, out_dtype=torch.float32)[:, : original.shape[1]]
    original_difference = int(
        (native_split.view(torch.int32) != native_fused.contiguous().view(torch.int32)).sum()
    )
    captured_difference = int(
        (
            native_fused.contiguous().view(torch.int32) != original.contiguous().view(torch.int32)
        ).sum()
    )
    assert original_difference > 0
    assert captured_difference == 0
    fixed = compressor_projection(x, fused_weight)
    split = torch.cat([compressor_projection(x, w) for w in weights], -1)
    assert fixed.dtype == torch.float32 and torch.isfinite(fixed).all()
    assert torch.equal(fixed.view(torch.int32), split.view(torch.int32))
    orders = [
        torch.arange(len(x), device=x.device),
        torch.arange(len(x) - 1, -1, -1, device=x.device),
        torch.tensor([115, 0, 161], device=x.device),
        torch.tensor([115], device=x.device),
    ]
    mismatches = []
    for order in orders:
        for _ in range(2):
            result = compressor_projection(x[order], fused_weight)
            mismatch = int(
                (result.view(torch.int32) != fixed[order].contiguous().view(torch.int32)).sum()
            )
            mismatches.append(mismatch)
            assert mismatch == 0
    record = dict(
        original_split_fused_mismatches=original_difference,
        original_fused_capture_mismatches=captured_difference,
        fixed_split_fused_mismatches=0,
        fixed_reordered_and_repeated_mismatches=mismatches,
        shape=list(x.shape),
        weight_names=names,
        original_dtype=str(x.dtype),
    )
    if os.environ.get("DSV4_COMPRESSOR_TEST_OUTPUT"):
        Path(os.environ["DSV4_COMPRESSOR_TEST_OUTPUT"]).write_text(json.dumps(record, indent=2))


def test_original_bf16_leaf_parameters_receive_finite_effective_vjp(operands):
    x, weights, _, _ = operands
    x = x[:3].clone().requires_grad_()
    weight = weights[0].clone().requires_grad_()
    original_weight = weight.detach().clone()
    output = compressor_projection(x, weight)
    coefficient = torch.linspace(-1, 1, output.numel(), device=x.device).reshape_as(output)
    (output * coefficient).sum().backward()
    for leaf in (x, weight):
        assert leaf.grad is not None and leaf.grad.dtype == torch.bfloat16
        assert torch.isfinite(leaf.grad).all() and torch.count_nonzero(leaf.grad) > 0
    assert torch.equal(weight.detach().view(torch.int16), original_weight.view(torch.int16))
