# SPDX-License-Identifier: Apache-2.0
"""Original captured DSV4 QNorm regression; no fake model or BF16 weight substitute."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from rl_engine.backends.rocm.norm.qnorm_fp8 import FusedQNorm

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_qnorm_quantizes_before_bf16_store():
    from aiter import QuantType, get_hip_quant

    if not os.getenv("RLK_DSV4_TRACE_ROOT") or not os.getenv("RLK_DSV4_CHECKPOINT"):
        pytest.skip("set original checkpoint and real-worker trace paths")
    trace = Path(os.environ["RLK_DSV4_TRACE_ROOT"])
    checkpoint = Path(os.environ["RLK_DSV4_CHECKPOINT"])
    root = trace / "generation-trace"
    entries = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines()]

    def load(name):
        e = next(e for e in entries if e["name"] == name and e["forward"] == 0)
        return torch.load(root / e["file"], weights_only=True)

    qkv = load("layer.0.attn.method._fused_wqa_wkv_gemm")
    expected_codes, expected_scales, _ = load("layer.0.attn.method._split_qkv_and_norm")
    if isinstance(qkv, (tuple, list)):
        qkv = qkv[0]
    x = qkv[:, :1024].contiguous().cuda().requires_grad_(True)
    with safe_open(
        str(next(checkpoint.glob("*.safetensors"))), framework="pt", device="cpu"
    ) as source:
        weight = source.get_tensor("layers.0.attn.q_norm.weight").cuda()
    # Construct a real ordinary RMSNorm with the original Parameter; the new
    # module preserves that Parameter rather than installing dense model weights.
    original = torch.nn.RMSNorm(1024, eps=1e-6, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        original.weight.copy_(weight)
    norm = FusedQNorm(original, 1e-6)
    y = norm(x)
    codes, scales = norm.quantized_input(y.detach())
    assert torch.equal(codes.cpu().view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scales.cpu().view(torch.uint8), expected_scales.view(torch.uint8))
    old_codes, old_scales = get_hip_quant(QuantType.per_1x128)(
        y, quant_dtype=torch.float8_e4m3fnuz, transpose_scale=False
    )
    assert not torch.equal(old_codes.view(torch.uint8), codes.view(torch.uint8))
    assert not torch.equal(old_scales.view(torch.uint8), scales.view(torch.uint8))
    y.float().square().sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all()) and bool(x.grad.ne(0).any())
    assert original.weight.grad is not None and bool(original.weight.grad.ne(0).any())
    for rows in (2, 17):
        for position in (0, rows - 1):
            batch = x.detach()[0:1].repeat(rows, 1)
            norm(batch)
            assert torch.equal(
                norm._activation[1][position].view(torch.uint8), codes[0].view(torch.uint8)
            )
            assert torch.equal(
                norm._activation[2][position].view(torch.uint8), scales[0].view(torch.uint8)
            )
