# SPDX-License-Identifier: Apache-2.0
"""Original checkpoint FP8 bytes/scales, production AITER forward and input VJP."""

import os
from pathlib import Path

import pytest
import torch

from rl_engine.backends.rocm.gemm.checkpoint_fp8 import CheckpointFP8Linear

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_original_projection_forward_input_gradient_and_bytes():
    directory = os.getenv("RLK_DSV4_CHECKPOINT")
    if not directory:
        pytest.skip("set RLK_DSV4_CHECKPOINT to the original checkpoint")
    from safetensors import safe_open

    source = safe_open(str(next(Path(directory).glob("*.safetensors"))), framework="pt")
    weight = source.get_tensor("layers.0.attn.wq_a.weight")
    scale = source.get_tensor("layers.0.attn.wq_a.scale")
    projection = CheckpointFP8Linear(weight.cuda(), scale.cuda())
    torch.manual_seed(42)
    x = torch.randn(2, 4096, device="cuda").bfloat16().requires_grad_()
    actual = projection(x)[0]
    repeated = projection(x)[0]
    assert torch.equal(actual.view(torch.uint8), repeated.view(torch.uint8))
    assert torch.equal(projection.codes.cpu(), weight.view(torch.uint8))
    assert torch.equal(projection.scales.cpu(), scale.view(torch.uint8))
    # AITER owns activation quantization. Condition the independent CPU FP32
    # GEMM oracle on its actual quantized operands; do not substitute a codec
    # whose reciprocal rounding differs at FP8 ties.
    from aiter import QuantType, get_hip_quant

    q, scales = get_hip_quant(QuantType.per_1x128)(
        x.detach(), quant_dtype=torch.float8_e4m3fnuz, transpose_scale=False
    )
    a = (q.cpu().float().reshape(2, 32, 128) * scales.cpu()[..., None]).reshape(2, 4096)
    w = weight.float() * scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
    torch.testing.assert_close(actual.cpu().float(), a @ w.t(), atol=0.003, rtol=0.01)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    torch.testing.assert_close(
        x.grad.cpu().float(), grad.cpu().float() @ w.bfloat16().float(), atol=0.003, rtol=0.01
    )
    assert torch.isfinite(x.grad).all() and bool(x.grad.count_nonzero())
