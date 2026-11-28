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
        x.detach(),
        quant_dtype=(
            torch.float8_e4m3fnuz
            if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0]
            == "gfx942"
            else torch.float8_e4m3fn
        ),
        transpose_scale=False,
    )
    a = (q.cpu().float().reshape(2, 32, 128) * scales.cpu()[..., None]).reshape(2, 4096)
    w = weight.float() * scale.float().repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    torch.testing.assert_close(actual.cpu().float(), a @ w.t(), atol=0.003, rtol=0.01)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    torch.testing.assert_close(
        x.grad.cpu().float(),
        grad.cpu().float() @ w.bfloat16().float(),
        atol=0.003,
        rtol=0.01,
    )
    assert torch.isfinite(x.grad).all() and bool(x.grad.count_nonzero())


def test_preserved_long_q_projection_matches_actor_across_batch_rows():
    """Original failed R1: 3899 real prompt/response rows, not synthetic input."""
    directory = os.getenv("RLK_DSV4_CHECKPOINT")
    trace_directory = os.getenv("RLK_DSV4_LONG_Q_TRACES")
    if not directory or not trace_directory:
        pytest.skip("set checkpoint and preserved long-Q trace paths")
    import json
    from safetensors import safe_open
    from examples.operators.dsv4_4l.trace_store import load_capture
    from rl_engine.backends.rocm.gemm.checkpoint_fp8 import original_block_fp8_gemm

    def capture(role, name):
        path = Path(trace_directory) / f"{role}-rank-0-failure-math-R1-b1"
        entry = next(
            e
            for e in json.loads((path / "manifest.json").read_text())
            if e["name"] == name
        )
        return load_capture(path / entry["file"])

    q, scale = capture("rollout", "layer.0.attn.method._wq_b_proj.input")["args"]
    expected = capture("actor", "layer.0.attn.linear_q_up_proj.output")[0][:, 0]
    with safe_open(
        str(next(Path(directory).glob("*.safetensors"))), framework="pt"
    ) as source:
        codes = source.get_tensor("layers.0.attn.wq_b.weight").view(torch.uint8).cuda()
        scales = source.get_tensor("layers.0.attn.wq_b.scale").view(torch.uint8).cuda()
    q, scale = q.cuda(), scale.cuda()
    assert q.shape == (3899, 1024)
    for rows in [1, 16, 1607, 3899]:
        actual = original_block_fp8_gemm(q[:rows], scale[:rows], codes, scales).cpu()
        assert torch.equal(
            actual.view(torch.uint8), expected[:rows].contiguous().view(torch.uint8)
        )
