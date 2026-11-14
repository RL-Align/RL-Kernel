# SPDX-License-Identifier: Apache-2.0
"""GPU regression for the packed training provider, including empty experts."""

import os
from pathlib import Path

import pytest
import torch

from rl_engine.backends.rocm.moe.packed_mx import RocmPackedMXBackend
from rl_engine.contracts.operators.expert_training import ExpertTrainingConfig, PackedExpertWeight
from rl_engine.reference.moe.expert_training import ReferenceExpertTrainingBackend, decode_weights

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def test_quantization_grouped_projection_and_input_vjp():
    torch.manual_seed(43)
    base = RocmPackedMXBackend()
    base.validate(ExpertTrainingConfig(64, 32, 3, mode="base_only"), torch.device("cuda:0"))
    x = torch.randn(5, 64).bfloat16()
    cpu_q = ReferenceExpertTrainingBackend().quantize(x)
    q = base.quantize(x.cuda())
    assert torch.equal(q.codes.cpu(), cpu_q.codes)
    assert torch.equal(q.scales.cpu(), cpu_q.scales)
    codes = torch.randint(0, 256, (3, 48, 32), dtype=torch.uint8)
    scales = torch.randint(121, 125, (3, 48, 2), dtype=torch.uint8)
    cpu_w = PackedExpertWeight(codes, scales, (3, 48, 64))
    w = PackedExpertWeight(codes.cuda(), scales.cuda(), cpu_w.shape)
    offsets = torch.tensor([0, 2, 2, 5], device="cuda", dtype=torch.int64)
    we, ws = decode_weights(cpu_w)
    wd = (we.reshape(3, 48, 2, 32) * ws[..., None]).reshape(3, 48, 64)
    ae = cpu_q.codes.view(torch.float8_e4m3fn).float().reshape(5, 2, 32)
    sa = torch.ldexp(torch.ones_like(cpu_q.scales, dtype=torch.float32), cpu_q.scales.int() - 127)
    a = (ae * sa[..., None]).reshape(5, 64)
    expected = torch.cat((a[:2] @ wd[0].t(), a[2:] @ wd[2].t()))
    y = base.forward(q, w, offsets)
    torch.testing.assert_close(y.cpu(), expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(y.view(torch.uint8), base.forward(q, w, offsets).view(torch.uint8))
    dy = torch.randn(5, 48).bfloat16()
    dx = base.backward(dy.cuda(), w, offsets)
    expected_dx = torch.cat(
        (dy[:2].float() @ wd[0].bfloat16().float(), dy[2:].float() @ wd[2].bfloat16().float())
    )
    torch.testing.assert_close(dx.cpu(), expected_dx, atol=2e-6, rtol=2e-5)
    assert torch.equal(dx.view(torch.uint8), base.backward(dy.cuda(), w, offsets).view(torch.uint8))
    empty = base.quantize(x[:0].cuda())
    assert base.forward(empty, w, offsets.new_zeros(4)).shape == (0, 48)
    with pytest.raises(ValueError, match="offsets"):
        base.forward(q, w, offsets.new_tensor([0, 3, 2, 5]))
    with pytest.raises(ValueError, match="finite"):
        base.quantize(torch.full((1, 64), float("nan"), device="cuda", dtype=torch.bfloat16))


def test_original_checkpoint_expert_forward_backward():
    checkpoint = os.getenv("RLK_DSV4_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set RLK_DSV4_CHECKPOINT to the original four-layer checkpoint")
    from safetensors import safe_open

    from rl_engine.backends.rocm.moe.expert_training import RocmExpertTrainingBackend
    from rl_engine.ops.moe.expert_training import routed_expert_training

    source = safe_open(str(next(Path(checkpoint).glob("*.safetensors"))), framework="pt")

    def packed(n):
        prefix = "layers.0.ffn.experts.0." + n
        c = source.get_tensor(prefix + ".weight").view(torch.uint8).unsqueeze(0).cuda()
        s = source.get_tensor(prefix + ".scale").view(torch.uint8).unsqueeze(0).cuda()
        return PackedExpertWeight(c, s, (1, c.shape[1], c.shape[-1] * 2))

    gate, down, up = [packed(n) for n in ("w1", "w2", "w3")]
    w1 = PackedExpertWeight(
        torch.cat((gate.codes, up.codes), 1),
        torch.cat((gate.scales, up.scales), 1),
        (1, 4096, 4096),
    )
    x = torch.randn(2, 4096, device="cuda").bfloat16().requires_grad_()
    offsets = torch.tensor([0, 2], device="cuda")
    config = ExpertTrainingConfig(4096, 2048, 1, mode="base_only")
    base = RocmPackedMXBackend()
    provider = RocmExpertTrainingBackend(base=base)
    y = routed_expert_training(
        config, provider, x, torch.ones(2, device="cuda"), offsets, w1, down, (None,) * 4
    )
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and bool(x.grad.count_nonzero())
    again = routed_expert_training(
        config, provider, x.detach(), torch.ones(2, device="cuda"), offsets, w1, down, (None,) * 4
    )
    assert torch.equal(y.view(torch.uint8), again.view(torch.uint8))
