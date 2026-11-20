# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from rl_engine.backends.rocm.attention.v41_transforms import fp4_roundtrip, fp8_roundtrip, rotary
from rl_engine.backends.rocm.moe.v41_router import route_top6
from rl_engine.models.deepseek_v41.bitwise import compare_bits
from rl_engine.models.deepseek_v41.reference import quant_dequant_reference


@pytest.fixture(autouse=True, scope="module")
def require_device():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("real ROCm device required")
        pytest.skip("ROCm device required")


@pytest.mark.parametrize("group,e4scale", [(32, False), (16, True)])
def test_fp4_codec_reference_and_ste(group, e4scale):
    torch.manual_seed(44)
    x = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = fp4_roundtrip(x, group=group, e4m3_scale=e4scale)
    blocks = x.detach().float().reshape(-1, group)
    if e4scale:
        scale = (blocks.abs().amax(-1).clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        scale = 2.0 ** torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(6 * 2**-126) / 6))
    normalized = (blocks / scale[:, None]).clamp(-6, 6)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    # even-code tie rule of E2M1, expressed independently as closest-level distance.
    distance = (normalized.abs()[..., None] - levels).abs()
    minimum = distance.amin(-1, keepdim=True)
    even = torch.arange(8, device="cuda") % 2 == 0
    choices = distance == minimum
    favored = choices & even
    use_even = favored.any(-1, keepdim=True)
    selected = torch.where(use_even, favored, choices).int().argmax(-1)
    ref = (
        (torch.copysign(levels[selected], normalized) * scale[:, None]).reshape(x.shape).bfloat16()
    )
    assert compare_bits(y, ref)["status"] == "PASS"
    dy = torch.randn_like(x)
    assert compare_bits(torch.autograd.grad(y, x, dy)[0], dy)["status"] == "PASS"
    with torch.inference_mode():
        assert (
            compare_bits(y, fp4_roundtrip(x, group=group, e4m3_scale=e4scale))["status"] == "PASS"
        )


def test_fp8_roundtrip_and_ste():
    x = torch.randn(2, 3, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = fp8_roundtrip(x)
    assert compare_bits(y, quant_dequant_reference(x.detach()))["status"] == "PASS"
    dy = torch.randn_like(x)
    assert compare_bits(torch.autograd.grad(y, x, dy)[0], dy)["status"] == "PASS"


def test_rope_forward_inverse_and_transposed_gradient():
    x = torch.randn(2, 3, 4, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    angle = torch.randn(2, 3, 32, device="cuda")
    cos, sin = angle.cos(), angle.sin()
    for inverse in (False, True):
        y = rotary(x, cos, sin, inverse=inverse)
        tail = torch.view_as_complex(x.float()[..., -64:].reshape(2, 3, 4, 32, 2))
        freq = torch.complex(cos, -sin if inverse else sin).unsqueeze(-2)
        ref = torch.cat((x[..., :448], torch.view_as_real(tail * freq).flatten(-2).bfloat16()), -1)
        torch.testing.assert_close(y, ref, rtol=0.01, atol=0.01)
        dy = torch.randn_like(y)
        dx = torch.autograd.grad(y, x, dy, retain_graph=True)[0]
        expected = torch.autograd.grad(ref, x, dy, retain_graph=True)[0]
        torch.testing.assert_close(dx, expected, rtol=0.01, atol=0.01)


def test_384_expert_routing_gradients_and_tie_rule():
    torch.manual_seed(55)
    logits = torch.randn(7, 384, device="cuda", requires_grad=True) * 3
    bias = torch.randn(384, device="cuda") * 0.1
    w, ids = route_top6(logits, bias)
    scores = torch.nn.functional.softplus(logits).sqrt()
    expected_ids = (scores + bias).topk(6, -1).indices
    assert torch.equal(ids, expected_ids)
    chosen = scores.gather(1, ids)
    expected = chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * 1.5
    torch.testing.assert_close(w, expected, rtol=2e-6, atol=2e-7)
    dw = torch.randn_like(w)
    dx = torch.autograd.grad(w, logits, dw, retain_graph=True)[0]
    ref = torch.autograd.grad(expected, logits, dw)[0]
    torch.testing.assert_close(dx, ref, rtol=2e-5, atol=2e-6)
    again = route_top6(logits, bias)
    assert compare_bits(w, again[0])["status"] == "PASS"
    ties = route_top6(torch.zeros(1, 384, device="cuda"), torch.zeros(384, device="cuda"))
    assert torch.equal(ties[1], torch.arange(6, device="cuda")[None, :])


def test_top_six_combine_order_and_backward():
    from rl_engine.backends.rocm.moe.v41_combine import combine

    torch.manual_seed(192)
    x = torch.randn(5, 6, 5120, device="cuda", dtype=torch.bfloat16).requires_grad_()
    shared = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16).requires_grad_()
    expected = torch.zeros_like(shared, dtype=torch.float32)
    for slot in range(6):
        expected = expected + x[:, slot].float()
    expected = (expected + shared.float()).bfloat16()
    y = combine(x, shared)
    assert compare_bits(y, expected)["status"] == "PASS"
    dy = torch.randn_like(y)
    dx, ds = torch.autograd.grad(y, (x, shared), dy)
    assert compare_bits(dx, dy[:, None].expand_as(dx))["status"] == "PASS"
    assert compare_bits(ds, dy)["status"] == "PASS"


def test_candidate_blocks_partial_newest_and_unreachable():
    from rl_engine.models.deepseek_v41.attention import candidate_blocks

    scores = torch.tensor([[[20.0, 19.0, 10.0, 9.0, -1.0, -2.0, -torch.inf]]], device="cuda")
    keep = candidate_blocks(scores, torch.tensor([[6]], device="cuda"), 2, 2)
    expected = torch.tensor([[[True, True, False, False, True, True, False]]], device="cuda")
    assert torch.equal(keep, expected)
