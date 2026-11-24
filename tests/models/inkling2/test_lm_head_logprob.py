# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.lm_head_logprob import (
    lm_head_logprob,
    selected_logprob,
    softcap_valid_logits,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_released_vocab_softcap_and_padding(device):
    raw = torch.zeros(2, 201024, device=device, requires_grad=True)
    with torch.no_grad():
        raw[:, 200058:] = 1e5
        raw[0, 0] = 90
    capped = softcap_valid_logits(raw)
    assert capped.shape == (2, 200058) and capped.dtype == torch.float32
    assert capped[0, 0].item() == pytest.approx(30 * torch.tanh(torch.tensor(3.)).item())
    y = selected_logprob(capped, torch.tensor([0, -100], device=device))
    expected = capped[0].log_softmax(-1)[0]
    torch.testing.assert_close(y[0], expected, atol=4e-6, rtol=2e-5)
    assert y[1].item() == 0
    y.sum().backward()
    assert torch.count_nonzero(raw.grad[:, 200058:]) == 0
    assert torch.count_nonzero(raw.grad[1]) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_lm_head_and_bf16_gradient_boundary(device):
    torch.manual_seed(32)
    h = torch.randint(-2, 3, (3, 16), device=device).bfloat16().requires_grad_()
    w = torch.randint(-2, 3, (24, 16), device=device).float().requires_grad_()
    targets = torch.tensor([2, 5, -100], device=device)
    out = lm_head_logprob(h, w, targets, valid_vocab_size=20)
    raw = (h.detach().float() @ w.detach()[:20].bfloat16().float().t()).requires_grad_()
    ref = selected_logprob(softcap_valid_logits(raw, valid_vocab_size=20), targets)
    torch.testing.assert_close(out, ref)
    assert torch.equal(out[1:2], lm_head_logprob(h[1:2], w, targets[1:2], valid_vocab_size=20))
    g = torch.autograd.grad(ref.sum(), raw)[0].bfloat16().float()
    out.sum().backward()
    expected_h = (g @ w.detach()[:20].bfloat16().float()).bfloat16()
    expected_w = (g.t() @ h.detach().float()).bfloat16().float()
    assert torch.equal(h.grad, expected_h)
    assert torch.equal(w.grad[:20], expected_w)
    assert torch.count_nonzero(w.grad[20:]) == 0


def test_invalid_target_and_fp32_logit_contract():
    with pytest.raises(ValueError, match="outside valid"):
        selected_logprob(torch.ones(1, 4), torch.tensor([4]))
    with pytest.raises(TypeError, match="FP32"):
        softcap_valid_logits(torch.ones(1, 4).bfloat16(), valid_vocab_size=4)
