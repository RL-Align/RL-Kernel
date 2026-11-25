# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.guidance import NativeCfgCombineRenormOp

CONTRACT = load_contract()
MODES = ("global", "channel", "text_channel")


def assert_accuracy(actual, expected, judgment):
    tol = resolve_tolerance(CONTRACT, judgment=judgment, op_class="reduction", dtype=actual.dtype)
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=tol.atol, rtol=tol.rtol)


def oracle(main, text, image, mode, minimum):
    def renorm(value):
        dims = (1, 2) if mode == "global" else (-1,)
        numerator = torch.linalg.vector_norm(main, dim=dims, keepdim=True)
        denominator = torch.linalg.vector_norm(value, dim=dims, keepdim=True) + 1e-8
        return value * (numerator / denominator).clamp(minimum, 1)

    mixed = text + 4 * (main - text)
    if mode == "text_channel":
        mixed = renorm(mixed)
    if image is not None:
        mixed = image + 1.5 * (mixed - image)
    return mixed if mode == "text_channel" else renorm(mixed)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("branch_count", (2, 3))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
def test_forward_and_all_branch_gradients_against_fp64(mode, branch_count, dtype):
    rng = torch.Generator().manual_seed(435)
    leaves = [
        torch.randn(2, 3, 7, generator=rng).to(dtype).requires_grad_() for _ in range(branch_count)
    ]
    op = NativeCfgCombineRenormOp()
    actual = op(*leaves, renorm_type=mode)
    gold = op.forward_fp32(*leaves, renorm_type=mode)
    assert gold.dtype == torch.float32
    assert actual.dtype == dtype and torch.equal(actual, gold.to(dtype))
    doubles = [x.detach().double().requires_grad_() for x in leaves]
    expected = oracle(doubles[0], doubles[1], doubles[2] if branch_count == 3 else None, mode, 0)
    assert_accuracy(gold, expected, "forward_accuracy")
    upstream = torch.randn(actual.shape, generator=rng).to(dtype)
    grads = torch.autograd.grad(actual, leaves, upstream)
    refs = torch.autograd.grad(expected, doubles, upstream.double())
    for grad, ref in zip(grads, refs):
        assert_accuracy(grad, ref, "gradient_accuracy")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cfg_img_scale", [1.0, 0.5])
def test_disabled_image_branch_matches_text_only(mode, cfg_img_scale):
    main = torch.ones(1, 2, 3)
    text = torch.full_like(main, 3)
    image = torch.arange(6.0).reshape_as(main).requires_grad_()
    op = NativeCfgCombineRenormOp()
    expected = op(main, text, renorm_type=mode)
    actual = op(main, text, image, renorm_type=mode, cfg_img_scale=cfg_img_scale)
    assert torch.equal(actual, expected)
    (grad_image,) = torch.autograd.grad(actual.sum(), image)
    assert torch.equal(grad_image, torch.zeros_like(image))


def test_identity_profile_at_unit_text_scale_without_image():
    main = torch.arange(6.0).reshape(1, 2, 3).requires_grad_()
    text = torch.full_like(main, 9)
    op = NativeCfgCombineRenormOp()
    actual = op(main, text, v_img=None, cfg_text_scale=1.0, cfg_img_scale=1.5)
    assert torch.equal(actual, main)
    (grad_main,) = torch.autograd.grad(actual.sum(), main)
    assert torch.equal(grad_main, torch.ones_like(main))


@pytest.mark.parametrize(
    "main_value,text_value,minimum,scale",
    [(1, -4, 0.5, 0.5), (4, 7, 0.0, 1.0), (1, 1, 0.5, 0.5), (2, 3, 0.0, 1.0)],
)
def test_clamp_saturation_including_exact_endpoints(main_value, text_value, minimum, scale):
    main = torch.full((1, 1, 1), float(main_value), requires_grad=True)
    text = torch.full_like(main, float(text_value), requires_grad=True)
    out = NativeCfgCombineRenormOp()(main, text, cfg_text_scale=2, cfg_renorm_min=minimum, eps=1)
    assert out.item() == (2 * main_value - text_value) * scale
    dmain, dtext = torch.autograd.grad(out.sum(), (main, text))
    assert (dmain.item(), dtext.item()) == (2 * scale, -scale)


@pytest.mark.parametrize("mode", MODES)
def test_all_zero_is_finite(mode):
    leaves = [torch.zeros(2, 3, 7, requires_grad=True) for _ in range(3)]
    out = NativeCfgCombineRenormOp()(*leaves, renorm_type=mode)
    assert torch.equal(out, torch.zeros_like(out))
    assert all(torch.isfinite(x).all() for x in torch.autograd.grad(out.sum(), leaves))
