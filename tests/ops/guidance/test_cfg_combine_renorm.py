# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.ops.guidance.cfg_combine_renorm import cfg_combine_renorm


def test_gate_compares_fp32_open_lower_closed_upper_per_request():
    main = torch.ones(4, 1, 3, requires_grad=True)
    text = torch.full_like(main, 3, requires_grad=True)
    image = torch.full_like(main, 2, requires_grad=True)
    times = torch.tensor([0.4, 0.400000001, 1.0, 1.001], dtype=torch.float64)
    out = cfg_combine_renorm(main, text, image, timestep=times)
    expected = cfg_combine_renorm(main, text, image)
    assert torch.equal(out[[0, 1, 3]], main[[0, 1, 3]])
    assert torch.equal(out[2], expected[2]) and not torch.equal(out[2], main[2])
    grads = torch.autograd.grad(out.sum(), (main, text, image))
    for grad, value in zip(grads, (1, 0, 0)):
        assert (grad[[0, 1, 3]] == value).all()


@pytest.mark.parametrize(
    "policy", [{"cfg_text_scale": 0.5}, {"cfg_text_scale": 1}, {"timestep": 0.4}]
)
def test_identity_profile_with_image_and_zero_branch_gradients(policy):
    main = torch.arange(6.0).reshape(1, 2, 3).requires_grad_()
    text = torch.full_like(main, float("nan"), requires_grad=True)
    image = torch.full_like(main, float("nan"), requires_grad=True)
    out = cfg_combine_renorm(main, text, image, cfg_img_scale=2, **policy)
    upstream = torch.arange(6.0).reshape_as(main)
    dv, dt, di = torch.autograd.grad(out, (main, text, image), upstream)
    assert torch.equal(out, main) and torch.equal(dv, upstream)
    assert torch.count_nonzero(dt) == torch.count_nonzero(di) == 0


def test_identity_profile_at_unit_text_scale_without_image():
    main = torch.arange(6.0).reshape(1, 2, 3).requires_grad_()
    text = torch.full_like(main, 9)
    out = cfg_combine_renorm(main, text, v_img=None, cfg_text_scale=1.0, cfg_img_scale=1.5)
    assert torch.equal(out, main)
    (grad_main,) = torch.autograd.grad(out.sum(), main)
    assert torch.equal(grad_main, torch.ones_like(main))


@pytest.mark.parametrize("mode", ["global", "channel", "text_channel"])
def test_packed_neighbors_and_order_are_bitwise_invariant(mode):
    rng = torch.Generator().manual_seed(65)
    requests = [[torch.randn(n, 17, generator=rng) for _ in range(4)] for n in (17, 2, 25)]

    def run(order):
        tensors = [torch.cat([requests[i][k] for i in order]) for k in range(4)]
        offsets = [0]
        for i in order:
            offsets.append(offsets[-1] + requests[i][0].shape[0])
        leaves = [x.requires_grad_() for x in tensors[:3]]
        out = cfg_combine_renorm(*leaves, cu_seqlens=torch.tensor(offsets), renorm_type=mode)
        gradients = torch.autograd.grad(out, leaves, tensors[3])
        idx = order.index(0)
        return [x[offsets[idx] : offsets[idx + 1]] for x in (out, *gradients)]

    alone = run([0])
    for order in ([1, 0, 2], [2, 1, 0]):
        assert all(torch.equal(a, b) for a, b in zip(alone, run(order)))


@pytest.mark.parametrize(
    "offsets",
    [None, [1, 3], [0, 2], [0, 4, 3], [0, 1, 1, 3], [[0, 3]], [0.0, 3.0]],
)
def test_strict_packed_requires_valid_integer_offsets(offsets):
    main = torch.ones(3, 5)
    cu = None if offsets is None else torch.tensor(offsets)
    with pytest.raises((ValueError, TypeError)):
        cfg_combine_renorm(main, main, cu_seqlens=cu)


@pytest.mark.parametrize(
    "kwargs",
    [{"renorm_type": "local"}, {"backend": "triton"}, {"timestep": torch.ones(3)}],
)
def test_unknown_policy_backend_and_misaligned_timestep_rejected(kwargs):
    main = torch.ones(2, 3, 5)
    with pytest.raises((ValueError, RuntimeError)):
        cfg_combine_renorm(main, main, **kwargs)


@pytest.mark.parametrize("other", [torch.ones(1, 3, 5), torch.ones(2, 3, 5, dtype=torch.float16)])
def test_mismatched_branch_is_rejected(other):
    main = torch.ones(2, 3, 5)
    with pytest.raises((TypeError, ValueError)):
        cfg_combine_renorm(main, other)


def test_optional_branches_follow_effective_scales():
    main = torch.ones(1, 2, 3)
    assert torch.equal(cfg_combine_renorm(main, timestep=0.0), main)
    with pytest.raises(ValueError):
        cfg_combine_renorm(main.expand(2, -1, -1), timestep=torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("cfg_img_scale", [1.0, 0.5])
def test_disabled_image_branch_matches_text_only(cfg_img_scale):
    main = torch.ones(1, 2, 3)
    text = torch.full_like(main, 3)
    image = torch.arange(6.0).reshape_as(main).requires_grad_()
    two = cfg_combine_renorm(main, text)
    disabled_image = cfg_combine_renorm(main, text, image, cfg_img_scale=cfg_img_scale)
    assert torch.equal(two, disabled_image)
    (grad_image,) = torch.autograd.grad(disabled_image.sum(), image)
    assert torch.count_nonzero(grad_image) == 0


def test_packed_mixed_gates_match_dense_requests():
    rng = torch.Generator().manual_seed(7)
    tensors = [torch.randn(5, 7, generator=rng) for _ in range(3)]
    mixed = cfg_combine_renorm(
        *tensors, cu_seqlens=torch.tensor([0, 2, 5]), timestep=torch.tensor([0.4, 1.0])
    )
    assert torch.equal(mixed[:2], tensors[0][:2])
    dense = cfg_combine_renorm(*(x[2:].unsqueeze(0) for x in tensors), timestep=1.0)
    assert torch.equal(mixed[2:], dense.squeeze(0))


def test_non_strict_global_treats_packed_tensor_as_one_request():
    rng = torch.Generator().manual_seed(81)
    tensors = [torch.randn(5, 7, generator=rng) for _ in range(3)]
    implicit = cfg_combine_renorm(*tensors, strict=False)
    explicit = cfg_combine_renorm(*tensors, cu_seqlens=torch.tensor([0, 5]))
    assert torch.equal(implicit, explicit)
