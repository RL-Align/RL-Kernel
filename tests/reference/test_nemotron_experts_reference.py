# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract, cast points and differentiability of the experts reference."""

import math

import pytest
import torch

from rl_engine.reference.moe.nemotron_experts import MAX_TOKENS, experts_reference


def _toy(
    seed=434,
    tokens=6,
    width=8,
    intermediate=5,
    shared_width=7,
    experts=4,
    slots=3,
    dtype=torch.float64,
    weight_dtype=None,
):
    generator = torch.Generator().manual_seed(seed)
    exact = torch.float64
    x = torch.randn(tokens, width, generator=generator, dtype=exact)
    w_up = torch.randn(experts, intermediate, width, generator=generator, dtype=exact) * 0.5
    w_down = torch.randn(experts, width, intermediate, generator=generator, dtype=exact) * 0.5
    w_sh1 = torch.randn(
        shared_width, width, generator=torch.Generator().manual_seed(seed + 1), dtype=exact
    )
    w_sh2 = torch.randn(
        width, shared_width, generator=torch.Generator().manual_seed(seed + 2), dtype=exact
    )
    ids = torch.stack(
        [torch.arange(slots) + (index % (experts - slots + 1)) for index in range(tokens)]
    )
    if tokens == 0:
        ids = torch.zeros((0, slots), dtype=torch.int64)
    weights = torch.rand(
        tokens, slots, generator=torch.Generator().manual_seed(seed + 3), dtype=exact
    )
    return (
        x.to(dtype),
        ids,
        weights.to(dtype if weight_dtype is None else weight_dtype),
        w_up.to(dtype),
        w_down.to(dtype),
        w_sh1.to(dtype),
        w_sh2.to(dtype),
    )


def _fsum_dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _scalar_oracle(x, ids, weights, w_up, w_down, w_sh1=None, w_sh2=None):
    """Row-local scalar composition in ascending expert/slot order."""
    tokens, width = x.shape
    out = [[0.0] * width for _ in range(tokens)]
    for token in range(tokens):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            up = [_fsum_dot(x[token], w_up[expert][index]) for index in range(w_up.shape[1])]
            gate = [max(value, 0.0) ** 2 for value in up]
            down = [_fsum_dot(gate, w_down[expert][index]) for index in range(width)]
            for index in range(width):
                out[token][index] += float(weights[token, slot]) * down[index]
        if w_sh1 is not None:
            up = [_fsum_dot(x[token], w_sh1[index]) for index in range(w_sh1.shape[0])]
            gate = [max(value, 0.0) ** 2 for value in up]
            down = [_fsum_dot(gate, w_sh2[index]) for index in range(width)]
            for index in range(width):
                out[token][index] += down[index]
    return torch.tensor(out, dtype=torch.float64).reshape(tokens, width)


def test_scalar_oracle_toy_geometry():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy()
    actual = experts_reference(x, ids, weights, w_up, w_down, w_sh1, w_sh2)
    expected = _scalar_oracle(x, ids, weights, w_up, w_down, w_sh1, w_sh2)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


def test_shared_expert_is_additive_and_optional():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy()
    routed = experts_reference(x, ids, weights, w_up, w_down)
    both = experts_reference(x, ids, weights, w_up, w_down, w_sh1, w_sh2)
    shared_only = _scalar_oracle(
        x,
        torch.zeros((x.shape[0], 0), dtype=torch.int64),
        weights[:, :0],
        w_up,
        w_down,
        w_sh1,
        w_sh2,
    )
    torch.testing.assert_close(both - routed, shared_only, atol=1e-13, rtol=1e-13)
    with pytest.raises(ValueError):
        experts_reference(x, ids, weights, w_up, w_down, w_sh1, None)


def test_empty_expert_and_zero_tokens():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy(experts=10, slots=2)
    base = experts_reference(x, ids, weights, w_up, w_down)
    used = torch.zeros(w_up.shape[0], dtype=torch.bool)
    used[ids.reshape(-1)] = True
    assert int((~used).sum()) >= 2  # The toy must actually leave segments empty.
    unused_perturbed = w_up.clone()
    unused_perturbed[~used] += 1.0
    assert torch.equal(base, experts_reference(x, ids, weights, unused_perturbed, w_down))
    used_perturbed = w_up.clone()
    used_perturbed[used] += 1.0
    assert not torch.equal(base, experts_reference(x, ids, weights, used_perturbed, w_down))
    for shared in ((w_sh1, w_sh2), (None, None)):
        empty_out = experts_reference(
            torch.zeros(0, x.shape[1], dtype=torch.float64),
            torch.zeros((0, ids.shape[1]), dtype=torch.int64),
            torch.zeros((0, ids.shape[1]), dtype=torch.float64),
            w_up,
            w_down,
            *shared,
        )
        assert empty_out.shape == (0, x.shape[1])
        assert empty_out.dtype == torch.float64


def test_single_expert_matches_row_local_oracle():
    x, ids, weights, w_up, w_down, _, _ = _toy(experts=2, slots=1)
    actual = experts_reference(x, ids, weights, w_up, w_down)
    expected = _scalar_oracle(x, ids, weights, w_up, w_down)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


def _cast_point_error(payload_dtype, weight_dtype=None):
    """Relative L2 of the reference against the scalar oracle for one pairing."""
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy(
        dtype=payload_dtype, weight_dtype=weight_dtype
    )
    actual = experts_reference(x, ids, weights, w_up, w_down, w_sh1, w_sh2)
    exact = _scalar_oracle(
        x.double(),
        ids,
        weights.double(),
        w_up.double(),
        w_down.double(),
        w_sh1.double(),
        w_sh2.double(),
    )
    assert actual.dtype == torch.bfloat16
    return (actual.double() - exact).norm() / exact.norm()


@pytest.mark.parametrize(
    "payload_dtype, weight_dtype",
    [
        pytest.param(torch.bfloat16, torch.float32, id="canonical-bf16-fp32-routes"),
        pytest.param(torch.bfloat16, None, id="bf16-payload-bf16-routes"),
        pytest.param(torch.float32, torch.float32, id="fp32-payload"),
    ],
)
def test_cast_points_round_leaves_and_exit_only(payload_dtype, weight_dtype):
    error = _cast_point_error(payload_dtype, weight_dtype)
    # The BF16 rounding points (up result, gate, exit cast) are dtype-independent:
    # the payload dtype only decides the matmul precision, so every accepted
    # pairing sits in the same 1e-3 band.  An unrounded path would be at 1e-13
    # (that is what the FP64 cases assert), hence the two-sided bound.
    assert 1e-5 < error < 5e-3


def test_payload_dtype_does_not_move_the_rounding_band():
    """BF16 and FP32 payloads agree to well within the bound, unlike FP64.

    Measured on the toy geometry: FP32 payload 3.1e-3, BF16 payload 2.4e-3, FP64
    payload 1e-13.  A change that dropped a `_leaf_round` would move a BF16/FP32
    case towards the FP64 figure, so the band is what pins the cast points.
    """
    fp32_payload = _cast_point_error(torch.float32, torch.float32)
    bf16_payload = _cast_point_error(torch.bfloat16, torch.float32)
    assert max(fp32_payload, bf16_payload) < 3 * min(fp32_payload, bf16_payload)
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy()
    exact = _scalar_oracle(x, ids, weights, w_up, w_down, w_sh1, w_sh2)
    fp64 = experts_reference(x, ids, weights, w_up, w_down, w_sh1, w_sh2).double()
    assert (fp64 - exact).norm() / exact.norm() < 1e-12


def test_rejects_a_payload_weight_dtype_mismatch():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy(dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="share dtype and device"):
        experts_reference(x, ids, weights, w_up.double(), w_down, w_sh1, w_sh2)
    with pytest.raises(ValueError, match="weights must be float32"):
        experts_reference(x, ids, weights.double(), w_up, w_down, w_sh1, w_sh2)


def test_token_permutation_invariance():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy(tokens=9)
    order = torch.tensor([8, 0, 5, 2, 7, 1, 6, 4, 3])
    base = experts_reference(x, ids, weights, w_up, w_down, w_sh1, w_sh2)
    moved = experts_reference(
        x.index_select(0, order),
        ids.index_select(0, order),
        weights.index_select(0, order),
        w_up,
        w_down,
        w_sh1,
        w_sh2,
    )
    # The reference accumulates with torch.matmul, whose per-row reduction is
    # not row-order stable; the operator's row-local kernels are (tests on GPU).
    torch.testing.assert_close(base.index_select(0, order), moved, atol=1e-12, rtol=1e-12)


def test_expert_id_mapping_is_not_shared_between_experts():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy()
    swapped = w_up.flip(0)
    base = experts_reference(x, ids, weights, w_up, w_down)
    other = experts_reference(x, ids, weights, swapped, w_down)
    assert not torch.equal(base.to(torch.float32), other.to(torch.float32))
    assert torch.equal(experts_reference(x, ids, weights, w_up, w_down), base)


def test_gradcheck_weights_and_payload():
    x, ids, weights, w_up, w_down, _, _ = _toy(
        tokens=2, width=3, intermediate=2, experts=2, slots=2
    )
    x = x.clone().requires_grad_()
    w_up = w_up.clone().requires_grad_()
    w_down = w_down.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda hidden, up, down: experts_reference(hidden, ids, weights, up, down),
        (x, w_up, w_down),
    )
    weights_grad = weights.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda route: experts_reference(x.detach(), ids, route, w_up.detach(), w_down.detach()),
        (weights_grad,),
    )


def test_shared_expert_is_differentiable():
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy(
        tokens=2, width=3, intermediate=2, shared_width=2, experts=2, slots=2
    )
    x = x.clone().requires_grad_()
    w_sh1 = w_sh1.clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda hidden, up, down: experts_reference(hidden, ids, weights, w_up, w_down, up, down),
        (x, w_sh1, w_sh2.clone().requires_grad_()),
    )


def test_weights_scale_each_route_only():
    x, ids, weights, w_up, w_down, _, _ = _toy(slots=2, experts=2)
    weights = weights.clone()
    weights[:, 0] = 0.0
    actual = experts_reference(x, ids, weights, w_up, w_down)
    expected = _scalar_oracle(x, ids, weights, w_up, w_down)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda tensors: (tensors[0][:, :-1],) + tensors[1:], "widths must agree"),
        (
            lambda tensors: tensors[:2]
            + (tensors[2], tensors[3], tensors[4][:2], tensors[5], tensors[6]),
            "same expert set",
        ),
        (
            lambda tensors: tensors[:1] + (tensors[1].to(torch.int32),) + tensors[2:],
            "int64 on the input device",
        ),
    ],
)
def test_reject_mismatched_geometry(mutation, message):
    x, ids, weights, w_up, w_down, w_sh1, w_sh2 = _toy()
    mutated = mutation((x, ids, weights, w_up, w_down, w_sh1, w_sh2))
    with pytest.raises(ValueError, match=message):
        experts_reference(mutated[0], mutated[1], mutated[2], mutated[3], mutated[4])
    with pytest.raises(ValueError):
        experts_reference(x, ids, weights, w_up, w_down, w_sh1, None)


def test_token_bound_is_documented():
    assert MAX_TOKENS == 65536
