# SPDX-License-Identifier: Apache-2.0
"""F2 reference tests; error bounds here are provisional reference smoke bounds."""

import pytest
import torch

from rl_engine.reference.rope.qknorm_scaled_rope import (
    build_rope_table,
    fixed_dot_128,
    fixed_sum_128,
    fp64_math,
    qknorm_scaled_rope_backward_reference,
    qknorm_scaled_rope_fixed_tree_reference,
    qknorm_scaled_rope_fp64_reference,
    qknorm_scaled_rope_reference,
)


def inputs(tokens=5):
    gen = torch.Generator().manual_seed(501)
    q = torch.randn(tokens, 32, 128, generator=gen).bfloat16()
    k = torch.randn(tokens, 2, 128, generator=gen).bfloat16()
    return q, k


@pytest.mark.parametrize("local", [False, True])
def test_model_source_expression_and_cast_sensitivity(local):
    q, k = inputs()
    positions = torch.tensor([0, 17, 2047, 2048, 2049])
    cos, sin = build_rope_table(2050)
    actual = qknorm_scaled_rope_reference(q, k, positions, cos, sin, is_local=local)
    for x, scale, out in zip((q, k), (3.87, 1.0), actual):
        x32 = x.float()
        normalized = x32 * (x32.pow(2).mean(-1, keepdim=True) + 1e-5).pow(-0.5)
        y = normalized.bfloat16() * scale
        if scale != 1:
            assert not torch.equal(y, (normalized * scale).bfloat16())
        if local:
            c = cos[positions].repeat(1, 2)[:, None]
            s = sin[positions].repeat(1, 2)[:, None]
            rotated = torch.cat((-y[..., 64:], y[..., :64]), -1)
            expected = y * c + rotated * s
            late_cast = (y.float() * c.float() + rotated.float() * s.float()).bfloat16()
            # This fixture must distinguish missing intermediate casts.
            assert not torch.equal(expected, late_cast)
        else:
            expected = y
        assert torch.equal(out, expected)


def test_scale_and_rotate_half_pairing():
    q = torch.ones(1, 1, 128, dtype=torch.bfloat16)
    k = q.clone()
    global_q, global_k = qknorm_scaled_rope_reference(q, k, is_local=False)
    assert torch.equal(global_k, k)
    assert torch.equal(global_q, (k.float() * 3.87).bfloat16())
    cos = torch.zeros(1, 64, dtype=torch.bfloat16)
    sin = torch.ones_like(cos)
    local = qknorm_scaled_rope_reference(q, k, torch.tensor([0]), cos, sin, is_local=True)
    for out, original in zip(local, (global_q, global_k)):
        assert torch.equal(out, torch.cat((-original[..., 64:], original[..., :64]), -1))


@pytest.mark.parametrize("local", [False, True])
def test_partition_permutation_and_no_mutation(local):
    q, k = inputs()
    before = q.clone(), k.clone()
    pos = torch.tensor([2049, 0, 17, 2047, 2048])
    c, s = build_rope_table(2050)
    full = qknorm_scaled_rope_reference(q, k, pos, c, s, is_local=local)
    pieces = [
        qknorm_scaled_rope_reference(q[a:b], k[a:b], pos[a:b], c, s, is_local=local)
        for a, b in [(0, 1), (1, 3), (3, 5)]
    ]
    order = torch.tensor([4, 1, 3, 0, 2])
    permuted = qknorm_scaled_rope_reference(q[order], k[order], pos[order], c, s, is_local=local)
    for i in range(2):
        assert torch.equal(full[i], torch.cat([part[i] for part in pieces]))
        assert torch.equal(full[i][order], permuted[i])
    assert torch.equal(q, before[0]) and torch.equal(k, before[1])


@pytest.mark.parametrize("local", [False, True])
def test_fp64_output_and_gradient_smoke(local):
    q, k = inputs(3)
    q.requires_grad_()
    k.requires_grad_()
    pos = torch.tensor([0, 2048, 131071])
    c, s = build_rope_table(131072)
    c.requires_grad_()
    s.requires_grad_()
    actual = qknorm_scaled_rope_reference(q, k, pos, c, s, is_local=local)
    q64, k64 = (
        q.detach().double().requires_grad_(),
        k.detach().double().requires_grad_(),
    )
    expected = fp64_math(q64, k64, pos, is_local=local)
    gen = torch.Generator().manual_seed(228)
    upstream = [torch.randn(y.shape, generator=gen).bfloat16() for y in actual]
    grads_all = torch.autograd.grad(actual, (q, k, c, s), grad_outputs=upstream, allow_unused=True)
    grads = grads_all[:2]
    grads64 = torch.autograd.grad(expected, (q64, k64), grad_outputs=[g.double() for g in upstream])
    # Provisional broad smoke bounds, not kernel acceptance thresholds.
    for y, ref in zip(actual, expected):
        relative_l2 = torch.linalg.vector_norm(y.double() - ref) / torch.linalg.vector_norm(ref)
        assert relative_l2 < 0.02
    for g, ref in zip(grads, grads64):
        relative_l2 = torch.linalg.vector_norm(g.double() - ref) / torch.linalg.vector_norm(ref)
        assert relative_l2 < 0.03
    assert grads_all[2:] == (None, None)


def test_zero_empty_and_table_validation():
    q = torch.zeros(2, 1, 128, dtype=torch.bfloat16)
    k = q.clone()
    for y in qknorm_scaled_rope_fp64_reference(q, k, is_local=False):
        assert torch.equal(y, torch.zeros_like(y))
    c, s = build_rope_table(2)
    out = qknorm_scaled_rope_reference(
        q[:0], k[:0], torch.empty(0, dtype=torch.int64), c, s, is_local=True
    )
    assert all(y.shape == (0, 1, 128) for y in out)
    with pytest.raises(ValueError, match="positions"):
        qknorm_scaled_rope_reference(q, k, torch.tensor([0, 2]), c, s, is_local=True)
    with pytest.raises(ValueError, match="BF16"):
        qknorm_scaled_rope_reference(q.float(), k, is_local=False)
    with pytest.raises(ValueError, match="tables"):
        qknorm_scaled_rope_reference(q, k, torch.tensor([0, 1]), is_local=True)


def test_fixed_tree_order_witness():
    # The declared half-split tree gives 2; an adjacent-pair tree gives 0.
    x = torch.zeros(128, dtype=torch.float32)
    x[0], x[1], x[64], x[65] = 2**25, 1, -(2**25), 1
    assert fixed_sum_128(x).item() == 2
    adjacent = x
    for _ in range(7):
        adjacent = adjacent[::2] + adjacent[1::2]
    assert adjacent.item() == 0
    assert fixed_dot_128(x, torch.ones_like(x)).item() == 2
    assert fixed_sum_128(torch.empty(0, 2, 128)).shape == (0, 2, 1)
    with pytest.raises(ValueError, match="FP32"):
        fixed_sum_128(x.bfloat16())


def test_fixed_tree_positive_sum_error_bound():
    gen = torch.Generator().manual_seed(558)
    x = torch.randn(31, 128, generator=gen)
    squares = x * x
    exact = squares.double().sum(-1, keepdim=True)
    got = fixed_sum_128(squares).double()
    # Seven rounded adds on each path; bound applies to finite positive values
    # here, without underflow/overflow. It isolates reduction from product error.
    u = 2.0**-24
    gamma7 = 7 * u / (1 - 7 * u)
    assert torch.all((got - exact).abs() <= gamma7 * exact)


@pytest.mark.parametrize("local", [False, True])
def test_fixed_tree_reference_partition_and_fp64(local):
    q, k = inputs()
    pos = torch.tensor([0, 17, 2047, 2048, 2049])
    c, s = build_rope_table(2050)
    run = qknorm_scaled_rope_fixed_tree_reference
    whole = run(q, k, pos, c, s, is_local=local)
    order = torch.tensor([4, 2, 0, 3, 1])
    permuted = run(q[order], k[order], pos[order], c, s, is_local=local)
    pieces = [
        run(q[i : i + 1], k[i : i + 1], pos[i : i + 1], c, s, is_local=local) for i in range(5)
    ]
    expected = qknorm_scaled_rope_fp64_reference(q, k, pos, is_local=local)
    for index in range(2):
        # Integer views verify actual BF16 bit patterns, including signed zero.
        assert torch.equal(
            whole[index].view(torch.int16),
            torch.cat([p[index] for p in pieces]).view(torch.int16),
        )
        assert torch.equal(whole[index][order].view(torch.int16), permuted[index].view(torch.int16))
        error = torch.linalg.vector_norm(whole[index].double() - expected[index])
        assert error / torch.linalg.vector_norm(expected[index]) < 0.02


@pytest.mark.parametrize("local", [False, True])
def test_declared_backward_against_independent_autograd(local):
    q, k = inputs(3)
    pos = torch.tensor([0, 2048, 2049])
    c, s = build_rope_table(2050)
    saved = tuple(
        (fixed_sum_128(x.float() * x.float()) / 128 + 1e-5).pow(-0.5).squeeze(-1) for x in (q, k)
    )
    gen = torch.Generator().manual_seed(558)
    grads = tuple(torch.randn(x.shape, generator=gen).bfloat16() for x in (q, k))
    actual = qknorm_scaled_rope_backward_reference(q, k, *grads, *saved, pos, c, s, is_local=local)
    q64, k64 = q.double().requires_grad_(), k.double().requires_grad_()
    expected = torch.autograd.grad(
        fp64_math(q64, k64, pos, is_local=local),
        (q64, k64),
        grad_outputs=tuple(g.double() for g in grads),
    )
    for out, ref in zip(actual, expected):
        error = torch.linalg.vector_norm(out.double() - ref, dim=-1)
        assert torch.all(error / torch.linalg.vector_norm(ref, dim=-1) < 0.03)


@pytest.mark.parametrize("exponent", [50, 55, 60])
def test_backward_large_constant_input_preserves_correction(exponent):
    x = torch.full((1, 1, 128), 2.0**exponent, dtype=torch.bfloat16)
    g = torch.zeros_like(x)
    g[..., :64] = 1
    r = (fixed_sum_128(x.float() * x.float()) / 128 + 1e-5).pow(-0.5).squeeze(-1)
    actual = qknorm_scaled_rope_backward_reference(
        x, x, torch.zeros_like(g), g, r, r, is_local=False
    )[1]
    # Analytic derivative: (g - mean(g))/x. Epsilon is negligible at these scales.
    expected = torch.full_like(x, -(2.0 ** (-exponent - 1)))
    expected[..., :64] *= -1
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
