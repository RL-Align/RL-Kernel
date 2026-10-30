# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from rl_engine.p2 import oracle
from rl_engine.p2.contract import ContractError, Status
from rl_engine.p2.fixtures import materialize_weight, selector_linear, selector_recipe, values


@pytest.mark.parametrize("n", [0, 1, 2, 3, 32, 64, 128, 513])
def test_fixed_tree(n):
    x = values((3, n))
    expected = []
    for row in x.tolist():
        if not row:
            expected.append(0.0)
            continue
        row += [0.0] * ((1 << (len(row) - 1).bit_length()) - len(row))
        while len(row) > 1:
            row = [
                float(torch.tensor(row[i], dtype=torch.float32) + row[i + 1])
                for i in range(0, len(row), 2)
            ]
        expected.append(row[0])
    assert torch.equal(oracle.fixed_sum(x), torch.tensor(expected))


def test_scale_forward_backward_and_global_head_count():
    u = values((2, 64)).requires_grad_()
    out = oracle.scale(u)
    assert torch.equal(out, (u * 0.125) * (128**-0.5))
    out.sum().backward()
    assert torch.equal(u.grad, oracle.scale(torch.ones_like(u), backward=True))
    assert not torch.equal(out, (u * (8**-0.5)) * (128**-0.5))


@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("position", [0, 1, 4, 128])
def test_rope_partial_inverse_and_backward(dim, position):
    x = values((1, 3, dim)).requires_grad_()
    original = x.detach().clone()
    cos = torch.full((129, 32), 0.6)
    sin = torch.full((129, 32), 0.8)
    pos = torch.tensor([position])
    y = oracle.rope(x, cos, sin, pos)
    assert y.data_ptr() != x.data_ptr()
    assert torch.equal(x, original)
    assert torch.equal(y[..., :-64], x[..., :-64])
    torch.testing.assert_close(oracle.rope(y, cos, sin, pos, inverse=True), x, rtol=1e-6, atol=1e-6)
    grad = values(y.shape, 7)
    y.backward(grad)
    torch.testing.assert_close(
        x.grad, oracle.rope(grad, cos, sin, pos, inverse=True), rtol=0, atol=0
    )


def test_rope_negative():
    x, c, s, p = values((1, 128)), torch.ones(1, 32), torch.zeros(1, 32), torch.tensor([0])
    with pytest.raises(ContractError, match=Status.INVALID_ROPE_VARIANT.value):
        oracle.rope(x, c, s, p, variant="neox")
    with pytest.raises(ContractError, match=Status.INVALID_GLOBAL_POSITION.value):
        oracle.rope(x, c, s, torch.tensor([1]))


def test_hadamard_self_inverse_and_backward():
    x = values((2, 128)).requires_grad_()
    y = oracle.hadamard(x)
    torch.testing.assert_close(oracle.hadamard(y), x, atol=1e-6, rtol=1e-6)
    y.sum().backward()
    torch.testing.assert_close(x.grad, oracle.hadamard(torch.ones_like(x)), atol=1e-6, rtol=1e-6)


def test_fp4_literal_golden_nibbles_and_zero_block():
    x = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6] * 2)
    packed, scales = oracle.pack_mxfp4(x)
    assert packed.tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2
    assert scales.tolist() == [127]
    unpacked = oracle.unpack_mxfp4(packed, scales)
    assert torch.equal(unpacked, x)
    assert torch.signbit(unpacked)[8]
    z, s = oracle.pack_mxfp4(torch.zeros(64))
    assert z.tolist() == [0] * 32
    assert s.tolist() == [1, 1]


def test_fp4_ties_even_and_block_row_isolation():
    block = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4)
    x = torch.stack((torch.cat((block, block * 2)), torch.cat((block * 4, block))))
    p, s = oracle.pack_mxfp4(x)
    assert s.tolist() == [[127, 128], [129, 127]]
    assert p[0, :4].tolist() == [0x20, 0x42, 0x64, 0x76]
    for row in range(2):
        for b in range(2):
            pp, ss = oracle.pack_mxfp4(x[row, b * 32 : (b + 1) * 32])
            assert torch.equal(pp, p[row, b * 16 : (b + 1) * 16])
            assert ss.item() == s[row, b].item()


@pytest.mark.parametrize("number", [2.0**-149, 2.0**-126, torch.finfo(torch.float32).max])
def test_fp4_subnormal_and_scale_clip(number):
    p, s = oracle.pack_mxfp4(torch.full((32,), number))
    assert 1 <= s.item() <= 254
    if number == torch.finfo(torch.float32).max:
        with pytest.raises(ContractError, match=Status.NON_FINITE.value):
            oracle.unpack_mxfp4(p, s)
    else:
        assert torch.isfinite(oracle.unpack_mxfp4(p, s)).all()


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_nonfinite_rejected(number):
    with pytest.raises(ContractError, match=Status.NON_FINITE.value):
        oracle.pack_mxfp4(torch.full((32,), number))


def test_c4_ape_before_overlap_and_first_group_gradient():
    k = values((3, 4, 2, 8), 1).requires_grad_()
    s = values(k.shape, 3).requires_grad_()
    ape = values(k.shape[1:], 7).requires_grad_()
    out, alpha = oracle.c4_pool(k, s, ape)
    assert torch.count_nonzero(alpha[0, :4]) == 0
    for g in range(3):
        keys = torch.cat((torch.zeros_like(k[0, :, 0]) if g == 0 else k[g - 1, :, 0], k[g, :, 1]))
        logits = torch.cat(
            (
                torch.full_like(s[0, :, 0], -math.inf) if g == 0 else s[g - 1, :, 0] + ape[:, 0],
                s[g, :, 1] + ape[:, 1],
            )
        )
        expected = (torch.softmax(logits, dim=0) * keys).sum(0)
        torch.testing.assert_close(out[g], expected)
    out.sum().backward()
    assert torch.count_nonzero(k.grad[-1, :, 0]) == 0
    assert torch.isfinite(s.grad).all() and torch.isfinite(ape.grad).all()
    # Putting current second-half APE on the previous first half is observably wrong.
    wrong = ape.detach().clone()
    wrong[:, 0] = wrong[:, 1].flip(0)
    assert not torch.equal(oracle.c4_pool(k.detach(), s.detach(), wrong)[0][1:], out.detach()[1:])


def test_c128_forward_backward_formula():
    k = values((2, 128, 8), 1).requires_grad_()
    s = values(k.shape, 3).requires_grad_()
    ape = values(k.shape[1:], 7).requires_grad_()
    out, alpha = oracle.c128_pool(k, s, ape)
    g = values(out.shape, 5)
    out.backward(g)
    torch.testing.assert_close(k.grad, alpha * g[:, None, :])
    da = alpha * g[:, None, :] * (k.detach() - out.detach()[:, None, :])
    torch.testing.assert_close(s.grad, da, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(ape.grad, da.sum(0), atol=2e-7, rtol=2e-5)
    with pytest.raises(ContractError, match=Status.INVALID_COMPRESSION_PLAN.value):
        oracle.c128_pool(k[:, :127], s[:, :127], ape[:127])


@pytest.mark.parametrize("n", [0, 1, 511, 512, 513])
def test_icv_score_topk_shapes_and_gradients(n):
    q = (values((1, 64, 128), 2) / 32).requires_grad_()
    k = (values((n, 128), 3) / 32).requires_grad_()
    u = values((1, 64), 7).requires_grad_()
    a = oracle.icv(q, k)
    assert a.shape == (1, 64, n)
    torch.testing.assert_close(a, torch.einsum("thd,nd->thn", q, k))
    score = oracle.relu_score(a, oracle.scale(u))
    chosen, count = oracle.topk512(score, torch.ones_like(score, dtype=torch.bool), torch.arange(n))
    assert count.item() == min(n, 512)
    assert chosen.shape == (1, 512)
    score.sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()
    assert torch.isfinite(u.grad).all()


def test_topk_exact_ties_near_ties_future_padding_and_duplicate():
    ids = torch.arange(512, -1, -1)
    score = torch.ones(1, 513)
    valid = torch.ones_like(score, dtype=torch.bool)
    chosen, count = oracle.topk512(score, valid, ids)
    assert chosen[0].tolist() == list(range(512))
    assert count.item() == 512
    score[0, 0] = torch.nextafter(score[0, 0], torch.tensor(math.inf))
    assert oracle.topk512(score, valid, ids)[0][0, 0] == 512
    valid[0, 0] = False
    assert oracle.topk512(score, valid, ids)[0][0, 0] == 0
    valid[:] = False
    out, count = oracle.topk512(score, valid, ids)
    assert out.eq(-1).all() and count.item() == 0
    with pytest.raises(ContractError, match=Status.AMBIGUOUS_LOGICAL_INDEX.value):
        oracle.topk512(score, valid, torch.zeros_like(ids))


def test_relu_zero_derivative():
    a = torch.zeros(1, 64, 2, requires_grad=True)
    oracle.relu_score(a, torch.ones(1, 64)).sum().backward()
    assert a.grad.eq(0).all()


@pytest.mark.parametrize("n,sink_value", [(0, 0.0), (1, 0.0), (7, 0.0), (7, 100.0)])
def test_joint_attention_one_denominator_and_backward(n, sink_value):
    q = (values((64, 512)) / 32).requires_grad_()
    kv = (values((n, 512), 4) / 32).requires_grad_()
    sink = torch.full((64,), sink_value, requires_grad=True)
    out, saved = oracle.joint_attention(q, kv, sink)
    logits = q @ kv.T / math.sqrt(512)
    weights = torch.softmax(torch.cat((logits, sink[:, None]), -1), -1)[:, :-1]
    torch.testing.assert_close(out, weights @ kv, atol=1e-7, rtol=1e-5)
    grad = values(out.shape, 7)
    explicit = oracle.joint_attention_backward(q.detach(), kv.detach(), saved, grad)
    out.backward(grad)
    for key, actual in (("dQ", q.grad), ("dKV", kv.grad), ("dsink", sink.grad)):
        torch.testing.assert_close(explicit[key], actual, rtol=2e-5, atol=2e-6)


def test_two_softmax_is_not_joint_attention():
    q, kv, sink = values((64, 512)) / 32, values((7, 512), 3) / 32, torch.zeros(64)
    joined = oracle.joint_attention(q, kv, sink)[0]
    wrong = oracle.joint_attention(q, kv[:3], sink)[0] + oracle.joint_attention(q, kv[3:], sink)[0]
    assert not torch.equal(joined, wrong)


@pytest.mark.parametrize("salt", [0, 1, 7])
def test_projection_recipe_forward_backward_and_group_order(salt):
    x = values((2, 16)).requires_grad_()
    recipe = selector_recipe(32, 16, salt)
    w = materialize_weight(recipe).requires_grad_()
    actual = selector_linear(x, recipe)
    expected = x @ w.T
    assert torch.equal(actual, expected)
    actual.sum().backward()
    grad = x.grad.clone()
    x.grad = None
    expected.sum().backward()
    assert torch.equal(x.grad, grad)
    assert torch.equal(w.grad, x.detach().sum(0).expand_as(w))
