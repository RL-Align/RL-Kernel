# SPDX-License-Identifier: Apache-2.0
"""INK-16 native ROCm dynamic K/V convolution tests."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)

from rl_engine.backends.extension import _C, _EXT_AVAILABLE  # noqa: E402
from rl_engine.models.inkling2.dynamic_kv_conv import (  # noqa: E402
    dynamic_kv_conv,
    dynamic_kv_conv_rocm,
)

if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_dynamic_kv_conv_forward_with_state"):
    pytest.skip("Inkling 2 ROCm extension is unavailable", allow_module_level=True)

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# (batch, time, hidden): KV8 x D128 = 1024 plus awkward shapes.
SHAPES = (
    (2, 17, 1024),
    (1, 9, 2304),
    (3, 5, 5),
    (1, 1, 1),
    (2, 3, 65),
    (2, 33, 127),
    (1, 2, 64),
)


def _inputs(shape, dtype, seed=0, with_state=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b, t, h = shape
    x = torch.randn(b, t, 2, h, device="cuda", generator=gen).to(dtype)
    c = torch.randn(b, t, 2, h, 4, device="cuda", generator=gen).to(dtype)
    s = None
    if with_state:
        s = torch.randn(b, 2, h, 3, device="cuda", generator=gen).to(dtype)
    return x, c, s


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_state", [False, True])
def test_forward_and_state_bitwise_match_reference(shape, dtype, with_state):
    x, c, s = _inputs(shape, dtype, with_state=with_state)
    ref_out, ref_state = dynamic_kv_conv(x, c, s)
    out, state = dynamic_kv_conv_rocm(x, c, s)
    assert torch.equal(out, ref_out)
    assert torch.equal(state, ref_state)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sequences_shorter_than_state(dtype):
    for t in (1, 2, 3):
        x, c, s = _inputs((2, t, 70), dtype, seed=t)
        ref_out, ref_state = dynamic_kv_conv(x, c, s)
        out, state = dynamic_kv_conv_rocm(x, c, s)
        assert torch.equal(out, ref_out)
        assert torch.equal(state, ref_state)


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunked_execution_matches_full_sequence(dtype):
    x, c, _ = _inputs((2, 21, 1024), dtype)
    full, full_state = dynamic_kv_conv_rocm(x, c)
    state = None
    pieces = []
    for lo, hi in ((0, 1), (1, 5), (5, 6), (6, 15), (15, 21)):
        piece, state = dynamic_kv_conv_rocm(
            x[:, lo:hi].contiguous(), c[:, lo:hi].contiguous(), state
        )
        pieces.append(piece)
    assert torch.equal(torch.cat(pieces, dim=1), full)
    assert torch.equal(state, full_state)
    ref, ref_state = dynamic_kv_conv(x, c)
    assert torch.equal(full, ref)
    assert torch.equal(full_state, ref_state)


def test_forward_binding_matches_with_state_binding():
    x, c, s = _inputs((2, 6, 100), torch.bfloat16)
    assert torch.equal(
        _C.inkling2_dynamic_kv_conv_forward(x, c, s),
        _C.inkling2_dynamic_kv_conv_forward_with_state(x, c, s)[0],
    )


def test_empty_inputs():
    x = torch.empty(2, 0, 2, 64, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(2, 0, 2, 64, 4, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(2, 2, 64, 3, device="cuda", dtype=torch.bfloat16)
    out, state = dynamic_kv_conv_rocm(x, c, s)
    _, ref_state = dynamic_kv_conv(x, c, s)
    assert out.shape == x.shape
    assert torch.equal(state, ref_state)
    x0 = torch.empty(0, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
    c0 = torch.empty(0, 3, 2, 64, 4, device="cuda", dtype=torch.bfloat16)
    s0 = torch.empty(0, 2, 64, 3, device="cuda", dtype=torch.bfloat16)
    out0, _ = dynamic_kv_conv_rocm(x0, c0, s0)
    assert out0.shape == x0.shape


@pytest.mark.parametrize("dtype", DTYPES)
def test_run_to_run_bitwise_determinism(dtype):
    x, c, s = _inputs((3, 33, 1024), dtype)
    outs = [dynamic_kv_conv_rocm(x, c, s)[0] for _ in range(3)]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    grads = []
    for _ in range(3):
        xg = x.clone().requires_grad_()
        cg = c.clone().requires_grad_()
        o, _ = dynamic_kv_conv_rocm(xg, cg, s)
        o.float().square().sum().backward()
        grads.append((xg.grad, cg.grad))
    for gx, gc in grads[1:]:
        assert torch.equal(gx, grads[0][0]) and torch.equal(gc, grads[0][1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_invariance(dtype):
    x, c, s = _inputs((4, 11, 1024), dtype)
    full, full_state = dynamic_kv_conv_rocm(x, c, s)
    for i in range(4):
        one, one_state = dynamic_kv_conv_rocm(
            x[i : i + 1].contiguous(), c[i : i + 1].contiguous(), s[i : i + 1].contiguous()
        )
        assert torch.equal(one, full[i : i + 1])
        assert torch.equal(one_state, full_state[i : i + 1])
    x2 = x.clone()
    c2 = c.clone()
    x2[1:] = torch.randn_like(x2[1:])
    c2[1:] = torch.randn_like(c2[1:])
    out2, _ = dynamic_kv_conv_rocm(x2, c2, s)
    assert torch.equal(out2[0], full[0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 9, 70), (2, 17, 1024), (1, 1, 3)])
def test_backward_matches_reference_autograd(shape, dtype):
    x, c, s = _inputs(shape, dtype)
    results = []
    for fn in (dynamic_kv_conv, dynamic_kv_conv_rocm):
        xg = x.clone().requires_grad_()
        cg = c.clone().requires_grad_()
        sg = s.clone().requires_grad_()
        o, _ = fn(xg, cg, sg)
        coeff = torch.linspace(-1, 1, o.numel(), device="cuda").reshape(o.shape)
        (o.float() * coeff).sum().backward()
        results.append((xg.grad, cg.grad, sg.grad))
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 5e-2}[dtype]
    for ref, got in zip(*results, strict=True):
        torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol)


def test_backward_binding_gradient_shapes_and_tuple_return():
    x, c, s = _inputs((2, 5, 33), torch.float32)
    go = torch.randn_like(x)
    grads = _C.inkling2_dynamic_kv_conv_backward(go, x, c, s)
    assert len(grads) == 3
    assert grads[0].shape == x.shape
    assert grads[1].shape == c.shape
    assert grads[2].shape == s.shape


def test_rejects_cpu_tensors():
    x = torch.randn(1, 4, 2, 8)
    c = torch.randn(1, 4, 2, 8, 4)
    with pytest.raises(ValueError):
        dynamic_kv_conv_rocm(x, c)
