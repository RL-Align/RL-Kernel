# SPDX-License-Identifier: Apache-2.0
"""INK-10 native ROCm short-convolution tests."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)

from rl_engine.backends.extension import _C, _EXT_AVAILABLE  # noqa: E402
from rl_engine.models.inkling2.short_conv import (  # noqa: E402
    STREAMS,
    causal_short_conv,
    causal_short_conv_rocm,
    causal_short_conv_streams,
)

if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_short_conv_forward_with_state"):
    pytest.skip("Inkling 2 ROCm extension is unavailable", allow_module_level=True)

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# (batch, time, hidden): real Inkling hidden/Q widths plus awkward shapes.
SHAPES = (
    (2, 33, 2304),
    (1, 17, 3072),
    (2, 7, 1024),
    (3, 5, 5),
    (1, 1, 1),
    (2, 3, 65),
    (4, 129, 127),
    (1, 2, 64),
)


def _inputs(shape, dtype, seed=0, with_state=True):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    b, t, h = shape
    x = torch.randn(b, t, h, device="cuda", dtype=torch.float32, generator=gen).to(dtype)
    w = torch.randn(h, 4, device="cuda", dtype=torch.float32, generator=gen).to(dtype)
    s = None
    if with_state:
        s = torch.randn(b, h, 3, device="cuda", dtype=torch.float32, generator=gen).to(dtype)
    return x, w, s


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_state", [False, True])
def test_forward_and_state_bitwise_match_reference(shape, dtype, with_state):
    x, w, s = _inputs(shape, dtype, with_state=with_state)
    ref_out, ref_state = causal_short_conv(x, w, s)
    out, state = causal_short_conv_rocm(x, w, s)
    assert torch.equal(out, ref_out)
    assert torch.equal(state, ref_state)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sequences_shorter_than_state(dtype):
    for t in (1, 2, 3):
        x, w, s = _inputs((2, t, 70), dtype, seed=t)
        ref_out, ref_state = causal_short_conv(x, w, s)
        out, state = causal_short_conv_rocm(x, w, s)
        assert torch.equal(out, ref_out)
        assert torch.equal(state, ref_state)


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunked_execution_matches_full_sequence(dtype):
    x, w, _ = _inputs((2, 41, 2304), dtype)
    full, full_state = causal_short_conv_rocm(x, w)
    state = None
    pieces = []
    for lo, hi in ((0, 1), (1, 5), (5, 6), (6, 30), (30, 41)):
        piece, state = causal_short_conv_rocm(x[:, lo:hi].contiguous(), w, state)
        pieces.append(piece)
    assert torch.equal(torch.cat(pieces, dim=1), full)
    assert torch.equal(state, full_state)
    ref, ref_state = causal_short_conv(x, w)
    assert torch.equal(full, ref)
    assert torch.equal(full_state, ref_state)


def test_four_streams_independent():
    inputs = {n: _inputs((1, 9, 2304), torch.bfloat16, seed=i)[0] for i, n in enumerate(STREAMS)}
    weights = {
        n: _inputs((1, 9, 2304), torch.bfloat16, seed=10 + i)[1] for i, n in enumerate(STREAMS)
    }
    ref, ref_states = causal_short_conv_streams(inputs, weights)
    for name in STREAMS:
        out, state = causal_short_conv_rocm(inputs[name], weights[name])
        assert torch.equal(out, ref[name])
        assert torch.equal(state, ref_states[name])


def test_empty_inputs():
    w = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)
    x = torch.empty(2, 0, 64, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(2, 64, 3, device="cuda", dtype=torch.bfloat16)
    out, state = causal_short_conv_rocm(x, w, s)
    _, ref_state = causal_short_conv(x, w, s)
    assert out.shape == x.shape
    assert torch.equal(state, ref_state)
    x0 = torch.empty(0, 4, 64, device="cuda", dtype=torch.bfloat16)
    s0 = torch.empty(0, 64, 3, device="cuda", dtype=torch.bfloat16)
    out0, _ = causal_short_conv_rocm(x0, w, s0)
    assert out0.shape == x0.shape


@pytest.mark.parametrize("dtype", DTYPES)
def test_run_to_run_bitwise_determinism(dtype):
    x, w, s = _inputs((3, 65, 2304), dtype)
    outs = [causal_short_conv_rocm(x, w, s)[0] for _ in range(3)]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    grads = []
    for _ in range(3):
        xg = x.clone().requires_grad_()
        wg = w.clone().requires_grad_()
        o, _ = causal_short_conv_rocm(xg, wg, s)
        o.float().square().sum().backward()
        grads.append((xg.grad, wg.grad))
    for gx, gw in grads[1:]:
        assert torch.equal(gx, grads[0][0]) and torch.equal(gw, grads[0][1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_invariance(dtype):
    x, w, s = _inputs((5, 19, 2304), dtype)
    full, full_state = causal_short_conv_rocm(x, w, s)
    for i in range(5):
        one, one_state = causal_short_conv_rocm(
            x[i : i + 1].contiguous(), w, s[i : i + 1].contiguous()
        )
        assert torch.equal(one, full[i : i + 1])
        assert torch.equal(one_state, full_state[i : i + 1])
    # Perturbing other rows leaves a row unchanged.
    x2 = x.clone()
    x2[1:] = torch.randn_like(x2[1:])
    out2, _ = causal_short_conv_rocm(x2, w, s)
    assert torch.equal(out2[0], full[0])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 9, 70), (3, 40, 2304), (1, 1, 3)])
def test_backward_matches_reference_autograd(shape, dtype):
    x, w, s = _inputs(shape, dtype)
    tol = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 1e-1}[dtype]
    grads = []
    for fn in (causal_short_conv, causal_short_conv_rocm):
        xg = x.clone().requires_grad_()
        wg = w.clone().requires_grad_()
        sg = s.clone().requires_grad_()
        o, _ = fn(xg, wg, sg)
        coeff = torch.linspace(-1, 1, o.numel(), device="cuda").reshape(o.shape)
        (o.float() * coeff).sum().backward()
        grads.append((xg.grad, wg.grad, sg.grad))
    for ref, got in zip(*grads, strict=True):
        torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol)


def test_backward_fp32_matches_reference_autograd_tightly():
    x, w, s = _inputs((2, 11, 130), torch.float32)
    go = torch.randn_like(x)
    gx, gw, gs = _C.inkling2_short_conv_backward(go, x, w, s)
    xr = x.clone().requires_grad_()
    wr = w.clone().requires_grad_()
    sr = s.clone().requires_grad_()
    o, _ = causal_short_conv(xr, wr, sr)
    o.backward(go)
    torch.testing.assert_close(gx, xr.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gs, sr.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(gw, wr.grad, atol=1e-4, rtol=1e-4)


def test_rejects_cpu_tensors():
    x = torch.randn(1, 4, 8)
    w = torch.randn(8, 4)
    with pytest.raises(ValueError):
        causal_short_conv_rocm(x, w)
