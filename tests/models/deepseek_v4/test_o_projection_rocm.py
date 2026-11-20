from dataclasses import replace

import pytest
import torch

from rl_engine.backends.rocm.gemm.o_projection import AiterOProjectionBackend
from rl_engine.contracts.operators.indexer_projection import IndexerProjectionError
from rl_engine.contracts.operators.o_projection import OProjectionContract
from rl_engine.models.deepseek_v4.o_projection import OProjection, merge_tp_partials
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from .o_projection_helpers import (
    make_case,
    oracle_backward,
    oracle_forward,
    same_bytes,
    select_rows,
    rotation,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="MI300X ROCm native acceptance",
)


@pytest.fixture(scope="module")
def backend():
    return AiterOProjectionBackend()


@pytest.mark.parametrize("m", [0, 1, 3, 8, 31, 32, 33, 65])
@pytest.mark.parametrize("padding", [False, True])
def test_native_forward_backward_fp64(m, padding, backend):
    op, x, w, rows = make_case(device="cuda", m=m, train=True, padding=padding, backend=backend)
    with canonical_backward_session() as session:
        result = op(x, w, rows=rows)
        inv, z, y = oracle_forward(op.contract, x, w, rows)
        # The provider uses FP32 products; a FP64 oracle can round to the
        # adjacent BF16 value at a midpoint. Exact BI is tested separately.
        torch.testing.assert_close(result.inverse_rope, inv, rtol=0.008, atol=1e-5)
        torch.testing.assert_close(result.groups, z, rtol=0.016, atol=0.016)
        # Compare wo_b separately so BF16 boundary rounding is not conflated.
        y_at_actual_z = result.groups.flatten(1).double() @ w.b.double().t()
        torch.testing.assert_close(result.partial.double(), y_at_actual_z, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(result.partial, y, rtol=0.03, atol=0.04)
        grad = torch.randn_like(result.partial)
        want = oracle_backward(op.contract, result, w, grad)
        result.partial.backward(grad)
        session.validate_complete()
    for actual, expected in zip((x.grad, w.a.grad, w.b.grad), want, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)
    if padding and m:
        assert result.partial[-1].eq(0).all() and x.grad[-1].eq(0).all()


@pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1, 1, 1], [3, 4], [7]])
@pytest.mark.parametrize("reverse_backward", [False, True])
def test_bi_chunks_permutation_and_backward_order(chunks, reverse_backward, backend):
    op, x, w, rows = make_case(device="cuda", m=7, train=True, backend=backend, padding=True)
    dy = torch.randn(7, op.contract.hidden_size, device="cuda")
    with canonical_backward_session() as session:
        full = op(x, w, rows=rows)
        full.partial.backward(dy)
        session.validate_complete()
    expected = [x.grad.clone(), w.a.grad.clone(), w.b.grad.clone()]
    x.grad = w.a.grad = w.b.grad = None
    perm = torch.tensor([6, 4, 0, 5, 1, 3, 2], device="cuda")
    with canonical_backward_session() as session:
        pieces, start = [], 0
        for count in chunks:
            index = perm[start : start + count]
            out = op(x[index].contiguous(), w, rows=select_rows(op.contract, rows, index))
            assert same_bytes(out.partial, full.partial[index])
            assert same_bytes(out.groups, full.groups[index])
            assert same_bytes(out.inverse_rope, full.inverse_rope[index])
            pieces.append((out, index))
            start += count
        for out, index in reversed(pieces) if reverse_backward else pieces:
            out.partial.backward(dy[index])
        session.validate_complete()
    for actual, want in zip((x.grad, w.a.grad, w.b.grad), expected, strict=True):
        assert same_bytes(actual, want)
    with torch.no_grad():
        assert same_bytes(op(x, w, rows=rows).partial, full.partial)
    with torch.inference_mode():
        assert same_bytes(op(x, w, rows=rows).partial, full.partial)


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_tp_fixed_order_numerics_and_gradients(tp, backend):
    torch.manual_seed(231)
    results, params, grads = [], [], []
    for rank in range(tp):
        c = OProjectionContract(32, 128, 64, 128, 128, tp_size=tp, tp_rank=rank)
        op, x, w, rows = make_case(device="cuda", m=3, train=True, backend=backend, contract=c)
        params.append((op, x, w, rows))
    dy = torch.randn(3, 128, device="cuda")
    with canonical_backward_session() as session:
        for op, x, w, rows in params:
            result = op(x, w, rows=rows)
            results.append(result)
            grads.append(oracle_backward(op.contract, result, w, dy))
        merged = merge_tp_partials(list(reversed(results)))
        want = results[0].partial.clone()
        for result in results[1:]:
            want = want + result.partial
        assert same_bytes(merged, want)
        merged.backward(dy)
        session.validate_complete()
    for (_, x, w, _), want in zip(params, grads, strict=True):
        for a, b in zip((x.grad, w.a.grad, w.b.grad), want, strict=True):
            torch.testing.assert_close(a, b, rtol=0.03, atol=0.04)


def checkpoint(master):
    data = (master.float() * 16).to(torch.float8_e4m3fn)
    scales = torch.full(
        (*master.shape[:-2], master.shape[-2] // 128, master.shape[-1] // 128),
        1 / 16,
        device="cuda",
    )
    return data, scales


@pytest.mark.parametrize("a_fp8,b_fp8", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("master", [False, True])
def test_native_fp8_bi_and_master_vjp(a_fp8, b_fp8, master, backend):
    c = OProjectionContract(
        16,
        128,
        64,
        128,
        128,
        a_precision="block_fp8_ue8m0" if a_fp8 else "bf16",
        b_precision="block_fp8_ue8m0" if b_fp8 else "bf16",
    )
    base, x, bw, rows = make_case(device="cuda", m=7, train=master, backend=backend)
    op = OProjection(c, layer_id="layer23", backend=backend)
    rows = replace(rows, contract_fingerprint=c.fingerprint)
    w = op.prepare_weights(
        bw.a if master or not a_fp8 else None,
        bw.b if master or not b_fp8 else None,
        version="v1",
        a_checkpoint=checkpoint(bw.a.detach()) if a_fp8 else None,
        b_checkpoint=checkpoint(bw.b.detach()) if b_fp8 else None,
    )
    with canonical_backward_session() as session:
        full = op(x, w, rows=rows)
        if master:
            wanted = oracle_backward(c, full, w, torch.ones_like(full.partial))
            full.partial.sum().backward()
            session.validate_complete()
            expected_grads = [v.grad.clone() for v in (x, w.a, w.b)]
            for actual, expected in zip(expected_grads, wanted, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)
    if master:
        x.grad = w.a.grad = w.b.grad = None
        with canonical_backward_session() as session:
            chunks = []
            for start, end in ((0, 3), (3, 4), (4, 7)):
                out = op(x[start:end].contiguous(), w, rows=select_rows(c, rows, slice(start, end)))
                chunks.append(out)
            for out in reversed(chunks):
                out.partial.sum().backward()
            session.validate_complete()
        for actual, expected in zip((x.grad, w.a.grad, w.b.grad), expected_grads, strict=True):
            assert same_bytes(actual, expected)
    with torch.no_grad():
        for i in range(x.shape[0]):
            solo = op(x[i : i + 1].contiguous(), w, rows=select_rows(c, rows, slice(i, i + 1)))
            assert same_bytes(solo.groups, full.groups[i : i + 1])
            assert same_bytes(solo.partial, full.partial[i : i + 1])
    # FP8 profile is validated against dequantized *actual native activation*
    # and checkpoint, not against a falsely equivalent BF16 checkpoint.
    inv = full.inverse_rope.reshape(7, c.local_groups, c.group_width)
    if a_fp8:
        for g, pack in enumerate(w.packed_a):
            qx, sx = backend.quantize_activation(inv[:, g].contiguous())
            xd = qx.double() * sx.double().repeat_interleave(128, 1)
            wd = pack.data.double() * pack.scales.double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            expected = (xd @ wd.t()).bfloat16()
            torch.testing.assert_close(full.groups[:, g], expected, rtol=0.016, atol=0.016)
    if b_fp8:
        qx, sx = backend.quantize_activation(full.groups.flatten(1))
        xd = qx.double() * sx.double().repeat_interleave(128, 1)
        p = w.packed_b
        wd = p.data.double() * p.scales.double().repeat_interleave(128, 0).repeat_interleave(128, 1)
        torch.testing.assert_close(full.partial.double(), xd @ wd.t(), rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("fault", ["pack", "scales", "source", "master", "no_master_vjp"])
def test_fp8_staleness(fault, backend):
    op0, x, w0, rows = make_case(device="cuda", backend=backend)
    c = replace(op0.contract, a_precision="block_fp8_ue8m0")
    op = OProjection(c, layer_id="L", backend=backend)
    rows = replace(rows, contract_fingerprint=c.fingerprint)
    source = checkpoint(w0.a)
    a = None if fault == "no_master_vjp" else w0.a
    w = op.prepare_weights(a, w0.b, version="v1", a_checkpoint=source)
    if fault == "pack":
        w.packed_a[0].data.zero_()
    if fault == "scales":
        w.packed_a[0].scales.mul_(2)
    if fault == "source":
        source[0].zero_()
    if fault == "master":
        w0.a.add_(1)
    if fault == "no_master_vjp":
        x.requires_grad_()
    with pytest.raises(IndexerProjectionError):
        op(x, w, rows=rows)


@pytest.mark.parametrize(
    "geometry",
    [
        (8, 512, 64, 1024, 4096),
        (64, 512, 64, 1024, 4096),
        (16, 128, 128, 128, 128),
        (8, 128, 2, 65, 129),
    ],
)
def test_large_and_tail_geometry(geometry, backend):
    c = OProjectionContract(*geometry)
    op, x, w, rows = make_case(device="cuda", m=3, backend=backend, contract=c)
    full = op(x, w, rows=rows)
    for i in range(3):
        solo = op(x[i : i + 1].contiguous(), w, rows=select_rows(c, rows, slice(i, i + 1)))
        assert same_bytes(solo.partial, full.partial[i : i + 1])
    inv, z, y = oracle_forward(c, x, w, rows)
    torch.testing.assert_close(full.inverse_rope, inv, rtol=0.008, atol=1e-5)
    torch.testing.assert_close(full.groups, z, rtol=0.016, atol=0.02)
    torch.testing.assert_close(full.partial, y, rtol=0.04, atol=0.06)


def test_native_inverse_adjoint_prefix_and_canary(backend):
    for m in (1, 3, 8, 33):
        op, x, w, rows = make_case(device="cuda", m=m, backend=backend)
        storage = torch.full((m * 16 * 128 + 512,), 17.0, dtype=torch.bfloat16, device="cuda")
        dy = storage[256:-256].view(m, 16, 128)
        result = backend.inverse_rope_gradient(dy, rows.cos, rows.sin)
        want = rotation(dy, rows.cos, rows.sin, inverse=False).bfloat16()
        torch.testing.assert_close(result, want, rtol=0.008, atol=1e-5)
        assert storage[:256].eq(17).all() and storage[-256:].eq(17).all()
        assert same_bytes(result[..., :64], dy[..., :64])


def test_nondefault_stream_and_repeated_allocations(backend):
    op, x, w, rows = make_case(device="cuda", m=33, backend=backend)
    expected = op(x, w, rows=rows).partial
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(8):
            result = op(x, w, rows=rows).partial
    stream.synchronize()
    assert same_bytes(result, expected)


@pytest.mark.parametrize("m", [0, 1, 33])
def test_fp8_padding_and_empty(m, backend):
    base, x, bw, rows = make_case(device="cuda", m=m, backend=backend, padding=True)
    c = replace(base.contract, a_precision="block_fp8_ue8m0", b_precision="block_fp8_ue8m0")
    op = OProjection(c, layer_id="L", backend=backend)
    rows = replace(rows, contract_fingerprint=c.fingerprint)
    w = op.prepare_weights(
        None, None, version="v1", a_checkpoint=checkpoint(bw.a), b_checkpoint=checkpoint(bw.b)
    )
    out = op(x, w, rows=rows)
    assert out.partial.shape == (m, c.hidden_size)
    if m:
        assert out.partial[-1].eq(0).all() and out.groups[-1].eq(0).all()


def test_native_input_and_weight_gradient_modes(backend):
    for x_grad, a_grad, b_grad in (
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ):
        op, x, w, rows = make_case(device="cuda", m=3, backend=backend)
        x.requires_grad_(x_grad)
        w.a.requires_grad_(a_grad)
        w.b.requires_grad_(b_grad)
        with canonical_backward_session() as session:
            out = op(x, w, rows=rows)
            out.partial.sum().backward()
            session.validate_complete()
        for tensor, needs in ((x, x_grad), (w.a, a_grad), (w.b, b_grad)):
            assert (tensor.grad is not None) == needs


def test_fp8_native_fnuz_checkpoint_matches_normalized_ocp(backend):
    base, x, bw, rows = make_case(device="cuda", m=3, backend=backend)
    c = replace(base.contract, a_precision="block_fp8_ue8m0")
    op = OProjection(c, layer_id="L", backend=backend)
    rows = replace(rows, contract_fingerprint=c.fingerprint)
    w = op.prepare_weights(None, bw.b, version="v1", a_checkpoint=checkpoint(bw.a))
    expected = op(x, w, rows=rows).partial
    data = torch.stack([p.data for p in w.packed_a])
    scales = torch.stack([p.scales for p in w.packed_a])
    second = op.prepare_weights(None, bw.b, version="v1", a_checkpoint=(data, scales))
    assert same_bytes(expected, op(x, second, rows=rows).partial)
