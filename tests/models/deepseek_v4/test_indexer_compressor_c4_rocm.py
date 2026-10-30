# SPDX-License-Identifier: Apache-2.0
"""Actual upstream MI300X kernels; exact BI tests separated from numerical error."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from rl_engine.backends.rocm.attention.indexer_compressor_c4 import AiterIndexerC4Backend
from rl_engine.backends.rocm.attention.kv_compressor_c4 import AiterC4Backend
from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.contracts.operators.kv_compressor_c4 import C4Error
from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c4_helpers import byte_equal
from tests.models.deepseek_v4.indexer_c4_helpers import fixture, pre_rotation_reference, hadamard64

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]


@pytest.fixture(scope="module")
def backend():
    return AiterIndexerC4Backend()


def operator(backend, precision="mxfp4"):
    return IndexerC4Compressor(IndexerC4Contract(precision=precision), backend=backend)


def assert_bf16_close(actual, exact):
    exponent = torch.floor(torch.log2(exact.abs().clamp_min(2.0**-126)))
    tolerance = 2 * torch.exp2(exponent - 7) + 1e-5
    error = (actual.double() - exact).abs()
    assert bool((error <= tolerance).all()), (error.max().item(), tolerance.max().item())


def assert_quantization(out):
    x = out.values.double()
    if out.precision == "mxfp4":
        assert out.packed.dtype == out.scales.dtype == torch.uint8
        assert bool((out.scales != 255).all())
        table = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float64,
            device=x.device,
        )
        code = torch.stack((out.packed & 15, out.packed >> 4), -1).flatten(-2)
        scale = torch.exp2(out.scales.double() - 127).repeat_interleave(32, -1)
        decoded = table[code.long()] * scale
        # Independent nearest-representable check for the emitted E8M0 scale,
        # including saturation. Does not call the upstream quantizer again.
        distance = (x[..., None] - scale[..., None] * table).abs().amin(-1)
        torch.testing.assert_close((decoded - x).abs(), distance, rtol=0, atol=1e-10)
    else:
        assert out.packed.dtype == torch.float8_e4m3fnuz
        assert out.scales.dtype == torch.float32
        assert bool((torch.frexp(out.scales)[0] == 0.5).all())
        assert bool((out.scales > 0).all())
        expected = (out.values.float() / out.scales).clamp(-240, 240).to(torch.float8_e4m3fnuz)
        byte_equal(out.packed, expected)


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
@pytest.mark.parametrize("groups", [0, 1, 2, 31, 65])
def test_real_compression_rotation_and_quantized_layout(backend, precision, groups):
    op = operator(backend, precision)
    x, ids, a, n, co, si = fixture(groups, device="cuda")
    b = op.bind(x, ids, version="v1")
    out = op(b, a, n, co, si, version="v1")
    assert out.values.shape == (groups, 128)
    assert not out.packed.requires_grad and not out.scales.requires_grad
    if not groups:
        assert out.scales.shape == (0, 4 if precision == "mxfp4" else 1)
        return
    before_h = AiterC4Backend.forward(backend, b, a, n, co, si, op.contract)
    exact = torch.stack(
        [
            pre_rotation_reference(ctx.double(), a.double(), n.double(), group, co, si, 1e-6)
            for ctx, (_r, group) in zip(b.context_values, b.logical_keys, strict=True)
        ]
    )
    assert_bf16_close(before_h, exact)
    assert_bf16_close(out.values, before_h.double() @ hadamard64(x.device).T)
    assert_quantization(out)


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
@pytest.mark.parametrize("seed", range(5))
def test_exact_bytes_across_chunks_permutations_padding_and_cp(backend, precision, seed):
    op = operator(backend, precision)
    x, ids, a, n, co, si = fixture(17, device="cuda", seed=seed)
    baseline = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1")
    perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed)).cuda()
    extra = torch.full((3, 512), torch.nan, dtype=x.dtype, device=x.device)
    padded_x = torch.cat((x[perm], extra))
    padded_ids = torch.cat((ids[perm], torch.full((3, 2), -1, dtype=ids.dtype, device=x.device)))
    owners = torch.cat(
        (ids[perm, 1].remainder(3), torch.zeros(3, dtype=torch.int64, device=x.device))
    )
    seen = set()
    for rank in range(3):
        b = op.bind(padded_x, padded_ids, owners=owners, cp_size=3, rank=rank, version="v1")
        out = op(b, a, n, co, si, version="v1")
        for i, key in enumerate(b.logical_keys):
            assert key not in seen
            seen.add(key)
            g = key[1]
            for attr in ("values", "packed", "scales"):
                byte_equal(getattr(out, attr)[i], getattr(baseline, attr)[g])
    assert len(seen) == 17
    for first, end in [(0, 1), (1, 4), (4, 17)]:
        lo = max(0, first - 1)
        b = op.bind(
            x[lo * 4 : end * 4],
            ids[lo * 4 : end * 4],
            version="v1",
            emit_group_ids=[(7, g) for g in range(first, end)],
        )
        out = op(b, a, n, co, si, version="v1")
        for attr in ("values", "packed", "scales"):
            byte_equal(getattr(out, attr), getattr(baseline, attr)[first:end])


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_train_and_rollout_expose_identical_bytes_and_scales(backend, precision):
    op = operator(backend, precision)
    x, ids, a, n, co, si = fixture(4, device="cuda", requires_grad=True)
    b = op.bind(x, ids, version="v1")
    with torch.no_grad():
        rollout = op(b, a, n, co, si, version="v1")
    with canonical_backward_session() as session:
        train = op(b, a, n, co, si, version="v1", parameter_id="indexer")
        for attr in ("values", "packed", "scales"):
            byte_equal(getattr(train, attr), getattr(rollout, attr))
        train.values.sum().backward()
        session.validate_complete()
    assert x.grad is not None and a.grad is not None and n.grad is not None


def training_run(backend, chunks, order, seed):
    op = operator(backend)
    x, ids, a, n, co, si = fixture(8, device="cuda", requires_grad=True, seed=seed)
    grad = torch.randn((8, 128), generator=torch.Generator().manual_seed(808)).bfloat16().cuda()
    with canonical_backward_session() as session:
        calls = []
        for first, end in chunks:
            lo = max(0, first - 1)
            b = op.bind(
                x[lo * 4 : end * 4],
                ids[lo * 4 : end * 4],
                version="v1",
                emit_group_ids=[(7, g) for g in range(first, end)],
            )
            out = op(b, a, n, co, si, version="v1", parameter_id="layer1")
            calls.append((out.values, grad[first:end]))
        for i in order:
            calls[i][0].backward(calls[i][1])
        session.validate_complete()
    return (x.grad, a.grad, n.grad), (x, ids, a, n, co, si, grad)


@pytest.mark.parametrize("seed", [0, 20, 73])
def test_real_vjp_exact_across_microbatches_and_independent_fp64(backend, seed):
    baseline, inputs = training_run(backend, [(0, 8)], [0], seed)
    for chunks, order in [([(0, 4), (4, 8)], [1, 0]), ([(4, 8), (0, 2), (2, 4)], [0, 2, 1])]:
        actual, _ = training_run(backend, chunks, order, seed)
        for a, b in zip(actual, baseline, strict=True):
            byte_equal(a, b)
    x, ids, a, n, co, si, grad = inputs
    # Check the explicit BF16 H-transpose dY boundary independently; keep the
    # compressor reference itself FP64 for its forward and gradient arithmetic.
    unrotated = backend._rotate(grad)
    assert_bf16_close(unrotated, grad.double() @ hadamard64(x.device).T)
    xd, ad, nd = [t.detach().double().requires_grad_() for t in (x, a, n)]
    outputs = []
    for g in range(8):
        ctx = torch.cat(
            (
                torch.zeros_like(xd[:4]) if g == 0 else xd[(g - 1) * 4 : g * 4],
                xd[g * 4 : (g + 1) * 4],
            ),
            0,
        )
        outputs.append(pre_rotation_reference(ctx, ad, nd, g, co, si, 1e-6))
    expected = torch.autograd.grad(torch.stack(outputs), (xd, ad, nd), unrotated.double())
    torch.testing.assert_close(baseline[0].double(), expected[0], rtol=0.0045, atol=3e-5)
    for actual, exact in zip(baseline[1:], expected[1:], strict=True):
        torch.testing.assert_close(actual.double(), exact, rtol=3e-4, atol=5e-5)


def test_actual_native_calls_no_matmul_fallback(backend, monkeypatch):
    op = operator(backend)
    x, ids, a, n, co, si = fixture(4, device="cuda")
    b = op.bind(x, ids, version="v1")
    observed = {"rotate": 0, "quant": 0}
    rotate, quant = backend.rotate_activation, backend.quant_mxfp4

    def counted_rotate(*args, **kwargs):
        observed["rotate"] += 1
        return rotate(*args, **kwargs)

    def counted_quant(*args, **kwargs):
        observed["quant"] += 1
        return quant(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("generic matmul is not the native Hadamard provider")

    with monkeypatch.context() as patch:
        patch.setattr(backend, "rotate_activation", counted_rotate)
        patch.setattr(backend, "quant_mxfp4", counted_quant)
        patch.setattr(torch, "matmul", forbidden)
        patch.setattr(torch.Tensor, "__matmul__", forbidden)
        op(b, a, n, co, si, version="v1")
    assert observed == {"rotate": 1, "quant": 1}


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 8, 9, 15, 17, 65])
def test_native_hadamard_tail_uses_full_wave_allocations_and_preserves_live_outputs(
    backend, monkeypatch, rows
):
    rotate = backend.rotate_activation
    shapes = []

    def checked(out, inp):
        assert out.shape == inp.shape and inp.shape[0] % 8 == 0
        shapes.append(tuple(inp.shape))
        return rotate(out, inp)

    generator = torch.Generator().manual_seed(rows)
    x = torch.randn((rows, 128), generator=generator).bfloat16().cuda()
    with monkeypatch.context() as patch:
        patch.setattr(backend, "rotate_activation", checked)
        previous = backend._rotate(x)
        saved = previous.clone()
        canary = torch.full((9, 128), 3.25, dtype=torch.bfloat16, device="cuda")
        for count in (1, 7, 9, 17):
            backend._rotate(torch.ones((count, 128), dtype=x.dtype, device=x.device))
        torch.cuda.synchronize()
        byte_equal(previous, saved)
        assert bool((canary == 3.25).all())
    assert shapes[0] == ((rows + 7) // 8 * 8, 128)
    assert_bf16_close(previous, x.double() @ hadamard64(x.device).T)


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_zero_rows_first_group_extreme_gate_and_stream_reuse(backend, precision):
    op = operator(backend, precision)
    x, ids, a, n, co, si = fixture(4, device="cuda")
    x.zero_()
    x[:, 256:] = -(2.0**30)
    a.zero_()
    b = op.bind(x, ids, version="v1")
    baseline = op(b, a, n, co, si, version="v1")
    assert bool((baseline.values == 0).all())
    assert_quantization(baseline)
    producer = torch.cuda.current_stream()
    for stream in [torch.cuda.Stream(), torch.cuda.Stream()]:
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            out = op(b, a, n, co, si, version="v1")
        producer.wait_stream(stream)
        for attr in ("values", "packed", "scales"):
            byte_equal(getattr(out, attr), getattr(baseline, attr))


def test_graph_rejected_and_predecessor_mutations_detected(backend):
    op = operator(backend)
    x, ids, a, n, co, si = fixture(2, device="cuda")
    b = op.bind(x, ids, version="v1")
    with pytest.raises(C4Error, match="Graph"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            op(b, a, n, co, si, version="v1")
    b.context_values.zero_()
    with pytest.raises(C4Error, match="storage changed"):
        op(b, a, n, co, si, version="v1")


def test_fresh_process_fingerprints(backend, tmp_path):
    del backend
    script = Path(__file__).with_name("indexer_c4_process_probe.py")
    records = []
    for index in range(2):
        output = tmp_path / f"record-{index}.json"
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
            "TRITON_CACHE_DIR": str(tmp_path / f"triton-{index}"),
        }
        proc = subprocess.run(
            [sys.executable, str(script), str(output)],
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr[-8000:]
        records.append(json.loads(output.read_text()))
    assert records[0] == records[1]
    (tmp_path / "process-evidence.json").write_text(json.dumps(records, indent=2))


@pytest.mark.parametrize("precision", ["mxfp4", "fp8"])
def test_nonzero_global_positions_across_cache_page_boundary(backend, precision):
    op = operator(backend, precision)
    x, ids, a, n, co, si = fixture(67, device="cuda")
    baseline = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1")
    b = op.bind(
        x[62 * 4 : 67 * 4],
        ids[62 * 4 : 67 * 4],
        version="v1",
        emit_group_ids=[(7, g) for g in range(63, 67)],
    )
    out = op(b, a, n, co, si, version="v1")
    assert b.cache_slots_host == ((0, 63), (1, 0), (1, 1), (1, 2))
    for attr in ("values", "packed", "scales"):
        byte_equal(getattr(out, attr), getattr(baseline, attr)[63:67])


def test_native_main_and_indexer_gradients_are_isolated_in_one_session(backend):
    from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract
    from rl_engine.models.deepseek_v4.kv_compressor_c4 import C4Compressor
    from tests.models.deepseek_v4.c4_helpers import fixture as main_fixture

    main = C4Compressor(C4Contract(), backend=AiterC4Backend())
    index = operator(backend)
    m = main_fixture(2, device="cuda", requires_grad=True)
    i = fixture(2, device="cuda", requires_grad=True)
    with canonical_backward_session() as session:
        results = []
        for op, (x, ids, a, n, co, si) in [(main, m), (index, i)]:
            results.append(
                op(
                    op.bind(x, ids, version="v1"),
                    a,
                    n,
                    co,
                    si,
                    version="v1",
                    parameter_id="same-layer",
                )
            )
        results[1].values.sum().backward()
        assert all(t.grad is None for t in (m[0], m[2], m[3]))
        saved = [t.grad.clone() for t in (i[0], i[2], i[3])]
        results[0].values.sum().backward()
        session.validate_complete()
        assert len(session.uses) == 4
    for t, expected in zip((i[0], i[2], i[3]), saved, strict=True):
        byte_equal(t.grad, expected)


def test_native_vjp_finite_difference_ape_and_norm(backend):
    op = operator(backend)
    x, ids, a, n, co, si = fixture(1, device="cuda", requires_grad=True)
    grad = torch.randn((1, 128), generator=torch.Generator().manual_seed(13)).bfloat16().cuda()
    with canonical_backward_session() as session:
        out = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1", parameter_id="fd")
        out.values.backward(grad)
        session.validate_complete()
    context = torch.cat((torch.zeros_like(x), x), 0).detach().double()
    ad, nd = a.detach().double(), n.detach().double()
    dg = backend._rotate(grad)[0].double()

    def loss():
        return (pre_rotation_reference(context, ad, nd, 0, co, si, 1e-6) * dg).sum()

    for target, idx, analytic in [
        (ad, (0, 128), a.grad[0, 128]),
        (ad, (3, 255), a.grad[3, 255]),
        (nd, (0,), n.grad[0]),
        (nd, (127,), n.grad[127]),
    ]:
        epsilon = 1e-5
        value = target[idx].item()
        target[idx] = value + epsilon
        plus = loss()
        target[idx] = value - epsilon
        minus = loss()
        target[idx] = value
        torch.testing.assert_close(
            analytic.double(), (plus - minus) / (2 * epsilon), rtol=5e-4, atol=3e-5
        )


def test_inference_empty_autograd_and_workspace_reuse(backend, tmp_path):
    import gc

    op = operator(backend)
    with torch.inference_mode():
        x, ids, a, n, co, si = fixture(0, device="cuda")
        out = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1")
        assert out.values.shape == (0, 128)
    x, ids, a, n, co, si = fixture(0, device="cuda", requires_grad=True)
    with canonical_backward_session() as session:
        out = op(op.bind(x, ids, version="v1"), a, n, co, si, version="v1", parameter_id="empty")
        out.values.sum().backward()
        session.validate_complete()
        assert n.grad is not None and bool((n.grad == 0).all())
    x, ids, a, n, co, si = fixture(17, device="cuda")
    b = op.bind(x, ids, version="v1")
    for _ in range(3):
        op(b, a, n, co, si, version="v1")
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(50):
        op(b, a, n, co, si, version="v1")
    gc.collect()
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after <= before + 512 * 1024
    (tmp_path / "allocation-evidence.json").write_text(
        json.dumps(
            {
                "calls": 50,
                "live_before": before,
                "live_after": after,
            },
            indent=2,
        )
    )
