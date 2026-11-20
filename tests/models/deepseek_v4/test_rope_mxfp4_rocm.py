# SPDX-License-Identifier: Apache-2.0
"""Real-kernel items 11/15: byte-level BI/CP/MXFP4 and FP64 accuracy, checked separately."""

import json
import os
import subprocess
import sys

import pytest
import torch

from rl_engine.backends.rocm.attention.partial_rope import (
    AiterIndexerQBackend,
    AiterPartialRoPEBackend,
)
from rl_engine.contracts.operators.indexer_q_preprocess import IndexerQPreprocessContract
from rl_engine.contracts.operators.indexer_training import IndexerAuxContract
from rl_engine.contracts.operators.partial_rope import PartialRoPEContract, PartialRoPEError
from rl_engine.models.deepseek_v4.cp_ownership import Dsv4CpPlan
from rl_engine.models.deepseek_v4.indexer_q_preprocess import IndexerQPreprocess
from rl_engine.models.deepseek_v4.indexer_training import IndexerAuxiliary
from rl_engine.models.deepseek_v4.partial_rope import PartialRoPE
from tests.models.deepseek_v4.indexer_c4_helpers import hadamard64
from tests.models.deepseek_v4.icv_helpers import decode_fp4
from tests.models.deepseek_v4.rope_mxfp4_helpers import (
    angles,
    byte_equal,
    mxfp4_even,
    rope_fp64,
    selected,
)

pytestmark = [
    pytest.mark.ws1,
    pytest.mark.skipif(
        torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
    ),
]

TABLE = 8192
BF16 = {"rtol": 0.016, "atol": 0.016}


@pytest.fixture(scope="module")
def tables():
    return angles(TABLE, device="cuda")


@pytest.fixture(scope="module")
def rope_backend():
    return AiterPartialRoPEBackend()


@pytest.fixture(scope="module")
def q_backend():
    return AiterIndexerQBackend()


@pytest.fixture(scope="module", params=[128, 512])
def rope(request, rope_backend):
    return PartialRoPE(
        PartialRoPEContract(head_dim=request.param, rope_dim=64), backend=rope_backend
    )


@pytest.fixture(scope="module")
def q_op(q_backend):
    return IndexerQPreprocess(IndexerQPreprocessContract(num_heads=64), backend=q_backend)


def rows(t, heads, dim, *, seed=0, start=0, grad=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, heads, dim, generator=g, device="cuda").bfloat16()
    positions = start + torch.randperm(TABLE - start, generator=g, device="cuda")[:t]
    return x.requires_grad_(grad), positions


def heads_for(dim):
    return 64 if dim == 128 else 16


@pytest.mark.parametrize("t", [0, 1, 3, 33, 65])
def test_rope_nope_bytes_unchanged_and_matches_fp64(rope, tables, t):
    dim = rope.contract.head_dim
    x, positions = rows(t, heads_for(dim), dim, seed=t)
    y = rope(x, positions, *tables)
    assert y.shape == x.shape and y.dtype == torch.bfloat16
    byte_equal(y[..., :-64], x[..., :-64])
    torch.testing.assert_close(y.double(), rope_fp64(x, positions, *tables), **BF16)


def test_rope_angles_are_fp32_from_global_positions(rope, tables):
    dim = rope.contract.head_dim
    x, positions = rows(9, heads_for(dim), dim, seed=1, start=4000)
    fp32 = tuple(t.float() for t in tables)
    byte_equal(rope(x, positions, *tables), rope(x, positions, *fp32))
    shifted = rope(x, positions - 1, *tables)
    assert not torch.equal(shifted[..., -64:], rope(x, positions, *tables)[..., -64:])


def test_padding_rows_are_zero_and_gradient_free(rope, tables):
    dim = rope.contract.head_dim
    x, positions = rows(6, heads_for(dim), dim, seed=2, grad=True)
    positions[[1, 4]] = -1
    y = rope(x, positions, *tables)
    assert y[[1, 4]].eq(0).all()
    y.backward(torch.randn_like(y))
    assert x.grad[[1, 4]].eq(0).all()


def test_rope_backward_is_transpose_and_matches_627(rope, tables):
    from rl_engine.backends.rocm.gemm.o_projection import AiterOProjectionBackend

    dim = rope.contract.head_dim
    x, positions = rows(33, heads_for(dim), dim, seed=3, grad=True)
    y = rope(x, positions, *tables)
    g = torch.randn_like(y)
    y.backward(g)
    torch.testing.assert_close(
        x.grad.double(), rope_fp64(g, positions, *tables, inverse=True), **BF16
    )
    inverse = AiterOProjectionBackend()
    cos, sin = selected(tables[0], positions), selected(tables[1], positions)
    byte_equal(x.grad, inverse.inverse_rope(g, cos, sin))
    torch.testing.assert_close(inverse.inverse_rope(y.detach(), cos, sin), x.detach(), **BF16)
    byte_equal(inverse.inverse_rope_gradient(g, cos, sin), rope.backend.rotate(g, cos, sin))


def test_rope_batch_invariant(rope, tables):
    dim = rope.contract.head_dim
    x, positions = rows(65, heads_for(dim), dim, seed=4)
    full = rope(x, positions, *tables)
    for part in (slice(0, 1), slice(7, 8), slice(10, 43), slice(64, 65)):
        byte_equal(rope(x[part], positions[part], *tables), full[part])
    order = torch.randperm(65, device="cuda")
    byte_equal(rope(x[order].contiguous(), positions[order], *tables), full[order])


@pytest.mark.parametrize("ranks", [2, 4])
def test_rope_cp_shards_match_unsharded(rope, tables, ranks):
    dim = rope.contract.head_dim
    x, _ = rows(67, heads_for(dim), dim, seed=5)
    positions = torch.arange(1000, 1067, device="cuda")
    full = rope(x, positions, *tables)
    plan = Dsv4CpPlan(0, 67, tuple(range(ranks)), 4, 67, "v1")
    parts = [slice(*plan.bounds(r)) for r in plan.ranks]
    shards = [rope(x[p].contiguous(), positions[p], *tables) for p in parts]
    byte_equal(torch.cat(shards), full)


def test_rope_rejects_bad_inputs(rope, tables):
    dim = rope.contract.head_dim
    x, positions = rows(3, heads_for(dim), dim)
    with pytest.raises(PartialRoPEError, match="int64"):
        rope(x, positions.int(), *tables)
    with pytest.raises(PartialRoPEError, match="outside"):
        rope(x, positions.clamp_max(0) + TABLE, *tables)
    with pytest.raises(PartialRoPEError, match="BF16"):
        rope(x.float(), positions, *tables)
    with pytest.raises(PartialRoPEError, match="device"):
        rope(x.cpu(), positions.cpu(), *(t.cpu() for t in tables))


def staged(q_backend, q, positions, tables):
    cos, sin = selected(tables[0], positions), selected(tables[1], positions)
    masked = torch.where((positions >= 0)[:, None, None], q, 0)
    rotated = q_backend.rotate(masked, cos, sin)
    return q_backend.codec._rotate(rotated.reshape(-1, 128)).reshape_as(q)


@pytest.mark.parametrize("t", [0, 1, 3, 9, 33])
def test_indexer_q_matches_staged_and_reference(q_op, q_backend, tables, t):
    q, positions = rows(t, 64, 128, seed=10 + t)
    out = q_op(q, positions, *tables)
    assert out.values.shape == (t, 64, 128) and out.values.dtype == torch.bfloat16
    assert out.packed.shape == (t, 64, 64) and out.packed.dtype == torch.uint8
    assert out.scales.shape == (t, 64, 4) and out.scales.dtype == torch.uint8
    byte_equal(out.values, staged(q_backend, q, positions, tables))
    expected = rope_fp64(q, positions, *tables) @ hadamard64("cuda").T
    torch.testing.assert_close(out.values.double(), expected, **BF16)
    packed, scales = mxfp4_even(out.values)
    byte_equal(out.packed, packed)
    byte_equal(out.scales, scales)


def test_indexer_q_matches_631_transform_and_encode(q_op, tables):
    aux = IndexerAuxiliary(
        IndexerAuxContract(9, 0.01, 64, 8, 128, selected_capacity=2), layer_id="q-equivalence"
    )
    q, positions = rows(9, 64, 128, seed=20, grad=True)
    positions[2] = -1
    ours = q_op(q, positions, *tables)
    q631 = q.detach().clone().requires_grad_()
    theirs = aux.transform_queries(q631, positions, *tables)
    byte_equal(ours.values, theirs)
    operand = aux.prepare_operand(theirs.detach().contiguous(), role="q", version="v1")
    byte_equal(ours.packed, operand.quantized.data)
    byte_equal(ours.scales, operand.quantized.scales)
    g = torch.randn_like(theirs)
    ours.values.backward(g)
    theirs.backward(g)
    byte_equal(q.grad, q631.grad)


def test_indexer_q_identity_ste_gradient(q_op, tables):
    q, positions = rows(9, 64, 128, seed=21, grad=True)
    out = q_op(q, positions, *tables)
    assert not out.packed.requires_grad and not out.scales.requires_grad
    g = torch.randn_like(out.values)
    out.values.backward(g)
    h = hadamard64("cuda")
    expected = rope_fp64((g.double() @ h).bfloat16(), positions, *tables, inverse=True)
    torch.testing.assert_close(q.grad.double(), expected, **BF16)


def test_indexer_q_decoded_values_are_the_quantized_operand(q_op, tables):
    q, positions = rows(5, 64, 128, seed=22)
    out = q_op(q, positions, *tables)
    decoded = decode_fp4(out.packed, out.scales, 128)
    error = (decoded - out.values.cpu().double()).abs()
    bound = torch.exp2(out.scales.cpu().double() - 127).repeat_interleave(32, -1)
    assert bool((error <= bound + 1e-30).all())


def test_indexer_q_batch_invariant_and_cp(q_op, tables):
    q, _ = rows(41, 64, 128, seed=23)
    positions = torch.arange(500, 541, device="cuda")
    full = q_op(q, positions, *tables)
    order = torch.randperm(41, device="cuda")
    shuffled = q_op(q[order].contiguous(), positions[order], *tables)
    for part in (slice(0, 1), slice(3, 12), slice(40, 41)):
        piece = q_op(q[part].contiguous(), positions[part], *tables)
        for name in ("values", "packed", "scales"):
            byte_equal(getattr(piece, name), getattr(full, name)[part])
            byte_equal(getattr(shuffled, name), getattr(full, name)[order])
    for ranks in (2, 4):
        plan = Dsv4CpPlan(0, 41, tuple(range(ranks)), 4, 41, "v1")
        parts = [slice(*plan.bounds(r)) for r in plan.ranks]
        shards = [q_op(q[p].contiguous(), positions[p], *tables) for p in parts]
        for name in ("values", "packed", "scales"):
            byte_equal(torch.cat([getattr(s, name) for s in shards]), getattr(full, name))


@pytest.mark.parametrize(
    "amax, scale_byte, code",
    [(6.5, 127, 7), (4.0, 127, 6), (8.0, 128, 6), (6.0, 127, 7), (0.0, 0, 0), (2.0**-130, 0, 0)],
)
def test_mxfp4_extreme_groups(q_backend, q_op, amax, scale_byte, code):
    values = torch.zeros(2, 128, dtype=torch.bfloat16, device="cuda")
    values[0, 5] = amax
    values[1, 37] = -amax
    data, scales = q_backend.codec.quantize(values, q_op.contract)
    ref_data, ref_scales = mxfp4_even(values)
    byte_equal(data, ref_data)
    byte_equal(scales, ref_scales)
    assert scales[0, 0].item() == scale_byte and scales[1, 1].item() == scale_byte
    assert scales[0, 1:].eq(0).all() and scales[1, 0].item() == 0
    assert (data[0, 2] >> 4).item() == code and (data[1, 18] >> 4).item() == (code | 8)


def test_mxfp4_rne_ties_and_nibble_order(q_backend, q_op):
    grid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, 6.0, 0.5])
    values = torch.zeros(1, 128)
    values[0, : grid.numel()] = grid
    values = values.bfloat16().cuda()
    data, scales = q_backend.codec.quantize(values, q_op.contract)
    byte_equal(data, mxfp4_even(values)[0])
    assert scales[0, 0].item() == 127
    codes = torch.stack((data[0, :5] & 15, data[0, :5] >> 4), -1).flatten().tolist()
    assert codes == [0, 2, 2, 4, 4, 6, 6, 8, 7, 1]


def test_fresh_processes_with_distinct_jit_caches_produce_identical_bytes(tmp_path):
    outputs = []
    for run in range(2):
        output = tmp_path / f"run{run}.json"
        env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / f"triton{run}"))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.models.deepseek_v4.rope_mxfp4_process_probe",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(json.loads(output.read_text()))
    assert outputs[0] == outputs[1]
