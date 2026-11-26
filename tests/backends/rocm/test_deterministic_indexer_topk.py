# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic indexer Top-K: rule tests (CPU) and production-kernel set parity (ROCm + vLLM)."""

import importlib.util
import pathlib

import pytest
import torch

_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "rl_engine/backends/rocm/attention/deterministic_indexer_topk.py"
)
# Load by path: the attention package __init__ pulls in ROCm-only imports.
_spec = importlib.util.spec_from_file_location("deterministic_indexer_topk_under_test", _PATH)
det = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(det)


def i32(*values):
    return torch.tensor(list(values), dtype=torch.int32)


def reference_row(scores, start, end, topk):
    """Brute-force rule: largest score first, lower column on ties, relative ascending."""
    cols = list(range(start, end))
    cols.sort(key=lambda c: (-float(scores[c]), c))
    chosen = sorted(c - start for c in cols[:topk])
    return chosen + [-1] * (topk - len(chosen))


def reference_prefill(logits, starts, ends, topk):
    cpu = logits.float().cpu()
    return torch.tensor(
        [reference_row(cpu[i], int(starts[i]), int(ends[i]), topk) for i in range(cpu.shape[0])],
        dtype=torch.int32,
    )


def run_prefill(logits, starts, ends, topk):
    out = torch.full((logits.shape[0], topk), -7, dtype=torch.int32, device=logits.device)
    det.deterministic_topk_prefill(logits, starts, ends, out, topk)
    return out


def test_ties_lower_index_wins():
    # All tied except column 4 (largest); K=3 -> {4, 0, 1}.
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    assert run_prefill(logits, i32(0), i32(6), 3).tolist() == [[0, 1, 4]]


def test_fewer_candidates_than_k_padded():
    logits = torch.randn(2, 8)
    out = run_prefill(logits, i32(0, 0), i32(3, 0), 5)
    assert out.tolist() == [[0, 1, 2, -1, -1], [-1] * 5]


def test_row_start_offsets_relative():
    logits = torch.tensor([[9.0, 9.0, 1.0, 5.0, 5.0, 2.0, 9.0]])
    # valid [2, 6): scores 1,5,5,2 -> top2 are columns 3,4 -> relative 1,2.
    assert run_prefill(logits, i32(2), i32(6), 2).tolist() == [[1, 2]]


def test_invalid_columns_never_selected_even_if_larger():
    logits = torch.tensor([[100.0, 100.0, 0.0, 0.0, 100.0]])
    assert run_prefill(logits, i32(2), i32(4), 4).tolist() == [[0, 1, -1, -1]]


def test_valid_neg_inf_beats_invalid_columns():
    # Candidate masking leaves -inf inside the valid range; still selected before pads.
    ninf = float("-inf")
    logits = torch.tensor([[7.0, 7.0, ninf, 1.0, ninf]])
    assert run_prefill(logits, i32(2), i32(5), 3).tolist() == [[0, 1, 2]]


def test_nan_in_valid_range_raises():
    logits = torch.tensor([[1.0, float("nan"), 2.0]])
    with pytest.raises(RuntimeError):
        run_prefill(logits, i32(0), i32(3), 2)


def test_nan_outside_valid_range_ignored():
    logits = torch.tensor([[float("nan"), 1.0, 2.0]])
    assert run_prefill(logits, i32(1), i32(3), 2).tolist() == [[0, 1]]


def test_topk_wider_than_columns():
    logits = torch.randn(1, 4)
    assert run_prefill(logits, i32(0), i32(4), 6).tolist() == [[0, 1, 2, 3, -1, -1]]


@pytest.mark.parametrize("quantized", [False, True])
def test_random_matches_reference(quantized):
    torch.manual_seed(0)
    rows, cols, topk = 64, 90, 16
    logits = torch.randn(rows, cols)
    if quantized:
        logits = torch.relu(logits.round())  # many exact ties incl. 0.0
    starts = torch.randint(0, 30, (rows,), dtype=torch.int32)
    ends = (starts + torch.randint(0, 60, (rows,), dtype=torch.int32)).clamp(max=cols)
    expected = reference_prefill(logits, starts, ends, topk)
    assert torch.equal(run_prefill(logits, starts, ends, topk), expected)


def test_chunking_does_not_change_result(monkeypatch):
    torch.manual_seed(1)
    logits = torch.relu(torch.randn(37, 50).round())
    starts = torch.zeros(37, dtype=torch.int32)
    ends = torch.randint(0, 51, (37,), dtype=torch.int32)
    whole = run_prefill(logits, starts, ends, 8)
    monkeypatch.setattr(det, "_CHUNK_ELEMENTS", 150)  # 3 rows per chunk
    assert torch.equal(run_prefill(logits, starts, ends, 8), whole)


def test_output_sorted_ascending():
    torch.manual_seed(2)
    logits = torch.randn(10, 40)
    ends = torch.full((10,), 40, dtype=torch.int32)
    out = run_prefill(logits, torch.zeros(10, dtype=torch.int32), ends, 12)
    assert (out[:, 1:] > out[:, :-1]).all()


def test_decode_per_row_lengths():
    torch.manual_seed(3)
    logits = torch.relu(torch.randn(5, 40).round())
    lens = i32(1, 5, 12, 40, 0)
    out = torch.full((5, 8), -7, dtype=torch.int32)
    det.deterministic_topk_decode(logits, 1, lens, out, 8)
    assert torch.equal(out, reference_prefill(logits, torch.zeros(5, dtype=torch.int32), lens, 8))


def test_decode_next_n_expands_request_lengths():
    torch.manual_seed(4)
    next_n = 3
    logits = torch.relu(torch.randn(6, 30).round())
    out = torch.full((6, 4), -7, dtype=torch.int32)
    det.deterministic_topk_decode(logits, next_n, i32(10, 25), out, 4)
    row_ends = i32(8, 9, 10, 23, 24, 25)  # len - next_n + n + 1
    assert torch.equal(out, reference_prefill(logits, torch.zeros(6, dtype=torch.int32), row_ends, 4))


def test_decode_bad_lens_shape_raises():
    logits = torch.randn(4, 10)
    with pytest.raises(ValueError):
        det.deterministic_topk_decode(logits, 1, i32(3, 4), torch.empty(4, 2, dtype=torch.int32), 2)


# ---------------------------------------------------------------- GPU / vLLM parity

ON_ROCM = torch.version.hip is not None and torch.cuda.is_available()
gpu = pytest.mark.skipif(not ON_ROCM, reason="requires ROCm GPU")


def _vllm_ops():
    pytest.importorskip("vllm")
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse

    return rocm_aiter_ops, sparse


def _row_sets(indices):
    return [{v for v in row if v >= 0} for row in indices.cpu().tolist()]


def _prefill_case(case):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    if case == 0:
        # Actor shape: rows 2180, 545 columns, starts 0, ends=(i+1)//4.
        rows = 2180
        ends = torch.div(torch.arange(1, rows + 1, device=dev, dtype=torch.int32), 4, rounding_mode="floor")
        return torch.randn(rows, 545, device=dev, generator=g), torch.zeros(rows, dtype=torch.int32, device=dev), ends
    # Multi-sequence chunk: nonzero starts, some rows with > 512 candidates.
    klens, nrows = [300, 700, 650], 200
    starts, ends, off = [], [], 0
    grow = torch.arange(nrows, dtype=torch.int32)
    for klen in klens:
        starts.append(torch.full((nrows,), off, dtype=torch.int32))
        ends.append(off + ((grow + 1) * klen) // nrows)
        off += klen
    return (
        torch.randn(len(klens) * nrows, off, device=dev, generator=g),
        torch.cat(starts).to(dev),
        torch.cat(ends).to(dev),
    )


@gpu
@pytest.mark.parametrize("case", [0, 1])
def test_prefill_set_matches_production_kernels(case):
    ops, sparse = _vllm_ops()
    logits, starts, ends = _prefill_case(case)
    rows = logits.shape[0]
    mine = run_prefill(logits, starts, ends, 512)
    checked = 0
    try:  # aiter + localize (the production prefill path)
        aiter_out = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
        ops.indexer_top_k_prefill(logits, starts, ends, aiter_out, 512)
        sparse._localize_aiter_prefill_topk(aiter_out, starts)
    except Exception as exc:  # aiter unavailable on this arch
        print("aiter prefill unavailable:", repr(exc))
    else:
        assert _row_sets(mine) == _row_sets(aiter_out)
        checked += 1
    try:  # vLLM _C (writes row-relative indices directly)
        c_out = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
        torch.ops._C.top_k_per_row_prefill(
            logits, starts, ends, c_out, rows, logits.stride(0), logits.stride(1), 512
        )
    except Exception as exc:
        print("_C prefill unavailable:", repr(exc))
    else:
        assert _row_sets(mine) == _row_sets(c_out)
        checked += 1
    if not checked:
        pytest.skip("no production prefill top-k kernel available")


@gpu
def test_decode_set_matches_production_kernels():
    ops, _ = _vllm_ops()
    g = torch.Generator(device="cuda").manual_seed(1)
    rows, cols = 128, 2048
    logits = torch.randn(rows, cols, device="cuda", generator=g)
    seq_lens = torch.randint(1, cols + 1, (rows,), dtype=torch.int32, device="cuda", generator=g)
    seq_lens[:5] = torch.tensor([1, 511, 512, 513, cols], dtype=torch.int32)
    mine = torch.full((rows, 512), -7, dtype=torch.int32, device="cuda")
    det.deterministic_topk_decode(logits, 1, seq_lens, mine, 512)
    checked = 0
    try:
        a = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
        ops.indexer_top_k_decode(logits, 1, seq_lens, a, 512)
    except Exception as exc:
        print("aiter decode unavailable:", repr(exc))
    else:
        assert _row_sets(mine) == _row_sets(a)
        checked += 1
    try:
        c = torch.full((rows, 512), -1, dtype=torch.int32, device="cuda")
        torch.ops._C.top_k_per_row_decode(logits, 1, seq_lens, c, rows, logits.stride(0), logits.stride(1), 512)
    except Exception as exc:
        print("_C decode unavailable:", repr(exc))
    else:
        assert _row_sets(mine) == _row_sets(c)
        checked += 1
    if not checked:
        pytest.skip("no production decode top-k kernel available")


@gpu
def test_decode_next_n_matches_c_kernel_length_convention():
    _vllm_ops()
    g = torch.Generator(device="cuda").manual_seed(2)
    batch, next_n, cols = 16, 2, 1500
    logits = torch.randn(batch * next_n, cols, device="cuda", generator=g)
    seq_lens = torch.randint(next_n, cols + 1, (batch,), dtype=torch.int32, device="cuda", generator=g)
    mine = torch.full((batch * next_n, 512), -7, dtype=torch.int32, device="cuda")
    det.deterministic_topk_decode(logits, next_n, seq_lens, mine, 512)
    c = torch.full((batch * next_n, 512), -1, dtype=torch.int32, device="cuda")
    try:
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, seq_lens, c, batch * next_n, logits.stride(0), logits.stride(1), 512
        )
    except Exception as exc:
        pytest.skip(f"_C decode unavailable: {exc!r}")
    assert _row_sets(mine) == _row_sets(c)


@gpu
def test_gpu_ties_deterministic_and_lower_index():
    g = torch.Generator(device="cuda").manual_seed(3)
    rows, cols = 64, 545
    # ReLU-like: many exact zeros, few distinct values -> heavy ties.
    logits = torch.relu(torch.randn(rows, cols, device="cuda", generator=g).mul(2).round() - 1.0)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    first = run_prefill(logits, starts, ends, 512)
    for _ in range(20):
        assert torch.equal(run_prefill(logits, starts, ends, 512), first)
    assert torch.equal(first.cpu(), reference_prefill(logits, starts, ends, 512))
    # 513 candidates, 10 ones then zeros: the dropped key is the highest-index zero.
    row = torch.zeros(1, 513, device="cuda")
    row[0, :10] = 1.0
    out = run_prefill(row, starts[:1], torch.tensor([513], dtype=torch.int32, device="cuda"), 512)
    assert out.cpu().tolist()[0] == list(range(512))
