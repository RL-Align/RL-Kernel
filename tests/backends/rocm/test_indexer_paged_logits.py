# SPDX-License-Identifier: Apache-2.0
"""C4 indexer paged logits: paging math (CPU) and bitwise invariance (ROCm + vLLM)."""

import importlib.util
import pathlib

import pytest
import torch

_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "rl_engine/backends/rocm/attention/indexer_paged_logits.py"
)
# Load by path: the attention package __init__ pulls in ROCm-only imports.
_spec = importlib.util.spec_from_file_location("indexer_paged_logits_under_test", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

gpu = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


# ---- CPU: paging math ----


def test_segment_layout_groups_by_start():
    starts = torch.tensor([0, 0, 0, 10, 10], dtype=torch.int32)
    ends = torch.tensor([3, 5, 4, 12, 20], dtype=torch.int32)
    seg_id, seg_start, seg_end = mod.segment_layout(starts, ends)
    assert seg_id.tolist() == [0, 0, 0, 1, 1]
    assert seg_start.tolist() == [0, 10]
    assert seg_end.tolist() == [5, 20]


def test_page_count():
    assert [mod.page_count(n) for n in (0, 1, 64, 65, 128, 129)] == [1, 1, 1, 2, 2, 3]


@pytest.mark.parametrize("seed", [None, 3])
def test_build_paged_cache_layout(seed):
    torch.manual_seed(0)
    n = 300
    k = torch.randint(0, 256, (n, 128), dtype=torch.uint8)
    scale = torch.rand(n, dtype=torch.float32) + 0.5
    seg_start, seg_end = torch.tensor([0, 100]), torch.tensor([70, 300])
    cache, table = mod.build_paged_cache(k, scale, seg_start, seg_end, placement_seed=seed, device="cpu")
    assert cache.shape == (2 + 4, 64, 132) and table.shape == (2, 4)
    assert sorted(table[0, :2].tolist() + table[1, :4].tolist()) == list(range(6))
    flat = cache.view(cache.shape[0], -1)
    for s, (start, end) in enumerate(zip(seg_start.tolist(), seg_end.tolist())):
        for j in range(end - start):
            page, row = int(table[s, j // 64]), j % 64
            assert torch.equal(flat[page, row * 128 : (row + 1) * 128], k[start + j])
            sc = flat[page, 8192 + row * 4 : 8192 + row * 4 + 4].view(torch.float32)
            assert sc.item() == scale[start + j].item()
        # padding rows of the last page are zero
        used = (end - start - 1) % 64 + 1
        last = int(table[s, (end - start - 1) // 64])
        assert not flat[last, used * 128 : 8192].any()
        assert not flat[last, 8192 + used * 4 :].any()


# ---- GPU ----


def _fp8():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import FP8_DTYPE

    return FP8_DTYPE


def _operands(n, rows, heads=64, dim=128, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(rows, heads, dim, device="cuda", generator=g) * 2).to(_fp8())
    k = (torch.randn(n, dim, device="cuda", generator=g) * 2).to(_fp8())
    scale = torch.rand(n, device="cuda", generator=g) * 0.05 + 0.001
    weights = torch.randn(rows, heads, device="cuda", generator=g) * 0.1
    return q, k, scale, weights


def _causal(rows):
    ends = torch.div(torch.arange(1, rows + 1, device="cuda", dtype=torch.int32), 4, rounding_mode="floor")
    return torch.zeros_like(ends), ends


def _direct(q, k, scale, weights, ends, n_splits):
    """Launch the vLLM kernel itself on an equivalent paged cache (identity placement)."""
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as sparse

    rows, n = q.shape[0], k.shape[0]
    pages = (n + 63) // 64
    cache = torch.zeros(pages, 64, 132, dtype=torch.uint8, device="cuda")
    flat = cache.view(pages, -1)
    kb = torch.zeros(pages * 64, 128, dtype=torch.uint8, device="cuda")
    kb[:n] = k.view(torch.uint8)
    sb = torch.zeros(pages * 64, dtype=torch.float32, device="cuda")
    sb[:n] = scale
    flat[:, :8192] = kb.view(pages, 8192)
    flat[:, 8192:] = sb.view(torch.uint8).view(pages, 256)
    kv_val, kv_scale = flat.view(sparse.FP8_DTYPE), flat.view(torch.float32)
    table = torch.arange(pages, dtype=torch.int32, device="cuda").expand(rows, -1)
    out = torch.full((rows, n), float("-inf"), device="cuda")
    q4 = q.unsqueeze(1)
    sparse._fp8_paged_mqa_logits_decode_kernel[(rows, n_splits)](
        q4, kv_val, kv_scale, weights, ends, table, out,
        q4.stride(0), q4.stride(1), q4.stride(2), weights.stride(0),
        kv_val.stride(0), kv_scale.stride(0), (64 * 128) // 4,
        table.stride(0), out.stride(0), pages, n,
        NUM_HEADS=q.shape[1], HEAD_SIZE=128, BLOCK_SIZE=64, BLOCK_KV=64,
        N_SPLITS=n_splits, NEXT_N=1, CTX_PER_ROW=True, num_warps=4, num_stages=2,
    )  # fmt: skip
    return out


@gpu
@pytest.mark.parametrize("n", [1, 63, 64, 65, 700, 2000])
def test_contiguous_matches_vllm_decode_kernel(n):
    rows = 4 * n
    q, k, scale, weights = _operands(n, rows)
    starts, ends = _causal(rows)
    got = mod.paged_c4_logits_contiguous(q, weights, k, scale.view(-1, 1), starts, ends)
    expected = _direct(q, k, scale, weights, ends, mod.N_SPLITS)
    bits(got, expected[:, : got.shape[1]])


@gpu
def test_invariance_batch_placement_splits():
    n = 1500
    rows = 4 * n
    q, k, scale, weights = _operands(n, rows, seed=1)
    starts, ends = _causal(rows)
    full = mod.paged_c4_logits_contiguous(q, weights, k, scale, starts, ends)
    # late rows (longest contexts) in batches of 1 and 7
    for batch in (1, 7):
        for lo in range(rows - 63, rows, batch):
            hi = min(rows, lo + batch)
            part = mod.paged_c4_logits_contiguous(
                q[lo:hi], weights[lo:hi], k, scale, starts[lo:hi], ends[lo:hi]
            )
            bits(part, full[lo:hi, : part.shape[1]])
    bits(mod.paged_c4_logits_contiguous(q, weights, k, scale, starts, ends, placement_seed=11), full)
    for splits in (1, 3, 32):
        bits(mod.paged_c4_logits_contiguous(q, weights, k, scale, starts, ends, n_splits=splits), full)


@gpu
def test_multi_segment_absolute_columns():
    # two "requests" in one key buffer, as the rollout prefill gathers them
    n0, n1 = 300, 500
    q, k, scale, weights = _operands(n0 + n1, 40, seed=2)
    ks = torch.tensor([0] * 20 + [n0] * 20, dtype=torch.int32, device="cuda")
    ends = list(range(281, 301)) + list(range(n0 + 481, n0 + 501))
    ke = torch.tensor(ends, dtype=torch.int32, device="cuda")
    absolute = mod.paged_c4_logits_contiguous(
        q, weights, k, scale, ks, ke, absolute_columns=True, num_columns=n0 + n1
    )
    assert absolute.shape == (40, n0 + n1)
    for r in (0, 19, 20, 39):
        s, e = int(ks[r]), int(ke[r])
        zero = torch.zeros(1, dtype=torch.int32, device="cuda")
        single = mod.paged_c4_logits_contiguous(
            q[r : r + 1], weights[r : r + 1], k[s:e], scale[s:e], zero, zero + (e - s)
        )
        bits(absolute[r, s:e], single[0, : e - s])
        assert torch.isinf(absolute[r, :s]).all() and torch.isinf(absolute[r, e:]).all()


@gpu
def test_weights_proj_det_gemm_m_invariant():
    from rl_engine.backends.rocm.attention.compressor_projection import compressor_projection

    torch.manual_seed(5)
    x = torch.randn(2000, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) * 0.05
    full = compressor_projection(x, w).to(torch.bfloat16)
    for m in (1, 3, 64, 2000):
        for lo in (0, 2000 - m):
            part = compressor_projection(x[lo : lo + m].contiguous(), w).to(torch.bfloat16)
            bits(part, full[lo : lo + m])
