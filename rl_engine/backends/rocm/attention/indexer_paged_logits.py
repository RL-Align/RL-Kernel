# SPDX-License-Identifier: Apache-2.0
"""C4 indexer logits through vLLM's paged decode Triton kernel, for actor and rollout alike.

Decode scores keys with ``_fp8_paged_mqa_logits_decode_kernel`` (k-scale applied after the
head sum). The flydsl prefill kernel applies it per head before ReLU, so prefill and decode
rounded differently and Top-512 flipped rarely. Routing every prefill (actor and rollout)
through this one kernel makes each (query, key) logit a pure function of the Q row, the
weights row, and the key bytes/scale: tiles are disjoint 64-key pages, so neither N_SPLITS,
the batch, nor page placement changes the bits.
"""

import inspect

import torch

PAGE = 64
HEAD_DIM = 128
ROW_BYTES = HEAD_DIM + 4
# Fixed (never shape-derived) so that every caller launches the same grid recipe.
N_SPLITS = 8
BLOCK_KV = 64
NUM_WARPS = 4
NUM_STAGES = 2

_KERNEL_PARAMS = [
    "q_ptr", "kv_val_ptr", "kv_scale_ptr", "weights_ptr", "ctx_lens_ptr", "block_tables_ptr",
    "logits_ptr", "stride_q_b", "stride_q_n", "stride_q_h", "stride_w_row", "stride_kvblk_fp8",
    "stride_kvblk_f32", "scale_region_off", "stride_bt_b", "stride_logits_row", "max_blocks",
    "max_model_len", "NUM_HEADS", "HEAD_SIZE", "BLOCK_SIZE", "BLOCK_KV", "N_SPLITS", "NEXT_N",
    "CTX_PER_ROW",
]  # fmt: skip
_KERNEL_LINES = (
    "for kv_start in tl.range(split * BLOCK_KV, seq_len, N_SPLITS * BLOCK_KV):",
    "logical_blk = kv_start // BLOCK_SIZE",
    "val_off = page * stride_kvblk_fp8 + pos_in_blk[:, None] * HEAD_SIZE + d[None, :]",
    "kv_scale_ptr + page * stride_kvblk_f32 + scale_region_off + pos_in_blk",
    "s = tl.dot(kv, tl.trans(q))",
    "s = tl.maximum(s, 0.0)",
    "s = s * w[None, :]",
    "s = tl.sum(s, axis=1)",
    "s = s * sc",
    "seq_len = tl.load(ctx_lens_ptr + row)",
)
_LAUNCHER_LINES = (
    "BLOCK_KV = 64",
    "(block_size * head_size) // 4,",
    "NEXT_N=next_n,",
    "CTX_PER_ROW=ctx_per_row,",
    "num_warps=4,",
    "num_stages=2,",
)
_CHECKED = {}


def _sparse_module():
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as sparse

    return sparse


def _kernel():
    """vLLM decode kernel, after a structural check that it is the one this module mirrors."""
    sparse = _sparse_module()
    kernel = sparse._fp8_paged_mqa_logits_decode_kernel
    if id(kernel) not in _CHECKED:
        source = inspect.getsource(kernel.fn)
        params = list(inspect.signature(kernel.fn).parameters)
        launcher = inspect.getsource(sparse.rocm_fp8_paged_mqa_logits_triton)
        missing = [x for x in _KERNEL_LINES if x not in source]
        missing += [x for x in _LAUNCHER_LINES if x not in launcher]
        if params != _KERNEL_PARAMS or missing:
            raise RuntimeError(f"vLLM paged MQA logits kernel ABI changed: {params} {missing}")
        _CHECKED[id(kernel)] = True
    return kernel


def _fp8_dtype():
    return _sparse_module().FP8_DTYPE


def paged_c4_logits(q_fp8, weights, cache_u8, block_table, ctx_lens, max_len, out, *, n_splits=N_SPLITS):
    """Logits [R, max_len] into ``out``; columns >= ctx are -inf.

    q_fp8 [R,H,D] fp8, weights [R,H] fp32, cache_u8 [P,64,132] uint8 block-flat
    (64*128 fp8 bytes then 64 fp32 scales per page), block_table [R, max_pages] int32
    (column stride 1; row stride may be 0), ctx_lens [R] int32.
    """
    kernel = _kernel()
    rows, heads, dim = q_fp8.shape
    if (
        dim != HEAD_DIM
        or heads & (heads - 1)
        or q_fp8.dtype != _fp8_dtype()
        or q_fp8.stride(-1) != 1
        or weights.dtype != torch.float32
        or weights.shape != (rows, heads)
        or weights.stride(-1) != 1
        or cache_u8.dtype != torch.uint8
        or cache_u8.shape[1:] != (PAGE, ROW_BYTES)
        or not cache_u8.is_contiguous()
        or block_table.dtype != torch.int32
        or block_table.shape[0] != rows
        or block_table.stride(1) != 1
        or ctx_lens.dtype != torch.int32
        or ctx_lens.shape != (rows,)
        or out.dtype != torch.float32
        or out.shape != (rows, max_len)
        or out.stride(1) != 1
    ):
        raise ValueError("unexpected paged C4 logits operands")
    out.fill_(float("-inf"))
    if rows == 0 or max_len == 0:
        return out
    pages = cache_u8.shape[0]
    flat = cache_u8.view(pages, -1)
    kv_val = flat.view(_fp8_dtype())
    kv_scale = flat.view(torch.float32)
    q4 = q_fp8.unsqueeze(1)  # [R, NEXT_N=1, H, D]
    kernel[(rows, n_splits)](
        q4,
        kv_val,
        kv_scale,
        weights,
        ctx_lens,
        block_table,
        out,
        q4.stride(0),
        q4.stride(1),
        q4.stride(2),
        weights.stride(0),
        kv_val.stride(0),
        kv_scale.stride(0),
        (PAGE * dim) // 4,
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        max_len,
        NUM_HEADS=heads,
        HEAD_SIZE=dim,
        BLOCK_SIZE=PAGE,
        BLOCK_KV=BLOCK_KV,
        N_SPLITS=n_splits,
        NEXT_N=1,
        CTX_PER_ROW=True,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def segment_layout(starts, ends):
    """Host-side grouping of rows by segment start (one segment per request).

    starts/ends: int CPU tensors [R]. Returns (seg_id [R] int64, seg_start [S], seg_end [S])
    where a segment spans [start, max end over its rows). Rows with ctx <= 0 still join one.
    """
    starts = starts.to(torch.int64)
    ends = ends.to(torch.int64)
    seg_start, seg_id = torch.unique(starts, return_inverse=True)
    seg_end = seg_start.clone()
    seg_end.scatter_reduce_(0, seg_id, ends, reduce="amax", include_self=True)
    return seg_id, seg_start, seg_end


def page_count(length):
    return max(1, (length + PAGE - 1) // PAGE)


def build_paged_cache(k_u8, k_scale_f32, seg_start, seg_end, *, placement_seed=None, device=None):
    """Block-flat paged copy of each segment; page 0 row 0 is the segment's first key.

    k_u8 [N,128] uint8 (fp8 bytes), k_scale_f32 [N] fp32. Padding rows are zero bytes/scales
    and are never inside any row's ctx. Returns (cache [P,64,132] uint8,
    seg_table [S, max_pages] int32 on device).
    """
    device = k_u8.device if device is None else device
    lengths = [int(e - s) for s, e in zip(seg_start.tolist(), seg_end.tolist())]
    seg_pages = [page_count(n) for n in lengths]
    total = sum(seg_pages)
    ids = torch.arange(total, dtype=torch.int64)
    if placement_seed is not None:
        generator = torch.Generator().manual_seed(int(placement_seed))
        ids = torch.randperm(total, generator=generator)
    cache = torch.zeros((total, PAGE, ROW_BYTES), dtype=torch.uint8, device=device)
    flat = cache.view(total, -1)
    scale_u8 = k_scale_f32.contiguous().view(torch.uint8).view(-1, 4)  # [N,4]
    max_pages = max(seg_pages) if seg_pages else 1
    table = torch.zeros((len(lengths), max_pages), dtype=torch.int32)
    offset = 0
    for s, (start, n, pages) in enumerate(zip(seg_start.tolist(), lengths, seg_pages)):
        page_ids = ids[offset : offset + pages]
        offset += pages
        table[s, :pages] = page_ids.to(torch.int32)
        if n <= 0:
            continue
        vals = torch.zeros((pages * PAGE, HEAD_DIM), dtype=torch.uint8, device=device)
        vals[:n] = k_u8[start : start + n]
        scls = torch.zeros((pages * PAGE, 4), dtype=torch.uint8, device=device)
        scls[:n] = scale_u8[start : start + n]
        ids_dev = page_ids.to(device)
        flat[ids_dev, : PAGE * HEAD_DIM] = vals.view(pages, PAGE * HEAD_DIM)
        flat[ids_dev, PAGE * HEAD_DIM :] = scls.view(pages, PAGE * 4)
    return cache, table.to(device)


def paged_c4_logits_contiguous(
    q_fp8,
    weights,
    k_fp8,
    k_scale,
    row_starts,
    row_ends,
    *,
    absolute_columns=False,
    num_columns=None,
    placement_seed=None,
    n_splits=N_SPLITS,
):
    """Prefill-shaped entry: row i attends keys [row_starts[i], row_ends[i]) of k_fp8 [N,D].

    Default: logits [R, max_ctx] request-relative (valid [0, ctx_i)), -inf elsewhere.
    ``absolute_columns``: logits [R, num_columns (default N)] with row i's logits at columns
    [start_i, end_i) (the layout of the flydsl ``rocm_fp8_mqa_logits``), -inf elsewhere.
    Rows are grouped by start; each group's keys are copied byte-exactly into fresh pages.
    One host sync (starts/ends).
    """
    fp8 = _fp8_dtype()
    rows = q_fp8.shape[0]
    if (
        q_fp8.ndim != 3
        or k_fp8.ndim != 2
        or k_fp8.dtype != fp8
        or k_fp8.shape[1] != HEAD_DIM
        or q_fp8.dtype != fp8
        or k_scale.dtype != torch.float32
        or k_scale.numel() != k_fp8.shape[0]
        or row_starts.shape != (rows,)
        or row_ends.shape != (rows,)
        or weights.shape != q_fp8.shape[:2]
    ):
        raise ValueError("unexpected contiguous C4 logits operands")
    n_keys = k_fp8.shape[0]
    device = q_fp8.device
    starts_cpu, ends_cpu = row_starts.cpu(), row_ends.cpu()
    if rows and (int(starts_cpu.min()) < 0 or int(ends_cpu.max()) > n_keys):
        raise ValueError("row ranges outside the key buffer")
    ctx_cpu = (ends_cpu.to(torch.int64) - starts_cpu.to(torch.int64)).clamp(min=0)
    max_ctx = int(ctx_cpu.max()) if rows else 0
    width = max_ctx
    if absolute_columns:
        width = n_keys if num_columns is None else int(num_columns)
    weights = weights.contiguous()
    if rows == 0 or max_ctx == 0:
        return torch.full((rows, width), float("-inf"), dtype=torch.float32, device=device)
    seg_id, seg_start, seg_end = segment_layout(starts_cpu, ends_cpu)
    k_u8 = k_fp8.contiguous().view(torch.uint8)
    cache, seg_table = build_paged_cache(
        k_u8, k_scale.reshape(-1), seg_start, seg_end, placement_seed=placement_seed, device=device
    )
    if seg_table.shape[0] == 1:
        table = seg_table.expand(rows, -1)
    else:
        table = seg_table[seg_id.to(device)]
    rel = torch.empty((rows, max_ctx), dtype=torch.float32, device=device)
    paged_c4_logits(
        q_fp8, weights, cache, table, ctx_cpu.to(torch.int32).to(device), max_ctx, rel, n_splits=n_splits
    )
    if not absolute_columns:
        return rel
    if int(seg_end.max()) > width:
        raise ValueError("segment extends past num_columns")
    out = torch.full((rows, width), float("-inf"), dtype=torch.float32, device=device)
    for s, (start, end) in enumerate(zip(seg_start.tolist(), seg_end.tolist())):
        span = min(end - start, max_ctx)
        if span <= 0:
            continue
        idx = (seg_id == s).nonzero().reshape(-1).to(device)
        out[idx, start : start + span] = rel[idx, :span]
    return out
