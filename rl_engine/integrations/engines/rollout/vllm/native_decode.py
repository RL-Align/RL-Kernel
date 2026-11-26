# SPDX-License-Identifier: Apache-2.0
"""Real paged-cache decode operands into the shared fixed sparse attention math."""

import ast
import inspect
import os
import textwrap
import time
import types

import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention
from rl_engine.backends.rocm.fp8 import fp8_dtype

_MAX_SELECTED = 640
_ROW_STRIDE = 1 << 32  # > any logical compressed position; keeps (row, logical) keys ordered


def _dequant(out, flat, lengths, table, block, fnuz):
    from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache

    dequantize_and_gather_k_cache(out, flat, lengths, None, table, block, 0, use_fnuz=fnuz)


def _flat_pages(cache, block):
    # The real hybrid allocator interleaves layer payloads within each
    # physical block. Its stride is larger than this layer's packed page.
    page_bytes = ((block * 584 + 575) // 576) * 576
    return cache.as_strided((cache.shape[0], page_bytes), (cache.stride(0), 1))


def _gather(cache, slots, fnuz):
    if slots.numel() == 0:
        return torch.empty((0, 512), device=cache.device, dtype=torch.bfloat16)
    if cache.ndim != 3 or cache.dtype != torch.uint8:
        raise ValueError("original paged FP8-DS-MLA cache is required")
    block = cache.shape[1]
    flat = _flat_pages(cache, block)
    pages, inverse = torch.unique(slots // block, sorted=True, return_inverse=True)
    out = torch.empty((pages.numel(), block, 512), device=cache.device, dtype=torch.bfloat16)
    lengths = torch.full((pages.numel(),), block, device=cache.device, dtype=torch.int32)
    table = pages.int()[:, None].contiguous()
    _dequant(out, flat, lengths, table, block, fnuz)
    return out.flatten(0, 1).index_select(0, inverse * block + slots % block)


def _slots(kwargs, name, row):
    ragged = kwargs.get(name + "_ragged_indices")
    ptr = kwargs.get(name + "_ragged_indptr")
    lengths = kwargs.get(name + "_lens")
    expected = None if lengths is None else int(lengths[row])
    if expected == 0:
        return torch.empty(0, device=kwargs["q"].device, dtype=torch.int64)
    if ragged is not None and ptr is not None:
        result = ragged[int(ptr[row]) : int(ptr[row + 1])].long()
    else:
        dense = kwargs.get(name + "_indices")
        if dense is None:
            raise ValueError("nonempty sparse cache segment has no real indices")
        result = dense[row].reshape(-1).long()
        result = result[result >= 0]
    if expected is not None and result.numel() != expected:
        raise ValueError("actual sparse cache segment disagrees with recorded lengths")
    return result


def _build_operands_reference(kwargs, block_table, device, batch, fnuz):
    """Original per-request loop (many host syncs); kept as the equivalence oracle."""
    selected = []
    vectors = []
    offset = 0
    for i in range(batch):
        recent = _slots(kwargs, "swa", i)
        kv = _gather(kwargs["swa_k_cache"], recent, fnuz)
        if not kwargs["swa_only"]:
            ids = _slots(kwargs, "topk", i)
            if ids.numel():
                block = kwargs["kv_cache"].shape[1]
                table = block_table[i].long()
                matches = (ids[:, None] // block) == table[None, :]
                if not bool(matches.any(-1).all()):
                    raise ValueError("compressed cache slots absent from real request block table")
                logical = matches.int().argmax(-1) * block + ids % block
                ids = ids.index_select(0, logical.argsort(stable=True))
                extra = _gather(kwargs["kv_cache"], ids, False)
                kv = torch.cat((kv, extra), 0)
        n = kv.shape[0]
        if n > _MAX_SELECTED:
            raise ValueError("actual sparse set exceeds the fixed original TopK/window profile")
        vectors.append(kv)
        row = torch.arange(offset, offset + n, device=device, dtype=torch.int32)
        selected.append(F.pad(row, (0, _MAX_SELECTED - n), value=-1))
        offset += n
    kv = torch.cat(vectors, 0).contiguous()
    if kv.shape[0] == 0:
        # Startup metadata may describe sink-only rows. Keep one physical
        # workspace row; every index remains -1, so it is never attended.
        kv = torch.zeros((1, 512), device=device, dtype=torch.bfloat16)
    return kv, torch.stack(selected)


def _segment(kwargs, name, batch, device):
    """Batched ``_slots``: (flat values, per-row start in flat, per-row length, error code).

    Row i's slots are ``flat[start[i] : start[i] + length[i]]`` in today's order. The error
    code (0-d device int64) is 0 ok, 1 lengths disagree, 2 no real indices; no host read here.
    """
    ragged = kwargs.get(name + "_ragged_indices")
    ptr = kwargs.get(name + "_ragged_indptr")
    lengths = kwargs.get(name + "_lens")
    if lengths is not None:
        lengths = lengths[:batch].long()
    error = torch.zeros((), device=device, dtype=torch.int64)
    if ragged is not None and ptr is not None:
        flat = ragged
        ptr = ptr[: batch + 1].long()
        start = ptr[:-1].clamp(0, flat.numel())
        actual = (ptr[1:].clamp(0, flat.numel()) - start).clamp(min=0)
    else:
        dense = kwargs.get(name + "_indices")
        if dense is None:
            if lengths is None:
                raise ValueError("nonempty sparse cache segment has no real indices")
            flat = torch.empty(0, device=device, dtype=torch.int64)
            start = torch.zeros(batch, device=device, dtype=torch.int64)
            actual = torch.zeros(batch, device=device, dtype=torch.int64)
            error = error + 2 * (lengths != 0).any().long()
        else:
            dense = dense[:batch].reshape(batch, -1)
            valid = dense >= 0
            actual = valid.sum(1)
            # Stable: valid entries first, original order == ``result[result >= 0]`` per row.
            order = torch.sort((~valid).to(torch.int8), dim=1, stable=True).indices
            flat = dense.gather(1, order).reshape(-1)
            start = torch.arange(batch, device=device, dtype=torch.int64) * dense.shape[1]
    if lengths is None:
        length = actual
    else:
        length = torch.where(lengths == 0, torch.zeros_like(actual), actual)
        error = error + ((lengths != 0) & (actual != lengths)).any().long()
    return flat, start, length, error


def _expand(flat, start, length, total):
    """Flatten ragged rows: (row of each element, position within its row, slot value)."""
    rows = torch.arange(length.numel(), device=length.device).repeat_interleave(
        length, output_size=total
    )
    pos = torch.arange(total, device=length.device) - (torch.cumsum(length, 0) - length)[rows]
    return rows, pos, flat[start[rows] + pos].long()


def _page_plan(cache, slots):
    """Device-side sorted-unique of ``slots // block`` (same pages/inverse as torch.unique)."""
    if cache.ndim != 3 or cache.dtype != torch.uint8:
        raise ValueError("original paged FP8-DS-MLA cache is required")
    block = cache.shape[1]
    sorted_pages, perm = torch.sort(slots // block)
    index = torch.zeros_like(sorted_pages)
    index[1:] = (sorted_pages[1:] != sorted_pages[:-1]).cumsum(0)
    return block, sorted_pages, perm, index


def _gather_planned(cache, slots, fnuz, plan, count):
    block, sorted_pages, perm, index = plan
    pages = torch.empty((count,), device=slots.device, dtype=torch.int64)
    pages[index] = sorted_pages  # duplicates write equal values
    inverse = torch.empty_like(index)
    inverse[perm] = index
    out = torch.empty((count, block, 512), device=cache.device, dtype=torch.bfloat16)
    lengths = torch.full((count,), block, device=cache.device, dtype=torch.int32)
    table = pages.int()[:, None].contiguous()
    _dequant(out, _flat_pages(cache, block), lengths, table, block, fnuz)
    return out.flatten(0, 1).index_select(0, inverse * block + slots % block)


def _build_operands(kwargs, block_table, device, batch, fnuz):
    """Vectorized ``_build_operands_reference`` with identical values: 2 host syncs per call.

    Layout is unchanged: [row0 swa, row0 topk, row1 swa, ...]; indices are [B, 640] int32.
    """
    if batch == 0:
        return _build_operands_reference(kwargs, block_table, device, batch, fnuz)
    swa_only = kwargs["swa_only"]
    swa_flat, swa_start, swa_len, swa_err = _segment(kwargs, "swa", batch, device)
    zero = torch.zeros((), device=device, dtype=torch.int64)
    if swa_only:
        top_flat, top_start, top_len, top_err = None, None, torch.zeros_like(swa_len), zero
    else:
        top_flat, top_start, top_len, top_err = _segment(kwargs, "topk", batch, device)
    count = swa_len + top_len
    # Host sync 1: sizes plus the deferred per-row consistency flags.
    n_swa, n_top, e_swa, e_top, too_big = torch.stack(
        (swa_len.sum(), top_len.sum(), swa_err, top_err, (count.max() > _MAX_SELECTED).long())
    ).tolist()
    for code in (e_swa, e_top):
        if code == 2:
            raise ValueError("nonempty sparse cache segment has no real indices")
        if code:
            raise ValueError("actual sparse cache segment disagrees with recorded lengths")
    if too_big:
        raise ValueError("actual sparse set exceeds the fixed original TopK/window profile")

    if n_swa:
        swa_rows, swa_pos, swa_ids = _expand(swa_flat, swa_start, swa_len, n_swa)
    if n_top:
        top_rows, top_pos, top_ids = _expand(top_flat, top_start, top_len, n_top)
        cblock = kwargs["kv_cache"].shape[1]
        if block_table is None:
            raise ValueError("compressed cache slots need the real request block table")
        table = block_table[:batch].long()
        logical = torch.empty_like(top_ids)
        present = torch.ones((), device=device, dtype=torch.bool)
        step = max(1, (1 << 24) // max(table.shape[1], 1))
        for a in range(0, n_top, step):
            ids = top_ids[a : a + step]
            matches = (ids[:, None] // cblock) == table[top_rows[a : a + step]]
            present = present & matches.any(-1).all()
            logical[a : a + step] = matches.int().argmax(-1) * cblock + ids % cblock
        # Device-side stand-in for the per-row host check (aborts on violation, no sync).
        torch._assert_async(present, "compressed cache slots absent from real request block table")
        # One global stable sort == per-row stable argsort (rows stay contiguous, row-major).
        top_ids = top_ids.index_select(0, (top_rows * _ROW_STRIDE + logical).argsort(stable=True))

    jobs = []
    if n_swa:
        cache = kwargs["swa_k_cache"]
        jobs.append((cache, swa_ids, fnuz, _page_plan(cache, swa_ids)))
    if n_top:
        cache = kwargs["kv_cache"]
        jobs.append((cache, top_ids, False, _page_plan(cache, top_ids)))
    gathered = []
    if jobs:
        # Host sync 2: unique page counts of both caches (plan[-1] is the page index per slot).
        counts = torch.stack([job[3][3][-1] + 1 for job in jobs]).tolist()
        gathered = [
            _gather_planned(cache, ids, flag, plan, c)
            for (cache, ids, flag, plan), c in zip(jobs, counts)
        ]

    offsets = torch.cumsum(count, 0) - count
    if n_swa + n_top == 0:
        # Startup metadata may describe sink-only rows. Keep one physical
        # workspace row; every index remains -1, so it is never attended.
        kv = torch.zeros((1, 512), device=device, dtype=torch.bfloat16)
    else:
        kv = torch.empty((n_swa + n_top, 512), device=device, dtype=torch.bfloat16)
        parts = iter(gathered)
        if n_swa:
            kv.index_copy_(0, offsets[swa_rows] + swa_pos, next(parts))
        if n_top:
            kv.index_copy_(0, offsets[top_rows] + swa_len[top_rows] + top_pos, next(parts))
    column = torch.arange(_MAX_SELECTED, device=device, dtype=torch.int64)[None, :]
    indices = torch.where(column < count[:, None], offsets[:, None] + column, -1).to(torch.int32)
    return kv, indices


class _DecodeBoundary(ast.NodeTransformer):
    def __init__(self):
        self.calls = 0

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "rocm_sparse_attn_decode":
            self.calls += 1
            node.func = ast.Attribute(
                ast.Name("self", ast.Load()), "_rlk_decode_sparse", ast.Load()
            )
            node.keywords += [
                ast.keyword("rlk_metadata", ast.Name("attn_metadata", ast.Load())),
                ast.keyword("rlk_swa_metadata", ast.Name("swa_metadata", ast.Load())),
            ]
        return self.generic_visit(node)


def bind_rollout_decode(attention, observer=None, record_if=None):
    if attention._use_aiter_sparse_mla_decode or attention._wo_a_fp8_weight is not None:
        raise ValueError("fixed decode requires the pinned ordinary cache path")
    original = type(attention)._forward_decode
    native = original.__globals__["rocm_sparse_attn_decode"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform = _DecodeBoundary()
    tree = transform.visit(tree)
    if transform.calls != 1:
        raise ValueError("unsupported pinned vLLM decode ABI")
    operator = NativeSparseAttention()
    recorded = [False]
    # Opt-in (off by default): synchronizes around each phase, so it perturbs timing.
    timing = os.environ.get("RLK_DECODE_STEP_TIMING") == "1"
    timer = {"calls": 0, "build": 0.0, "attn": 0.0}

    def decode(*, rlk_metadata, rlk_swa_metadata, **kwargs):
        q = kwargs["q"]
        output = kwargs["output"]
        if q.shape[0] != rlk_swa_metadata.num_decodes:
            raise ValueError("ordinary one-token-per-request decode only")
        recording = observer is not None and not recorded[0] and (record_if is None or record_if())
        if recording:
            legacy = torch.empty_like(output)
            native(**dict(kwargs, output=legacy))
            observer("sparse_decode.legacy_fused_output", legacy)
        if timing:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        kv, indices = _build_operands(
            kwargs,
            # Warm-up/profile batches carry no metadata; only real top-k rows need the table.
            None if rlk_metadata is None else rlk_metadata.block_table,
            q.device,
            q.shape[0],
            fp8_dtype(q.device) == torch.float8_e4m3fnuz,
        )
        if timing:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        sink = kwargs["attn_sink"][: q.shape[1]].contiguous()
        result = operator(q, kv, sink, indices, kwargs["scale"])
        if timing:
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            timer["calls"] += 1
            timer["build"] += t1 - t0
            timer["attn"] += t2 - t1
            if timer["calls"] % 100 == 0:
                print(
                    "RLK_DECODE_STEP_TIMING",
                    f"calls={timer['calls']} B={q.shape[0]}",
                    f"build_ms_avg={1e3 * timer['build'] / 100:.3f}",
                    f"attn_ms_avg={1e3 * timer['attn'] / 100:.3f}",
                    "(last 100 layer-calls)",
                    flush=True,
                )
                timer["build"] = timer["attn"] = 0.0
        if recording:
            observer(
                "sparse_decode.operands",
                dict(q=q, kv=kv, indices=indices, sink=sink, scale=kwargs["scale"]),
            )
            observer("sparse_decode.unrotated_output", result)
            recorded[0] = True
        output.copy_(result)
        # The native prefill profile has a BF16 attention boundary. Report zero
        # rotated rows so forward_mqa applies its original standalone inverse
        # exactly once, preserving that same boundary for decode.
        return 0

    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    attention._rlk_decode_sparse = decode
    attention._forward_decode = types.MethodType(namespace[original.__name__], attention)
    return attention
