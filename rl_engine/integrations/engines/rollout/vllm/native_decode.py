# SPDX-License-Identifier: Apache-2.0
"""Real paged-cache decode operands into the shared fixed sparse attention math."""

import ast
import inspect
import textwrap
import types

import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention


def _gather(cache, slots, fnuz):
    from vllm.models.deepseek_v4.common.ops.cache_utils import dequantize_and_gather_k_cache

    if slots.numel() == 0:
        return torch.empty((0, 512), device=cache.device, dtype=torch.bfloat16)
    if cache.ndim != 3 or cache.dtype != torch.uint8:
        raise ValueError("original paged FP8-DS-MLA cache is required")
    block = cache.shape[1]
    # The real hybrid allocator interleaves layer payloads within each
    # physical block. Its stride is larger than this layer's packed page.
    page_bytes = ((block * 584 + 575) // 576) * 576
    flat = cache.as_strided((cache.shape[0], page_bytes), (cache.stride(0), 1))
    pages, inverse = torch.unique(slots // block, sorted=True, return_inverse=True)
    out = torch.empty((pages.numel(), block, 512), device=cache.device, dtype=torch.bfloat16)
    lengths = torch.full((pages.numel(),), block, device=cache.device, dtype=torch.int32)
    table = pages.int()[:, None].contiguous()
    dequantize_and_gather_k_cache(out, flat, lengths, None, table, block, 0, use_fnuz=fnuz)
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
        raise ValueError("initial fixed decode requires the pinned gfx942 ordinary cache path")
    original = type(attention)._forward_decode
    native = original.__globals__["rocm_sparse_attn_decode"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform = _DecodeBoundary()
    tree = transform.visit(tree)
    if transform.calls != 1:
        raise ValueError("unsupported pinned vLLM decode ABI")
    operator = NativeSparseAttention()
    recorded = [False]

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
        selected = []
        vectors = []
        offset = 0
        for i in range(q.shape[0]):
            recent = _slots(kwargs, "swa", i)
            kv = _gather(kwargs["swa_k_cache"], recent, True)
            if not kwargs["swa_only"]:
                ids = _slots(kwargs, "topk", i)
                if ids.numel():
                    block = kwargs["kv_cache"].shape[1]
                    table = rlk_metadata.block_table[i].long()
                    matches = (ids[:, None] // block) == table[None, :]
                    if not bool(matches.any(-1).all()):
                        raise ValueError(
                            "compressed cache slots absent from real request block table"
                        )
                    logical = matches.int().argmax(-1) * block + ids % block
                    ids = ids.index_select(0, logical.argsort(stable=True))
                    extra = _gather(kwargs["kv_cache"], ids, False)
                    kv = torch.cat((kv, extra), 0)
            n = kv.shape[0]
            if n > 640:
                raise ValueError("actual sparse set exceeds the fixed original TopK/window profile")
            vectors.append(kv)
            row = torch.arange(offset, offset + n, device=q.device, dtype=torch.int32)
            selected.append(F.pad(row, (0, 640 - n), value=-1))
            offset += n
        kv = torch.cat(vectors, 0).contiguous()
        if kv.shape[0] == 0:
            # Startup metadata may describe sink-only rows. Keep one physical
            # workspace row; every index remains -1, so it is never attended.
            kv = torch.zeros((1, 512), device=q.device, dtype=torch.bfloat16)
        indices = torch.stack(selected)
        sink = kwargs["attn_sink"][: q.shape[1]].contiguous()
        result = operator(q, kv, sink, indices, kwargs["scale"])
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
