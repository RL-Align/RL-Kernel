# SPDX-License-Identifier: Apache-2.0
"""Instance-local sparse prefill math; retain vLLM's real scheduler and caches."""

import ast
import inspect
import textwrap
import types

import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention


class _PrefillBoundary(ast.NodeTransformer):
    def __init__(self):
        self.calls = 0

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "rocm_sparse_attn_prefill":
            self.calls += 1
            node.func = ast.Attribute(
                ast.Name("self", ast.Load()), "_rlk_prefill_sparse", ast.Load()
            )
            node.keywords += [
                ast.keyword("chunk_N", ast.Name("chunk_N", ast.Load())),
                ast.keyword("chunk_M", ast.Name("chunk_M", ast.Load())),
            ]
        return self.generic_visit(node)


def bind_rollout_sparse(attention, observer=None):
    if attention._use_aiter_sparse_mla_prefill or attention._use_aiter_sparse_mla_decode:
        raise ValueError("initial native sparse binding expects the pinned gather-prefill path")
    original = type(attention)._forward_prefill
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    transform = _PrefillBoundary()
    tree = transform.visit(tree)
    if transform.calls != 1:
        raise ValueError("unsupported vLLM sparse prefill boundary ABI")
    operator = NativeSparseAttention()

    def prefill(
        *,
        q,
        kv,
        indices,
        topk_length,
        scale,
        head_dim,
        nope_head_dim,
        rope_head_dim,
        attn_sink,
        output,
        chunk_N,
        chunk_M,
    ):
        if (head_dim, nope_head_dim, rope_head_dim) != (512, 448, 64):
            raise ValueError("initial native sparse profile requires original DSV4 dimensions")
        selected = indices.contiguous()
        if not torch.equal((selected >= 0).sum(-1).to(topk_length.dtype), topk_length):
            raise ValueError("sparse selected lengths do not match actual vLLM metadata")
        # Main-window rows precede compressed rows in the shared arithmetic
        # order, independently of their physical positions in this workspace.
        keys = torch.where(selected.remainder(chunk_M) >= chunk_N, selected, selected + kv.shape[0])
        keys = torch.where(selected >= 0, keys, torch.iinfo(torch.int32).max)
        selected = selected.gather(1, keys.argsort(dim=-1, stable=True))
        selected = F.pad(selected, (0, 640 - selected.shape[-1]), value=-1)
        if observer:
            observer(
                "sparse_prefill.input",
                dict(q=q, kv=kv, indices=selected, attn_sink=attn_sink, scale=scale),
            )
        result = operator(
            q, kv[:, 0].contiguous(), attn_sink[: q.shape[1]].contiguous(), selected, scale
        )
        output.copy_(result)
        if observer:
            observer("sparse_prefill.output", output)

    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), __file__, "exec"), namespace)
    attention._rlk_prefill_sparse = prefill
    attention._forward_prefill = types.MethodType(namespace[original.__name__], attention)
    attention._rlk_sparse_operator = operator
    return attention
