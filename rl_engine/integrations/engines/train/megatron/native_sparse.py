# SPDX-License-Identifier: Apache-2.0
"""Keep native CSA compression/indexing and bind only its sparse math boundary."""

import types

import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.native_sparse import NativeSparseAttention


def bind_native_sparse(core):
    original = core.forward.__func__
    if "unfused_compressed_sparse_attn" not in original.__code__.co_names or core.use_fused_kernels:
        raise ValueError("expected native ordinary unfused CSA sparse boundary")
    operator = NativeSparseAttention()

    def sparse(query, kv_full, sink, indices, scale):
        t, batch, heads, dim = query.shape
        outputs = []
        for b in range(batch):
            selected = indices[b].contiguous()
            keys = torch.where(selected >= 0, selected, torch.iinfo(torch.int32).max)
            selected = selected.gather(1, keys.argsort(dim=-1, stable=True))
            selected = F.pad(selected, (0, 640 - selected.shape[-1]), value=-1)
            out = operator(
                query[:, b].contiguous(), kv_full[:, b].contiguous(), sink, selected, scale
            )
            outputs.append(out)
        return torch.stack(outputs, dim=1).reshape(t, batch, heads * dim)

    cloned = types.FunctionType(
        original.__code__,
        dict(original.__globals__, unfused_compressed_sparse_attn=sparse),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    cloned.__kwdefaults__ = original.__kwdefaults__
    core.forward = types.MethodType(cloned, core)
    return core
