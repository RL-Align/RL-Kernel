# SPDX-License-Identifier: Apache-2.0
"""Original sparse operands with the existing fixed AITER forward and CSR VJP."""

from pathlib import Path

import torch

from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend
from rl_engine.contracts.operators.joint_attention import JointAttentionContract


class _Sparse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, indices, scale, operator):
        if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2:
            raise ValueError("sparse boundary requires Q[M,H,D], KV[N,D], indices[M,K]")
        contract = JointAttentionContract(
            q.shape[1],
            q.shape[2],
            q.shape[0],
            kv.shape[0],
            0,
            indices.shape[1],
            scale=scale,
            graph=False,
        )
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        operator.forward_backend.forward_into(q, kv, sink, indices, out, lse, contract)
        ctx.save_for_backward(q, kv, sink, indices, out, lse)
        ctx.operator, ctx.scale = operator, scale
        return out

    @staticmethod
    def backward(ctx, dy):
        q, kv, sink, indices, out, lse = ctx.saved_tensors
        backend = ctx.operator.backward_backend
        ptr, data = backend.build_csr(indices, kv.shape[0])
        ws = backend.allocate(
            m=q.shape[0],
            h=q.shape[1],
            d=q.shape[2],
            selected=indices.shape[1],
            n_kv=kv.shape[0],
            max_bytes=1 << 30,
        )
        backend.backward_into(
            q, kv, out, dy.contiguous(), indices, lse, sink, ptr, data, ws, ctx.scale
        )
        return ws.dq.to(q.dtype), ws.dkv.to(kv.dtype), ws.dsink, None, None, None


class NativeSparseAttention:
    """One operator profile for actual native/vLLM sparse attention operands."""

    def __init__(self):
        import aiter

        root = Path(aiter.__file__).parent
        self.forward_backend = AiterJointAttentionBackend(
            kernel_source=root / "ops/triton/_triton_kernels/attention/sparse_mla_dsv4_train.py"
        )
        self.backward_backend = AiterJointBackwardBackend(
            self.forward_backend,
            wrapper_source=root / "ops/triton/attention/sparse_mla_dsv4_train.py",
        )
        self.executed = False

    def __call__(self, q, kv, sink, indices, scale):
        out = _Sparse.apply(q, kv, sink, indices, scale, self)
        if not self.executed:
            print(
                "RLK_NATIVE_SPARSE_EXECUTED",
                __file__,
                self.forward_backend.provenance,
                self.backward_backend.provenance,
                flush=True,
            )
            self.executed = True
        return out
