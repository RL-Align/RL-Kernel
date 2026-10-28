# SPDX-License-Identifier: Apache-2.0
"""Reuse aiter backward/CSR and MHA delta kernels, with no floating sink atomic."""

import ast
from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path

import torch

from rl_engine.contracts.operators.joint_attention import require

BACKWARD_PROFILE = "aiter.326c.h16.k32.csr64.sink32.canonical_queries.v1"
WRAPPER_SHA256 = "ede08814720b9c783c7375cab9f95ee38bf6212ca77ba23149e33309b72842ff"


def load_upstream_csr(path):
    """Load just the verified upstream CSR helper, without global module patches."""
    path = Path(path)
    require(path.is_file(), "pinned aiter sparse wrapper source is required")
    source = path.read_bytes()
    require(
        hashlib.sha256(source).hexdigest() == WRAPPER_SHA256, "aiter sparse wrapper hash mismatch"
    )
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_inverted_topk"
    ]
    require(len(nodes) == 1, "pinned aiter CSR ABI is missing")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["_build_inverted_topk"]


@dataclass(frozen=True)
class JointBackwardWorkspace:
    delta: torch.Tensor
    dq: torch.Tensor
    p: torch.Tensor
    dp: torch.Tensor
    intermediate: torch.Tensor
    dkv: torch.Tensor
    dsink: torch.Tensor
    inactive_sink: torch.Tensor


class AiterJointBackwardBackend:
    profile = BACKWARD_PROFILE

    def __init__(self, forward_backend, *, wrapper_source=None):
        self.forward_backend = forward_backend
        self.device, self.native = forward_backend.device, forward_backend.module
        triton = forward_backend.triton
        self.triton = triton
        source = wrapper_source or os.environ.get("RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE", "")
        require(bool(source), "set RLK_AITER_DSV4_SPARSE_WRAPPER_SOURCE explicitly")
        self.build_csr = load_upstream_csr(source)
        preprocessing = importlib.import_module(
            "aiter.ops.triton._triton_kernels.attention.mha_fused_bwd"
        )
        self._delta = preprocessing._bwd_preprocess
        self.provenance = {
            "profile": self.profile,
            "forward": forward_backend.provenance,
            "wrapper_sha256": WRAPPER_SHA256,
            "delta_source_sha256": hashlib.sha256(
                Path(preprocessing.__file__).read_bytes()
            ).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    @staticmethod
    def workspace_bytes(m, h, d, selected, n_kv):
        # Native resident outputs/intermediates only; record snapshots, CSR,
        # graph pool and Torch sink-reduction temporaries are separate.
        return 4 * (m * h + m * h * d + m * h * selected + m * selected * d + n_kv * d + 2 * h)

    def allocate(self, *, m, h, d, selected, n_kv, max_bytes):
        require(type(max_bytes) is int and max_bytes > 0, "positive backward budget required")
        require(
            self.workspace_bytes(m, h, d, selected, n_kv) <= max_bytes,
            "backward workspace budget exceeded",
        )

        def empty(shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)

        with torch.inference_mode(False):
            return JointBackwardWorkspace(
                empty((m, h)),
                empty((m, h, d)),
                empty((m, h, selected), torch.bfloat16),
                empty((m, h, selected), torch.bfloat16),
                empty((m, selected, d)),
                empty((n_kv, d)),
                empty((h,)),
                empty((h,)),
            )

    def local_parts_into(self, q, kv, out, dy, indices, lse, sink, ws, scale):
        """One writer per dQ/query contribution; no reduction across queries here."""
        m, h, d = q.shape
        k = indices.shape[1]
        if not m:
            return
        self._delta[(self.triton.cdiv(m, 16), 1, h)](
            out,
            dy,
            ws.delta,
            m * h * d,
            out.stride(1),
            out.stride(0),
            out.stride(2),
            m * h * d,
            dy.stride(1),
            dy.stride(0),
            dy.stride(2),
            m * h,
            1,
            h,
            0,
            None,
            m,
            None,
            BLOCK_M=16,
            BLOCK_D_MODEL=d,
            BLOCK_D_MODEL_POW2=d,
            IS_VARLEN=False,
            IS_FP8=False,
            num_warps=4,
        )
        self.native._bwd_dq_store_dp_kernel[(m, self.triton.cdiv(h, 16))](
            q,
            kv,
            dy,
            indices,
            lse,
            ws.delta,
            sink,
            ws.dq,
            ws.inactive_sink,
            ws.p,
            ws.dp,
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            dy.stride(0),
            dy.stride(1),
            indices.stride(0),
            lse.stride(0),
            ws.delta.stride(0),
            ws.dq.stride(0),
            ws.dq.stride(1),
            ws.p.stride(0),
            ws.p.stride(1),
            h,
            d,
            kv.shape[0],
            k,
            scale,
            # False removes ONLY the upstream sink atomic branch. The real
            # sink-inclusive LSE remains an input to dQ/P/dP.
            HAS_ATTN_SINK=False,
            BLOCK_H=16,
            BLOCK_D=d,
            BLOCK_K=32,
            num_warps=4,
        )
        self.native._bwd_dkv_interm_kernel[(m,)](
            q,
            dy,
            ws.p,
            ws.dp,
            ws.intermediate,
            q.stride(0),
            q.stride(1),
            dy.stride(0),
            dy.stride(1),
            ws.p.stride(0),
            ws.p.stride(1),
            ws.intermediate.stride(0),
            ws.intermediate.stride(1),
            h,
            d,
            k,
            BLOCK_H=16,
            BLOCK_D=d,
            BLOCK_K=32,
            NUM_HG=self.triton.cdiv(h, 16),
            num_warps=4,
        )

    def gather_intermediate_into(self, intermediate, indices, ptr, data, dkv):
        m, k = indices.shape
        d = dkv.shape[1]
        if not m:
            dkv.zero_()
            return
        if dkv.shape[0]:
            self.native._bwd_dkv_gather_kernel[(dkv.shape[0],)](
                intermediate,
                ptr,
                data,
                dkv,
                intermediate.stride(0),
                intermediate.stride(1),
                dkv.stride(0),
                d,
                k,
                m,
                BLOCK_D=d,
                BLOCK_G=64,
                num_warps=4,
            )

    @staticmethod
    def sink_contributions(delta, lse, sink):
        return torch.where(
            torch.isfinite(lse) & torch.isfinite(sink[None]),
            -delta * torch.exp(sink[None] - lse),
            0.0,
        )

    @staticmethod
    def reduce_sink_rows(rows):
        m, h = rows.shape
        if not m:
            return torch.zeros(h, dtype=torch.float32, device=rows.device)
        # Reuse native Torch reductions on explicit 32-row tiles. Both trees
        # see the same canonically ordered full query set, regardless of
        # microbatch partition/backward arrival order. No floating atomic.
        padded = torch.nn.functional.pad(rows, (0, 0, 0, (-m) % 32))
        return padded.reshape(-1, 32, h).sum(1).sum(0)

    def reduce_into(self, indices, lse, sink, ptr, data, ws):
        """Input query order must already be canonical, with no padded queries."""
        self.gather_intermediate_into(ws.intermediate, indices, ptr, data, ws.dkv)
        ws.dsink.copy_(self.reduce_sink_rows(self.sink_contributions(ws.delta, lse, sink)))

    def backward_into(self, q, kv, out, dy, indices, lse, sink, ptr, data, ws, scale):
        self.local_parts_into(q, kv, out, dy, indices, lse, sink, ws, scale)
        self.reduce_into(indices, lse, sink, ptr, data, ws)
