# SPDX-License-Identifier: Apache-2.0
"""Fixed-record native VJP workspace; autograd/session integration is separate."""

from dataclasses import dataclass

import torch

from rl_engine.contracts.operators.joint_attention import require
from rl_engine.models.deepseek_v4.joint_attention import JointForwardRecord, eager, owned, stamp


@dataclass(frozen=True)
class JointGradients:
    dq: torch.Tensor
    dkv: torch.Tensor
    dsink: torch.Tensor


@dataclass(frozen=True)
class JointLocalParts:
    dq: torch.Tensor
    intermediate: torch.Tensor
    sink_rows: torch.Tensor


class JointBackwardPlan:
    """Prepare CSR once outside Graph; replay dy with owned output snapshots.

    dQ follows the forward's original query order. dKV follows canonical
    logical KV order, as saved in the forward record. All three outputs are
    FP32; a BF16-master autograd bridge must explicitly cast q/KV VJPs.
    """

    def __init__(self, forward_plan, record, *, backend=None, max_workspace_bytes=1 << 30):
        require(isinstance(record, JointForwardRecord), "a saved forward record is required")
        record.validate_saved()
        require(
            record.implementation == forward_plan.implementation,
            "forward/backward provider mismatch",
        )
        if backend is None:
            from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend

            backend = AiterJointBackwardBackend(forward_plan.backend)
        require(
            backend.forward_backend is forward_plan.backend, "backward provider ownership mismatch"
        )
        self.forward_plan, self.record, self.backend = forward_plan, record, backend
        self.device = forward_plan.device
        eager(self.device)
        with forward_plan._transaction(), torch.inference_mode(False), torch.no_grad():
            keys = record.metadata.query_keys.cpu().tolist()
            order = sorted((i for i, key in enumerate(keys) if key[0] >= 0), key=lambda i: keys[i])
            self.order = torch.tensor(order, dtype=torch.int64, device=self.device)
            self.q = owned(record.q.index_select(0, self.order))
            self.kv = record.kv
            self.out = owned(record.output.index_select(0, self.order))
            self.lse = owned(record.lse.index_select(0, self.order))
            self.indices = owned(record.indices.index_select(0, self.order))
            self.sink = record.sink
            self.dy = torch.zeros_like(self.q)
            self.dq_original = torch.zeros_like(record.q, dtype=torch.float32)
            self.ptr, self.data = backend.build_csr(self.indices, record.kv.shape[0])
            self.workspace = backend.allocate(
                m=len(order),
                h=record.q.shape[1],
                d=record.q.shape[2],
                selected=record.indices.shape[1],
                n_kv=record.kv.shape[0],
                max_bytes=max_workspace_bytes,
            )
            self._frozen = (
                self.order,
                self.q,
                self.out,
                self.lse,
                self.indices,
                self.ptr,
                self.data,
            )
            self._stamps = tuple(stamp(t) for t in self._frozen)
        self._graph = None

    def _validate(self):
        self.record.validate_saved()
        require(
            tuple(stamp(t) for t in self._frozen) == self._stamps,
            "backward metadata or immutable inputs were mutated",
        )

    def _launch(self):
        self.backend.backward_into(
            self.q,
            self.kv,
            self.out,
            self.dy,
            self.indices,
            self.lse,
            self.sink,
            self.ptr,
            self.data,
            self.workspace,
            self.forward_plan.contract.softmax_scale,
        )
        self.dq_original.zero_()
        self.dq_original.index_copy_(0, self.order, self.workspace.dq)

    def capture(self):
        require(self.forward_plan.contract.graph, "Graph profile is disabled")
        self._validate()
        eager(self.device)
        with self.forward_plan._transaction(), torch.no_grad():
            for _ in range(3):
                self._launch()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._launch()
            self._graph = graph

    @torch.no_grad()
    def backward(self, dy, *, use_graph=False):
        eager(self.device)
        self._validate()
        require(
            dy.shape == self.record.output.shape
            and dy.device == self.device
            and dy.dtype in (torch.bfloat16, torch.float32)
            and dy.is_contiguous(),
            "dO must be contiguous BF16/FP32 in original query order",
        )
        require(bool(torch.isfinite(dy).all()), "dO must be finite")
        require(not use_graph or self._graph is not None, "capture backward before replay")
        with self.forward_plan._transaction(), torch.inference_mode(False), torch.no_grad():
            self.dy.copy_(dy.index_select(0, self.order).to(torch.bfloat16))
            if use_graph:
                self._graph.replay()
            else:
                self._launch()
            tensors = tuple(
                owned(t) for t in (self.dq_original, self.workspace.dkv, self.workspace.dsink)
            )
        return JointGradients(*tensors)

    @torch.no_grad()
    def local_parts(self, dy):
        """Return row-local contributions for the existing canonical session.

        No per-microbatch dKV/d_sink reduction is performed. The session later
        invokes the same upstream CSR and fixed sink reducers on the full,
        logically sorted query set.
        """
        eager(self.device)
        self._validate()
        require(
            dy.shape == self.record.output.shape
            and dy.device == self.device
            and dy.dtype in (torch.bfloat16, torch.float32)
            and dy.is_contiguous(),
            "dO ABI mismatch",
        )
        require(bool(torch.isfinite(dy).all()), "dO must be finite")
        with self.forward_plan._transaction(), torch.inference_mode(False), torch.no_grad():
            self.dy.copy_(dy.index_select(0, self.order).bfloat16())
            self.backend.local_parts_into(
                self.q,
                self.kv,
                self.out,
                self.dy,
                self.indices,
                self.lse,
                self.sink,
                self.workspace,
                self.forward_plan.contract.softmax_scale,
            )
            self.dq_original.zero_()
            self.dq_original.index_copy_(0, self.order, self.workspace.dq)
            intermediate = torch.zeros(
                (self.record.q.shape[0], self.indices.shape[1], self.q.shape[2]),
                dtype=torch.float32,
                device=self.device,
            )
            intermediate.index_copy_(0, self.order, self.workspace.intermediate)
            sink_rows = torch.zeros(
                self.record.q.shape[:2], dtype=torch.float32, device=self.device
            )
            sink_rows.index_copy_(
                0,
                self.order,
                self.backend.sink_contributions(self.workspace.delta, self.lse, self.sink),
            )
            dq = owned(self.dq_original)
        return JointLocalParts(dq, intermediate, sink_rows)
