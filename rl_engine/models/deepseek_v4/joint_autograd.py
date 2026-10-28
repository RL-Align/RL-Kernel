# SPDX-License-Identifier: Apache-2.0
"""Native sparse VJPs with the existing canonical row-aggregation session."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import torch

from rl_engine.contracts.operators.joint_attention import require
from rl_engine.models.deepseek_v4.joint_attention import stamp
from rl_engine.models.deepseek_v4.joint_backward import JointBackwardPlan
from rl_engine.ops.autograd.canonical_backward import active_session


@dataclass
class _Binding:
    tensor: torch.Tensor
    tensor_stamp: tuple
    version: str
    implementation: str
    backward_implementation: str
    plan: object
    metadata: list = field(default_factory=list)
    keys: set = field(default_factory=set)


def _register(session, prefix, tensor, saved, operator):
    bindings = getattr(session, "_joint_backward_bindings", None)
    if bindings is None:
        bindings = {}
        session._joint_backward_bindings = bindings
    binding = bindings.setdefault(
        prefix,
        _Binding(
            tensor,
            stamp(tensor),
            saved.metadata.version,
            saved.implementation,
            operator.backward_implementation,
            operator.plan,
        ),
    )
    require(
        binding.tensor is tensor and binding.tensor_stamp == stamp(tensor),
        "canonical joint parameter identity/version mismatch",
    )
    require(
        binding.version == saved.metadata.version
        and binding.implementation == saved.implementation
        and binding.backward_implementation == operator.backward_implementation,
        "joint arithmetic/version mismatch",
    )
    host_keys = [tuple(row) for row in saved.metadata.query_keys.cpu().tolist() if row[0] >= 0]
    require(
        binding.plan is operator.plan,
        "one canonical parameter session must share one event-owned plan",
    )
    require(
        not binding.keys.intersection(host_keys),
        "duplicate joint logical query across microbatches",
    )
    require(
        len(binding.keys) + len(host_keys) <= operator.plan.contract.max_queries,
        "canonical joint query capacity exceeded",
    )
    c = operator.plan.contract
    require(
        operator.backend.workspace_bytes(
            len(binding.keys) + len(host_keys),
            c.num_heads,
            c.head_dim,
            c.selected_capacity,
            saved.kv.shape[0],
        )
        <= operator.max_workspace_bytes,
        "canonical joint workspace budget exceeded",
    )
    if binding.metadata:
        previous = binding.metadata[0]
        require(
            torch.equal(previous.kv_keys, saved.metadata.kv_keys)
            and torch.equal(previous.permutation, saved.metadata.permutation),
            "canonical session must use one KV identity/loader order",
        )
    binding.keys.update(host_keys)
    if not host_keys:
        return binding, None
    binding.metadata.append(saved.metadata)
    return binding, session.register(prefix, saved.metadata.query_keys)


def _canonical_indices(binding):
    keys = torch.cat([meta.query_keys for meta in binding.metadata], 0)
    indices = torch.cat([meta.indices for meta in binding.metadata], 0)
    for meta in binding.metadata:
        require(
            tuple(stamp(t) for t in (meta.query_keys, meta.kv_keys, meta.permutation, meta.indices))
            == meta.stamps,
            "canonical joint metadata was mutated",
        )
    valid = keys[:, 0] >= 0
    keys, indices = keys[valid], indices[valid]
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in (1, 0):
        order = order.index_select(0, torch.argsort(keys[order, column], stable=True))
    return indices.index_select(0, order).contiguous()


class _JointAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, metadata, version, mode, use_graph, operator):
        saved = operator.plan.forward_record(
            q, kv, sink, metadata=metadata, version=version, mode=mode, use_graph=use_graph
        )
        ctx.save_for_backward(q, kv, sink)
        ctx.saved, ctx.operator = saved, operator
        ctx.session = active_session()
        ctx.bindings = {}
        for name, tensor in (("kv", kv), ("sink", sink)):
            if tensor.requires_grad:
                require(
                    ctx.session is not None,
                    "joint KV/sink gradients require canonical_backward_session",
                )
                prefix = f"{operator.layer_id}.joint.{name}"
                binding, slot = _register(ctx.session, prefix, tensor, saved, operator)
                ctx.bindings[name] = (prefix, binding, slot)
        ctx.mark_non_differentiable(saved.lse)
        return saved.output, saved.lse

    @staticmethod
    def backward(ctx, grad, grad_lse):
        require(not torch.is_grad_enabled(), "higher-order joint derivatives are not supported")
        q, kv, sink = ctx.saved_tensors  # PyTorch version checks on original inputs.
        ctx.saved.validate_saved()
        operator = ctx.operator
        plan = JointBackwardPlan(
            operator.plan,
            ctx.saved,
            backend=operator.backend,
            max_workspace_bytes=operator.max_workspace_bytes,
        )
        parts = plan.local_parts(grad.contiguous())
        dq = parts.dq.to(q.dtype) if ctx.needs_input_grad[0] else None
        dkv, dsink = None, None
        if ctx.needs_input_grad[1]:
            prefix, binding, slot = ctx.bindings["kv"]
            if slot is None:
                dkv = torch.zeros_like(kv)
            else:

                def reduce_kv(rows):
                    indices = _canonical_indices(binding)
                    require(rows.shape[0] == indices.shape[0], "canonical CSR row mismatch")
                    intermediate = rows.reshape(
                        rows.shape[0],
                        operator.plan.contract.selected_capacity,
                        operator.plan.contract.head_dim,
                    ).contiguous()
                    with operator.plan._transaction():
                        ptr, data = operator.backend.build_csr(indices, kv.shape[0])
                        result = torch.empty_like(ctx.saved.kv, dtype=torch.float32)
                        operator.backend.gather_intermediate_into(
                            intermediate, indices, ptr, data, result
                        )
                    return result

                dkv = ctx.session.submit_rows(prefix, slot, parts.intermediate, reduce_kv)
                if dkv is not None:
                    dkv = dkv.index_select(0, torch.argsort(ctx.saved.metadata.permutation)).to(
                        kv.dtype
                    )
        if ctx.needs_input_grad[2]:
            prefix, _, slot = ctx.bindings["sink"]
            if slot is None:
                dsink = torch.zeros_like(sink)
            else:
                dsink = ctx.session.submit_rows(
                    prefix, slot, parts.sink_rows, operator.backend.reduce_sink_rows
                )
        return dq, dkv, dsink, None, None, None, None, None


@dataclass(frozen=True)
class JointTrainingResult:
    output: torch.Tensor
    lse: torch.Tensor


class JointAttentionAutograd:
    def __init__(
        self,
        plan,
        *,
        layer_id,
        backend=None,
        max_workspace_bytes=1 << 30,
        expected_backward_implementation=None,
    ):
        require(isinstance(layer_id, str) and bool(layer_id), "stable joint layer ID required")
        if backend is None:
            from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend

            backend = AiterJointBackwardBackend(plan.backend)
        require(
            backend.forward_backend is plan.backend, "joint backward backend ownership mismatch"
        )
        self.plan, self.backend = plan, backend
        self.layer_id, self.max_workspace_bytes = layer_id, max_workspace_bytes
        require(
            type(max_workspace_bytes) is int and max_workspace_bytes > 0,
            "positive backward workspace budget required",
        )
        identity = {
            "forward": plan.implementation,
            "backward": backend.provenance,
            "autograd_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "canonical_sha256": hashlib.sha256(
                Path(active_session.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "plan_sha256": hashlib.sha256(
                Path(JointBackwardPlan.__init__.__code__.co_filename).read_bytes()
            ).hexdigest(),
        }
        self.backward_implementation = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        require(
            expected_backward_implementation is None
            or expected_backward_implementation == self.backward_implementation,
            "backward implementation handshake mismatch",
        )

    def __call__(self, q, kv, sink, *, metadata, version, mode="train", use_graph=False):
        if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, kv, sink)):
            saved = self.plan.forward_record(
                q, kv, sink, metadata=metadata, version=version, mode=mode, use_graph=use_graph
            )
            return JointTrainingResult(saved.output, saved.lse)
        output, lse = _JointAutograd.apply(q, kv, sink, metadata, version, mode, use_graph, self)
        return JointTrainingResult(output, lse)
