# SPDX-License-Identifier: Apache-2.0
"""Contract-bound sparse auxiliary loss and quantized-operand training VJPs."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading

import torch

from rl_engine.contracts.operators.indexer_training import IndexerAuxContract, require
from rl_engine.models.deepseek_v4.indexer_projection import (
    IndexerLogicalRows,
    eager_only,
    tensor_stamp,
)
from rl_engine.ops.autograd.canonical_backward import active_session


@dataclass(frozen=True)
class IndexerAuxMetadata:
    queries: IndexerLogicalRows
    key_ids: torch.Tensor = field(repr=False)
    key_owners: torch.Tensor = field(repr=False)
    selected_indices: torch.Tensor = field(repr=False)
    selected_group_ids: torch.Tensor = field(repr=False)
    valid_rows: torch.Tensor = field(repr=False)
    stamps: tuple = field(repr=False)
    version: str
    contract_fingerprint: str

    @classmethod
    def bind(cls, contract, query_ids, key_ids, topk_group_ids, *, version, key_owners=None):
        """Top-K IDs are frozen logical group IDs, never differentiable indices.

        Canonical slot order is increasing logical key ID, preserving duplicate
        occurrences. Invalid slots follow the active entries. The caller owns
        alignment of all query/key-aligned numerical inputs with these IDs.
        """
        require(isinstance(contract, IndexerAuxContract), "typed auxiliary contract required")
        require(isinstance(version, str) and bool(version), "model/cache version required")
        require(
            query_ids.ndim == 2
            and query_ids.shape[1] == 2
            and query_ids.dtype == torch.int64
            and query_ids.shape[0] <= contract.max_queries,
            "query IDs must be int64 [M,2] within capacity",
        )
        require(
            key_ids.ndim == 2
            and key_ids.shape[1] == 2
            and key_ids.dtype == torch.int64
            and key_ids.shape[0] <= contract.max_keys,
            "key IDs must be int64 [N,2]",
        )
        require(
            topk_group_ids.shape == (query_ids.shape[0], contract.selected_capacity)
            and topk_group_ids.dtype in (torch.int32, torch.int64),
            "Top-K must be integer logical group IDs with fixed capacity",
        )
        device = query_ids.device
        require(key_ids.device == topk_group_ids.device == device, "metadata device mismatch")
        eager_only(device)
        queries = IndexerLogicalRows.bind(query_ids)
        # Validate exactly the owned bytes consumed later, not mutable caller
        # storage that could change while host-side admission is running.
        with torch.inference_mode(False), torch.no_grad():
            keys = key_ids.detach().clone().contiguous()
            choices = topk_group_ids.detach().clone().contiguous()
        q_host = queries.keys.cpu().tolist()
        k_host = keys.cpu().tolist()
        topk = choices.cpu().tolist()
        require(
            all((r >= 0 and 0 <= p < 2**31) or (r == p == -1) for r, p in q_host),
            "invalid global query ID/padding",
        )
        require(
            len(queries.active_keys) <= contract.global_query_count,
            "local query count exceeds declared global normalization",
        )
        require(
            all(r >= 0 and 0 <= g < (2**31 // contract.ratio) for r, g in k_host),
            "invalid global compressed-key ID",
        )
        require(len(k_host) == len(set(map(tuple, k_host))), "duplicate source KV identity")
        if key_owners is None:
            require(contract.cp_size == 1, "CP ownership must be supplied explicitly")
            key_owners = torch.zeros(key_ids.shape[0], dtype=torch.int64, device=device)
        require(
            key_owners.shape == (key_ids.shape[0],)
            and key_owners.dtype == torch.int64
            and key_owners.device == device,
            "key ownership ABI mismatch",
        )
        with torch.inference_mode(False), torch.no_grad():
            owners = key_owners.detach().clone().contiguous()
        require(
            bool(((owners >= 0) & (owners < contract.cp_size)).all()),
            "key owner outside declared CP group",
        )
        lookup = {tuple(key): index for index, key in enumerate(k_host)}
        selections, canonical_ids, active = [], [], []
        for (request, position), ids in zip(q_host, topk, strict=True):
            require(all(g >= -1 for g in ids), "only -1 may mark an invalid Top-K slot")
            chosen = sorted(g for g in ids if g >= 0)
            require(request >= 0 or not chosen, "padding query cannot select KV")
            for group in chosen:
                require(
                    (request, group) in lookup, "selected KV is missing (including remote CP keys)"
                )
                require(
                    (group + 1) * contract.ratio - 1 <= position, "future compressed KV selected"
                )
            selections.append(
                [lookup[request, g] for g in chosen]
                + [-1] * (contract.selected_capacity - len(chosen))
            )
            canonical_ids.append(chosen + [-1] * (contract.selected_capacity - len(chosen)))
            active.append(request >= 0 and bool(chosen))
        with torch.inference_mode(False), torch.no_grad():
            indices = torch.tensor(selections, dtype=torch.int32, device=device).reshape(
                query_ids.shape[0], contract.selected_capacity
            )
            group_ids = torch.tensor(canonical_ids, dtype=torch.int64, device=device).reshape_as(
                indices
            )
            valid_rows = torch.tensor(active, dtype=torch.bool, device=device)
        tensors = (keys, owners, indices, group_ids, valid_rows)
        return cls(
            queries,
            *tensors,
            tuple(tensor_stamp(t) for t in tensors),
            version,
            contract.fingerprint,
        )

    def validate(self, contract, queries, keys, device, version):
        require(
            self.contract_fingerprint == contract.fingerprint and self.version == version,
            "auxiliary contract/version mismatch",
        )
        self.queries.validate(queries, device)
        require(
            self.key_ids.shape == (keys, 2) and self.key_ids.device == device,
            "source KV metadata mismatch",
        )
        tensors = (
            self.key_ids,
            self.key_owners,
            self.selected_indices,
            self.selected_group_ids,
            self.valid_rows,
        )
        require(all(t.device == device for t in tensors), "auxiliary metadata device mismatch")
        require(
            tuple(tensor_stamp(t) for t in tensors) == self.stamps,
            "bound auxiliary metadata was mutated",
        )


@dataclass(frozen=True)
class PreparedIndexerTrainingOperand:
    master: torch.Tensor = field(repr=False)
    quantized: object = field(repr=False)
    prepared: torch.Tensor = field(repr=False)
    scale: torch.Tensor | None = field(repr=False)
    real: torch.Tensor = field(repr=False)
    role: str
    version: str
    implementation: str
    stamps: tuple = field(repr=False)

    def validate(self, *, role, version, implementation):
        require(
            (self.role, self.version, self.implementation) == (role, version, implementation),
            "training operand role/version/provider mismatch",
        )
        self.quantized.validate()
        tensors = (self.master, self.prepared, self.real)
        current = tuple(tensor_stamp(t) for t in tensors)
        if self.scale is not None:
            current += (tensor_stamp(self.scale),)
        require(current == self.stamps, "training master/decoded operand changed; prepare again")


@dataclass(frozen=True)
class IndexerAuxForwardRecord:
    per_query_loss: torch.Tensor
    raw_icv: torch.Tensor
    relu: torch.Tensor
    predicted: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor = field(repr=False)
    q: PreparedIndexerTrainingOperand = field(repr=False)
    k: PreparedIndexerTrainingOperand = field(repr=False)
    metadata: IndexerAuxMetadata
    implementation: str
    stamps: tuple = field(repr=False)

    def validate_saved(self):
        tensors = (
            self.per_query_loss,
            self.raw_icv,
            self.relu,
            self.predicted,
            self.target,
            self.weights,
        )
        require(
            tuple(tensor_stamp(t) for t in tensors) == self.stamps,
            "saved Indexer forward state was mutated",
        )
        self.metadata.queries.validate(self.per_query_loss.shape[0], self.q.master.device)
        meta_tensors = (
            self.metadata.key_ids,
            self.metadata.key_owners,
            self.metadata.selected_indices,
            self.metadata.selected_group_ids,
            self.metadata.valid_rows,
        )
        require(
            tuple(tensor_stamp(t) for t in meta_tensors) == self.metadata.stamps,
            "saved Indexer metadata was mutated",
        )
        self.q.validate(role="q", version=self.metadata.version, implementation=self.implementation)
        self.k.validate(role="k", version=self.metadata.version, implementation=self.implementation)


class IndexerAuxiliary:
    """Native sparse auxiliary arithmetic and explicit identity-STE autograd."""

    def __init__(self, contract, *, layer_id, backend=None, expected_implementation=None):
        require(isinstance(contract, IndexerAuxContract), "typed auxiliary contract required")
        require(isinstance(layer_id, str) and bool(layer_id), "stable Indexer layer ID required")
        if backend is None:
            from rl_engine.backends.rocm.attention.indexer_training import AiterIndexerAuxBackend

            backend = AiterIndexerAuxBackend(contract)
        require(backend.contract == contract, "auxiliary backend contract mismatch")
        self.contract, self.layer_id, self.backend = contract, layer_id, backend
        from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend
        from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend

        identity = {
            "contract": contract.fingerprint,
            "provider": backend.provenance,
            "bridge": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "canonical": hashlib.sha256(
                Path(active_session.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "reduction": hashlib.sha256(
                Path(AiterJointBackwardBackend.reduce_sink_rows.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "pinned_csr_loader": hashlib.sha256(
                Path(AiterJointAttentionBackend.__init__.__code__.co_filename).read_bytes()
            ).hexdigest(),
        }
        self.implementation = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        require(
            expected_implementation is None or expected_implementation == self.implementation,
            "Indexer auxiliary implementation handshake mismatch",
        )

    def autograd(self, q, k, weights, teacher_q, teacher_k, *, metadata, version):
        require(
            isinstance(q, PreparedIndexerTrainingOperand)
            and isinstance(k, PreparedIndexerTrainingOperand)
            and isinstance(weights, torch.Tensor),
            "prepared operands and tensor weights required",
        )
        if torch.is_grad_enabled() and any(t.requires_grad for t in (q.master, k.master, weights)):
            if k.master.requires_grad:
                require(
                    active_session() is not None,
                    "shared auxiliary K gradients require canonical_backward_session",
                )
                self.backend.enable_backward()
            losses, predicted, target = _IndexerAuxAutograd.apply(
                q.master,
                k.master,
                weights,
                teacher_q.detach(),
                teacher_k.detach(),
                self,
                q,
                k,
                metadata,
                version,
            )
        else:
            saved = self.forward_record(
                q, k, weights, teacher_q, teacher_k, metadata=metadata, version=version
            )
            losses, predicted, target = saved.per_query_loss, saved.predicted, saved.target
        return IndexerAuxResult(losses, predicted, target, metadata, self.implementation)

    def transform_queries(self, q, positions, cos_cache, sin_cache):
        """Explicit Q RoPE -> normalized native Hadamard; no implicit RMSNorm.

        Reuses the task-20 tail-safe Hadamard adapter and native cached RoPE.
        This is training composition, not a claim to finish another owner's
        fused-query-preprocessing task.
        """
        require(
            q.ndim == 3
            and q.shape[1:] == (self.contract.num_heads, 128)
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
            and q.device == self.backend.device,
            "Q transform ABI mismatch",
        )
        require(
            positions.shape == (q.shape[0],)
            and positions.dtype == torch.int64
            and positions.device == q.device,
            "global Q positions required",
        )
        eager_only(q.device)
        require(
            cos_cache.ndim == 2
            and cos_cache.shape == sin_cache.shape
            and cos_cache.shape[1] == 32
            and cos_cache.shape[0] > 0,
            "Q RoPE64 requires [positions,32] cos/sin tables",
        )
        require(
            all(
                t.device == q.device and t.dtype == torch.bfloat16 and not t.requires_grad
                for t in (cos_cache, sin_cache)
            ),
            "read-only BF16 RoPE tables required",
        )
        require(
            bool(((positions >= -1) & (positions < cos_cache.shape[0])).all()),
            "Q RoPE position outside cache",
        )
        with torch.inference_mode(False), torch.no_grad():
            selected = positions.clamp_min(0)
            cos = cos_cache.index_select(0, selected).float().reshape(-1, 1, 1, 32)
            sin = sin_cache.index_select(0, selected).float().reshape(-1, 1, 1, 32)
        require(
            bool(
                (
                    torch.isfinite(cos) & torch.isfinite(sin) & (cos.abs() <= 1) & (sin.abs() <= 1)
                ).all()
            ),
            "invalid Q RoPE angles",
        )
        masked = torch.where((positions >= 0)[:, None, None], q, 0)
        return _IndexerQueryTransform.apply(masked, cos, sin, self.backend)

    def prepare_operand(self, master, *, role, version, packed=None):
        require(isinstance(master, torch.Tensor), "training master tensor required")
        require(
            role in ("q", "k") and isinstance(version, str) and bool(version),
            "operand role and version required",
        )
        require(not master.is_inference(), "training masters must have version counters")
        expected_rank = 3 if role == "q" else 2
        limit = self.contract.max_queries if role == "q" else self.contract.max_keys
        require(
            master.ndim == expected_rank and master.shape[0] <= limit and master.shape[-1] == 128,
            "master geometry/capacity mismatch",
        )
        if role == "q":
            require(master.shape[1] == self.contract.num_heads, "query head count mismatch")
        require(
            master.numel() * 16 <= self.contract.max_workspace_bytes,
            "operand preparation workspace budget exceeded",
        )
        eager_only(master.device)
        with torch.inference_mode(False), torch.no_grad():
            operand, prepared, scale, real = self.backend.encode(
                master, role=role, version=version, packed=packed
            )
        tensors = (master, prepared, real)
        stamps = tuple(tensor_stamp(t) for t in tensors)
        if scale is not None:
            stamps += (tensor_stamp(scale),)
        return PreparedIndexerTrainingOperand(
            master, operand, prepared, scale, real, role, version, self.implementation, stamps
        )

    def workspace_estimate(self, m, n):
        """Conservative tensor-workspace estimate, excluding library/JIT startup."""
        c = self.contract
        p, h, d, th, td = (
            c.selected_capacity,
            c.num_heads,
            c.head_dim,
            c.teacher_heads,
            c.teacher_dim,
        )
        return (
            m * p * (12 * h + 12 * th + 10 * d + 2 * td) + 12 * m * h * d + 8 * n * d + 32 * m * p
        )

    @torch.no_grad()
    def forward_record(self, q, k, weights, teacher_q, teacher_k, *, metadata, version):
        c = self.contract
        require(
            isinstance(q, PreparedIndexerTrainingOperand)
            and isinstance(k, PreparedIndexerTrainingOperand)
            and isinstance(metadata, IndexerAuxMetadata),
            "prepared operands/metadata required",
        )
        q.validate(role="q", version=version, implementation=self.implementation)
        k.validate(role="k", version=version, implementation=self.implementation)
        m, n = q.master.shape[0], k.master.shape[0]
        metadata.validate(c, m, n, self.backend.device, version)
        require(
            q.master.shape == (m, c.num_heads, 128) and k.master.shape == (n, 128),
            "training-master geometry mismatch",
        )
        require(
            q.master.device == k.master.device == self.backend.device,
            "training-master device mismatch",
        )
        for tensor, shape, dtype, name in (
            (weights, (m, c.num_heads), torch.float32, "weights"),
            (teacher_q, (m, c.teacher_heads, c.teacher_dim), torch.bfloat16, "teacher Q"),
            (teacher_k, (n, c.teacher_dim), torch.bfloat16, "teacher K"),
        ):
            require(
                tensor.shape == shape
                and tensor.dtype == dtype
                and tensor.is_contiguous()
                and tensor.device == self.backend.device,
                f"{name} ABI mismatch",
            )
            require(bool(torch.isfinite(tensor).all()), f"non-finite {name}")
        require(
            self.workspace_estimate(m, n) <= c.max_workspace_bytes,
            "auxiliary tensor-workspace budget exceeded",
        )
        eager_only(self.backend.device)
        with torch.cuda.device(self.backend.device), torch.inference_mode(False), torch.no_grad():
            owned_weights = weights.detach().clone()
            values = self.backend.forward_sparse(
                q.prepared,
                k.prepared,
                q.scale,
                k.scale,
                owned_weights,
                teacher_q.detach(),
                teacher_k.detach(),
                metadata,
            )
        tensors = (*values, owned_weights)
        return IndexerAuxForwardRecord(
            *tensors, q, k, metadata, self.implementation, tuple(tensor_stamp(t) for t in tensors)
        )


@dataclass(frozen=True)
class IndexerAuxResult:
    per_query_loss: torch.Tensor
    predicted: torch.Tensor
    target: torch.Tensor
    metadata: IndexerAuxMetadata
    implementation: str


class _IndexerQueryTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cos, sin, backend):
        ctx.save_for_backward(cos, sin)
        ctx.backend = backend
        return backend.query_transform(q, cos, sin)

    @staticmethod
    def backward(ctx, grad):
        require(not torch.is_grad_enabled(), "higher-order Q-transform gradients are not supported")
        cos, sin = ctx.saved_tensors
        return ctx.backend.query_transform_vjp(grad.contiguous(), cos, sin), None, None, None


@dataclass
class _KeyBinding:
    master: torch.Tensor
    operand: PreparedIndexerTrainingOperand
    stamp: tuple
    implementation: str
    version: str
    keys: set = field(default_factory=set)
    metadata: list = field(default_factory=list)
    submitted: set = field(default_factory=set)
    event: object = None
    lock: object = field(default_factory=threading.RLock)


def _register_keys(session, operator, operand, metadata):
    require(not session.backward_started, "register all auxiliary forwards before backward")
    master = operand.master
    bindings = getattr(session, "_indexer_aux_key_bindings", None)
    if bindings is None:
        bindings = {}
        session._indexer_aux_key_bindings = bindings
    prefix = f"{operator.layer_id}.indexer_aux.keys"
    binding = bindings.setdefault(
        prefix,
        _KeyBinding(
            master, operand, tensor_stamp(master), operator.implementation, metadata.version
        ),
    )
    require(
        binding.master is master and tensor_stamp(master) == binding.stamp,
        "reuse one immutable prepared KV master within a canonical auxiliary session",
    )
    require(binding.operand is operand, "reuse the same encoded KV snapshot within a session")
    require(
        binding.implementation == operator.implementation and binding.version == metadata.version,
        "auxiliary arithmetic/version changed within a session",
    )
    if binding.metadata:
        previous = binding.metadata[0]
        require(
            torch.equal(previous.key_ids, metadata.key_ids)
            and torch.equal(previous.key_owners, metadata.key_owners),
            "auxiliary KV identity/CP ownership changed",
        )
    keys = metadata.queries.active_keys
    require(not binding.keys.intersection(keys), "duplicate auxiliary logical query")
    require(
        len(binding.keys) + len(keys) <= operator.contract.global_query_count,
        "canonical query count exceeds global normalization",
    )
    require(
        operator.workspace_estimate(len(binding.keys) + len(keys), master.shape[0])
        <= operator.contract.max_workspace_bytes,
        "canonical auxiliary workspace budget exceeded",
    )
    binding.keys.update(keys)
    if not keys:
        return prefix, binding, None
    binding.metadata.append(metadata)
    return prefix, binding, session.register(prefix, metadata.queries.keys)


def _sorted_selection(binding):
    for meta in binding.metadata:
        meta.queries.validate(meta.queries.keys.shape[0], meta.key_ids.device)
        tensors = (
            meta.key_ids,
            meta.key_owners,
            meta.selected_indices,
            meta.selected_group_ids,
            meta.valid_rows,
        )
        require(
            tuple(tensor_stamp(t) for t in tensors) == meta.stamps,
            "canonical auxiliary metadata was mutated",
        )
    keys = torch.cat([meta.queries.keys for meta in binding.metadata])
    selected = torch.cat([meta.selected_indices for meta in binding.metadata])
    valid = keys[:, 0] >= 0
    keys, selected = keys[valid], selected[valid]
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in (1, 0):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return selected.index_select(0, order).contiguous()


class _IndexerAuxAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q_master, k_master, weights, teacher_q, teacher_k, operator, q, k, metadata, version
    ):
        saved = operator.forward_record(
            q, k, weights, teacher_q, teacher_k, metadata=metadata, version=version
        )
        ctx.save_for_backward(q_master, k_master, weights)
        ctx.saved, ctx.operator = saved, operator
        ctx.session, ctx.registration = active_session(), None
        if k_master.requires_grad:
            require(
                ctx.session is not None,
                "shared auxiliary K gradients require canonical_backward_session",
            )
            ctx.registration = _register_keys(ctx.session, operator, k, metadata)
        ctx.mark_non_differentiable(saved.predicted, saved.target)
        return saved.per_query_loss, saved.predicted, saved.target

    @staticmethod
    def backward(ctx, grad_loss, _grad_predicted, _grad_target):
        require(not torch.is_grad_enabled(), "higher-order auxiliary gradients are not supported")
        q_master, k_master, weights = ctx.saved_tensors
        ctx.saved.validate_saved()
        operator = ctx.operator
        with torch.cuda.device(operator.backend.device):
            dq, dk_rows, dw = operator.backend.backward_parts(
                ctx.saved, grad_loss.float().contiguous()
            )
            dk = None
            if ctx.needs_input_grad[1]:
                prefix, binding, slot = ctx.registration
                if slot is None:
                    dk = torch.zeros_like(k_master)
                else:
                    with binding.lock:
                        require(
                            slot not in binding.submitted,
                            "auxiliary backward was already submitted for this forward",
                        )
                        binding.submitted.add(slot)
                        if binding.event is not None:
                            torch.cuda.current_stream().wait_event(binding.event)

                        def reduce(rows):
                            indices = _sorted_selection(binding)
                            shaped = rows.reshape(indices.shape[0], indices.shape[1], 128)
                            return operator.backend.reduce_keys(shaped, indices, k_master.shape[0])

                        dk = ctx.session.submit_rows(prefix, slot, dk_rows, reduce)
                        binding.event = torch.cuda.Event()
                        binding.event.record()
                    if dk is not None:
                        dk = dk.to(k_master.dtype)
        return (
            dq.to(q_master.dtype) if ctx.needs_input_grad[0] else None,
            dk,
            dw.to(weights.dtype) if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _CanonicalAuxSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        from rl_engine.backends.rocm.attention.joint_backward import AiterJointBackwardBackend

        ctx.shape = values.shape
        return AiterJointBackwardBackend.reduce_sink_rows(values.reshape(-1, 1))[0]

    @staticmethod
    def backward(ctx, grad):
        return grad.expand(ctx.shape)


def canonical_indexer_auxiliary_loss(results):
    """One scalar/backward over all roots, not a sum of rounded chunk losses.

    This also keeps a shared nonleaf compressed-K graph alive until the whole
    canonical set has contributed. Separate backward calls on shared nonleaf
    graphs require the caller to manage retain_graph explicitly.
    """
    require(bool(results), "at least one (possibly empty) auxiliary result is required")
    first = results[0]
    require(
        all(
            r.implementation == first.implementation
            and r.metadata.version == first.metadata.version
            for r in results
        ),
        "cannot merge different auxiliary objectives/providers/versions",
    )
    for result in results:
        result.metadata.queries.validate(
            result.per_query_loss.shape[0], result.per_query_loss.device
        )
    keys = torch.cat([r.metadata.queries.keys for r in results])
    losses = torch.cat([r.per_query_loss for r in results])
    logical = [key for r in results for key in r.metadata.queries.active_keys]
    require(len(logical) == len(set(logical)), "duplicate query in auxiliary loss merge")
    valid = keys[:, 0] >= 0
    keys, losses = keys[valid], losses[valid]
    order = torch.arange(keys.shape[0], device=keys.device)
    for column in (1, 0):
        order = order[torch.argsort(keys[order, column], stable=True)]
    return _CanonicalAuxSum.apply(losses.index_select(0, order))
