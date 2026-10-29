# SPDX-License-Identifier: Apache-2.0
"""Complete-group C4 admission, fixed overlap state and canonical VJP wiring."""

from dataclasses import dataclass, field
import hashlib
import json

import torch

from rl_engine.contracts.operators.kv_compressor_c4 import C4Contract, require, source_hash
from rl_engine.ops.autograd.canonical_backward import active_session


def eager(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            require(not torch.cuda.is_current_stream_capturing(), "Graph capture is not validated")


def stamp(tensor):
    return (
        tensor.data_ptr(),
        None if tensor.is_inference() else tensor._version,
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
    )


@dataclass(frozen=True)
class C4Groups:
    source: torch.Tensor = field(repr=False)
    values: torch.Tensor = field(repr=False)
    context_values: torch.Tensor = field(repr=False)
    context_indices: torch.Tensor = field(repr=False)
    selector: torch.Tensor = field(repr=False)
    group_ids: torch.Tensor = field(repr=False)
    plan: torch.Tensor = field(repr=False)
    block_table: torch.Tensor = field(repr=False)
    kv_state: torch.Tensor = field(repr=False)
    score_state: torch.Tensor = field(repr=False)
    state_slots: torch.Tensor = field(repr=False)
    logical_keys: tuple[tuple[int, int], ...]
    request_spans: tuple[tuple[int, int], ...]
    incomplete_groups: tuple[tuple[int, int], ...]
    unowned_groups: tuple[tuple[int, int], ...]
    context_indices_host: tuple[tuple[int, ...], ...]
    cache_slots_host: tuple[tuple[int, int], ...]
    version: str
    owner_rank: int
    cp_size: int
    contract: C4Contract
    _stamps: tuple = field(repr=False)
    _source_stamp: tuple = field(repr=False)

    @property
    def tensors(self):
        return (
            self.values,
            self.context_values,
            self.context_indices,
            self.selector,
            self.group_ids,
            self.plan,
            self.block_table,
            self.kv_state,
            self.score_state,
            self.state_slots,
        )

    def validate(self):
        require(tuple(stamp(t) for t in self.tensors) == self._stamps, "bound C4 storage changed")
        if self.source.requires_grad:
            require(stamp(self.source) == self._source_stamp, "training C4 source changed")


@dataclass(frozen=True)
class C4Output:
    values: torch.Tensor
    group_ids: torch.Tensor
    owner_rank: int
    version: str
    contract_fingerprint: str
    implementation_fingerprint: str


def bind_complete_groups(
    values,
    token_ids,
    *,
    version,
    contract,
    owners=None,
    cp_size=1,
    rank=0,
    emit_group_ids=None,
):
    require(isinstance(contract, C4Contract), "typed C4 contract required")
    require(
        isinstance(version, str) and bool(version),
        "explicit model/position/request version required",
    )
    require(
        isinstance(values, torch.Tensor)
        and values.dtype == torch.bfloat16
        and values.ndim == 2
        and values.shape[1] == contract.input_width
        and values.is_contiguous(),
        "projected [tokens, 4*head_dim] contiguous BF16 input required",
    )
    eager(values.device)
    require(values.shape[0] <= contract.max_input_rows, "input capacity exceeded")
    require(
        isinstance(token_ids, torch.Tensor)
        and token_ids.dtype == torch.int64
        and tuple(token_ids.shape) == (values.shape[0], 2)
        and token_ids.device == values.device,
        "token IDs must be int64 [tokens,2] on the input device",
    )
    require(
        type(cp_size) is int and cp_size > 0 and type(rank) is int and 0 <= rank < cp_size,
        "invalid CP size/rank",
    )
    host_ids = tuple(tuple(row) for row in token_ids.cpu().tolist())
    if owners is None:
        require(cp_size == 1, "CP token ownership metadata required")
        host_owners = (0,) * len(host_ids)
    else:
        require(
            isinstance(owners, torch.Tensor)
            and owners.dtype == torch.int64
            and tuple(owners.shape) == (values.shape[0],)
            and owners.device == values.device,
            "owners must be int64 [tokens] on the input device",
        )
        host_owners = tuple(owners.cpu().tolist())
    emit = None if emit_group_ids is None else set(tuple(key) for key in emit_group_ids)
    if emit is not None:
        require(
            all(len(key) == 2 and key[0] >= 0 and key[1] >= 0 for key in emit),
            "emit_group_ids must contain nonnegative [request,group] keys",
        )
    groups, seen = {}, set()
    for physical, (key, owner) in enumerate(zip(host_ids, host_owners, strict=True)):
        if key == (-1, -1):
            continue
        request, position = key
        require(request >= 0 and 0 <= position < 2**31, "invalid logical token ID")
        require(key not in seen, "duplicate logical token ID")
        require(0 <= owner < cp_size, "token owner outside CP group")
        seen.add(key)
        group, offset = divmod(position, 4)
        groups.setdefault((request, group), {})[offset] = (physical, owner)

    complete = {key: entries for key, entries in groups.items() if len(entries) == 4}
    incomplete = [key for key, entries in groups.items() if len(entries) != 4]
    selected, unowned = [], []
    for key, entries in sorted(complete.items()):
        request, group = key
        if emit is not None and key not in emit:
            continue
        if group > 0 and (request, group - 1) not in complete:
            incomplete.append(key)
            continue
        if entries[3][1] != rank:
            unowned.append(key)
            continue
        selected.append(key)
    selected = sorted(selected)
    require(len(selected) <= contract.max_groups, "complete-group capacity exceeded")
    request_order = {request: i for i, request in enumerate(sorted({key[0] for key in selected}))}
    max_block_column = max((key[1] // 64 + 1 for key in selected), default=0)
    require(
        len(request_order) * max_block_column * 4 <= contract.max_block_table_bytes,
        "C4 block-table byte budget exceeded",
    )
    block_order = {
        key: index
        for index, key in enumerate(sorted({(request, group // 64) for request, group in selected}))
    }

    current_indices = []
    context_indices_host = []
    for request, group in selected:
        current = [complete[(request, group)][offset][0] for offset in range(4)]
        previous = (
            [complete[(request, group - 1)][offset][0] for offset in range(4)]
            if group > 0
            else [-1] * 4
        )
        current_indices.extend(current)
        context_indices_host.append(tuple(previous + current))
    cache_slots_host = tuple(
        (block_order[(request, group // 64)], group % 64) for request, group in selected
    )
    with torch.inference_mode(False), torch.no_grad():
        selector = torch.tensor(current_indices, dtype=torch.int64, device=values.device)
        packed = values.index_select(0, selector).detach()
        context_indices = torch.tensor(
            context_indices_host, dtype=torch.int64, device=values.device
        ).reshape(-1, 8)
        safe_context = context_indices.clamp_min(0)
        context = (
            values.index_select(0, safe_context.reshape(-1))
            .reshape(-1, 8, contract.input_width)
            .detach()
        )
        if context.numel():
            context[:, :4] = torch.where(
                (context_indices[:, :4] >= 0).unsqueeze(-1),
                context[:, :4],
                torch.zeros_like(context[:, :4]),
            )
        require(
            bool((torch.isfinite(context.float()) & (context.float().abs() <= 2.0**30)).all()),
            "selected input outside finite C4 FP32 accumulation domain",
        )
        group_ids = torch.tensor(selected, dtype=torch.int64, device=values.device).reshape(-1, 2)
        plan = torch.tensor(
            [
                [i * 8 + 7, request_order[request], group * 4 + 3, 4 if group == 0 else 0]
                for i, (request, group) in enumerate(selected)
            ],
            dtype=torch.int32,
            device=values.device,
        ).reshape(-1, 4)
        table = torch.full(
            (len(request_order), max_block_column),
            -1,
            dtype=torch.int32,
            device=values.device,
        )
        for (request, group), (block, _slot) in zip(selected, cache_slots_host, strict=True):
            table[request_order[request], group // 64] = block
        slots = torch.arange(len(request_order), dtype=torch.int32, device=values.device)
        state_kv = torch.zeros(
            (len(request_order), 8, 2 * contract.head_dim),
            dtype=torch.float32,
            device=values.device,
        )
        state_score = torch.zeros_like(state_kv)
    spans = []
    cursor = 0
    for request in sorted({key[0] for key in selected}):
        length = sum(key[0] == request for key in selected)
        spans.append((cursor, cursor + length))
        cursor += length
    tensors = (
        packed,
        context,
        context_indices,
        selector,
        group_ids,
        plan,
        table,
        state_kv,
        state_score,
        slots,
    )
    return C4Groups(
        values,
        *tensors,
        tuple(selected),
        tuple(spans),
        tuple(sorted(set(incomplete))),
        tuple(unowned),
        tuple(context_indices_host),
        cache_slots_host,
        version,
        rank,
        cp_size,
        contract,
        tuple(stamp(tensor) for tensor in tensors),
        stamp(values),
    )


def canonical_sum(rows):
    result = torch.zeros(rows.shape[1:], dtype=rows.dtype, device=rows.device)
    for row in rows.unbind(0):
        result = torch.add(result, row)
    return result


def register_parameters(session, prefix, parameters, bound, implementation):
    require(session is not None, "trainable C4 parameters require canonical_backward_session")
    require(not session.backward_started, "register all C4 calls before backward")
    require(isinstance(prefix, str) and bool(prefix), "stable layer parameter_id required")
    bindings = getattr(session, "_c4_bindings", {})
    signature = (implementation, bound.version)
    for suffix, parameter in parameters.items():
        if not parameter.requires_grad or not bound.logical_keys:
            continue
        entry = bindings.get(prefix + "." + suffix)
        if entry is not None:
            require(
                entry[0] is parameter and entry[1] == signature,
                "canonical parameter identity/version mismatch",
            )
            require(
                not entry[2].intersection(bound.logical_keys),
                "duplicate C4 group across microbatches",
            )
            require(
                len(entry[2]) + len(bound.logical_keys) <= bound.contract.max_backward_groups,
                "canonical backward group budget exceeded",
            )
    slots = {}
    for suffix, parameter in parameters.items():
        if parameter.requires_grad and bound.logical_keys:
            key = prefix + "." + suffix
            entry = bindings.setdefault(key, (parameter, signature, set()))
            entry[2].update(bound.logical_keys)
            slots[suffix] = (key, session.register(key, bound.group_ids))
    session._c4_bindings = bindings
    return slots


class _C4Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, ape, norm, cos, sin, operator, bound, parameter_id):
        ctx.operator, ctx.bound = operator, bound
        ctx.session = active_session()
        ctx.slots = {}
        if (ape.requires_grad or norm.requires_grad) and bound.logical_keys:
            ctx.slots = register_parameters(
                ctx.session,
                parameter_id,
                {"ape": ape, "norm": norm},
                bound,
                operator.implementation_fingerprint,
            )
        ctx.save_for_backward(source, ape, norm, cos, sin)
        return operator.backend.forward(
            bound, ape.detach(), norm.detach(), cos, sin, operator.contract
        )

    @staticmethod
    def backward(ctx, grad):
        require(not torch.is_grad_enabled(), "higher-order C4 gradients are not validated")
        source, ape, norm, cos, sin = ctx.saved_tensors
        op, bound = ctx.operator, ctx.bound
        eager(bound.values.device)
        bound.validate()
        if not bound.logical_keys:
            return (
                torch.zeros_like(source),
                torch.zeros_like(ape),
                torch.zeros_like(norm),
                None,
                None,
                None,
                None,
                None,
            )
        contributions, da, dn = op.backend.vjp(bound, ape, norm, cos, sin, grad, op.contract)
        dsource = torch.zeros_like(source)
        for indices, rows in contributions:
            for physical, row in zip(indices, rows, strict=True):
                if physical >= 0:
                    dsource[physical] = dsource[physical] + row.to(source.dtype)
        parameter_grads = []
        for suffix, rows, parameter in (("ape", da, ape), ("norm", dn, norm)):
            if suffix not in ctx.slots:
                parameter_grads.append(None)
                continue
            key, slot = ctx.slots[suffix]
            reduced = ctx.session.submit_rows(key, slot, rows, canonical_sum)
            parameter_grads.append(None if reduced is None else reduced.reshape(parameter.shape))
        return dsource, *parameter_grads, None, None, None, None, None


class C4Compressor:
    contract_type = C4Contract

    @staticmethod
    def _make_backend():
        from rl_engine.backends.rocm.attention.kv_compressor_c4 import AiterC4Backend

        return AiterC4Backend()

    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(type(contract) is self.contract_type, "typed C4 contract required")
        if backend is None:
            backend = self._make_backend()
        require(backend.backend_id == contract.backend_id, "C4 backend identity mismatch")
        self.contract, self.backend = contract, backend
        identity = {
            "contract": contract.fingerprint,
            "arithmetic": contract.arithmetic,
            "backend": getattr(backend, "provenance", {"test_provider": type(backend).__name__}),
            "bridge": source_hash(__file__),
            "canonical": source_hash(active_session.__code__.co_filename),
        }
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        if expected_implementation is not None:
            require(
                expected_implementation == self.implementation_fingerprint,
                "C4 implementation fingerprint mismatch",
            )

    def bind(
        self,
        values,
        token_ids,
        *,
        version,
        owners=None,
        cp_size=1,
        rank=0,
        emit_group_ids=None,
    ):
        require(
            isinstance(values, torch.Tensor) and values.device.type == self.backend.device_type,
            "C4 input/backend device mismatch",
        )
        if hasattr(self.backend, "device"):
            require(values.device == self.backend.device, "C4 device index mismatch")
        return bind_complete_groups(
            values,
            token_ids,
            version=version,
            contract=self.contract,
            owners=owners,
            cp_size=cp_size,
            rank=rank,
            emit_group_ids=emit_group_ids,
        )

    def __call__(self, bound, ape, norm, cos, sin, *, version, parameter_id=""):
        require(
            isinstance(bound, C4Groups) and bound.contract == self.contract,
            "C4 bound contract mismatch",
        )
        require(bound.version == version, "C4 model/position/request version mismatch")
        require(bound.values.device.type == self.backend.device_type, "C4 bound backend mismatch")
        if hasattr(self.backend, "device"):
            require(bound.values.device == self.backend.device, "C4 bound device index mismatch")
        eager(bound.values.device)
        bound.validate()
        require(
            isinstance(ape, torch.Tensor)
            and tuple(ape.shape) == (4, 2 * self.contract.head_dim)
            and ape.dtype == torch.float32
            and ape.is_contiguous()
            and ape.device == bound.values.device,
            f"APE must be contiguous FP32 [4,{2 * self.contract.head_dim}]",
        )
        require(
            isinstance(norm, torch.Tensor)
            and tuple(norm.shape) == (self.contract.head_dim,)
            and norm.dtype == torch.float32
            and norm.is_contiguous()
            and norm.device == bound.values.device,
            f"RMS weight must be contiguous FP32 [{self.contract.head_dim}]",
        )
        for value, name in ((ape, "APE"), (norm, "RMS weight")):
            require(
                bool((torch.isfinite(value) & (value.abs() <= 2.0**20)).all()),
                f"{name} outside finite C4 parameter domain",
            )
        require(
            isinstance(cos, torch.Tensor)
            and isinstance(sin, torch.Tensor)
            and cos.shape == sin.shape
            and cos.ndim == 2
            and cos.shape[1] == 32
            and cos.dtype == torch.bfloat16
            and sin.dtype == torch.bfloat16
            and cos.is_contiguous()
            and sin.is_contiguous()
            and cos.device == bound.values.device
            and sin.device == bound.values.device
            and not cos.requires_grad
            and not sin.requires_grad,
            "RoPE tables must be matching contiguous BF16 [positions,32]",
        )
        if bound.logical_keys:
            max_position = max(group * 4 for _request, group in bound.logical_keys)
            require(max_position < cos.shape[0], "RoPE table misses a global C4 position")
            for table in (cos, sin):
                # Do not create a transient GPU gather immediately before the
                # FlyDSL launch; the provider applies this global offset in-kernel.
                require(
                    bool((torch.isfinite(table) & (table.abs() <= 1)).all()), "invalid RoPE values"
                )
        training = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (bound.source, ape, norm)
        )
        if training:
            require(
                len(bound.logical_keys) <= self.contract.max_backward_groups,
                "C4 backward group budget exceeded",
            )
            output = _C4Autograd.apply(
                bound.source,
                ape,
                norm,
                cos,
                sin,
                self,
                bound,
                f"{self.contract.family}:{parameter_id}" if parameter_id else "",
            )
        else:
            with torch.no_grad():
                output = self.backend.forward(bound, ape, norm, cos, sin, self.contract)
        require(
            output.shape == (len(bound.logical_keys), self.contract.head_dim)
            and output.dtype == torch.bfloat16
            and output.device == bound.values.device,
            "C4 provider returned invalid output",
        )
        return C4Output(
            output,
            bound.group_ids,
            bound.owner_rank,
            version,
            self.contract.fingerprint,
            self.implementation_fingerprint,
        )
