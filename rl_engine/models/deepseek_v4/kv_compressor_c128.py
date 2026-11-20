# SPDX-License-Identifier: Apache-2.0
"""Complete-group admission and autograd wiring for existing C128 providers."""

from dataclasses import dataclass, field
import hashlib
import json

import torch

from rl_engine.contracts.operators.kv_compressor_c128 import (
    C128Contract,
    require,
    source_hash,
)
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
class C128Groups:
    source: torch.Tensor = field(repr=False)
    values: torch.Tensor = field(repr=False)
    selector: torch.Tensor = field(repr=False)
    group_ids: torch.Tensor = field(repr=False)
    plan: torch.Tensor = field(repr=False)
    block_table: torch.Tensor = field(repr=False)
    state_slots: torch.Tensor = field(repr=False)
    logical_keys: tuple[tuple[int, int], ...]
    incomplete_groups: tuple[tuple[int, int], ...]
    unowned_groups: tuple[tuple[int, int], ...]
    version: str
    owner_rank: int
    cp_size: int
    contract: C128Contract
    _stamps: tuple = field(repr=False)
    _source_stamp: tuple = field(repr=False)

    @property
    def tensors(self):
        return (
            self.values,
            self.selector,
            self.group_ids,
            self.plan,
            self.block_table,
            self.state_slots,
        )

    def validate(self):
        require(
            tuple(stamp(t) for t in self.tensors) == self._stamps, "bound group storage changed"
        )
        if self.source.requires_grad:
            require(
                stamp(self.source) == self._source_stamp, "training source changed after binding"
            )


@dataclass(frozen=True)
class C128Output:
    values: torch.Tensor
    group_ids: torch.Tensor
    owner_rank: int
    version: str
    contract_fingerprint: str
    implementation_fingerprint: str


def bind_complete_groups(values, token_ids, *, version, contract, owners=None, cp_size=1, rank=0):
    """Own complete [128,1024] groups; never pad an incomplete logical group.

    The CP caller must transport all 128 rows before binding. Only the rank
    owning the group's final token emits that row. No collective/cache side effect.
    """
    require(isinstance(contract, C128Contract), "typed C128 contract required")
    require(
        isinstance(version, str) and bool(version),
        "explicit model/position/request version required",
    )
    require(
        isinstance(values, torch.Tensor)
        and values.dtype == torch.bfloat16
        and values.ndim == 2
        and values.shape[1] == 1024
        and values.is_contiguous(),
        "projected [tokens, content512+score512] contiguous BF16 input required",
    )
    eager(values.device)
    require(values.shape[0] <= contract.max_input_rows, "input capacity exceeded")
    require(
        isinstance(token_ids, torch.Tensor)
        and token_ids.dtype == torch.int64
        and token_ids.shape == (values.shape[0], 2)
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
            and owners.shape == (values.shape[0],)
            and owners.device == values.device,
            "owners must be int64 [tokens] on the input device",
        )
        host_owners = tuple(owners.cpu().tolist())
    groups, seen = {}, set()
    for physical, (key, owner) in enumerate(zip(host_ids, host_owners, strict=True)):
        if key == (-1, -1):
            continue
        request, position = key
        require(request >= 0 and 0 <= position < 2**31, "invalid logical token ID")
        require(key not in seen, "duplicate logical token ID")
        require(0 <= owner < cp_size, "token owner outside CP group")
        seen.add(key)
        group, offset = divmod(position, 128)
        groups.setdefault((request, group), {})[offset] = (physical, owner)
    selected, incomplete, unowned, indices = [], [], [], []
    for key, entries in sorted(groups.items()):
        if len(entries) != 128:
            incomplete.append(key)
        elif entries[127][1] != rank:
            unowned.append(key)
        else:
            selected.append(key)
            indices.extend(entries[offset][0] for offset in range(128))
    require(len(selected) <= contract.max_groups, "complete-group capacity exceeded")
    request_order = {req: i for i, req in enumerate(sorted({key[0] for key in selected}))}
    width = 1 + max((key[1] for key in selected), default=-1)
    require(
        len(request_order) * width * 4 <= contract.max_block_table_bytes,
        "C128 provider block-table byte budget exceeded",
    )
    with torch.inference_mode(False), torch.no_grad():
        selector = torch.tensor(indices, dtype=torch.int64, device=values.device)
        packed = values.index_select(0, selector).detach()
        # Input values may span the BF16 range, but FP32 squared-norm arithmetic
        # must not overflow. This explicit bounded profile is not a clamp.
        require(
            bool((torch.isfinite(packed.float()) & (packed.float().abs() <= 2.0**30)).all()),
            "selected input outside finite C128 FP32 accumulation domain",
        )
        group_ids = torch.tensor(selected, dtype=torch.int64, device=values.device).reshape(-1, 2)
        plan = torch.tensor(
            [
                [i * 128 + 127, request_order[req], group * 128 + 127, 0]
                for i, (req, group) in enumerate(selected)
            ],
            dtype=torch.int32,
            device=values.device,
        ).reshape(-1, 4)
        table = torch.full((len(request_order), width), -1, dtype=torch.int32, device=values.device)
        for i, (req, group) in enumerate(selected):
            table[request_order[req], group] = i
        slots = torch.zeros(len(request_order), dtype=torch.int32, device=values.device)
    tensors = (packed, selector, group_ids, plan, table, slots)
    return C128Groups(
        values,
        *tensors,
        tuple(selected),
        tuple(incomplete),
        tuple(unowned),
        version,
        rank,
        cp_size,
        contract,
        tuple(stamp(t) for t in tensors),
        stamp(values),
    )


def canonical_sum(rows):
    """Reuse native FP32 vector adds in a prescribed logical left-to-right order."""
    result = torch.zeros(rows.shape[1], dtype=rows.dtype, device=rows.device)
    for row in rows.unbind(0):
        result = torch.add(result, row)
    return result


def register_parameters(session, prefix, parameters, bound, implementation):
    require(session is not None, "trainable C128 parameters require canonical_backward_session")
    require(not session.backward_started, "register all microbatches before backward")
    require(isinstance(prefix, str) and bool(prefix), "stable layer parameter_id required")
    bindings = getattr(session, "_c128_bindings", {})
    signature = (implementation, bound.version)
    # Validate both parameters before changing session registration state.
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
                "duplicate C128 group across microbatches",
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
    session._c128_bindings = bindings
    return slots


class _C128Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source, ape, norm, cos, sin, operator, bound, parameter_id):
        ctx.operator, ctx.bound = operator, bound
        ctx.session = active_session()
        ctx.slots = {}
        if ape.requires_grad or norm.requires_grad:
            ctx.slots = register_parameters(
                ctx.session,
                parameter_id,
                {"ape": ape, "norm": norm},
                bound,
                operator.implementation_fingerprint,
            )
        with torch.inference_mode(False):
            a, n = ape.detach().clone(), norm.detach().clone()
        ctx.save_for_backward(source, ape, norm, cos, sin, a, n)
        return operator.backend.forward(bound, a, n, cos, sin, operator.contract)

    @staticmethod
    def backward(ctx, grad):
        require(not torch.is_grad_enabled(), "higher-order C128 gradients are not validated")
        source, ape, norm, cos, sin, a, n = ctx.saved_tensors
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
        dx, da, dn = op.backend.vjp(bound, a, n, cos, sin, grad, op.contract)
        dsource = torch.zeros_like(source)
        # Every selector index is unique: overwrite, not scatter-add/atomics.
        dsource.index_copy_(0, bound.selector, dx.to(source.dtype))
        parameter_grads = []
        for suffix, rows, parameter in (("ape", da, ape), ("norm", dn, norm)):
            if suffix not in ctx.slots:
                parameter_grads.append(None)
                continue
            key, slot = ctx.slots[suffix]
            reduced = ctx.session.submit_rows(key, slot, rows, canonical_sum)
            parameter_grads.append(None if reduced is None else reduced.reshape(parameter.shape))
        return dsource, *parameter_grads, None, None, None, None, None


class C128Compressor:
    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(isinstance(contract, C128Contract), "typed C128 contract required")
        if backend is None:
            from rl_engine.backends.rocm.attention.kv_compressor_c128 import AiterC128Backend

            backend = AiterC128Backend()
        require(backend.backend_id == contract.backend_id, "C128 backend identity mismatch")
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
                "C128 implementation fingerprint mismatch",
            )

    def bind(self, values, token_ids, *, version, owners=None, cp_size=1, rank=0):
        require(
            isinstance(values, torch.Tensor) and values.device.type == self.backend.device_type,
            "C128 input/backend device mismatch",
        )
        if hasattr(self.backend, "device"):
            require(values.device == self.backend.device, "C128 device index mismatch")
        return bind_complete_groups(
            values,
            token_ids,
            version=version,
            contract=self.contract,
            owners=owners,
            cp_size=cp_size,
            rank=rank,
        )

    def __call__(self, bound, ape, norm, cos, sin, *, version, parameter_id=""):
        require(
            isinstance(bound, C128Groups) and bound.contract == self.contract,
            "C128 bound contract mismatch",
        )
        require(bound.version == version, "C128 model/position/request version mismatch")
        require(bound.values.device.type == self.backend.device_type, "C128 bound backend mismatch")
        if hasattr(self.backend, "device"):
            require(bound.values.device == self.backend.device, "C128 bound device index mismatch")
        eager(bound.values.device)
        bound.validate()
        for value, shape, name in ((ape, (128, 512), "APE"), (norm, (512,), "RMS weight")):
            require(
                isinstance(value, torch.Tensor)
                and value.shape == shape
                and value.dtype == torch.float32
                and value.is_contiguous()
                and value.device == bound.values.device,
                f"{name} must be contiguous FP32 {shape}",
            )
            require(
                bool((torch.isfinite(value) & (value.abs() <= 2.0**20)).all()),
                f"{name} outside finite C128 parameter domain",
            )
        require(
            isinstance(cos, torch.Tensor)
            and isinstance(sin, torch.Tensor)
            and cos.shape == sin.shape
            and cos.ndim == 2
            and cos.shape[1] == 32,
            "RoPE tables must be matching [positions,32]",
        )
        for table in (cos, sin):
            require(
                table.dtype == torch.bfloat16
                and table.is_contiguous()
                and table.device == bound.values.device
                and not table.requires_grad,
                "read-only contiguous BF16 RoPE tables required",
            )
        positions = [key[1] * 128 for key in bound.logical_keys]
        require(
            not positions or max(positions) < cos.shape[0],
            "RoPE table misses a global group position",
        )
        if positions:
            index = bound.group_ids[:, 1] * 128
            for table in (cos, sin):
                selected = table.index_select(0, index).float()
                require(
                    bool((torch.isfinite(selected) & (selected.abs() <= 1)).all()),
                    "RoPE values must be finite in [-1,1]",
                )
        training = torch.is_grad_enabled() and any(
            t.requires_grad for t in (bound.source, ape, norm)
        )
        if training:
            require(
                len(bound.logical_keys) <= self.contract.max_backward_groups,
                "C128 backward group budget exceeded",
            )
            values = _C128Autograd.apply(
                bound.source, ape, norm, cos, sin, self, bound, parameter_id
            )
        else:
            with torch.no_grad():
                values = self.backend.forward(bound, ape, norm, cos, sin, self.contract)
        require(
            values.shape == (len(bound.logical_keys), 512)
            and values.dtype == torch.bfloat16
            and values.device == bound.values.device,
            "C128 provider returned an invalid output",
        )
        return C128Output(
            values,
            bound.group_ids,
            bound.owner_rank,
            version,
            self.contract.fingerprint,
            self.implementation_fingerprint,
        )
