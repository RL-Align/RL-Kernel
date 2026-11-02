# SPDX-License-Identifier: Apache-2.0
"""Normal learned routing with immutable complete-domain capacity admission."""

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.learned_route import execution_record, implementation
from rl_engine.contracts.operators.learned_route import CAPACITY_POLICY, EXPERTS, SLOTS


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()


def _digest(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _version(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"nonempty {label} required")


def _logical_ids(value):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int32, torch.int64):
        raise TypeError("logical token IDs must be I32/I64 tensors")
    if value.ndim != 1 or value.stride(0) <= 0:
        raise ValueError("logical token IDs require a positive-stride vector")
    if value.is_cuda:
        with torch.cuda.device(value.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("capacity metadata does not support Graph capture")
    values = tuple(value.detach().cpu().long().tolist())
    if any(x < 0 for x in values) or len(set(values)) != len(values):
        raise ValueError("logical token IDs must be distinct nonnegative global identities")
    return values


@dataclass(frozen=True)
class CapacityDomain:
    absolute_layer: int
    model_version: str
    domain_version: str
    token_ids: tuple
    capacities: tuple
    capacity_mode: str
    sha256: str


def _domain_record(domain):
    return {
        "absolute_layer": domain.absolute_layer,
        "model_version": domain.model_version,
        "domain_version": domain.domain_version,
        "tokens": domain.token_ids,
        "capacities": domain.capacities,
        "capacity_mode": domain.capacity_mode,
        "policy": CAPACITY_POLICY,
    }


def _verify_domain(domain):
    if (
        not isinstance(domain, CapacityDomain)
        or not isinstance(domain.token_ids, tuple)
        or (not isinstance(domain.capacities, tuple) or len(domain.capacities) != EXPERTS)
    ):
        raise ValueError("immutable complete capacity domain required")
    if _digest(_domain_record(domain)) != domain.sha256:
        raise ValueError("capacity domain identity changed; prepare a new version")


def prepare_capacity_domain(
    logical_token_ids, capacity, *, absolute_layer, model_version, domain_version
):
    """Declare the COMPLETE routing domain independently of its compute batches."""
    _version(model_version, "model_version")
    _version(domain_version, "domain_version")
    if type(absolute_layer) is not int or not 3 <= absolute_layer < 43:
        raise ValueError("Flash-0731 learned routing uses absolute layers 3 through 42")
    ids = tuple(sorted(_logical_ids(logical_token_ids)))
    mode = "per_expert"
    if capacity is None:
        capacities = (len(ids),) * EXPERTS
        mode = "unbounded"
    elif type(capacity) is int:
        capacities = (capacity,) * EXPERTS
        mode = "scalar"
    elif isinstance(capacity, torch.Tensor):
        if capacity.dtype not in (torch.int32, torch.int64) or tuple(capacity.shape) != (EXPERTS,):
            raise ValueError("per-expert capacity requires an I32/I64 [256] vector")
        capacities = tuple(capacity.detach().cpu().tolist())
    elif isinstance(capacity, (tuple, list)):
        capacities = tuple(capacity)
    else:
        raise TypeError("capacity must be None, an integer or an integer [256] vector")
    if len(capacities) != EXPERTS or any(
        type(c) is not int or not 0 <= c < 2**63 for c in capacities
    ):
        raise ValueError("capacities must be 256 nonnegative I64 counts")
    digest = _digest(
        {
            "absolute_layer": absolute_layer,
            "model_version": model_version,
            "domain_version": domain_version,
            "tokens": ids,
            "capacities": capacities,
            "capacity_mode": mode,
            "policy": CAPACITY_POLICY,
        }
    )
    return CapacityDomain(
        absolute_layer, model_version, domain_version, ids, capacities, mode, digest
    )


@dataclass(frozen=True)
class PlannedToken:
    logical_id: int
    logit_sha256: str
    expert_ids: tuple
    keep: tuple
    dispatch_positions: tuple


@dataclass(frozen=True)
class LearnedRoutePlan:
    domain: CapacityDomain
    bias_version: str
    plan_version: str
    bias_sha256: str
    global_ids: tuple
    tokens: tuple
    selected_loads: tuple
    accepted_loads: tuple
    scale: float
    sha256: str


def _plan_record(plan):
    return {
        "domain": plan.domain.sha256,
        "bias_version": plan.bias_version,
        "plan_version": plan.plan_version,
        "bias_sha256": plan.bias_sha256,
        "scale": plan.scale,
        "tokens": [
            (p.logical_id, p.logit_sha256, p.expert_ids, p.keep, p.dispatch_positions)
            for p in plan.tokens
        ],
    }


def _validate(logits, logical_token_ids, bias, global_expert_ids=None):
    if not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32:
        raise TypeError("learned route requires FP32 gate logits")
    if logits.ndim != 2 or logits.shape[1] != EXPERTS or any(s <= 0 for s in logits.stride()):
        raise ValueError("learned route requires positive-stride logits [M,256]")
    if (
        not isinstance(bias, torch.Tensor)
        or bias.dtype != torch.float32
        or tuple(bias.shape) != (EXPERTS,)
    ):
        raise TypeError("selection bias must be FP32 [256] in canonical global expert order")
    if not logits.is_cuda or torch.version.hip is None or bias.device != logits.device:
        raise RuntimeError("learned route logits and selection bias require the same ROCm GPU")
    with torch.cuda.device(logits.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("learned route does not support Graph capture")
    arch = torch.cuda.get_device_properties(logits.device).gcnArchName.partition(":")[0]
    if arch not in {"gfx942", "gfx950"}:
        raise RuntimeError(f"unsupported learned route architecture: {arch}")
    if not torch.isfinite(logits).all().item() or not torch.isfinite(bias).all().item():
        raise ValueError("learned route logits and bias must be finite")
    tokens = _logical_ids(logical_token_ids)
    if len(tokens) != logits.shape[0]:
        raise ValueError("one logical token identity is required per logit row")
    if global_expert_ids is None:
        ids = tuple(range(EXPERTS))
    else:
        if not isinstance(global_expert_ids, torch.Tensor) or global_expert_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("global expert IDs must be I32/I64")
        if tuple(global_expert_ids.shape) != (EXPERTS,):
            raise ValueError("complete global expert ID map required")
        ids = tuple(global_expert_ids.detach().cpu().long().tolist())
        if tuple(sorted(ids)) != tuple(range(EXPERTS)):
            raise ValueError("global expert IDs must cover [0,256) exactly once")
    return tokens, ids


def _row_hashes(logits, global_ids):
    # Hash in canonical global-expert order so equivalent rank materializations
    # have the same logical plan identity.
    columns = torch.tensor(sorted(range(EXPERTS), key=lambda i: global_ids[i]), dtype=torch.int64)
    canonical = logits.detach().cpu()[:, columns].contiguous()
    return tuple(tensor_sha256(row) for row in canonical)


def prepare_learned_route_plan(
    logits,
    logical_token_ids,
    bias,
    domain,
    *,
    bias_version,
    plan_version,
    global_expert_ids=None,
    routed_scaling_factor=1.5,
):
    """Normal selection over every declared token, then deterministic capacity admission.

    Plan state is immutable Python scalars/tuples/strings, not caller-owned tensors.
    """
    if not isinstance(domain, CapacityDomain):
        raise TypeError("prepared complete CapacityDomain required")
    _verify_domain(domain)
    _version(bias_version, "bias_version")
    _version(plan_version, "plan_version")
    if type(routed_scaling_factor) not in (int, float) or not (
        math.isfinite(routed_scaling_factor)
        and 0 <= routed_scaling_factor <= torch.finfo(torch.float32).max
    ):
        raise ValueError("routing scale must be finite, nonnegative and representable in FP32")
    scale = float(torch.tensor(routed_scaling_factor, dtype=torch.float32))
    if routed_scaling_factor > 0 and scale == 0:
        raise ValueError("positive routing scale underflows FP32")
    tokens, gids = _validate(logits, logical_token_ids, bias, global_expert_ids)
    if tuple(sorted(tokens)) != domain.token_ids:
        raise ValueError("capacity planning must contain the complete declared routing domain")
    api, _ = implementation()
    ids_tensor = torch.tensor(gids, device=logits.device, dtype=torch.int64)
    with torch.no_grad(), torch.cuda.device(logits.device):
        _, selected, _ = api.learned_route_select(
            logits.contiguous(), bias.contiguous(), ids_tensor
        )
    selected = selected.cpu().tolist()
    hashes = _row_hashes(logits, gids)
    row_for = {token: row for row, token in enumerate(tokens)}
    selected_loads = [0] * EXPERTS
    accepted_loads = [0] * EXPERTS
    planned = []
    for token in domain.token_ids:
        row = row_for[token]
        chosen = tuple(selected[row])
        keep = []
        positions = []
        for expert in chosen:
            position = selected_loads[expert]
            selected_loads[expert] += 1
            admitted = position < domain.capacities[expert]
            keep.append(admitted)
            positions.append(position if admitted else -1)
            accepted_loads[expert] += int(admitted)
        planned.append(PlannedToken(token, hashes[row], chosen, tuple(keep), tuple(positions)))
    bias_hash = tensor_sha256(bias)
    semantic = {
        "domain": domain.sha256,
        "bias_version": bias_version,
        "plan_version": plan_version,
        "bias_sha256": bias_hash,
        "scale": scale,
        "tokens": [
            (p.logical_id, p.logit_sha256, p.expert_ids, p.keep, p.dispatch_positions)
            for p in planned
        ],
    }
    return LearnedRoutePlan(
        domain,
        bias_version,
        plan_version,
        bias_hash,
        gids,
        tuple(planned),
        tuple(selected_loads),
        tuple(accepted_loads),
        scale,
        _digest(semantic),
    )


class _LearnedRouteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, bias, gids, keep, expected_ids, scale):
        api, _ = implementation()
        with torch.cuda.device(logits.device):
            scores, ids, columns = api.learned_route_select(logits, bias, gids)
            if not torch.equal(ids, expected_ids):
                raise ValueError(
                    "normal learned selection differs from the capacity plan; rebuild the plan"
                )
            weights, denominator = api.learned_route_weights(scores, columns, keep, scale)
        if not torch.isfinite(weights).all().item():
            raise ValueError("learned weights overflowed FP32")
        ctx.save_for_backward(logits, bias, scores, ids, columns, keep, weights, denominator)
        ctx.scale = scale
        ctx.mark_non_differentiable(ids, columns, denominator)
        return weights, ids, columns, denominator

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_weights, _ids, _columns, _denominator):
        logits, _bias, scores, _ids, columns, keep, _weights, _denominator = ctx.saved_tensors
        if not torch.isfinite(grad_weights).all().item():
            raise ValueError("finite learned weight gradients required")
        dx = None
        if ctx.needs_input_grad[0]:
            api, _ = implementation()
            with torch.cuda.device(logits.device):
                dx = api.learned_route_bwd(logits, scores, columns, keep, grad_weights, ctx.scale)
            if not torch.isfinite(dx).all().item():
                raise ValueError("learned route VJP overflowed FP32")
        # Bias affects a discrete choice only; it never receives a weight gradient.
        return dx, None, None, None, None, None


@dataclass(frozen=True)
class LearnedRouteResult:
    weights: torch.Tensor
    expert_ids: torch.Tensor
    column_indices: torch.Tensor
    surviving_mask: torch.Tensor
    dispatch_positions: torch.Tensor
    denominators: torch.Tensor
    execution: dict


def learned_route(
    logits,
    logical_token_ids,
    bias,
    plan,
    *,
    absolute_layer,
    model_version,
    bias_version,
    domain_version,
    plan_version,
):
    if not isinstance(plan, LearnedRoutePlan):
        raise TypeError("prepared LearnedRoutePlan required")
    _verify_domain(plan.domain)
    if _digest(_plan_record(plan)) != plan.sha256:
        raise ValueError("capacity plan identity changed; rebuild the versioned plan")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("learned route requires FP32 gate logits")
    domain = plan.domain
    if type(absolute_layer) is not int or absolute_layer != domain.absolute_layer:
        raise ValueError("learned route absolute-layer barrier mismatch")
    if (model_version, bias_version, domain_version, plan_version) != (
        domain.model_version,
        plan.bias_version,
        domain.domain_version,
        plan.plan_version,
    ):
        raise ValueError("learned route model/bias/domain/plan version barrier mismatch")
    gid_metadata = torch.tensor(plan.global_ids, dtype=torch.int64)
    tokens, gids = _validate(logits, logical_token_ids, bias, gid_metadata)
    gid_tensor = gid_metadata.to(logits.device)
    if tensor_sha256(bias) != plan.bias_sha256:
        raise ValueError("selection bias changed; rebuild the versioned plan")
    rows = {token.logical_id: token for token in plan.tokens}
    if any(token not in rows for token in tokens):
        raise ValueError("token not in declared routing domain")
    hashes = _row_hashes(logits, gids)
    chosen = [rows[token] for token in tokens]
    if any(h != row.logit_sha256 for h, row in zip(hashes, chosen, strict=True)):
        raise ValueError("logical token logits changed; rebuild the capacity plan")
    device = logits.device
    keep = torch.tensor([p.keep for p in chosen], device=device, dtype=torch.bool).reshape(
        -1, SLOTS
    )
    positions = torch.tensor(
        [p.dispatch_positions for p in chosen], device=device, dtype=torch.int64
    ).reshape(-1, SLOTS)
    expected = torch.tensor(
        [p.expert_ids for p in chosen], device=device, dtype=torch.int64
    ).reshape(-1, SLOTS)
    record = execution_record(logits)
    record.update(
        {
            "absolute_layer": absolute_layer,
            "model_version": model_version,
            "bias_version": bias_version,
            "domain_version": domain_version,
            "plan_version": plan_version,
            "capacity_policy": CAPACITY_POLICY,
            "domain_sha256": domain.sha256,
            "plan_sha256": plan.sha256,
            "bias_sha256": plan.bias_sha256,
            "domain_token_count": len(domain.token_ids),
            "logical_token_ids": list(tokens),
            "capacities": list(domain.capacities),
            "capacity_mode": domain.capacity_mode,
            "selected_loads": list(plan.selected_loads),
            "accepted_loads": list(plan.accepted_loads),
            "routed_scaling_factor": plan.scale,
            "global_id_map_sha256": tensor_sha256(gid_tensor),
            "overflow_slots": int((~keep).sum()),
            "input_materialized_contiguous": not logits.is_contiguous(),
        }
    )
    weights, ids, columns, denominator = _LearnedRouteFn.apply(
        logits.contiguous(), bias.contiguous(), gid_tensor, keep, expected, plan.scale
    )
    return LearnedRouteResult(weights, ids, columns, keep, positions, denominator, record)


class LearnedRouteOp:
    def __call__(self, *args, **kwargs):
        return learned_route(*args, **kwargs)
