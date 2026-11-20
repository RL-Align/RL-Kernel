# SPDX-License-Identifier: Apache-2.0
"""Optional normal-routing auxiliary loss with canonical P4 transport/reduction."""

from dataclasses import dataclass
import math

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.router_aux_loss import implementation
from rl_engine.contracts.operators.router_aux_loss import (
    CONTRACT_ID,
    FORMULA,
    LAUNCH,
    REDUCTION_ORDER,
)
from rl_engine.distributed.transports.rccl import RCCLDeterministicCollective
from rl_engine.ops.moe.learned_route import (
    LearnedRoutePlan,
    _LearnedRouteFn,
    _digest,
    _logical_ids,
    _plan_record,
    _verify_domain,
    _version,
    learned_route,
    tensor_sha256,
)


@dataclass(frozen=True)
class AuxLossConfig:
    enabled: bool = False
    alpha: float | None = None
    version: str = "router_aux.disabled.v1"

    def validate(self):
        _version(self.version, "auxiliary loss config version")
        if type(self.enabled) is not bool:
            raise TypeError("auxiliary loss enabled must be boolean")
        if self.enabled and self.alpha is None:
            raise ValueError("enabled auxiliary loss requires an explicit alpha")
        if self.alpha is not None:
            if (
                type(self.alpha) not in (int, float)
                or not math.isfinite(self.alpha)
                or not 0 <= self.alpha <= torch.finfo(torch.float32).max
            ):
                raise ValueError("auxiliary loss alpha must be finite nonnegative FP32")
            alpha = float(torch.tensor(self.alpha, dtype=torch.float32))
            if self.alpha > 0 and alpha == 0:
                raise ValueError("positive alpha underflows FP32")
        else:
            alpha = 0.0
        return alpha


@dataclass(frozen=True)
class RouterAuxContext:
    ownership: tuple
    plan_sha256: str
    domain_sha256: str
    config: AuxLossConfig
    version: str
    collective: object
    sha256: str


def _context_record(context):
    return {
        "ownership": context.ownership,
        "plan": context.plan_sha256,
        "domain": context.domain_sha256,
        "enabled": context.config.enabled,
        "alpha": context.config.validate(),
        "config_version": context.config.version,
        "context_version": context.version,
        "reduction_order": REDUCTION_ORDER,
    }


def prepare_router_aux_context(plan, *, config, ownership=None, collective=None, version):
    """Bind disjoint valid-token ownership; same context must be used on all ranks.

    A local context owns the whole domain. A multi-rank context requires an
    existing P4 RCCL collective and complete ownership metadata for every rank.
    """
    if not isinstance(config, AuxLossConfig):
        raise TypeError("AuxLossConfig required")
    config.validate()
    _version(version, "auxiliary context version")
    if not config.enabled:
        raise ValueError("disabled auxiliary loss does not need a reduction context")
    if not isinstance(plan, LearnedRoutePlan):
        raise TypeError("learned normal-route plan required")
    _verify_domain(plan.domain)
    if _digest(_plan_record(plan)) != plan.sha256:
        raise ValueError("modified learned route plan")
    if collective is not None and not isinstance(collective, RCCLDeterministicCollective):
        raise TypeError("auxiliary cross-card transport requires the P4 RCCL collective")
    if collective is not None and collective.closed:
        raise ValueError("auxiliary collective is closed")
    world = 1 if collective is None else collective.world_size
    if ownership is None:
        if world != 1:
            raise ValueError("complete rank ownership required")
        ownership = (plan.domain.token_ids,)
    if not isinstance(ownership, (tuple, list)) or len(ownership) != world:
        raise ValueError("one ownership vector required per rank")
    owned = []
    for tokens in ownership:
        if not isinstance(tokens, (tuple, list)) or any(
            type(t) is not int or not 0 <= t < 2**63 for t in tokens
        ):
            raise ValueError("ownership requires nonnegative logical I64 IDs")
        owned.append(tuple(sorted(tokens)))
    flat = [t for rank_tokens in owned for t in rank_tokens]
    if len(flat) != len(set(flat)) or tuple(sorted(flat)) != plan.domain.token_ids:
        raise ValueError("rank ownership must cover the complete domain exactly once")
    context = RouterAuxContext(
        tuple(owned), plan.sha256, plan.domain.sha256, config, version, collective, ""
    )
    from dataclasses import replace

    return replace(context, sha256=_digest(_context_record(context)))


def _gather_canonical(probabilities, experts, tokens, context):
    collective = context.collective
    device = probabilities.device
    if collective is None:
        all_probs, all_experts = probabilities.detach(), experts
        all_tokens = tokens
    else:
        if collective.device != device or collective.closed:
            raise ValueError("auxiliary P4 collective device/lifecycle mismatch")
        digest = torch.tensor(list(bytes.fromhex(context.sha256)), device=device, dtype=torch.uint8)
        gathered = collective.all_gather(digest).reshape(collective.world_size, -1)
        if not torch.equal(gathered, gathered[0].expand_as(gathered)):
            raise ValueError("auxiliary ranks disagree on config/domain/plan/ownership")
        width = max(1, max(map(len, context.ownership)))
        padded_probs = torch.zeros(width, 6, device=device, dtype=torch.float32)
        padded_experts = torch.full((width, 6), -1, device=device, dtype=torch.int64)
        padded_tokens = torch.full((width,), -1, device=device, dtype=torch.int64)
        count = len(tokens)
        padded_probs[:count] = probabilities.detach()
        padded_experts[:count] = experts
        padded_tokens[:count] = torch.tensor(tokens, device=device, dtype=torch.int64)
        all_tokens_tensor, all_experts, all_probs = collective.all_gather_many(
            (padded_tokens, padded_experts, padded_probs)
        )
        all_tokens = all_tokens_tensor.cpu().tolist()
        real = [i for i, t in enumerate(all_tokens) if t >= 0]
        all_tokens = [all_tokens[i] for i in real]
        indices = torch.tensor(real, device=device, dtype=torch.int64)
        all_probs = all_probs[indices]
        all_experts = all_experts[indices]
    canonical = sorted(range(len(all_tokens)), key=lambda i: all_tokens[i])
    expected = tuple(sorted(t for group in context.ownership for t in group))
    if tuple(all_tokens[i] for i in canonical) != expected:
        raise ValueError("gathered auxiliary tokens differ from declared global ownership")
    index = torch.tensor(canonical, device=device, dtype=torch.int64)
    return all_probs[index].contiguous(), all_experts[index].contiguous()


class _GlobalAuxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, experts, tokens, context, alpha):
        api, _ = implementation()
        canonical_probs, canonical_experts = _gather_canonical(
            probabilities, experts, tokens, context
        )
        with torch.cuda.device(probabilities.device):
            loss, counts, masses, coefficients = api.aux_loss_fwd(
                canonical_probs, canonical_experts, alpha
            )
        if not torch.isfinite(loss).all().item() or not torch.isfinite(coefficients).all().item():
            raise ValueError("auxiliary loss/coefficient overflowed FP32")
        ctx.save_for_backward(probabilities, experts, coefficients)
        ctx.hashes = tuple(tensor_sha256(t) for t in (probabilities, experts, coefficients))
        ctx.collective = context.collective
        ctx.mark_non_differentiable(counts, masses)
        return loss, counts, masses

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _counts, _masses):
        probabilities, experts, coefficients = ctx.saved_tensors
        for tensor, digest in zip((probabilities, experts, coefficients), ctx.hashes, strict=True):
            if tensor_sha256(tensor) != digest:
                raise RuntimeError("auxiliary forward state changed before backward")
        collective = ctx.collective
        if collective is not None:
            gathered = collective.all_gather(grad.detach().contiguous().reshape(1)).view(
                torch.int32
            )
            if not torch.equal(gathered, gathered[0].expand_as(gathered)):
                raise ValueError(
                    "replicated auxiliary loss requires identical upstream scaling on all ranks"
                )
        if not torch.isfinite(grad).item():
            raise ValueError("finite auxiliary scalar gradient required")
        api, _ = implementation()
        with torch.cuda.device(probabilities.device):
            gradient = api.aux_loss_bwd(experts, coefficients, grad.contiguous())
        if not torch.isfinite(gradient).all().item():
            raise ValueError("auxiliary weight adjoint overflowed FP32")
        return gradient, None, None, None, None


class _InputGuard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, protected):
        ctx.save_for_backward(*protected)
        ctx.hashes = tuple(tensor_sha256(value) for value in protected)
        return probabilities

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        for tensor, digest in zip(ctx.saved_tensors, ctx.hashes, strict=True):
            if tensor_sha256(tensor) != digest:
                raise RuntimeError("auxiliary input state changed before backward")
        return grad, None


@dataclass(frozen=True)
class RouterAuxLossResult:
    loss: torch.Tensor | None
    probabilities: torch.Tensor | None
    expert_ids: torch.Tensor | None
    counts: torch.Tensor | None
    masses: torch.Tensor | None
    routes: tuple
    execution: dict


def router_aux_loss(
    logits,
    logical_ids,
    bias,
    plan,
    *,
    config=None,
    context=None,
    valid_mask=None,
    microbatch_size=None,
    **versions,
):
    """One global loss per complete domain, optionally split into compute batches.

    Loss is replicated on all ranks, but its VJP returns only each rank's local
    token gradients, with no world-size multiplier. All ranks (including empty
    ranks) must enter forward and backward in the same collective order.
    """
    if config is None:
        config = AuxLossConfig()
    if not isinstance(config, AuxLossConfig):
        raise TypeError("AuxLossConfig required")
    alpha = config.validate()
    record = {
        "task": 34,
        "contract_id": CONTRACT_ID,
        "enabled": config.enabled,
        "config_version": config.version,
        "alpha": alpha,
        "alpha_requested": config.alpha,
        "formula": FORMULA,
        "normal_routing": True,
        "diagnostic_replay": False,
    }
    if not config.enabled:
        return RouterAuxLossResult(
            None,
            None,
            None,
            None,
            None,
            (),
            {**record, "kernel_launched": False, "collective_called": False},
        )
    if not isinstance(plan, LearnedRoutePlan):
        raise TypeError("enabled auxiliary loss requires a learned-route plan")
    if (
        not isinstance(context, RouterAuxContext)
        or _digest(_context_record(context)) != context.sha256
    ):
        raise ValueError("immutable auxiliary reduction context required")
    if context.config != config or context.plan_sha256 != plan.sha256:
        raise ValueError("auxiliary configuration/plan version barrier mismatch")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.dtype != torch.float32
        or logits.ndim != 2
        or logits.shape[1] != 256
    ):
        raise TypeError("auxiliary loss requires FP32 logits [M,256]")
    if not logits.is_cuda or torch.version.hip is None:
        raise RuntimeError("auxiliary loss requires ROCm")
    with torch.cuda.device(logits.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("auxiliary loss does not support Graph capture")
    _, manifest = implementation()
    if (
        not isinstance(logical_ids, torch.Tensor)
        or logical_ids.dtype not in (torch.int32, torch.int64)
        or logical_ids.shape != (logits.shape[0],)
    ):
        raise ValueError("auxiliary loss needs one I32/I64 logical ID per input row")
    if valid_mask is None:
        rows = torch.arange(logits.shape[0], device=logits.device, dtype=torch.int64)
    else:
        if (
            not isinstance(valid_mask, torch.Tensor)
            or valid_mask.dtype != torch.bool
            or valid_mask.shape != (logits.shape[0],)
        ):
            raise ValueError("valid mask must be boolean [M]")
        rows = valid_mask.detach().to(logits.device).nonzero().flatten()
    current_ids = logical_ids.detach()[rows.to(logical_ids.device)]
    tokens = _logical_ids(current_ids)
    rank = 0 if context.collective is None else context.collective.rank
    if tuple(sorted(tokens)) != context.ownership[rank]:
        raise ValueError("auxiliary call must contain all locally owned valid tokens exactly once")
    if microbatch_size is not None and (type(microbatch_size) is not int or microbatch_size <= 0):
        raise ValueError("microbatch size must be a positive integer")
    x = logits[rows]
    step = microbatch_size or max(1, len(tokens))
    routes = []
    probabilities = []
    selected = []
    for start in range(0, max(1, len(tokens)), step):
        local = x[start : start + step]
        local_ids = current_ids[start : start + step]
        route = learned_route(local, local_ids, bias, plan, **versions)
        gids = torch.tensor(plan.global_ids, device=logits.device, dtype=torch.int64)
        # Recompute actual normal selection with exactly scale=1; never divide
        # scaled route weights by scale (zero scale is a valid main-route mode).
        unscaled, experts, _, _ = _LearnedRouteFn.apply(
            local.contiguous(), bias.contiguous(), gids, route.surviving_mask, route.expert_ids, 1.0
        )
        routes.append(route)
        probabilities.append(unscaled)
        selected.append(experts)
    probs = torch.cat(probabilities, dim=0)
    probs = _InputGuard.apply(probs, (logits, bias))
    experts = torch.cat(selected, dim=0)
    loss, counts, masses = _GlobalAuxFn.apply(probs, experts, tokens, context, alpha)
    n = sum(map(len, context.ownership))
    props = torch.cuda.get_device_properties(logits.device)
    import triton

    record.update(
        {
            "actual_backend": "aiter.triton.router_aux_loss",
            "source_sha256": manifest["source_sha256"],
            "config": dict(LAUNCH),
            "kernels": ["_expert_totals", "_loss_and_coefficients", "_slot_adjoint"],
            "gpu": props.name,
            "arch": props.gcnArchName,
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "triton": triton.__version__,
            "dtype": "float32",
            "patch_sha256": manifest["patch_sha256"],
            "reduction_order": REDUCTION_ORDER,
            "domain_sha256": context.domain_sha256,
            "plan_sha256": plan.sha256,
            "context_sha256": context.sha256,
            "domain_version": plan.domain.domain_version,
            "model_version": plan.domain.model_version,
            "bias_version": plan.bias_version,
            "plan_version": plan.plan_version,
            "absolute_layer": plan.domain.absolute_layer,
            "loss_context_version": context.version,
            "global_valid_tokens": n,
            "local_valid_tokens": len(tokens),
            "logical_token_ids": list(tokens),
            "expert_ids_sha256": tensor_sha256(experts),
            "probabilities_sha256": tensor_sha256(probs),
            "rank": rank,
            "world_size": len(context.ownership),
            "microbatch_size": microbatch_size,
            "count_denominator": 6 * n,
            "probability_denominator": n,
            "pre_capacity_counts": counts.cpu().tolist(),
            "route_scale_excluded": True,
            "main_route_scale": plan.scale,
            "loss_replicated": True,
            "backward_world_multiplier": 1,
            "acceptance_sample": n > 0,
            "kernel_launched": True,
            "collective_backend": None
            if context.collective is None
            else context.collective.backend_id,
            "transport_only_gather": context.collective is not None,
        }
    )
    return RouterAuxLossResult(loss, probs, experts, counts, masses, tuple(routes), record)


class RouterAuxLossOp:
    def __call__(self, *args, **kwargs):
        return router_aux_loss(*args, **kwargs)
