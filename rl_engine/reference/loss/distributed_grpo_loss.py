# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic DP-aware GRPO loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch

from rl_engine.contracts.operators.loss import (
    AdvantageNormalizer,
    GRPOLossContract,
    KLEstimator,
    LossContractError,
    TokenNormalizer,
)
from rl_engine.reference.logprob.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
    VocabParallelLogprobOp,
)

BACKEND_ID = "pytorch-distributed-grpo-loss-ws2"

# Channel layout of the packed per-sequence tensor moved by the single fp32
# all-gather; counts travel separately as integers so they stay exact.
_CH_POLICY = 0
_CH_KL = 1


@dataclass(frozen=True)
class GRPOLossResult:
    """Scalar loss terms, per-sequence diagnostics, and bound provenance.

    Unpacks as ``loss, policy_loss, kl`` so it can stand in for the tuple the
    single-GPU GRPO ops return.

    The ``per_sequence_*`` vectors are the reduction's last mesh-independent
    intermediate, detached and replicated on every rank.  They are the right
    surface for comparing two configurations: the scalar loss averages
    ``num_sequences`` totals into 24 mantissa bits and routinely rounds a real
    reordering away, whereas the per-sequence vector preserves it.  They also
    give a drift report somewhere to point when one sequence is responsible.
    """

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    advantages: torch.Tensor
    per_sequence_policy: torch.Tensor
    per_sequence_kl: torch.Tensor
    per_sequence_active_tokens: torch.Tensor
    provenance: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from (self.loss, self.policy_loss, self.kl)


def _require_distributed_initialized():
    import torch.distributed as dist

    if not dist.is_available():
        raise LossContractError("distributed GRPO loss requires torch.distributed.")
    if not dist.is_initialized():
        raise LossContractError(
            "distributed GRPO loss requires an initialized process group when "
            "the contract declares dp_world_size > 1."
        )
    return dist


def _validate_invocation(
    policy_local_logits: torch.Tensor,
    ref_local_logits: torch.Tensor | None,
    action_ids: torch.Tensor,
    old_logps: torch.Tensor,
    rewards: torch.Tensor,
    contract: GRPOLossContract,
    dp_group: Any,
) -> None:
    sharding = contract.sharding
    num_rows = sharding.local_num_token_slots
    if policy_local_logits.dim() != 2:
        raise LossContractError(
            "policy_local_logits must be 2-D [num_tokens, local_vocab]; got "
            f"{policy_local_logits.dim()}-D"
        )
    if policy_local_logits.shape[0] != num_rows:
        raise LossContractError(
            f"policy_local_logits has {policy_local_logits.shape[0]} rows but this rank owns "
            f"{num_rows} token slots"
        )
    if ref_local_logits is not None and ref_local_logits.shape != policy_local_logits.shape:
        raise LossContractError(
            f"ref_local_logits shape {tuple(ref_local_logits.shape)} must match "
            f"policy_local_logits shape {tuple(policy_local_logits.shape)}"
        )
    if ref_local_logits is None and contract.objective.uses_reference_model:
        raise LossContractError(
            f"objective.beta={contract.objective.beta} puts the reference KL in the loss, "
            "so ref_local_logits is required"
        )
    for name, tensor in (("action_ids", action_ids), ("old_logps", old_logps)):
        if tensor.dim() != 1 or tensor.shape[0] != num_rows:
            raise LossContractError(
                f"{name} must be 1-D with one entry per owned token slot; got shape "
                f"{tuple(tensor.shape)} for {num_rows} slots"
            )
    if rewards.dim() != 1 or rewards.shape[0] != sharding.local_num_sequences:
        raise LossContractError(
            "rewards must be 1-D with one entry per sequence this rank owns; got shape "
            f"{tuple(rewards.shape)} for {sharding.local_num_sequences} sequences"
        )

    if sharding.dp_world_size > 1:
        dist = _require_distributed_initialized()
        group_world = dist.get_world_size(group=dp_group)
        group_rank = dist.get_rank(group=dp_group)
        if group_world != sharding.dp_world_size:
            raise LossContractError(
                f"dp_group world size {group_world} does not match the contract "
                f"dp_world_size={sharding.dp_world_size}; pass the DP subgroup, "
                "not the global group"
            )
        if group_rank != sharding.dp_rank:
            raise LossContractError(
                f"dp_group rank {group_rank} does not match the contract dp_rank={sharding.dp_rank}"
            )


def _preflight_cross_rank_agreement(
    contract: GRPOLossContract, dp_group: Any, tp_group: Any, num_vocab_tiles: int
) -> None:
    """All-gather (fingerprint, backend id, vocab tile count) and abort on mismatch.

    Checked over the DP group *and* the TP group.  Neither alone is sufficient:
    the loss is replicated across TP, so two TP siblings that disagree on, say,
    ``beta`` would compute different losses and produce inconsistent gradients
    for one sharded model -- and the logprob path's own preflight cannot catch
    that, because ``beta`` is not part of the logprob contract.  Agreement
    within both groups implies agreement across the whole DP x TP grid by
    transitivity.

    Runs before any other collective, including the logprob path's own TP
    preflight, so a rank that joined the wrong logical invocation fails here
    rather than deadlocking a later reduction.
    """

    checks = [
        (axis, group, world_size)
        for axis, group, world_size in (
            ("DP", dp_group, contract.sharding.dp_world_size),
            ("TP", tp_group, contract.logprob.sharding.tp_world_size),
        )
        if world_size > 1
    ]
    if not checks:
        return

    dist = _require_distributed_initialized()
    payload = (contract.cross_rank_fingerprint(), BACKEND_ID, int(num_vocab_tiles))
    for axis, group, _ in checks:
        gathered: list[Any] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, payload, group=group)
        mismatched = [(rank, other) for rank, other in enumerate(gathered) if other != payload]
        if mismatched:
            rank, other = mismatched[0]
            raise LossContractError(
                f"cross-rank preflight failed on the {axis} axis: this rank has {payload} "
                f"but {axis} rank {rank} has {other}; every rank must agree on the contract "
                "fingerprint, backend id and num_vocab_tiles before any collective"
            )


def _gather_global_rewards(
    rewards: torch.Tensor, contract: GRPOLossContract, dp_group: Any
) -> torch.Tensor:
    """Assemble the global ``[num_sequences]`` reward vector on every rank.

    ``sequence_shard_bounds`` is a contiguous partition in DP-rank order, so
    concatenating the gathered slices in rank order reproduces the global vector
    exactly -- no ownership arbitration is needed.
    """

    sharding = contract.sharding
    local = rewards.float()
    if sharding.dp_world_size == 1:
        return local.contiguous()

    dist = _require_distributed_initialized()
    max_local = max(end - start for start, end in sharding.sequence_shard_bounds)
    padded = local.new_zeros(max_local)
    padded[: local.shape[0]] = local
    gathered = [torch.empty_like(padded) for _ in range(sharding.dp_world_size)]
    dist.all_gather(gathered, padded.contiguous(), group=dp_group)
    return torch.cat(
        [
            gathered[rank][: end - start]
            for rank, (start, end) in enumerate(sharding.sequence_shard_bounds)
        ],
        dim=0,
    )


def _group_advantages(global_rewards: torch.Tensor, contract: GRPOLossContract) -> torch.Tensor:
    """Group-relative advantages over the global reward vector.

    Evaluated identically on every rank from an identically shaped input, so no
    merge is involved and the result is bitwise-equal mesh-wide.  The variance
    is two-pass: centring before squaring keeps the result meaningful when the
    rewards share a large offset, which ``E[x^2] - E[x]^2`` does not.
    """

    advantage = contract.objective.advantage
    boundaries = contract.sharding.group_boundaries
    parts: list[torch.Tensor] = []
    for index in range(len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        group = global_rewards[start:end]
        count = float(end - start)
        centered = group - group.sum() / count
        if advantage.normalizer is AdvantageNormalizer.MEAN_ONLY:
            parts.append(centered)
            continue
        variance = (centered * centered).sum() / count
        parts.append(centered / variance.clamp_min(advantage.std_eps**2).sqrt())
    return torch.cat(parts, dim=0)


def _sequence_totals(values: torch.Tensor, contract: GRPOLossContract) -> torch.Tensor:
    """Reduce this rank's per-token values to one total per owned sequence.

    The reduced extent is ``padded_seq_len``, which the contract fixes, so this
    sum is byte-for-byte the same work at every DP degree.
    """

    sharding = contract.sharding
    view = values.reshape(sharding.local_num_sequences, sharding.padded_seq_len)
    return view.sum(dim=1)


def _assemble_global_vector(
    local_totals: torch.Tensor, contract: GRPOLossContract, dp_group: Any
) -> torch.Tensor:
    """Place every rank's per-sequence totals into the fixed global vector.

    Returns ``[num_sequences, ...]``.  The length is a property of the contract,
    never of the DP degree, and filling it is pure placement -- no arithmetic
    touches the gathered values -- so the downstream reduction sees identical
    inputs at every degree.
    """

    sharding = contract.sharding
    if sharding.dp_world_size == 1:
        return local_totals

    dist = _require_distributed_initialized()
    trailing = local_totals.shape[1:]
    max_local = max(end - start for start, end in sharding.sequence_shard_bounds)
    padded = local_totals.new_zeros((max_local, *trailing))
    padded[: local_totals.shape[0]] = local_totals.detach()
    gathered = [torch.empty_like(padded) for _ in range(sharding.dp_world_size)]
    dist.all_gather(gathered, padded.contiguous(), group=dp_group)

    # This rank's own slice comes from the live tensor: all_gather severs the
    # graph, and the other ranks' slices are constants here anyway.
    pieces = []
    for rank, (start, end) in enumerate(sharding.sequence_shard_bounds):
        pieces.append(local_totals if rank == sharding.dp_rank else gathered[rank][: end - start])
    return torch.cat(pieces, dim=0)


def _normalized(
    per_sequence_totals: torch.Tensor,
    per_sequence_counts: torch.Tensor,
    contract: GRPOLossContract,
) -> torch.Tensor:
    """Apply the declared token normalizer to fixed-order sequence totals."""

    reduction = contract.reduction
    normalizer = reduction.token_normalizer
    if normalizer is TokenNormalizer.FIXED_CONSTANT:
        fixed_normalizer_constant = reduction.fixed_normalizer_constant
        assert fixed_normalizer_constant is not None
        return per_sequence_totals.sum() / float(fixed_normalizer_constant)
    if normalizer is TokenNormalizer.GLOBAL_ACTIVE_TOKENS:
        return per_sequence_totals.sum() / per_sequence_counts.sum().to(per_sequence_totals.dtype)

    # PER_SEQUENCE_THEN_MEAN: sequences with no active token contribute nothing
    # and are excluded from the outer denominator rather than counted as zero.
    live = per_sequence_counts > 0
    denominators = per_sequence_counts.to(per_sequence_totals.dtype).clamp_min(1.0)
    per_sequence_means = torch.where(
        live, per_sequence_totals / denominators, torch.zeros_like(per_sequence_totals)
    )
    return per_sequence_means.sum() / live.sum().to(per_sequence_totals.dtype)


class DistributedGRPOLossOp:
    """Deterministic GRPO loss over TP-sharded logits with a DP-invariant reduction.

    The WS2 reference (issue #241 PR5).  ``policy_local_logits`` is this rank's
    ``[n, local_vocab]`` vocabulary shard, not a dense ``[n, vocab]`` tensor:
    tensor parallelism is delegated to the vocab-parallel logprob path, and this
    operator owns the sum over tokens and sequences.
    """

    op_class = "grpo_loss"
    is_batch_invariant = True

    def __init__(self) -> None:
        self._logprob = VocabParallelLogprobOp()

    def __call__(self, *args: Any, **kwargs: Any) -> GRPOLossResult:
        return self.apply(*args, **kwargs)

    def apply(
        self,
        policy_local_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        rewards: torch.Tensor,
        *,
        contract: GRPOLossContract,
        ref_local_logits: torch.Tensor | None = None,
        tp_group: Any = None,
        dp_group: Any = None,
        num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES,
        validate: bool = True,
    ) -> GRPOLossResult:
        if not isinstance(contract, GRPOLossContract):
            raise LossContractError("contract must be a GRPOLossContract")
        _validate_invocation(
            policy_local_logits,
            ref_local_logits,
            action_ids,
            old_logps,
            rewards,
            contract,
            dp_group,
        )
        sharding = contract.sharding
        objective = contract.objective
        if validate:
            _preflight_cross_rank_agreement(contract, dp_group, tp_group, num_vocab_tiles)

        logp_policy, _ = self._logprob.apply(
            policy_local_logits,
            action_ids,
            contract=contract.logprob,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
        )
        active = torch.tensor(
            contract.logprob.mask.active_mask,
            dtype=torch.bool,
            device=policy_local_logits.device,
        )

        delta = (logp_policy - old_logps.float()).masked_fill(~active, 0.0)
        ratio = delta.exp()

        if ref_local_logits is None:
            kl_terms = torch.zeros_like(logp_policy)
        else:
            with torch.no_grad():
                logp_ref, _ = self._logprob.apply(
                    ref_local_logits,
                    action_ids,
                    contract=contract.logprob,
                    tp_group=tp_group,
                    num_vocab_tiles=num_vocab_tiles,
                    validate=False,
                )
            diff = (logp_ref - logp_policy).masked_fill(~active, 0.0)
            if objective.kl_estimator is KLEstimator.K3_UNBIASED:
                kl_terms = diff.exp() - diff - 1.0
            else:
                kl_terms = -diff
            kl_terms = kl_terms.masked_fill(~active, 0.0)

        global_rewards = _gather_global_rewards(rewards, contract, dp_group)
        advantages = _group_advantages(global_rewards, contract)
        adv_tokens = (
            advantages[sharding.local_sequence_start : sharding.local_sequence_end]
            .reshape(-1, 1)
            .expand(sharding.local_num_sequences, sharding.padded_seq_len)
            .reshape(-1)
        )

        clip = objective.clip
        unclipped = ratio * adv_tokens
        clipped = ratio.clamp(clip.lower_bound, clip.upper_bound) * adv_tokens
        policy_terms = (-torch.minimum(unclipped, clipped)).masked_fill(~active, 0.0)

        packed = torch.stack(
            (_sequence_totals(policy_terms, contract), _sequence_totals(kl_terms, contract)),
            dim=1,
        )
        # Fixed-extent reductions: [local_seqs, padded_seq_len] -> [local_seqs],
        # gathered into [num_sequences] -> scalar, at every DP degree.
        totals = _assemble_global_vector(packed, contract, dp_group)
        per_sequence_policy = totals[:, _CH_POLICY]
        per_sequence_kl = totals[:, _CH_KL]
        per_sequence_counts = _assemble_global_vector(
            _sequence_totals(active.to(torch.long), contract), contract, dp_group
        )

        total_active = int(per_sequence_counts.sum().item())
        if validate and total_active == 0:
            raise LossContractError(
                "the global batch holds no active tokens; the loss normalizer would divide by zero"
            )

        policy_loss = _normalized(per_sequence_policy, per_sequence_counts, contract)
        kl = _normalized(per_sequence_kl, per_sequence_counts, contract)
        loss = policy_loss + objective.beta * kl

        provenance = {
            "backend_id": BACKEND_ID,
            "implementation_kind": "reference",
            "num_vocab_tiles": int(num_vocab_tiles),
            "padded_seq_len": int(sharding.padded_seq_len),
            "num_sequences": int(sharding.num_sequences),
            "global_active_tokens": total_active,
            "reference_model_used": ref_local_logits is not None,
            "requested_contract": contract.to_dict(),
            "cross_rank_fingerprint": contract.cross_rank_fingerprint(),
        }
        return GRPOLossResult(
            loss=loss,
            policy_loss=policy_loss,
            kl=kl,
            advantages=advantages,
            per_sequence_policy=per_sequence_policy.detach(),
            per_sequence_kl=per_sequence_kl.detach(),
            per_sequence_active_tokens=per_sequence_counts.detach(),
            provenance=provenance,
        )


__all__ = [
    "BACKEND_ID",
    "GRPOLossResult",
    "DistributedGRPOLossOp",
]
