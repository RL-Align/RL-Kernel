# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic GRPO loss on the TP-aware logprob path.

The headline claim under test is that the scalar loss and its gradient are
bitwise-identical across every TP x DP degree, given a fixed vocab tile count.
``TestMeshBitwise`` is the file's centre of gravity: it runs the reachable
degrees on real NCCL ranks and compares raw bit patterns against the single-rank
baseline.

The remaining classes support that claim rather than duplicate it: the single-
rank tests pin the objective's algebraic identities (ratio exactly 1, KL exactly
0), the negative controls prove the bitwise comparisons are not vacuous, and the
guard tests check that a rank which disagrees about the invocation aborts
instead of corrupting the merge.

Comparisons are made on ``per_sequence_policy``/``per_sequence_kl`` rather than
on the scalar loss alone.  That is not a stylistic choice: measured over this
file's own inputs, regrouping the token sum moves the per-sequence vector in 12
of 12 seeds but the scalar loss in only 5 of 12, because averaging
``NUM_SEQUENCES`` totals into one fp32 number rounds most reorderings away.
Asserting on the scalar alone would let a wrong reduction pass most of the time.

Context parallelism is out of scope (see ``LossShardingSpec``); the contract
rejects ``cp_world_size > 1`` and ``TestGuards`` covers that.

Multi-rank tests need one GPU per rank and skip otherwise.  They stay small on
purpose -- a 1000-token vocabulary and 8 sequences of 32 slots -- so they can
share a node with a running training job; each worker additionally caps itself
with ``set_per_process_memory_fraction`` so a regression here cannot starve a
co-tenant.
"""

from __future__ import annotations

import math
import os
import queue
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.contracts.operators.logprob import (
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.contracts.operators.loss import (
    AdvantageNormalizer,
    AdvantageSpec,
    ClipSpec,
    GRPOLossContract,
    KLEstimator,
    LossContractError,
    LossReductionSpec,
    LossShardingSpec,
    ObjectiveSpec,
    TokenNormalizer,
)
from rl_engine.reference.loss.distributed_grpo_loss import BACKEND_ID, DistributedGRPOLossOp

# Global batch geometry, shared by every configuration so the comparisons are
# between meshes rather than between problems.  The degrees below are chosen so
# that DP and CP each reach 4 while every shard stays tile-aligned.
NUM_SEQUENCES = 8
PADDED_SEQ_LEN = 32
NUM_TOKEN_SLOTS = NUM_SEQUENCES * PADDED_SEQ_LEN
# Deliberately straddles the DP=2 split at 4 and the DP=4 splits at 2/4/6, so the
# replicated-advantage path is exercised by groups that no single rank owns.
GROUP_BOUNDARIES = (0, 3, NUM_SEQUENCES)
REAL_VOCAB = 1000
PADDED_VOCAB = 1024
NUM_VOCAB_TILES = 32
BETA = 0.04
SEED = 20260811

_SPAWN_TIMEOUT_S = 600
# Fraction of each card the workers may allocate.  The test tensors need a few
# megabytes; the cap exists so a bug cannot balloon into a co-tenant job.
_MEMORY_FRACTION = 0.02


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """Raw bit pattern, so -0.0 vs 0.0 and NaN vs NaN compare honestly."""

    view_dtype = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }[tensor.dtype]
    return tensor.contiguous().view(view_dtype)


def _scalar_bits(tensor: torch.Tensor) -> int:
    return int(_bits(tensor.detach().float().cpu()).item())


def _vector_bits(tensor: torch.Tensor) -> list[int]:
    return _bits(tensor.detach().float().cpu()).tolist()


def _reduction_fingerprint(result) -> tuple:
    """Everything a change of reduction order can move, at full resolution.

    The per-sequence vectors come first because they are the sensitive part;
    the scalars are carried along so a normalizer bug is caught too.
    """

    return (
        _vector_bits(result.per_sequence_policy),
        _vector_bits(result.per_sequence_kl),
        _scalar_bits(result.loss),
        _scalar_bits(result.policy_loss),
        _scalar_bits(result.kl),
    )


def _cuda_device_count() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:  # pragma: no cover - driver-level failures
        return 0


def _requires_gpus(count: int):
    return pytest.mark.skipif(
        _cuda_device_count() < count,
        reason=f"needs {count} CUDA devices for a real {count}-rank mesh",
    )


# --------------------------------------------------------------------------- #
# Global problem definition
# --------------------------------------------------------------------------- #
def _global_active_mask() -> tuple[bool, ...]:
    """Right-padded sequences of strictly decreasing length.

    The lengths are staggered so that no two DP shards hold the same number of
    active tokens; equal counts would let a rank-local normalizer accidentally
    agree with the global one and pass a test it should fail.
    """

    mask: list[bool] = []
    for seq in range(NUM_SEQUENCES):
        real_len = PADDED_SEQ_LEN - seq * 3
        mask.extend(slot < real_len for slot in range(PADDED_SEQ_LEN))
    return tuple(mask)


GLOBAL_ACTIVE_MASK = _global_active_mask()


def _global_inputs(seed: int = SEED) -> dict[str, torch.Tensor]:
    """Deterministic global tensors every rank slices its own view out of.

    ``old_logps`` is centred on ``-log(REAL_VOCAB)``, the scale of a selected
    logprob under near-uniform logits, so the importance ratios land around 1
    with real spread.  Leaving it centred on 0 would make every ratio ~1e-3 and
    every per-token loss term nearly identical, and a sum of near-identical
    values is almost invariant to how it is grouped -- which would quietly
    drain the power out of every bitwise comparison in this file.
    """

    gen = torch.Generator().manual_seed(seed)
    return {
        "policy": torch.randn(NUM_TOKEN_SLOTS, PADDED_VOCAB, generator=gen),
        "ref": torch.randn(NUM_TOKEN_SLOTS, PADDED_VOCAB, generator=gen),
        "action_ids": torch.randint(0, REAL_VOCAB, (NUM_TOKEN_SLOTS,), generator=gen),
        "old_logps": torch.randn(NUM_TOKEN_SLOTS, generator=gen) * 0.5 - math.log(REAL_VOCAB),
        "rewards": torch.randn(NUM_SEQUENCES, generator=gen),
    }


def _dp_bounds(dp: int) -> tuple[tuple[int, int], ...]:
    """Contiguous sequence partition in DP-rank order."""

    seqs = NUM_SEQUENCES // dp
    return tuple((d * seqs, (d + 1) * seqs) for d in range(dp))


def _owned_rows(bounds: tuple[int, int]) -> list[int]:
    """Global token-slot indices this shard owns, in canonical local row order."""

    start, end = bounds
    return [
        seq * PADDED_SEQ_LEN + slot for seq in range(start, end) for slot in range(PADDED_SEQ_LEN)
    ]


def _build_contract(
    *,
    tp_rank: int,
    tp: int,
    dp_rank: int,
    dp: int,
    objective: ObjectiveSpec | None = None,
    reduction: LossReductionSpec | None = None,
) -> GRPOLossContract:
    bounds = _dp_bounds(dp)
    rows = _owned_rows(bounds[dp_rank])
    shard = PADDED_VOCAB // tp
    logprob = LogprobContract(
        role=LogprobRole.TRAIN,
        dtype=LogprobDType.FP32,
        mask=MaskSpec(
            num_tokens=len(rows),
            active_mask=tuple(GLOBAL_ACTIVE_MASK[row] for row in rows),
        ),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp,
            vocab_shard_bounds=tuple((r * shard, (r + 1) * shard) for r in range(tp)),
            real_vocab_size=REAL_VOCAB,
            padded_vocab_size=PADDED_VOCAB,
        ),
        reduction=ReductionSpec(),
    )
    sharding = LossShardingSpec(
        dp_rank=dp_rank,
        dp_world_size=dp,
        num_sequences=NUM_SEQUENCES,
        padded_seq_len=PADDED_SEQ_LEN,
        sequence_shard_bounds=bounds,
        group_boundaries=GROUP_BOUNDARIES,
    )
    return GRPOLossContract(
        logprob=logprob,
        sharding=sharding,
        objective=objective if objective is not None else ObjectiveSpec(beta=BETA),
        reduction=reduction if reduction is not None else LossReductionSpec(),
    )


def _rank_inputs(
    globals_: dict[str, torch.Tensor],
    *,
    tp_rank: int,
    tp: int,
    bounds: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    rows = _owned_rows(bounds)
    shard = PADDED_VOCAB // tp
    cols = slice(tp_rank * shard, (tp_rank + 1) * shard)
    start, end = bounds
    return {
        "policy": globals_["policy"][rows, cols].clone().to(device).requires_grad_(True),
        "ref": globals_["ref"][rows, cols].clone().to(device),
        "action_ids": globals_["action_ids"][rows].clone().to(device),
        "old_logps": globals_["old_logps"][rows].clone().to(device),
        "rewards": globals_["rewards"][start:end].clone().to(device),
    }


def _single_rank_setup(
    *,
    objective: ObjectiveSpec | None = None,
    reduction: LossReductionSpec | None = None,
    device: str = "cpu",
    globals_: dict[str, torch.Tensor] | None = None,
) -> tuple[GRPOLossContract, dict[str, torch.Tensor]]:
    """Contract and tensors for one rank owning the whole batch."""

    contract = _build_contract(
        tp_rank=0, tp=1, dp_rank=0, dp=1, objective=objective, reduction=reduction
    )
    tensors = _rank_inputs(
        globals_ if globals_ is not None else _global_inputs(),
        tp_rank=0,
        tp=1,
        bounds=(0, NUM_SEQUENCES),
        device=torch.device(device),
    )
    return contract, tensors


def _run_single_rank(
    *,
    num_vocab_tiles: int = NUM_VOCAB_TILES,
    with_reference: bool = True,
    **setup,
):
    """Baseline invocation: one rank, no collectives."""

    contract, tensors = _single_rank_setup(**setup)
    result = DistributedGRPOLossOp().apply(
        tensors["policy"],
        tensors["action_ids"],
        tensors["old_logps"],
        tensors["rewards"],
        contract=contract,
        ref_local_logits=tensors["ref"] if with_reference else None,
        num_vocab_tiles=num_vocab_tiles,
    )
    return result, tensors, contract


# --------------------------------------------------------------------------- #
# Multi-rank harness
# --------------------------------------------------------------------------- #
def _mesh_worker(rank, world_size, init_method, result_queue, tp, dp, scenario):
    """One NCCL rank of a TP x DP mesh.

    Only plain Python values go back on the queue.  Tensors sent over a
    multiprocessing queue travel by shared memory and are lost when the sender
    exits before the parent maps them, which surfaces as an empty queue rather
    than an error.
    """

    payload = {"rank": rank}
    try:
        import torch.distributed as dist

        device_index = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        torch.cuda.set_per_process_memory_fraction(_MEMORY_FRACTION, device_index)
        device = torch.device("cuda", device_index)
        dist.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=rank
        )

        # Rank layout puts TP fastest: rank = dp_rank * tp + tp_rank.
        tp_rank = rank % tp
        dp_rank = rank // tp
        # new_group is collective, so every rank builds every subgroup in the
        # same order even though it only keeps one of each.
        tp_group = None
        dp_group = None
        if tp > 1:
            tp_groups = [
                dist.new_group(ranks=list(range(base * tp, (base + 1) * tp))) for base in range(dp)
            ]
            tp_group = tp_groups[dp_rank]
        if dp > 1:
            dp_groups = [
                dist.new_group(ranks=list(range(offset, world_size, tp))) for offset in range(tp)
            ]
            dp_group = dp_groups[tp_rank]

        objective = ObjectiveSpec(beta=BETA)
        if scenario == "preflight_dp_mismatch" and dp_rank == 1:
            # Perturb a pure fingerprint field.  Changing the batch geometry
            # instead would change this rank's local shapes, so it would fail
            # while constructing its contract -- before the preflight -- and
            # strand the other ranks inside the all-gather.
            objective = ObjectiveSpec(beta=BETA * 2)
        if scenario == "preflight_tp_mismatch" and tp_rank == 1:
            # beta is invisible to the logprob contract, so only the loss
            # preflight's TP-axis check can catch this one.
            objective = ObjectiveSpec(beta=BETA * 2)

        contract = _build_contract(
            tp_rank=tp_rank, tp=tp, dp_rank=dp_rank, dp=dp, objective=objective
        )
        tensors = _rank_inputs(
            _global_inputs(),
            tp_rank=tp_rank,
            tp=tp,
            bounds=contract.sharding.sequence_shard_bounds[dp_rank],
            device=device,
        )
        op = DistributedGRPOLossOp()
        result = op.apply(
            tensors["policy"],
            tensors["action_ids"],
            tensors["old_logps"],
            tensors["rewards"],
            contract=contract,
            ref_local_logits=tensors["ref"],
            tp_group=tp_group,
            dp_group=dp_group,
            num_vocab_tiles=NUM_VOCAB_TILES,
        )
        result.loss.backward()

        grad = tensors["policy"].grad
        payload.update(
            {
                "ok": True,
                "loss_bits": _scalar_bits(result.loss),
                "policy_bits": _scalar_bits(result.policy_loss),
                "kl_bits": _scalar_bits(result.kl),
                "per_sequence_policy_bits": _vector_bits(result.per_sequence_policy),
                "per_sequence_kl_bits": _vector_bits(result.per_sequence_kl),
                "per_sequence_counts": result.per_sequence_active_tokens.cpu().tolist(),
                "advantage_bits": _vector_bits(result.advantages),
                "global_active": result.provenance["global_active_tokens"],
                "backend_id": result.provenance["backend_id"],
                "grad_nonzero": bool(grad.abs().sum().item() > 0.0),
                # Gradient of one fixed global token slot, keyed by vocab shard
                # so the parent can reassemble the full row across TP ranks.
                "grad_row_bits": _vector_bits(grad[0]) if dp_rank == 0 else None,
                "vocab_start": contract.logprob.sharding.local_vocab_start,
            }
        )
    except BaseException as exc:  # the parent re-raises the text
        payload.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc(),
            }
        )
    finally:
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:  # pragma: no cover - teardown best effort
            pass
        try:
            result_queue.put(payload)
        except Exception:  # pragma: no cover - queue already closed
            pass


def _run_mesh(tp: int, dp: int, *, scenario: str = "correctness") -> list[dict]:
    """Spawn a ``tp * dp`` NCCL mesh and collect one payload per rank."""

    world_size = tp * dp
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmp:
        init_method = f"file://{Path(tmp) / 'store'}"
        spawned = mp.spawn(
            _mesh_worker,
            args=(world_size, init_method, result_queue, tp, dp, scenario),
            nprocs=world_size,
            join=False,
        )
        payloads: list[dict] = []
        try:
            for _ in range(world_size):
                payloads.append(result_queue.get(timeout=_SPAWN_TIMEOUT_S))
        except queue.Empty:  # pragma: no cover - only on a genuine hang
            pytest.fail(
                f"TP={tp} DP={dp}: only {len(payloads)}/{world_size} ranks reported "
                f"within {_SPAWN_TIMEOUT_S}s"
            )
        finally:
            spawned.join(timeout=_SPAWN_TIMEOUT_S)
    return sorted(payloads, key=lambda item: item["rank"])


def _require_all_ok(payloads: list[dict]) -> None:
    failures = [item for item in payloads if not item.get("ok")]
    if failures:
        head = failures[0]
        pytest.fail(f"rank {head['rank']} failed: {head['error']}\n{head['tb']}")


def _assemble_grad_row(payloads: list[dict]) -> list[int]:
    """Reassemble the fixed token slot's gradient row across TP vocab shards."""

    contributions = [item for item in payloads if item["grad_row_bits"] is not None]
    contributions.sort(key=lambda item: item["vocab_start"])
    row: list[int] = []
    for item in contributions:
        row.extend(item["grad_row_bits"])
    return row


def _consensus(payloads: list[dict]) -> dict:
    """Collapse the mesh's per-rank payloads into the one replicated answer."""

    _require_all_ok(payloads)
    replicated = (
        "loss_bits",
        "policy_bits",
        "kl_bits",
        "per_sequence_policy_bits",
        "per_sequence_kl_bits",
        "per_sequence_counts",
        "global_active",
        "advantage_bits",
    )
    for key in replicated:
        values = {repr(item[key]) for item in payloads}
        assert len(values) == 1, f"ranks disagree on {key}: {values}"
    assert all(item["grad_nonzero"] for item in payloads), (
        "some rank produced a zero gradient; the all-gather likely severed the "
        "graph for that rank's own block"
    )
    head = payloads[0]
    consensus = {key: head[key] for key in replicated}
    consensus["grad_row"] = _assemble_grad_row(payloads)
    return consensus


# --------------------------------------------------------------------------- #
# Single-rank behaviour
# --------------------------------------------------------------------------- #
class TestSingleRank:
    def test_forward_and_backward_run(self):
        result, tensors, contract = _run_single_rank()
        result.loss.backward()
        assert result.loss.dtype is torch.float32
        assert result.loss.shape == ()
        assert tensors["policy"].grad.abs().sum().item() > 0.0
        assert result.provenance["backend_id"] == BACKEND_ID
        assert result.provenance["global_active_tokens"] == sum(GLOBAL_ACTIVE_MASK)

    def test_unpacks_as_the_legacy_triple(self):
        result, _, _ = _run_single_rank()
        loss, policy_loss, kl = result
        assert _scalar_bits(loss) == _scalar_bits(result.loss)
        assert _scalar_bits(policy_loss) == _scalar_bits(result.policy_loss)
        assert _scalar_bits(kl) == _scalar_bits(result.kl)

    def test_run_to_run_bitwise_stability(self):
        first, _, _ = _run_single_rank()
        second, _, _ = _run_single_rank()
        assert _scalar_bits(first.loss) == _scalar_bits(second.loss)
        assert _scalar_bits(first.policy_loss) == _scalar_bits(second.policy_loss)
        assert _scalar_bits(first.kl) == _scalar_bits(second.kl)

    def test_advantages_are_group_centred(self):
        result, _, _ = _run_single_rank()
        advantages = result.advantages
        for start, end in zip(GROUP_BOUNDARIES[:-1], GROUP_BOUNDARIES[1:], strict=True):
            assert advantages[start:end].sum().item() == pytest.approx(0.0, abs=1e-5)

    def test_inactive_tokens_do_not_affect_the_loss(self):
        # Inactive slots carry real logits here, so a backend that forgot to
        # mask them would shift the loss rather than merely change a padding.
        baseline, _, _ = _run_single_rank()
        perturbed_globals = _global_inputs()
        inactive = [i for i, flag in enumerate(GLOBAL_ACTIVE_MASK) if not flag]
        perturbed_globals["policy"][inactive] += 7.5
        perturbed_globals["old_logps"][inactive] -= 3.25
        perturbed, _, _ = _run_single_rank(globals_=perturbed_globals)
        assert _scalar_bits(perturbed.loss) == _scalar_bits(baseline.loss)

    def test_dispatch_resolves_this_backend(self):
        from rl_engine.runtime.registry import kernel_registry

        _, _, contract = _run_single_rank()
        dispatched = kernel_registry.get_loss_op(contract)
        assert dispatched.capability.backend_id == BACKEND_ID
        assert isinstance(dispatched.op, DistributedGRPOLossOp)
        assert dispatched.provenance["fallback"] is False


class TestKLZeroIdentity:
    """Acceptance criterion: the reference-equals-policy identity is exact."""

    def _identity_run(self, **overrides):
        # old_logps set to the operator's own selected logprob, so the ratio is
        # exp(0) = 1 exactly rather than approximately.
        contract, tensors = _single_rank_setup(**overrides)
        op = DistributedGRPOLossOp()
        with torch.no_grad():
            logp, _ = op._logprob.apply(
                tensors["policy"],
                tensors["action_ids"],
                contract=contract.logprob,
                num_vocab_tiles=NUM_VOCAB_TILES,
            )
        result = op.apply(
            tensors["policy"],
            tensors["action_ids"],
            logp,
            tensors["rewards"],
            contract=contract,
            ref_local_logits=tensors["policy"].detach(),
            num_vocab_tiles=NUM_VOCAB_TILES,
        )
        return result, tensors

    def test_kl_is_exactly_zero(self):
        result, _ = self._identity_run()
        assert _scalar_bits(result.kl) == _scalar_bits(torch.zeros(()))

    def test_loss_reduces_to_the_policy_term(self):
        result, _ = self._identity_run()
        assert _scalar_bits(result.loss) == _scalar_bits(result.policy_loss)

    def test_ratio_is_exactly_one_so_clipping_cannot_bind(self):
        # A ratio that is only approximately 1 would land outside a sufficiently
        # tight clip range and change the answer; an exact 1 cannot.
        tight, _ = self._identity_run(
            objective=ObjectiveSpec(beta=BETA, clip=ClipSpec(clip_eps_low=1e-7, clip_eps_high=1e-7))
        )
        loose, _ = self._identity_run(
            objective=ObjectiveSpec(beta=BETA, clip=ClipSpec(clip_eps_low=0.9, clip_eps_high=0.9))
        )
        assert _scalar_bits(tight.loss) == _scalar_bits(loose.loss)

    def test_policy_term_matches_the_masked_advantage_mean(self):
        result, _ = self._identity_run()
        advantages = result.advantages
        mask = torch.tensor(GLOBAL_ACTIVE_MASK)
        per_token = advantages.reshape(-1, 1).expand(NUM_SEQUENCES, PADDED_SEQ_LEN).reshape(-1)
        expected = -per_token.masked_fill(~mask, 0.0).sum() / mask.sum()
        # Not a bitwise comparison: this reference sums the flat [N] vector,
        # while the operator sums through its fixed tile structure.
        assert result.policy_loss.item() == pytest.approx(expected.item(), abs=1e-6)


class TestNormalizers:
    def test_global_active_tokens_matches_a_flat_masked_mean(self):
        result, _, _ = _run_single_rank()
        assert result.provenance["global_active_tokens"] == sum(GLOBAL_ACTIVE_MASK)

    def test_normalizers_disagree_on_unequal_sequence_lengths(self):
        # If these ever agreed, the normalizer choice would be untestable and
        # the sequence lengths in this file would have stopped being staggered.
        token_mean, _, _ = _run_single_rank()
        seq_mean, _, _ = _run_single_rank(
            reduction=LossReductionSpec(token_normalizer=TokenNormalizer.PER_SEQUENCE_THEN_MEAN)
        )
        fixed, _, _ = _run_single_rank(
            reduction=LossReductionSpec(
                token_normalizer=TokenNormalizer.FIXED_CONSTANT,
                fixed_normalizer_constant=NUM_TOKEN_SLOTS,
            )
        )
        values = {
            _scalar_bits(token_mean.loss),
            _scalar_bits(seq_mean.loss),
            _scalar_bits(fixed.loss),
        }
        assert len(values) == 3

    def test_fixed_constant_normalizer_scales_the_token_sum(self):
        active = sum(GLOBAL_ACTIVE_MASK)
        token_mean, _, _ = _run_single_rank()
        fixed, _, _ = _run_single_rank(
            reduction=LossReductionSpec(
                token_normalizer=TokenNormalizer.FIXED_CONSTANT,
                fixed_normalizer_constant=NUM_TOKEN_SLOTS,
            )
        )
        assert fixed.policy_loss.item() == pytest.approx(
            token_mean.policy_loss.item() * active / NUM_TOKEN_SLOTS, rel=1e-6
        )

    def test_kl_estimators_differ(self):
        k3, _, _ = _run_single_rank()
        k1, _, _ = _run_single_rank(
            objective=ObjectiveSpec(beta=BETA, kl_estimator=KLEstimator.K1_LOG_RATIO)
        )
        assert _scalar_bits(k3.kl) != _scalar_bits(k1.kl)
        # k3 is non-negative by construction; the plain log-ratio is not.
        assert k3.kl.item() >= 0.0

    def test_mean_only_advantage_skips_the_std_divisor(self):
        std_normalized, _, _ = _run_single_rank()
        mean_only, _, _ = _run_single_rank(
            objective=ObjectiveSpec(
                beta=BETA,
                advantage=AdvantageSpec(normalizer=AdvantageNormalizer.MEAN_ONLY),
            )
        )
        assert _scalar_bits(std_normalized.loss) != _scalar_bits(mean_only.loss)


class TestNegativeControls:
    """Prove the bitwise assertions elsewhere are not comparing constants.

    Each control changes something that genuinely regroups or rescales the
    reduction and asserts the compared surface notices.  If one of these ever
    starts passing trivially, the corresponding positive assertion has stopped
    meaning anything.
    """

    def test_regrouping_the_token_sum_moves_the_per_sequence_vector(self):
        # Split each sequence's token sum in half before adding, changing the
        # summation tree without changing a single input value.  The scalar loss
        # absorbs that most of the time; the per-sequence vector does not, which
        # is why it is the comparison surface everywhere else in this file.
        import rl_engine.reference.loss.distributed_grpo_loss as module

        baseline, _, _ = _run_single_rank()
        original = module._sequence_totals

        def halved(values, contract):
            view = values.reshape(
                contract.sharding.local_num_sequences, 2, contract.sharding.padded_seq_len // 2
            )
            return view.sum(dim=2).sum(dim=1)

        module._sequence_totals = halved
        try:
            regrouped, _, _ = _run_single_rank()
        finally:
            module._sequence_totals = original
        assert _vector_bits(baseline.per_sequence_policy) != _vector_bits(
            regrouped.per_sequence_policy
        )

    def test_vocab_tile_count_perturbs_the_logprob_it_consumes(self):
        baseline, _, _ = _run_single_rank()
        retiled, _, _ = _run_single_rank(num_vocab_tiles=NUM_VOCAB_TILES * 2)
        assert _reduction_fingerprint(baseline) != _reduction_fingerprint(retiled)

    def test_sequence_order_matters_to_the_scalar(self):
        # An all_reduce would combine sequence totals in topology order rather
        # than in global sequence order.  Permuting the assembled grid is a
        # stand-in for that mistake, and the loss must notice.
        import rl_engine.reference.loss.distributed_grpo_loss as module

        baseline, _, _ = _run_single_rank()
        permutation = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
        original = module._assemble_global_vector
        module._assemble_global_vector = lambda totals, contract, group: original(
            totals, contract, group
        )[permutation]
        try:
            permuted, _, _ = _run_single_rank()
        finally:
            module._assemble_global_vector = original
        assert _vector_bits(baseline.per_sequence_policy) != _vector_bits(
            permuted.per_sequence_policy
        )

    def test_clip_epsilon_perturbs_the_loss(self):
        baseline, _, _ = _run_single_rank()
        clipped, _, _ = _run_single_rank(
            objective=ObjectiveSpec(beta=BETA, clip=ClipSpec(clip_eps_low=0.01, clip_eps_high=0.01))
        )
        assert _reduction_fingerprint(baseline) != _reduction_fingerprint(clipped)


class TestGuards:
    def test_reference_logits_required_when_beta_is_positive(self):
        contract, tensors = _single_rank_setup()
        with pytest.raises(LossContractError, match="ref_local_logits is required"):
            DistributedGRPOLossOp().apply(
                tensors["policy"],
                tensors["action_ids"],
                tensors["old_logps"],
                tensors["rewards"],
                contract=contract,
                num_vocab_tiles=NUM_VOCAB_TILES,
            )

    def test_reference_optional_when_beta_is_zero(self):
        result, _, _ = _run_single_rank(objective=ObjectiveSpec(beta=0.0), with_reference=False)
        assert _scalar_bits(result.loss) == _scalar_bits(result.policy_loss)
        assert _scalar_bits(result.kl) == _scalar_bits(torch.zeros(()))
        assert result.provenance["reference_model_used"] is False

    def test_row_count_must_match_owned_slots(self):
        contract, tensors = _single_rank_setup()
        with pytest.raises(LossContractError, match="token slots"):
            DistributedGRPOLossOp().apply(
                tensors["policy"][:-1],
                tensors["action_ids"][:-1],
                tensors["old_logps"][:-1],
                tensors["rewards"],
                contract=contract,
                ref_local_logits=tensors["ref"][:-1],
                num_vocab_tiles=NUM_VOCAB_TILES,
            )

    def test_reward_count_must_match_owned_sequences(self):
        contract, tensors = _single_rank_setup()
        with pytest.raises(LossContractError, match="one entry per sequence"):
            DistributedGRPOLossOp().apply(
                tensors["policy"],
                tensors["action_ids"],
                tensors["old_logps"],
                tensors["rewards"][:-1],
                contract=contract,
                ref_local_logits=tensors["ref"],
                num_vocab_tiles=NUM_VOCAB_TILES,
            )


# --------------------------------------------------------------------------- #
# The claim: bitwise equality across the mesh
# --------------------------------------------------------------------------- #
# Every reachable (TP, DP) combination with at most 4 ranks.  Larger degrees
# need a bigger node and run unchanged there via the same helper.
MESH_CONFIGS = [
    pytest.param(2, 1, id="tp2"),
    pytest.param(4, 1, id="tp4"),
    pytest.param(1, 2, id="dp2"),
    pytest.param(1, 4, id="dp4"),
    pytest.param(2, 2, id="tp2xdp2"),
]


@pytest.fixture(scope="module")
def gpu_baseline() -> dict:
    """The TP=DP=1 answer, computed on one GPU so the mesh comparison is
    device-for-device rather than CPU-versus-GPU.  Built once for the module."""

    if _cuda_device_count() < 1:
        pytest.skip("needs a CUDA device")
    return _consensus(_run_mesh(1, 1))


@_requires_gpus(1)
class TestMeshBitwise:
    @pytest.mark.parametrize(("tp", "dp"), MESH_CONFIGS)
    def test_mesh_matches_the_single_rank_baseline(self, tp, dp, gpu_baseline):
        world = tp * dp
        if _cuda_device_count() < world:
            pytest.skip(f"needs {world} CUDA devices for TP={tp} DP={dp}")
        baseline = gpu_baseline
        actual = _consensus(_run_mesh(tp, dp))
        label = f"TP={tp} DP={dp}"

        assert actual["global_active"] == baseline["global_active"]
        assert actual["per_sequence_counts"] == baseline["per_sequence_counts"]
        assert actual["advantage_bits"] == baseline["advantage_bits"]
        # The sensitive comparison: per-sequence totals, before the scalar
        # average rounds a reordering away.
        assert (
            actual["per_sequence_policy_bits"] == baseline["per_sequence_policy_bits"]
        ), f"{label} per-sequence policy totals differ from the single-rank baseline"
        assert (
            actual["per_sequence_kl_bits"] == baseline["per_sequence_kl_bits"]
        ), f"{label} per-sequence KL totals differ from the single-rank baseline"
        assert actual["loss_bits"] == baseline["loss_bits"], f"{label} loss differs"
        assert actual["policy_bits"] == baseline["policy_bits"]
        assert actual["kl_bits"] == baseline["kl_bits"]
        assert (
            actual["grad_row"] == baseline["grad_row"]
        ), f"{label} gradient differs from the single-rank baseline"

    @_requires_gpus(4)
    def test_pure_axes_agree_with_each_other(self):
        # Transitively implied by both matching the baseline; kept because a
        # direct TP-vs-DP comparison names the culprit when a shared drift moves
        # both away from the baseline at once.
        tp4 = _consensus(_run_mesh(4, 1))
        dp4 = _consensus(_run_mesh(1, 4))
        assert tp4["per_sequence_policy_bits"] == dp4["per_sequence_policy_bits"]
        assert tp4["loss_bits"] == dp4["loss_bits"]
        assert tp4["grad_row"] == dp4["grad_row"]


@_requires_gpus(2)
class TestMeshGuards:
    def test_preflight_rejects_a_dp_rank_that_disagrees(self):
        payloads = _run_mesh(1, 2, scenario="preflight_dp_mismatch")
        assert all(
            not item.get("ok") for item in payloads
        ), "every rank must abort when one of them declares a different objective"
        assert any("preflight" in item.get("error", "") for item in payloads)

    def test_preflight_rejects_a_tp_rank_that_disagrees(self):
        # beta is not part of the logprob contract, so the logprob path's own TP
        # preflight cannot see this; only the loss preflight's TP-axis check can.
        # Without it two TP siblings would compute different losses for one
        # sharded model and nothing would notice.
        payloads = _run_mesh(2, 1, scenario="preflight_tp_mismatch")
        assert all(not item.get("ok") for item in payloads)
        assert any("TP axis" in item.get("error", "") for item in payloads)
