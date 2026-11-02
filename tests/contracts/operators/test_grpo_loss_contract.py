# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unit tests for the WS2 deterministic GRPO loss contract.

These are pure-Python contract checks: no tensors, no collectives, no GPU.
The distributed behaviour they describe is exercised in
``tests/distributed/tp/test_distributed_grpo_loss.py``.
"""

from __future__ import annotations

import json

import pytest

from rl_engine.contracts.operators.logprob import (
    DeterminismScope,
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
    LossBackendCapability,
    LossContractError,
    LossReductionSpec,
    LossShardingSpec,
    ObjectiveSpec,
    TokenNormalizer,
)

# Global batch geometry shared by every fixture below: 8 sequences of 8 token
# slots, in two equal advantage groups.  Tests that care about variable or
# straddling groups override group_boundaries explicitly.
NUM_SEQUENCES = 8
PADDED_SEQ_LEN = 8
GROUP_BOUNDARIES = (0, 4, NUM_SEQUENCES)
REAL_VOCAB = 30
PADDED_VOCAB = 32


def _dp_bounds(dp: int) -> tuple[tuple[int, int], ...]:
    """Contiguous sequence partition in DP-rank order."""

    seqs = NUM_SEQUENCES // dp
    return tuple((d * seqs, (d + 1) * seqs) for d in range(dp))


def _logprob_contract(num_tokens: int, **overrides) -> LogprobContract:
    kwargs = {
        "role": LogprobRole.TRAIN,
        "dtype": LogprobDType.FP32,
        "mask": MaskSpec(num_tokens=num_tokens, active_mask=(True,) * num_tokens),
        "sharding": ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            vocab_shard_bounds=((0, PADDED_VOCAB),),
            real_vocab_size=REAL_VOCAB,
            padded_vocab_size=PADDED_VOCAB,
        ),
        "reduction": ReductionSpec(),
    }
    kwargs.update(overrides)
    return LogprobContract(**kwargs)


def _sharding(*, dp_rank: int = 0, dp: int = 1, **overrides) -> LossShardingSpec:
    kwargs = {
        "dp_rank": dp_rank,
        "dp_world_size": dp,
        "num_sequences": NUM_SEQUENCES,
        "padded_seq_len": PADDED_SEQ_LEN,
        "sequence_shard_bounds": _dp_bounds(dp),
        "group_boundaries": GROUP_BOUNDARIES,
    }
    kwargs.update(overrides)
    return LossShardingSpec(**kwargs)


def _contract(*, dp_rank: int = 0, dp: int = 1, **overrides) -> GRPOLossContract:
    sharding = overrides.pop("sharding", _sharding(dp_rank=dp_rank, dp=dp))
    kwargs = {
        "logprob": _logprob_contract(sharding.local_num_token_slots),
        "sharding": sharding,
        "objective": ObjectiveSpec(),
        "reduction": LossReductionSpec(),
    }
    kwargs.update(overrides)
    return GRPOLossContract(**kwargs)


class TestSequenceOwnership:
    def test_single_rank_owns_every_sequence(self):
        sharding = _sharding()
        assert sharding.local_sequence_start == 0
        assert sharding.local_sequence_end == NUM_SEQUENCES
        assert sharding.local_num_token_slots == NUM_SEQUENCES * PADDED_SEQ_LEN
        assert sharding.num_groups == 2
        assert sharding.group_sizes == (4, 4)

    @pytest.mark.parametrize("dp", [1, 2, 4, 8])
    def test_dp_shapes_partition_the_batch(self, dp):
        total = 0
        for dp_rank in range(dp):
            sharding = _sharding(dp_rank=dp_rank, dp=dp)
            assert sharding.local_num_sequences == NUM_SEQUENCES // dp
            total += sharding.local_num_token_slots
        assert total == NUM_SEQUENCES * PADDED_SEQ_LEN

    def test_bounds_must_be_contiguous_in_rank_order(self):
        with pytest.raises(LossContractError, match="contiguous"):
            _sharding(dp=2, sequence_shard_bounds=((0, 4), (5, 8)))

    def test_bounds_must_cover_every_sequence(self):
        with pytest.raises(LossContractError, match="cover num_sequences"):
            _sharding(dp=2, sequence_shard_bounds=((0, 3), (3, 6)))

    def test_bounds_count_must_match_dp_world_size(self):
        with pytest.raises(
            LossContractError, match="exactly one \\(start, end\\) pair per DP rank"
        ):
            _sharding(dp=2, sequence_shard_bounds=((0, NUM_SEQUENCES),))

    def test_empty_shard_rejected(self):
        with pytest.raises(LossContractError, match="end > start"):
            _sharding(dp=2, sequence_shard_bounds=((0, 0), (0, NUM_SEQUENCES)))

    def test_context_parallelism_is_rejected(self):
        # CP splits a sequence's tokens across ranks, which this contract does
        # not model; it must fail loudly rather than silently drop the rest.
        with pytest.raises(LossContractError, match="cp_world_size=2 is unsupported"):
            _sharding(cp_world_size=2)

    def test_cp_rank_must_be_zero(self):
        with pytest.raises(LossContractError, match="cp_rank must be 0"):
            _sharding(cp_rank=1)


class TestGroupBoundaries:
    @pytest.mark.parametrize(
        ("boundaries", "match"),
        [
            ((1, NUM_SEQUENCES), "must start at 0"),
            ((0, 3), "must start at 0"),
            ((0, 2, 2, NUM_SEQUENCES), "strictly increasing"),
            ((0,), "at least 2 entries"),
        ],
    )
    def test_malformed_boundaries_rejected(self, boundaries, match):
        with pytest.raises(LossContractError, match=match):
            _sharding(group_boundaries=boundaries)

    def test_variable_group_sizes_accepted(self):
        sharding = _sharding(group_boundaries=(0, 7, NUM_SEQUENCES))
        assert sharding.group_sizes == (7, 1)

    def test_groups_may_straddle_dp_shards(self):
        # A group split at 3 crosses the DP=2 shard boundary at 4, so no rank
        # owns that group alone.  The contract permits it; that is why the
        # operator replicates advantages instead of merging partial statistics.
        sharding = _sharding(dp=2, dp_rank=0, group_boundaries=(0, 3, NUM_SEQUENCES))
        assert sharding.sequence_shard_bounds == ((0, 4), (4, NUM_SEQUENCES))
        assert sharding.group_sizes == (3, 5)

    def test_population_std_rejects_singleton_groups(self):
        # A one-sequence group has zero population variance, so its advantage
        # would silently collapse to zero rather than fail.
        with pytest.raises(LossContractError, match="at least 2 sequences per group"):
            _contract(
                sharding=_sharding(group_boundaries=(0, 7, NUM_SEQUENCES)),
                objective=ObjectiveSpec(),
            )

    def test_mean_only_allows_singleton_groups(self):
        contract = _contract(
            sharding=_sharding(group_boundaries=(0, 7, NUM_SEQUENCES)),
            objective=ObjectiveSpec(
                advantage=AdvantageSpec(normalizer=AdvantageNormalizer.MEAN_ONLY)
            ),
        )
        assert contract.sharding.group_sizes == (7, 1)


class TestSpecValidation:
    def test_accumulation_must_be_fp32(self):
        with pytest.raises(LossContractError, match="must be fp32"):
            LossReductionSpec(acc_dtype=LogprobDType.BF16)

    def test_fixed_constant_normalizer_requires_its_constant(self):
        with pytest.raises(LossContractError, match="fixed_normalizer_constant"):
            LossReductionSpec(token_normalizer=TokenNormalizer.FIXED_CONSTANT)

    def test_constant_rejected_for_other_normalizers(self):
        with pytest.raises(LossContractError, match="only meaningful for"):
            LossReductionSpec(fixed_normalizer_constant=32)

    def test_fixed_constant_normalizer_accepts_its_constant(self):
        spec = LossReductionSpec(
            token_normalizer=TokenNormalizer.FIXED_CONSTANT, fixed_normalizer_constant=32
        )
        assert spec.fixed_normalizer_constant == 32

    def test_lower_clip_bound_must_stay_positive(self):
        with pytest.raises(LossContractError, match="must be smaller than 1.0"):
            ClipSpec(clip_eps_low=1.0)

    def test_asymmetric_clip_bounds(self):
        clip = ClipSpec(clip_eps_low=0.2, clip_eps_high=0.28)
        assert clip.lower_bound == pytest.approx(0.8)
        assert clip.upper_bound == pytest.approx(1.28)

    def test_std_eps_must_be_positive(self):
        with pytest.raises(LossContractError, match="strictly positive"):
            AdvantageSpec(std_eps=0.0)

    def test_negative_beta_rejected(self):
        with pytest.raises(LossContractError, match="non-negative"):
            ObjectiveSpec(beta=-0.01)

    def test_uses_reference_model_tracks_beta(self):
        assert not ObjectiveSpec(beta=0.0).uses_reference_model
        assert ObjectiveSpec(beta=0.04).uses_reference_model


class TestContractCoherence:
    def test_logprob_token_count_must_match_owned_slots(self):
        sharding = _sharding(dp=2, dp_rank=0)
        with pytest.raises(LossContractError, match="token slots this rank owns"):
            GRPOLossContract(
                logprob=_logprob_contract(sharding.local_num_token_slots + 1),
                sharding=sharding,
            )

    def test_determinism_scope_must_agree_with_logprob_path(self):
        tokens = _sharding().local_num_token_slots
        with pytest.raises(LossContractError, match="stronger or weaker determinism scope"):
            GRPOLossContract(
                logprob=_logprob_contract(
                    tokens,
                    reduction=ReductionSpec(determinism_scope=DeterminismScope.FIXED_TOPOLOGY),
                ),
                sharding=_sharding(),
            )

    def test_global_token_slots(self):
        assert _contract().global_token_slots == NUM_SEQUENCES * PADDED_SEQ_LEN


class TestFingerprint:
    @pytest.mark.parametrize("dp", [2, 4, 8])
    def test_every_dp_rank_agrees(self, dp):
        # This is the property the distributed preflight relies on.  Each rank
        # holds a different slice of sequences, so their nested logprob masks
        # genuinely differ -- the fingerprint must still match.
        fingerprints = {
            _contract(dp_rank=rank, dp=dp).cross_rank_fingerprint() for rank in range(dp)
        }
        assert len(fingerprints) == 1

    def test_dp_degree_changes_the_fingerprint(self):
        # A different partition is a different logical invocation, so a rank
        # that joined the wrong one must be caught rather than merged with.
        assert _contract(dp=1).cross_rank_fingerprint() != _contract(dp=2).cross_rank_fingerprint()

    @pytest.mark.parametrize(
        "overrides",
        [
            {
                "reduction": LossReductionSpec(
                    token_normalizer=TokenNormalizer.PER_SEQUENCE_THEN_MEAN
                )
            },
            {"objective": ObjectiveSpec(beta=0.04)},
            {"objective": ObjectiveSpec(clip=ClipSpec(clip_eps_high=0.28))},
            {"objective": ObjectiveSpec(kl_estimator=KLEstimator.K1_LOG_RATIO)},
            {
                "objective": ObjectiveSpec(
                    advantage=AdvantageSpec(normalizer=AdvantageNormalizer.MEAN_ONLY)
                )
            },
            {"sharding": _sharding(group_boundaries=(0, 3, NUM_SEQUENCES))},
        ],
        ids=["normalizer", "beta", "clip", "kl", "advantage", "groups"],
    )
    def test_numerical_identity_changes_the_fingerprint(self, overrides):
        assert (
            _contract().cross_rank_fingerprint() != _contract(**overrides).cross_rank_fingerprint()
        )

    def test_to_dict_is_json_serializable_and_stable(self):
        contract = _contract()
        first = json.dumps(contract.to_dict(), sort_keys=True)
        second = json.dumps(contract.to_dict(), sort_keys=True)
        assert first == second
        payload = contract.to_dict()
        assert payload["semantic_operator"] == "grpo_loss"
        assert payload["reduction"]["cp_is_merge_axis"] is False
        assert payload["reduction"]["dp_is_merge_axis"] is True
        assert payload["logprob"]["reduction"]["cp_is_merge_axis"] is False


def _capability(**overrides) -> LossBackendCapability:
    kwargs = {
        "backend_id": "test-loss-backend",
        "token_normalizers": frozenset({TokenNormalizer.GLOBAL_ACTIVE_TOKENS}),
        "kl_estimators": frozenset({KLEstimator.K3_UNBIASED}),
        "advantage_normalizers": frozenset({AdvantageNormalizer.MEAN_STD_POPULATION}),
        "determinism_scopes": frozenset({DeterminismScope.CROSS_TP_BITWISE}),
        "implementation_kind": "reference",
    }
    kwargs.update(overrides)
    return LossBackendCapability(**kwargs)


class TestBackendCapability:
    def test_matching_capability_supports_contract(self):
        assert _capability().supports(_contract())

    def test_backend_id_may_not_shadow_a_dispatch_policy(self):
        with pytest.raises(LossContractError, match="reserved dispatch policy"):
            _capability(backend_id="reference")

    def test_unsupported_normalizer_is_reported(self):
        contract = _contract(
            reduction=LossReductionSpec(token_normalizer=TokenNormalizer.PER_SEQUENCE_THEN_MEAN)
        )
        reasons = _capability().incompatibilities(contract)
        assert any("token_normalizer" in reason for reason in reasons)

    def test_unsupported_dp_degree_is_reported(self):
        capability = _capability(dp_world_sizes=(1,))
        reasons = capability.incompatibilities(_contract(dp=2))
        assert any("DP=2" in reason for reason in reasons)

    def test_variable_group_sizes_gated(self):
        contract = _contract(
            sharding=_sharding(group_boundaries=(0, 7, NUM_SEQUENCES)),
            objective=ObjectiveSpec(
                advantage=AdvantageSpec(normalizer=AdvantageNormalizer.MEAN_ONLY)
            ),
        )
        capability = _capability(advantage_normalizers=frozenset({AdvantageNormalizer.MEAN_ONLY}))
        assert any(
            "variable advantage group sizes" in reason
            for reason in capability.incompatibilities(contract)
        )

    def test_variable_group_sizes_allowed_when_declared(self):
        contract = _contract(
            sharding=_sharding(group_boundaries=(0, 7, NUM_SEQUENCES)),
            objective=ObjectiveSpec(
                advantage=AdvantageSpec(normalizer=AdvantageNormalizer.MEAN_ONLY)
            ),
        )
        capability = _capability(
            advantage_normalizers=frozenset({AdvantageNormalizer.MEAN_ONLY}),
            supports_variable_group_sizes=True,
        )
        assert capability.supports(contract)

    def test_asymmetric_clip_gated(self):
        contract = _contract(objective=ObjectiveSpec(clip=ClipSpec(clip_eps_high=0.28)))
        assert any(
            "asymmetric ratio clipping" in reason
            for reason in _capability().incompatibilities(contract)
        )
        assert _capability(supports_asymmetric_clip=True).supports(contract)

    def test_every_incompatibility_is_reported_at_once(self):
        contract = _contract(
            dp=2,
            reduction=LossReductionSpec(token_normalizer=TokenNormalizer.PER_SEQUENCE_THEN_MEAN),
            objective=ObjectiveSpec(
                kl_estimator=KLEstimator.K1_LOG_RATIO,
                clip=ClipSpec(clip_eps_high=0.28),
            ),
        )
        reasons = _capability(dp_world_sizes=(1,)).incompatibilities(contract)
        assert len(reasons) >= 4

    def test_to_dict_round_trips_declared_flags(self):
        payload = _capability(dp_world_sizes=(1, 2)).to_dict()
        assert payload["dp_world_sizes"] == [1, 2]
        assert payload["implementation_kind"] == "reference"
