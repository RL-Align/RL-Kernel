# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Coverage for the optional Vime WS2 strict attention adapter.

The validation-shaped cases need a real ROCm device with AITER present and are
skipped elsewhere.  The contract/fail-closed cases are pure metadata checks and
run anywhere, so a CUDA or CPU CI job still catches a provider that silently
widens what it accepts.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.orchestrators.vime.providers.attention import (
    AttentionProviderUnavailable,
    attention_provider,
)
from rl_engine.runtime.registry import _rocm_strict_attention_available

STRICT_ROCM = _rocm_strict_attention_available()
requires_strict_rocm = pytest.mark.skipif(
    not STRICT_ROCM,
    reason="strict ROCm attention requires a ROCm device with aiter.ops.mha",
)

# Qwen3-8B dense head layout, TP=1 local view.
GLOBAL_Q_HEADS = 32
GLOBAL_KV_HEADS = 8
HEAD_DIM = 128


def _metadata(**overrides):
    metadata = {
        "global_q_heads": GLOBAL_Q_HEADS,
        "global_kv_heads": GLOBAL_KV_HEADS,
        "tp_rank": 0,
        "tp_world_size": 1,
        "attention_mode": "prefill",
        "role": "train",
        "causal": True,
    }
    metadata.update(overrides)
    return metadata


def _request(
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    query_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_layout: str = "single",
    seed: int = 0,
    **metadata_overrides,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    shape_q = (batch_size, GLOBAL_Q_HEADS, query_len or seq_len, HEAD_DIM)
    shape_kv = (batch_size, GLOBAL_KV_HEADS, seq_len, HEAD_DIM)
    return SimpleNamespace(
        query=torch.randn(*shape_q, generator=generator, device=device, dtype=dtype),
        key=torch.randn(*shape_kv, generator=generator, device=device, dtype=dtype),
        value=torch.randn(*shape_kv, generator=generator, device=device, dtype=dtype),
        key_padding_mask=None,
        tensor_parallel_group=None,
        context_parallel=SimpleNamespace(world_size=cp_world_size, rank=cp_rank, layout=cp_layout),
        metadata=_metadata(**metadata_overrides),
    )


def _cpu_request(**kwargs):
    """A request whose rejection happens before any device work."""

    kwargs.setdefault("device", "cpu")
    kwargs.setdefault("seq_len", 8)
    return _request(**kwargs)


# ---------------------------------------------------------------------------
# Contract and fail-closed behavior (device independent)
# ---------------------------------------------------------------------------


def test_cp_zigzag_layout_fails_closed():
    """Only the contiguous-per-rank layout matches the strict CP block plan."""

    request = _cpu_request(cp_world_size=2, cp_rank=1, cp_layout="zigzag")

    with pytest.raises(AttentionProviderUnavailable, match="requires the 'allgather' layout"):
        attention_provider(request)


def test_cp_without_a_process_group_fails_closed():
    """CP>1 needs a group to build the RCCL transport; it must not fall back."""

    request = _cpu_request(cp_world_size=2, cp_rank=1, cp_layout="allgather")

    with pytest.raises(AttentionProviderUnavailable, match="context_parallel_group"):
        attention_provider(request)


def test_cp_contract_describes_one_contiguous_block_per_rank():
    """The CP sharding must place this rank's block at its global offset."""

    from rl_engine.integrations.orchestrators.vime.providers.attention import _contract_for_request

    seq_len = 8
    request = _cpu_request(
        seq_len=seq_len,
        cp_world_size=4,
        cp_rank=2,
        cp_layout="allgather",
    )
    contract, *_ = _contract_for_request(request)
    sharding = contract.sharding

    assert sharding.cp_world_size == 4
    assert sharding.global_sequence_length == seq_len * 4
    assert sharding.local_sequence_length == seq_len
    assert sharding.global_block_indices == (2,)
    assert sharding.global_block_token_starts == (2 * seq_len,)
    assert sharding.local_block_offsets == (0, seq_len)


def test_decode_without_kv_cache_identity_fails_closed():
    request = _cpu_request(attention_mode="decode")

    with pytest.raises(AttentionProviderUnavailable, match="KV-cache identity"):
        attention_provider(request)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dropout_p": 0.1},
        {"sliding_window": 128},
        {"logit_soft_cap": 30.0},
        {"alibi_slopes": [0.1]},
        {"window_size": (256, 0)},
    ],
)
def test_distribution_changing_knobs_fail_closed(overrides):
    request = _cpu_request(**overrides)

    with pytest.raises(AttentionProviderUnavailable):
        attention_provider(request)


def test_key_padding_mask_is_refused():
    request = _cpu_request()
    request.key_padding_mask = torch.ones(1, 8, dtype=torch.bool)

    with pytest.raises(AttentionProviderUnavailable, match="unpadded logical row"):
        attention_provider(request)


def test_fp32_is_refused():
    request = _cpu_request(dtype=torch.float32)

    with pytest.raises(AttentionProviderUnavailable, match="BF16/FP16"):
        attention_provider(request)


def test_head_counts_must_cover_the_tp_group_exactly():
    request = _cpu_request(global_q_heads=GLOBAL_Q_HEADS * 2)

    with pytest.raises(AttentionProviderUnavailable, match="do not cover global_q_heads"):
        attention_provider(request)


def test_declared_tp_rank_must_agree_with_the_group():
    request = _cpu_request(tp_rank=3)

    with pytest.raises(AttentionProviderUnavailable, match="disagrees with TP group rank"):
        attention_provider(request)


def test_cp_layout_must_describe_local_ownership():
    request = _cpu_request(cp_layout="unknown")

    with pytest.raises(AttentionProviderUnavailable, match="local CP token ownership"):
        attention_provider(request)


def test_non_contiguous_key_positions_are_refused():
    request = _cpu_request(key_position_ids=[0, 1, 2, 3, 9, 10, 11, 12])

    with pytest.raises(AttentionProviderUnavailable, match="contiguous increasing"):
        attention_provider(request)


# ---------------------------------------------------------------------------
# Strict arithmetic (requires a ROCm device with AITER)
# ---------------------------------------------------------------------------


@requires_strict_rocm
def test_provider_exports_attention_lse_and_strict_provenance():
    result = attention_provider(_request(seq_len=256))

    assert result.backend_id == "aiter.rocm.ck_dense_mha"
    assert result.out.shape == (1, GLOBAL_Q_HEADS, 256, HEAD_DIM)
    assert result.lse.shape == (1, GLOBAL_Q_HEADS, 256)
    assert result.lse.dtype == torch.float32
    assert result.provenance["fallback"] is False
    assert result.provenance["actual_backend"] == "aiter.rocm.ck_dense_mha"
    assert result.provenance["lse_domain"] == "attention"

    core = result.provenance["core"]
    assert core["native_attention_arithmetic"] is True
    assert core["deterministic_backward"] is True
    assert core["num_splits"] == 1
    assert core["fallback"] is False
    assert core["merge_order"] == "global_block_index"
    assert core["accum_dtype"] == "fp32"
    assert core["downcast_at"] == "final_write"


@requires_strict_rocm
def test_training_and_rollout_roles_are_bitwise_identical():
    train = attention_provider(_request(seq_len=256, role="train", seed=3))
    rollout = attention_provider(_request(seq_len=256, role="infer", seed=3))

    assert torch.equal(train.out, rollout.out)
    assert torch.equal(train.lse, rollout.lse)


@requires_strict_rocm
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("seq_len", [256, 512, 2048])
def test_batch_composition_is_bitwise_invariant(batch_size, seq_len):
    """A batch must equal the same rows submitted one at a time.

    Raw AITER does not provide this for every shape: measured on MI300X it is
    batch-composition sensitive in BF16 at ``S=256`` (B=4) and ``S=512`` (B=2
    and B=4), while holding at 128/1024/2048/4096.  Shape-dependent breakage is
    exactly what a per-row rule has to defend against, because the shapes that
    hold would otherwise make the bug look absent.  ``S=512`` is kept in this
    parametrization deliberately.
    """

    batched_request = _request(batch_size=batch_size, seq_len=seq_len, seed=11)
    batched = attention_provider(batched_request)

    for row in range(batch_size):
        single = SimpleNamespace(
            query=batched_request.query[row : row + 1],
            key=batched_request.key[row : row + 1],
            value=batched_request.value[row : row + 1],
            key_padding_mask=None,
            tensor_parallel_group=None,
            context_parallel=batched_request.context_parallel,
            metadata=batched_request.metadata,
        )
        row_result = attention_provider(single)
        assert torch.equal(batched.out[row : row + 1], row_result.out)
        assert torch.equal(batched.lse[row : row + 1], row_result.lse)


@requires_strict_rocm
def test_repeated_invocations_are_bitwise_identical():
    first = attention_provider(_request(seq_len=512, seed=5))
    second = attention_provider(_request(seq_len=512, seed=5))

    assert torch.equal(first.out, second.out)
    assert torch.equal(first.lse, second.lse)


@requires_strict_rocm
def test_backward_gradients_are_deterministic():
    def run():
        request = _request(seq_len=256, seed=17)
        request.query.requires_grad_(True)
        request.key.requires_grad_(True)
        request.value.requires_grad_(True)
        result = attention_provider(request)
        result.out.backward(torch.ones_like(result.out))
        return request.query.grad, request.key.grad, request.value.grad

    first = run()
    second = run()
    for lhs, rhs in zip(first, second, strict=True):
        assert torch.equal(lhs, rhs)


@requires_strict_rocm
def test_contract_fingerprint_is_rank_independent():
    left = attention_provider(_request(seq_len=128, seed=2))
    right = attention_provider(_request(seq_len=128, seed=2))

    assert left.contract_id == right.contract_id
    assert len(left.contract_id) == 64


@requires_strict_rocm
def test_explicit_scale_is_honored():
    scale = 1.0 / math.sqrt(HEAD_DIM)
    default = attention_provider(_request(seq_len=128, seed=8))
    explicit = attention_provider(_request(seq_len=128, seed=8, softmax_scale=scale))

    assert torch.equal(default.out, explicit.out)


@requires_strict_rocm
def test_provenance_records_the_launch_pinning():
    """Launch granularity is the mechanism behind the bitwise claim.

    A reader of a strict report has to be able to tell that the result came
    from pinned one-row/one-KV-group launches rather than a batched call that
    happened to agree.
    """

    result = attention_provider(_request(seq_len=256))
    execution = result.provenance["execution"]
    binding = result.provenance["cross_config_binding"]

    assert execution["launch_granularity"] == "one_batch_row_one_kv_group"
    assert execution["kv_groups_materialized_independently"] is True
    assert execution["batch_rows_materialized_independently"] is True
    # B=1 request over a 32Q/8KV layout -> one launch per KV group.
    assert execution["core_launches"] == GLOBAL_KV_HEADS
    assert binding["tp_degree_invariant"] is True
    assert binding["invariance_mechanism"] == "one_kv_group_per_launch"


@requires_strict_rocm
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("seq_len", [512, 2048])
def test_tp_degree_is_bitwise_invariant(tp, seq_len):
    """A TP head shard must equal the same slice of an unsharded run.

    TP performs no cross-rank reduction in attention, so this has to hold for
    train and rollout to compare across TP degrees. Raw AITER does not provide
    it (up to 7.8125e-03 drift); the per-KV-group launch rule is what does.
    """

    class _Group:
        def __init__(self, rank, size):
            self._rank, self._size = rank, size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    base = _request(seq_len=seq_len, seed=3)
    full = attention_provider(base)

    local_q, local_kv = GLOBAL_Q_HEADS // tp, GLOBAL_KV_HEADS // tp
    for rank in range(tp):
        shard = SimpleNamespace(
            query=base.query[:, rank * local_q : (rank + 1) * local_q],
            key=base.key[:, rank * local_kv : (rank + 1) * local_kv],
            value=base.value[:, rank * local_kv : (rank + 1) * local_kv],
            key_padding_mask=None,
            tensor_parallel_group=_Group(rank, tp),
            context_parallel=base.context_parallel,
            metadata=_metadata(tp_rank=rank, tp_world_size=tp),
        )
        result = attention_provider(shard)
        assert torch.equal(result.out, full.out[:, rank * local_q : (rank + 1) * local_q])
        assert torch.equal(result.lse, full.lse[:, rank * local_q : (rank + 1) * local_q])
