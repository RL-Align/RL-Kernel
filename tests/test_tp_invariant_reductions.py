# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import datetime
import math
import pathlib
import tempfile
from queue import Empty

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.kernels.ops.pytorch.loss.grpo_loss import NativeGRPOLossOp
from rl_engine.testing import (
    compute_policy_ratio,
    compute_reference_kl,
    distributed_active_token_count,
    distributed_masked_mean,
    distributed_masked_sum,
    make_synthetic_rl_kernel_batch,
    masked_mean,
    owner_ranks_for_token_ids,
    selected_logprobs_distributed_tp_reference,
    selected_logprobs_reference,
    selected_logprobs_tp_reference,
    shard_logits_by_vocab,
    sharded_active_token_count,
    sharded_masked_mean,
    sharded_masked_sum,
    summarize_tp_logprob_drift,
    vocab_shard_ranges,
)

requires_gloo = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="torch.distributed Gloo backend is unavailable.",
)


def _generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    gen = torch.Generator(device=torch.device(device))
    gen.manual_seed(seed)
    return gen


def _make_logits(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    scale: float = 3.0,
) -> torch.Tensor:
    logits = torch.randn(shape, generator=_generator(seed, device), device=device) * scale
    # Bias the last dimension slightly so max-reduction and owner-rank logic both do real work.
    vocab = shape[-1]
    ramp = torch.linspace(-2.0, 2.0, vocab, device=device).reshape(
        *((1,) * (len(shape) - 1)), vocab
    )
    return (logits + ramp).to(dtype=dtype)


def _force_tokens_on_every_shard(
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    shard_ranges: list[tuple[int, int]],
) -> None:
    flat_tokens = token_ids.reshape(-1)
    flat_mask = mask.reshape(-1)
    for rank, (start, end) in enumerate(shard_ranges):
        flat_tokens[2 * rank] = start
        flat_mask[2 * rank] = True
        flat_tokens[2 * rank + 1] = end - 1
        flat_mask[2 * rank + 1] = True


def _split_rows(values: torch.Tensor, parts: int) -> list[torch.Tensor]:
    # Uneven row splits simulate micro-batches with different valid-token counts.
    return list(torch.tensor_split(values, parts, dim=0))


def _distributed_tp_reference_worker(
    rank,
    world_size,
    init_method,
    full_logits,
    token_ids,
    completion_mask,
    shard_ranges,
    value_shards,
    mask_shards,
    queue,
):
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        start, end = shard_ranges[rank]
        local_logits = full_logits[..., start:end].contiguous()
        distributed_logps = selected_logprobs_distributed_tp_reference(
            local_logits,
            token_ids,
            completion_mask,
            vocab_start_index=start,
        )
        distributed_sum = distributed_masked_sum(value_shards[rank], mask_shards[rank])
        distributed_count = distributed_active_token_count(mask_shards[rank])
        distributed_mean = distributed_masked_mean(value_shards[rank], mask_shards[rank])
        if rank == 0:
            queue.put(
                {
                    "logps": distributed_logps.cpu().tolist(),
                    "sum": float(distributed_sum.cpu().item()),
                    "count": float(distributed_count.cpu().item()),
                    "mean": float(distributed_mean.cpu().item()),
                }
            )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
        raise
    finally:
        dist.destroy_process_group()


def test_vocab_sharded_logprob_reduction_matches_full_vocab_reference():
    logits = torch.tensor([[0.1, -0.2, 1.7, 0.3, 1.2, -0.5]])
    token_ids = torch.tensor([4])

    full = selected_logprobs_reference(logits, token_ids)
    tp = selected_logprobs_tp_reference(shard_logits_by_vocab(logits, tp_size=2), token_ids)

    assert full.item() == pytest.approx(-1.3395806963, abs=1e-7)
    assert torch.allclose(tp, full, atol=5e-7, rtol=0.0)

    rank1 = logits[:, 3:]
    owner_rank_local = 1.2 - torch.logsumexp(rank1, dim=-1)
    local_lses = torch.stack(
        [
            torch.logsumexp(logits[:, :3], dim=-1),
            torch.logsumexp(logits[:, 3:], dim=-1),
        ]
    )
    averaged_local_lse = 1.2 - local_lses.mean(dim=0)

    assert owner_rank_local.item() == pytest.approx(-0.4632642102, abs=2e-7)
    assert averaged_local_lse.item() == pytest.approx(-0.6322267505, abs=1e-7)
    assert not torch.allclose(owner_rank_local, full)
    assert not torch.allclose(averaged_local_lse, full)


def test_vocab_shard_ranges_cover_uneven_vocab_without_overlap():
    assert vocab_shard_ranges(10, 4) == [(0, 3), (3, 6), (6, 8), (8, 10)]

    ranges = vocab_shard_ranges(257, 8)
    assert ranges[0] == (0, 33)
    assert ranges[-1] == (225, 257)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 257
    assert all(prev[1] == cur[0] for prev, cur in zip(ranges, ranges[1:], strict=False))

    with pytest.raises(ValueError, match="tp_size"):
        vocab_shard_ranges(4, 5)


def test_owner_ranks_for_token_ids_marks_masked_and_uncovered_tokens():
    ranges = vocab_shard_ranges(10, 4)
    token_ids = torch.tensor([[0, 2, 3, 5, 6, 8, 9, 11]])
    mask = torch.tensor([[True, True, True, False, True, True, True, True]])

    owners = owner_ranks_for_token_ids(token_ids, ranges, mask)

    assert torch.equal(owners, torch.tensor([[0, 0, 1, -1, 2, 3, 3, -1]]))


@pytest.mark.parametrize("tp_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tp_selected_logprobs_match_full_reference_for_uneven_vocab(tp_size, dtype):
    vocab_size = 257
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=3,
        samples_per_prompt=5,
        prompt_len=17,
        completion_len=23,
        vocab_size=vocab_size,
        valid_density=0.73,
        dtype=dtype,
        seed=102,
    )
    logits = _make_logits(
        (batch.batch_size, batch.completion_len, vocab_size),
        seed=202,
        dtype=dtype,
    )
    ranges = vocab_shard_ranges(vocab_size, tp_size)
    _force_tokens_on_every_shard(batch.token_ids, batch.completion_mask, ranges)

    full = selected_logprobs_reference(
        logits,
        batch.token_ids,
        batch.completion_mask,
        output_dtype=torch.float32,
    )
    tp = selected_logprobs_tp_reference(
        shard_logits_by_vocab(logits, tp_size),
        batch.token_ids,
        batch.completion_mask,
        output_dtype=torch.float32,
    )
    summary = summarize_tp_logprob_drift(tp, full, batch.token_ids, ranges, batch.completion_mask)

    assert summary["active_count"] == int(batch.completion_mask.sum().item())
    assert summary["max_abs_error"] <= 2e-6
    assert torch.allclose(tp, full, atol=2e-6, rtol=0.0)
    assert torch.equal(tp[~batch.completion_mask], torch.zeros_like(tp[~batch.completion_mask]))


def test_tp_selected_logprobs_support_explicit_vocab_offsets_and_nonuniform_shards():
    logits = _make_logits((2, 5, 17), seed=303)
    token_ids = torch.tensor([[0, 2, 6, 11, 16], [1, 5, 7, 12, 15]])
    mask = torch.tensor([[True, True, False, True, True], [True, False, True, True, True]])
    shards = [logits[..., :2], logits[..., 2:7], logits[..., 7:13], logits[..., 13:]]
    starts = [0, 2, 7, 13]

    full = selected_logprobs_reference(logits, token_ids, mask)
    tp = selected_logprobs_tp_reference(shards, token_ids, mask, vocab_start_indices=starts)

    assert torch.allclose(tp, full, atol=5e-6, rtol=0.0)


def test_tp_selected_logprobs_allows_masked_ignore_index_but_rejects_active_missing_token():
    logits = torch.randn(1, 4, 9)
    token_ids = torch.tensor([[0, -100, 8, 99]])
    mask = torch.tensor([[True, False, True, False]])

    tp = selected_logprobs_tp_reference(shard_logits_by_vocab(logits, 3), token_ids, mask)
    full = selected_logprobs_reference(logits, token_ids.masked_fill(~mask, 0), mask)
    assert torch.allclose(tp, full, atol=5e-6, rtol=0.0)
    assert tp[0, 1] == 0.0
    assert tp[0, 3] == 0.0

    bad_mask = torch.tensor([[True, False, True, True]])
    with pytest.raises(ValueError, match="not covered"):
        selected_logprobs_tp_reference(shard_logits_by_vocab(logits, 3), token_ids, bad_mask)


def test_tp_selected_logprobs_are_temperature_invariant_against_full_reference():
    logits = _make_logits((4, 6, 41), seed=404, scale=8.0)
    token_ids = torch.randint(0, 41, (4, 6), generator=_generator(405))
    mask = torch.tensor(
        [
            [True, False, True, True, False, True],
            [True, True, True, False, False, True],
            [False, True, True, True, True, False],
            [True, True, False, True, True, True],
        ]
    )

    full = selected_logprobs_reference(logits, token_ids, mask, temperature=0.7)
    tp = selected_logprobs_tp_reference(
        shard_logits_by_vocab(logits, 4),
        token_ids,
        mask,
        temperature=0.7,
    )

    assert torch.allclose(tp, full, atol=5e-6, rtol=0.0)


def test_sharded_masked_reductions_use_global_denominator_not_average_of_local_means():
    values = torch.tensor(
        [
            [1.0, 1000.0, 3.0, 4.0],
            [5.0, 6.0, 700.0, 8.0],
            [9.0, 10.0, 11.0, 1200.0],
            [13.0, 14.0, 15.0, 16.0],
            [1700.0, 18.0, 19.0, 20.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, False],
            [False, True, True, False],
            [True, True, True, True],
            [False, True, False, True],
        ]
    )
    value_shards = _split_rows(values, 3)
    mask_shards = _split_rows(mask, 3)

    assert torch.equal(sharded_active_token_count(mask_shards), torch.tensor(13.0))
    expected_sum = masked_mean(values, mask) * 13
    assert torch.allclose(sharded_masked_sum(value_shards, mask_shards), expected_sum)
    assert torch.allclose(sharded_masked_mean(value_shards, mask_shards), masked_mean(values, mask))

    local_mean_average = torch.stack(
        [masked_mean(v, m) for v, m in zip(value_shards, mask_shards, strict=True)]
    ).mean()
    assert not torch.allclose(local_mean_average, masked_mean(values, mask))


def test_tp_logprob_drift_summary_reports_owner_rank_and_token_location():
    logits = _make_logits((2, 4, 19), seed=505)
    token_ids = torch.tensor([[0, 5, 9, 14], [18, 1, 7, 12]])
    mask = torch.tensor([[True, True, True, True], [True, False, True, True]])
    ranges = vocab_shard_ranges(19, 4)
    reference = selected_logprobs_reference(logits, token_ids, mask)
    candidate = reference.clone()
    candidate[0, 3] += 0.25

    summary = summarize_tp_logprob_drift(
        candidate,
        reference,
        token_ids,
        ranges,
        mask,
        backend="simulated-tp",
        reduction_name="unit-test-reduction",
        dtype=torch.float32,
    )

    assert summary["max_abs_error"] == pytest.approx(0.25, abs=1e-7)
    assert summary["max_rel_error"] > 0.0
    assert summary["multi_index"] == (0, 3)
    assert summary["token_id"] == 14
    assert summary["owner_rank"] == 2
    assert summary["owner_vocab_start"] == 10
    assert summary["owner_vocab_end"] == 15
    assert summary["tp_size"] == 4
    assert summary["backend"] == "simulated-tp"
    assert summary["reduction_name"] == "unit-test-reduction"
    assert summary["dtype"] == "torch.float32"


def test_tp_reference_gradient_matches_full_vocab_reference():
    vocab_size = 67
    logits = _make_logits((3, 7, vocab_size), seed=606).requires_grad_(True)
    token_ids = torch.randint(0, vocab_size, (3, 7), generator=_generator(607))
    mask = torch.tensor(
        [
            [True, False, True, True, True, False, True],
            [False, True, True, False, True, True, True],
            [True, True, False, True, False, True, True],
        ]
    )

    full_logps = selected_logprobs_reference(logits, token_ids, mask)
    full_loss = masked_mean(full_logps, mask)
    full_loss.backward()
    full_grad = logits.grad.detach().clone()

    ranges = vocab_shard_ranges(vocab_size, 4)
    shard_vars = [
        logits.detach()[..., start:end].clone().requires_grad_(True) for start, end in ranges
    ]
    tp_logps = selected_logprobs_tp_reference(shard_vars, token_ids, mask)
    tp_loss = sharded_masked_mean(_split_rows(tp_logps, 2), _split_rows(mask, 2))
    tp_loss.backward()
    tp_grad = torch.cat([shard.grad for shard in shard_vars], dim=-1)

    assert torch.allclose(tp_logps, full_logps, atol=2e-6, rtol=0.0)
    assert torch.allclose(tp_loss, full_loss, atol=2e-6, rtol=0.0)
    assert torch.allclose(tp_grad, full_grad, atol=2e-6, rtol=0.0)


def test_grpo_loss_pipeline_is_tp_invariant_under_microbatch_partitioning():
    vocab_size = 1027
    samples_per_prompt = 4
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=4,
        samples_per_prompt=samples_per_prompt,
        prompt_len=32,
        completion_len=31,
        vocab_size=vocab_size,
        valid_density=0.68,
        dtype=torch.float32,
        seed=707,
    )
    logits = _make_logits(
        (batch.batch_size, batch.completion_len, vocab_size),
        seed=708,
        scale=4.0,
    )
    ranges = vocab_shard_ranges(vocab_size, 8)
    _force_tokens_on_every_shard(batch.token_ids, batch.completion_mask, ranges)

    full_current = selected_logprobs_reference(logits, batch.token_ids, batch.completion_mask)
    tp_current = selected_logprobs_tp_reference(
        shard_logits_by_vocab(logits, 8),
        batch.token_ids,
        batch.completion_mask,
    )
    assert torch.allclose(tp_current, full_current, atol=2e-6, rtol=0.0)

    old_logps = full_current.detach() - 0.03
    ref_logps = full_current.detach() - 0.07
    op = NativeGRPOLossOp()
    full_loss, full_policy, full_kl = op.forward(
        full_current,
        old_logps,
        ref_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.05,
        samples_per_prompt=samples_per_prompt,
    )
    tp_loss, tp_policy, tp_kl = op.forward(
        tp_current,
        old_logps,
        ref_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.05,
        samples_per_prompt=samples_per_prompt,
    )

    assert torch.allclose(tp_loss, full_loss, atol=2e-6, rtol=0.0)
    assert torch.allclose(tp_policy, full_policy, atol=2e-6, rtol=0.0)
    assert torch.allclose(tp_kl, full_kl, atol=2e-6, rtol=0.0)

    sample_adv = op.group_advantages(batch.rewards, samples_per_prompt=samples_per_prompt)
    advantages = op.expand_advantages(sample_adv, batch.completion_mask)
    ratio = compute_policy_ratio(tp_current, old_logps, batch.completion_mask)
    unclipped = ratio * advantages.float()
    clipped = torch.clamp(ratio, 0.8, 1.2) * advantages.float()
    policy_terms = -torch.minimum(unclipped, clipped)
    kl_terms = compute_reference_kl(tp_current, ref_logps, batch.completion_mask)
    loss_terms = policy_terms + 0.05 * kl_terms

    value_shards = _split_rows(loss_terms, 5)
    mask_shards = _split_rows(batch.completion_mask, 5)
    sharded_loss = sharded_masked_mean(value_shards, mask_shards)

    assert torch.allclose(sharded_loss, full_loss, atol=2e-6, rtol=0.0)


def test_vocab_parallel_lm_head_shards_match_full_lm_head_logprobs():
    batch_size = 5
    completion_len = 9
    hidden_size = 64
    vocab_size = 521
    tp_size = 4
    gen = _generator(710)
    hidden = torch.randn(batch_size, completion_len, hidden_size, generator=gen)
    lm_head = torch.randn(vocab_size, hidden_size, generator=gen) / math.sqrt(hidden_size)
    bias = torch.randn(vocab_size, generator=gen) * 0.01
    token_ids = torch.randint(0, vocab_size, (batch_size, completion_len), generator=gen)
    mask = torch.rand(batch_size, completion_len, generator=gen) > 0.25
    ranges = vocab_shard_ranges(vocab_size, tp_size)
    _force_tokens_on_every_shard(token_ids, mask, ranges)

    full_logits = hidden @ lm_head.t() + bias
    shard_logits = [hidden @ lm_head[start:end].t() + bias[start:end] for start, end in ranges]

    full = selected_logprobs_reference(full_logits, token_ids, mask)
    tp = selected_logprobs_tp_reference(
        shard_logits,
        token_ids,
        mask,
        vocab_start_indices=[start for start, _end in ranges],
    )
    summary = summarize_tp_logprob_drift(
        tp,
        full,
        token_ids,
        ranges,
        mask,
        backend="vocab-parallel-lm-head",
        dtype=full_logits.dtype,
    )

    assert summary["max_abs_error"] <= 2e-6
    assert summary["backend"] == "vocab-parallel-lm-head"


@requires_gloo
def test_distributed_gloo_tp_reference_uses_real_all_reduce_collectives():
    world_size = 4
    vocab_size = 97
    full_logits = _make_logits((3, 5, vocab_size), seed=760, scale=2.7)
    token_ids = torch.randint(0, vocab_size, (3, 5), generator=_generator(761))
    completion_mask = torch.rand(3, 5, generator=_generator(762)) > 0.2
    shard_ranges = vocab_shard_ranges(vocab_size, world_size)
    _force_tokens_on_every_shard(token_ids, completion_mask, shard_ranges)

    values = torch.randn(7, 6, generator=_generator(763))
    value_mask = torch.rand(7, 6, generator=_generator(764)) > 0.35
    value_shards = _split_rows(values, world_size)
    mask_shards = _split_rows(value_mask, world_size)

    expected_logps = selected_logprobs_reference(full_logits, token_ids, completion_mask)
    expected_sum = float(values.masked_fill(~value_mask, 0.0).float().sum().item())
    expected_count = float(value_mask.sum().item())
    expected_mean = float(masked_mean(values, value_mask).item())

    context = mp.get_context("spawn")
    queue = context.Queue()
    with tempfile.TemporaryDirectory() as tmpdir:
        init_path = pathlib.Path(tmpdir, "gloo_init").resolve()
        init_method = init_path.as_uri()
        processes = [
            context.Process(
                target=_distributed_tp_reference_worker,
                args=(
                    rank,
                    world_size,
                    init_method,
                    full_logits,
                    token_ids,
                    completion_mask,
                    shard_ranges,
                    value_shards,
                    mask_shards,
                    queue,
                ),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)

        alive = [process for process in processes if process.is_alive()]
        for process in alive:
            process.terminate()
        assert not alive, "distributed TP smoke test timed out"
        assert all(process.exitcode == 0 for process in processes)

    try:
        result = queue.get(timeout=2)
    except Empty as exc:
        raise AssertionError("rank 0 did not report distributed TP result") from exc
    if "error" in result:
        raise AssertionError(f"distributed TP worker failed: {result}") from None

    actual_logps = torch.tensor(result["logps"], dtype=expected_logps.dtype)
    assert torch.allclose(actual_logps, expected_logps, atol=5e-6, rtol=0.0)
    assert result["sum"] == pytest.approx(expected_sum, abs=1e-6)
    assert result["count"] == pytest.approx(expected_count, abs=1e-6)
    assert result["mean"] == pytest.approx(expected_mean, abs=1e-6)


def test_large_rl_shaped_tp_matrix_matches_full_reference():
    vocab_size = 4099
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=6,
        samples_per_prompt=6,
        prompt_len=64,
        completion_len=48,
        vocab_size=vocab_size,
        valid_density=0.81,
        dtype=torch.float32,
        seed=808,
    )
    logits = _make_logits(
        (batch.batch_size, batch.completion_len, vocab_size),
        seed=809,
        scale=2.5,
    )

    for tp_size in (2, 4, 8):
        ranges = vocab_shard_ranges(vocab_size, tp_size)
        _force_tokens_on_every_shard(batch.token_ids, batch.completion_mask, ranges)
        full = selected_logprobs_reference(logits, batch.token_ids, batch.completion_mask)
        tp = selected_logprobs_tp_reference(
            shard_logits_by_vocab(logits, tp_size),
            batch.token_ids,
            batch.completion_mask,
        )
        summary = summarize_tp_logprob_drift(
            tp,
            full,
            batch.token_ids,
            ranges,
            batch.completion_mask,
        )

        assert summary["max_abs_error"] <= 2e-6
        assert math.isfinite(summary["mean_abs_error"])


def test_production_vocab_scale_tail_shard_smoke_matches_full_reference():
    vocab_size = 128257
    logits = _make_logits((1, 3, vocab_size), seed=811, scale=1.7)
    token_ids = torch.tensor([[0, 64000, vocab_size - 1]])
    mask = torch.tensor([[True, True, True]])
    ranges = vocab_shard_ranges(vocab_size, 8)

    full = selected_logprobs_reference(logits, token_ids, mask)
    tp = selected_logprobs_tp_reference(
        shard_logits_by_vocab(logits, 8),
        token_ids,
        mask,
    )
    summary = summarize_tp_logprob_drift(tp, full, token_ids, ranges, mask)

    assert summary["max_abs_error"] <= 2e-6
    assert summary["active_count"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tp_reference_cuda_fp16_smoke_matches_full_reference():
    vocab_size = 513
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=2,
        samples_per_prompt=3,
        prompt_len=8,
        completion_len=11,
        vocab_size=vocab_size,
        valid_density=0.7,
        dtype=torch.float16,
        device="cuda",
        seed=909,
    )
    logits = _make_logits(
        (batch.batch_size, batch.completion_len, vocab_size),
        seed=910,
        dtype=torch.float16,
        device="cuda",
        scale=3.5,
    )
    ranges = vocab_shard_ranges(vocab_size, 4)
    _force_tokens_on_every_shard(batch.token_ids, batch.completion_mask, ranges)

    full = selected_logprobs_reference(logits, batch.token_ids, batch.completion_mask)
    tp = selected_logprobs_tp_reference(
        shard_logits_by_vocab(logits, 4),
        batch.token_ids,
        batch.completion_mask,
    )

    assert torch.allclose(tp, full, atol=2e-5, rtol=0.0)
