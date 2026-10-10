# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic vocab-parallel TP logprob reference tests"""

from __future__ import annotations

import queue
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.backends.rocm.logprob.vocab_parallel_logp import RocmVocabParallelLogprobOp
from rl_engine.contracts.numerical import load_contract
from rl_engine.contracts.operators.logprob import (
    DeterminismScope,
    LogprobContract,
    LogprobContractError,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.reference.logprob.batch_invariant_logp import NativeBatchInvariantLogpOp
from rl_engine.reference.logprob.vocab_parallel_logp import BACKEND_ID, VocabParallelLogprobOp
from rl_engine.runtime.registry import KernelRegistry, OpBackend

REAL_VOCAB = 27
PADDED_VOCAB = 32
NUM_TILES = 8
NUM_TOKENS = 6
ACTIVE = (True, True, True, True, True, False)


def _even_bounds(padded: int, world: int) -> tuple[tuple[int, int], ...]:
    shard = padded // world
    return tuple(
        (rank * shard, padded if rank == world - 1 else (rank + 1) * shard) for rank in range(world)
    )


def _contract(
    *,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    bounds: tuple[tuple[int, int], ...] | None = None,
    real_vocab: int = REAL_VOCAB,
    padded_vocab: int = PADDED_VOCAB,
    num_tokens: int = NUM_TOKENS,
    active: tuple[bool, ...] = ACTIVE,
    dtype: str = "fp32",
) -> LogprobContract:
    return LogprobContract(
        role="train",
        dtype=dtype,
        mask=MaskSpec(num_tokens=num_tokens, active_mask=active),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            vocab_shard_bounds=(
                bounds if bounds is not None else _even_bounds(padded_vocab, tp_world_size)
            ),
            real_vocab_size=real_vocab,
            padded_vocab_size=padded_vocab,
        ),
        reduction=ReductionSpec(),
    )


def _inputs(dtype=torch.float32, seed: int = 2026):
    torch.manual_seed(seed)
    logits = torch.randn(NUM_TOKENS, PADDED_VOCAB, dtype=torch.float32).to(dtype)
    targets = torch.tensor([1, 5, REAL_VOCAB - 1, 0, 13, -100])
    return logits, targets


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    view_dtype = {
        torch.float32: torch.int32,
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
    }[tensor.dtype]
    return tensor.contiguous().view(view_dtype)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool((_bits(a) == _bits(b)).all())


def _case_shard_size_mismatch():
    logits, targets = _inputs()
    return logits, targets, _contract(tp_rank=0, tp_world_size=2), NUM_TILES, "vocab columns"


def _case_mask_length_mismatch():
    logits, targets = _inputs()
    contract = _contract(num_tokens=NUM_TOKENS + 1, active=ACTIVE + (True,))
    return logits, targets, contract, NUM_TILES, "num_tokens"


def _case_dtype_mismatch():
    logits, targets = _inputs()
    return logits, targets, _contract(dtype="bf16"), NUM_TILES, "dtype"


def _case_tile_misaligned_bounds():
    # Tile size is 32/8 = 4; a boundary at 6 is misaligned.
    logits, targets = _inputs()
    contract = _contract(tp_world_size=2, bounds=((0, 6), (6, 32)))
    return logits[:, :6], targets, contract, NUM_TILES, "tile"


def _case_bad_num_vocab_tiles():
    logits, targets = _inputs()
    return logits, targets, _contract(), 7, "num_vocab_tiles"


def _case_active_target_out_of_real_vocab():
    logits, targets = _inputs()
    bad_targets = targets.clone()
    bad_targets[0] = REAL_VOCAB  # padding column, active row
    return logits, bad_targets, _contract(), NUM_TILES, "real vocabulary"


def _case_all_inf_active_row():
    logits, targets = _inputs()
    poisoned = logits.clone()
    poisoned[0, :] = float("-inf")
    return poisoned, targets, _contract(), NUM_TILES, "non-finite"


@pytest.mark.parametrize(
    "case",
    [
        _case_shard_size_mismatch,
        _case_mask_length_mismatch,
        _case_dtype_mismatch,
        _case_tile_misaligned_bounds,
        _case_bad_num_vocab_tiles,
        _case_active_target_out_of_real_vocab,
        _case_all_inf_active_row,
    ],
    ids=lambda fn: fn.__name__.removeprefix("_case_"),
)
def test_invalid_invocations_fail_loudly(case):
    logits, targets, contract, num_tiles, match = case()
    with pytest.raises(LogprobContractError, match=match):
        VocabParallelLogprobOp()(logits, targets, contract=contract, num_vocab_tiles=num_tiles)


class TestSingleRank:
    def test_repeated_runs_are_bitwise_identical(self):
        contract = _contract()
        logits, targets = _inputs()
        op = VocabParallelLogprobOp()
        logp_a, lse_a = op(logits, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        logp_b, lse_b = op(logits, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        assert _bitwise_equal(logp_a, logp_b)
        assert _bitwise_equal(lse_a, lse_b)

    def test_batch_invariance_same_row_any_context(self):
        contract_full = _contract()
        logits, targets = _inputs()
        op = VocabParallelLogprobOp()
        logp_full, lse_full = op(logits, targets, contract=contract_full, num_vocab_tiles=NUM_TILES)

        contract_single = _contract(num_tokens=1, active=(True,))
        logp_one, lse_one = op(
            logits[2:3], targets[2:3], contract=contract_single, num_vocab_tiles=NUM_TILES
        )
        assert _bitwise_equal(logp_full[2:3], logp_one)
        assert _bitwise_equal(lse_full[2:3], lse_one)

    def test_matches_ws1_batch_invariant_logp_within_contract_tolerance(self):
        tolerance = load_contract()["accuracy"]["default"]["logprob"]["float32"]
        contract = _contract(padded_vocab=REAL_VOCAB + 5)
        # Use a real==padded contract so the WS1 op sees identical logits.
        contract = _contract(real_vocab=PADDED_VOCAB, padded_vocab=PADDED_VOCAB)
        logits, targets = _inputs()
        logp, _ = VocabParallelLogprobOp()(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        ws1 = NativeBatchInvariantLogpOp().apply(logits, targets)
        active = torch.tensor(ACTIVE)
        assert torch.allclose(
            logp[active], ws1[active], atol=tolerance["atol"], rtol=tolerance["rtol"]
        )

    def test_padding_columns_are_excluded_and_finite(self):
        contract = _contract()
        logits, targets = _inputs()
        boosted = logits.clone()
        boosted[:, REAL_VOCAB:] = 1e4  # huge padding logits must not leak into LSE
        logp, lse = VocabParallelLogprobOp()(
            boosted, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        ref_lse = torch.logsumexp(boosted[:, :REAL_VOCAB].float(), dim=-1)
        assert torch.isfinite(logp).all() and torch.isfinite(lse).all()
        assert torch.allclose(lse, ref_lse, atol=1e-5)

    def test_inactive_rows_zero_filled_lse_still_exported(self):
        contract = _contract()
        logits, targets = _inputs()
        logp, lse = VocabParallelLogprobOp()(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        assert logp[-1].item() == 0.0
        assert torch.isfinite(lse[-1])


class TestBackward:
    def test_grads_match_autograd_oracle(self):
        tolerance = load_contract()["accuracy"]["default"]["logprob"]["float32"]
        contract = _contract()
        logits, targets = _inputs()
        x = logits.clone().requires_grad_(True)
        logp, lse = VocabParallelLogprobOp()(
            x, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        (logp.sum() + 0.5 * lse.sum()).backward()

        y = logits.clone().requires_grad_(True)
        ref_lse = torch.logsumexp(y[:, :REAL_VOCAB].float(), dim=-1)
        safe = targets.clamp(0, REAL_VOCAB - 1)
        ref_logp = y[torch.arange(NUM_TOKENS), safe].float() - ref_lse
        ref_logp = torch.where(torch.tensor(ACTIVE), ref_logp, torch.zeros_like(ref_logp))
        (ref_logp.sum() + 0.5 * ref_lse.sum()).backward()

        assert torch.allclose(x.grad, y.grad, atol=tolerance["atol"], rtol=tolerance["rtol"])
        assert bool((x.grad[:, REAL_VOCAB:] == 0).all())

        # No grad requested -> outputs detached from autograd entirely.
        logp_ng, lse_ng = VocabParallelLogprobOp()(
            logits, targets, contract=contract, num_vocab_tiles=NUM_TILES
        )
        assert not logp_ng.requires_grad and not lse_ng.requires_grad

    def test_inactive_rows_grad_asymmetry(self):
        """The logp term is zeroed on inactive rows; the lse term still flows —
        lse is a row property exported (and differentiable) for every row."""

        contract = _contract()
        logits, targets = _inputs()

        x = logits.clone().requires_grad_(True)
        _, lse = VocabParallelLogprobOp()(x, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        lse.sum().backward()
        assert bool((x.grad[-1, :REAL_VOCAB].abs() > 0).any())

        z = logits.clone().requires_grad_(True)
        logp, _ = VocabParallelLogprobOp()(z, targets, contract=contract, num_vocab_tiles=NUM_TILES)
        logp.sum().backward()
        assert bool((z.grad[-1] == 0).all())

    def test_entropy_matches_full_vocab_oracle_and_backpropagates(self):
        contract = _contract()
        logits, targets = _inputs()
        x = logits.clone().requires_grad_(True)
        logp, _lse, entropy = VocabParallelLogprobOp().apply_with_entropy(
            x,
            targets,
            contract=contract,
            num_vocab_tiles=NUM_TILES,
        )

        y = logits.clone().requires_grad_(True)
        ref_log_probs = torch.log_softmax(y[:, :REAL_VOCAB].float(), dim=-1)
        safe = targets.clamp(0, REAL_VOCAB - 1)
        ref_logp = ref_log_probs[torch.arange(NUM_TOKENS), safe]
        ref_logp = torch.where(torch.tensor(ACTIVE), ref_logp, torch.zeros_like(ref_logp))
        ref_entropy = -(ref_log_probs.exp() * ref_log_probs).sum(dim=-1)

        torch.testing.assert_close(entropy, ref_entropy)
        (logp.sum() + entropy.sum()).backward()
        (ref_logp.sum() + ref_entropy.sum()).backward()
        torch.testing.assert_close(x.grad, y.grad, atol=2e-5, rtol=2e-5)


def test_dispatch_resolves_reference_and_leaves_legacy_untouched():
    registry = KernelRegistry()
    contract = _contract()

    result = registry.get_logprob_op(contract, requested_backend="reference")
    assert result.capability.backend_id == BACKEND_ID
    assert result.provenance["fallback"] is False
    assert isinstance(result.op, VocabParallelLogprobOp)
    assert (
        result.provenance["contract"]["reduction"]["determinism_scope"]
        == DeterminismScope.CROSS_TP_BITWISE.value
    )

    by_id = registry.get_logprob_op(contract, requested_backend=BACKEND_ID)
    assert by_id.capability.backend_id == BACKEND_ID
    by_kind = registry.get_logprob_op(contract, requested_backend="reference")
    assert by_kind.capability.backend_id == BACKEND_ID

    for ops in registry._priority_map.values():
        for candidates in ops.values():
            assert OpBackend.PYTORCH_VOCAB_PARALLEL_LOGP not in candidates


# Cross-TP bitwise determinism on real ranks (NCCL, one CUDA device per rank)
TP_REAL_VOCAB = 1000
TP_PADDED_VOCAB = 1024
TP_NUM_TILES = 32  # tile = 32 columns
TP_TILE = TP_PADDED_VOCAB // TP_NUM_TILES
TP_NUM_TOKENS = 48
TP_ACTIVE = tuple(index % 7 != 5 for index in range(TP_NUM_TOKENS))
TP_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_SPAWN_TIMEOUT_S = 300


def _cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _requires_gpus(count: int):
    return pytest.mark.skipif(
        _cuda_device_count() < count,
        reason=f"cross-TP determinism needs {count} CUDA devices to place one rank per device",
    )


def _tile_counts(world_size: int, uneven: bool) -> list[int]:
    """Tiles per rank; bounds are built from whole tiles so they stay tile-aligned."""

    counts = [TP_NUM_TILES // world_size for _ in range(world_size)]
    counts[-1] += TP_NUM_TILES % world_size
    if uneven:
        for rank in range(world_size - 1):
            if counts[rank] > 1:
                counts[rank] -= 1
                counts[-1] += 1
    return counts


def _tp_bounds(world_size: int, uneven: bool) -> tuple[tuple[int, int], ...]:
    bounds, cursor = [], 0
    for count in _tile_counts(world_size, uneven):
        bounds.append((cursor, cursor + count * TP_TILE))
        cursor += count * TP_TILE
    return tuple(bounds)


def _tp_contract(tp_rank: int, tp_world_size: int, bounds, dtype_name: str) -> LogprobContract:
    return _contract(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        bounds=bounds,
        real_vocab=TP_REAL_VOCAB,
        padded_vocab=TP_PADDED_VOCAB,
        num_tokens=TP_NUM_TOKENS,
        active=TP_ACTIVE,
        dtype=dtype_name,
    )


def _tp_inputs(device, dtype, seed: int = 2026):
    """Identical logits and targets on every rank, seeded on CPU."""

    gen = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(TP_NUM_TOKENS, TP_PADDED_VOCAB, generator=gen, dtype=torch.float32)
    targets = torch.randint(0, TP_REAL_VOCAB, (TP_NUM_TOKENS,), generator=gen)
    active = torch.tensor(TP_ACTIVE)
    # Inactive rows carry ignore_index; active_mask stays the sole authority.
    targets = torch.where(active, targets, torch.full_like(targets, -100))
    return logits.to(device=device, dtype=dtype), targets.to(device)


def _nccl_worker(
    rank,
    world_size,
    init_method,
    result_queue,
    scenario,
    uneven,
    dtype_name,
    backend_kind="pytorch",
):
    import torch.distributed as dist

    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=world_size
        )
        dtype = TP_DTYPES[dtype_name]
        if backend_kind == "rocm":
            op = RocmVocabParallelLogprobOp()
        elif backend_kind == "triton":
            from rl_engine.backends.shared.triton.logprob.vocab_parallel_logp import (
                TritonVocabParallelLogprobOp,
            )

            op = TritonVocabParallelLogprobOp()
        else:
            op = VocabParallelLogprobOp()
        bounds = _tp_bounds(world_size, uneven)
        logits, targets = _tp_inputs(device, dtype)
        tiles = TP_NUM_TILES

        if scenario in {"preflight", "misaligned"}:
            if scenario == "preflight":
                if rank == 0:
                    tiles = TP_NUM_TILES * 2
            else:
                # Nudge the first boundary off the tile grid, on every rank.
                split = bounds[0][1] + TP_TILE // 4
                bounds = ((0, split), (split, bounds[1][1])) + bounds[2:]

            start, end = bounds[rank]
            try:
                op(
                    logits[:, start:end].contiguous().clone(),
                    targets,
                    contract=_tp_contract(rank, world_size, bounds, dtype_name),
                    tp_group=dist.group.WORLD,
                    num_vocab_tiles=tiles,
                )
                result_queue.put({"ok": False, "rank": rank, "traceback": "no error raised"})
            except LogprobContractError as exc:
                result_queue.put({"ok": True, "rank": rank, "message": str(exc)})
            return

        start, end = bounds[rank]
        shard = logits[:, start:end].contiguous().clone().requires_grad_(True)
        tp_contract = _tp_contract(rank, world_size, bounds, dtype_name)
        logp_tp, lse_tp = op(
            shard,
            targets,
            contract=tp_contract,
            tp_group=dist.group.WORLD,
            num_vocab_tiles=TP_NUM_TILES,
        )
        (logp_tp.sum() + 0.5 * lse_tp.sum()).backward()

        # Same ranks, same inputs, run again: the collectives must not perturb bits.
        rerun = logits[:, start:end].contiguous().clone()
        logp_re, lse_re = op(
            rerun,
            targets,
            contract=tp_contract,
            tp_group=dist.group.WORLD,
            num_vocab_tiles=TP_NUM_TILES,
        )

        # In-process TP=1 run on the full logits: the cross-TP claim is that a
        # TP=n result equals the TP=1 result, bit for bit.
        full = logits.clone().requires_grad_(True)
        logp_one, lse_one = op(
            full,
            targets,
            contract=_tp_contract(0, 1, ((0, TP_PADDED_VOCAB),), dtype_name),
            num_vocab_tiles=TP_NUM_TILES,
        )
        (logp_one.sum() + 0.5 * lse_one.sum()).backward()

        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "logp_bits_match": _bitwise_equal(logp_tp, logp_one),
                "lse_bits_match": _bitwise_equal(lse_tp, lse_one),
                "grad_bits_match": _bitwise_equal(shard.grad, full.grad[:, start:end]),
                "rerun_bits_match": (
                    _bitwise_equal(logp_re, logp_tp) and _bitwise_equal(lse_re, lse_tp)
                ),
                "logp_bit_pattern": _bits(logp_tp.detach().float().cpu()).tolist(),
                "lse_bit_pattern": _bits(lse_tp.detach().float().cpu()).tolist(),
            }
        )
    except Exception:  # pragma: no cover - forwarded to the parent process
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def _run_nccl_scenario(
    world_size, scenario="correctness", uneven=False, dtype_name="fp32", backend_kind="pytorch"
):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "nccl_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_nccl_worker,
                args=(
                    rank,
                    world_size,
                    init_method,
                    result_queue,
                    scenario,
                    uneven,
                    dtype_name,
                    backend_kind,
                ),
            )
            for rank in range(world_size)
        ]
        results = []
        try:
            for process in processes:
                process.start()
            for _ in range(world_size):
                try:
                    results.append(result_queue.get(timeout=_SPAWN_TIMEOUT_S))
                except queue.Empty:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    pytest.fail(f"timed out waiting for NCCL workers (scenario={scenario})")
        finally:
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
    results.sort(key=lambda item: item["rank"])
    for result in results:
        assert result["ok"], result.get("traceback")
    for process in processes:
        assert process.exitcode == 0
    return results


class TestCrossTPBitwise:
    """TP=n output == TP=1 output, bit for bit, on real NCCL ranks."""

    @_requires_gpus(2)
    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
    @pytest.mark.parametrize("uneven", [False, True], ids=["even", "uneven"])
    def test_tp2_bitwise_identical_to_tp1(self, uneven, dtype_name):
        self._assert_matches_tp1(_run_nccl_scenario(2, uneven=uneven, dtype_name=dtype_name))

    @_requires_gpus(4)
    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
    @pytest.mark.parametrize("uneven", [False, True], ids=["even", "uneven"])
    def test_tp4_bitwise_identical_to_tp1(self, uneven, dtype_name):
        self._assert_matches_tp1(_run_nccl_scenario(4, uneven=uneven, dtype_name=dtype_name))

    @staticmethod
    def _assert_matches_tp1(results):
        for result in results:
            rank = result["rank"]
            assert result["logp_bits_match"], f"rank {rank} logp bits differ from TP=1"
            assert result["lse_bits_match"], f"rank {rank} lse bits differ from TP=1"
            assert result["grad_bits_match"], f"rank {rank} grad bits differ from TP=1"
            assert result["rerun_bits_match"], f"rank {rank} bits changed between identical runs"
        # Outputs are replicated: every rank must hold identical bits.
        for other in results[1:]:
            assert results[0]["logp_bit_pattern"] == other["logp_bit_pattern"]
            assert results[0]["lse_bit_pattern"] == other["lse_bit_pattern"]

    @_requires_gpus(2)
    def test_tp2_and_tp4_agree_with_each_other(self):
        """The claim is over TP degrees, so pin TP=2 against TP=4 directly."""

        if _cuda_device_count() < 4:
            pytest.skip("needs 4 CUDA devices to compare TP=2 against TP=4")
        tp2 = _run_nccl_scenario(2)
        tp4 = _run_nccl_scenario(4)
        assert tp2[0]["logp_bit_pattern"] == tp4[0]["logp_bit_pattern"]
        assert tp2[0]["lse_bit_pattern"] == tp4[0]["lse_bit_pattern"]


class TestCrossTPGuards:
    """A disagreement must abort loudly on every rank, not strand ranks in a collective."""

    @_requires_gpus(2)
    def test_preflight_rejects_mismatched_num_vocab_tiles(self):
        results = _run_nccl_scenario(2, scenario="preflight")
        for result in results:
            assert "cross-rank preflight failed" in result["message"]

    @_requires_gpus(2)
    def test_misaligned_shard_bounds_rejected(self):
        results = _run_nccl_scenario(2, scenario="misaligned")
        for result in results:
            assert "not aligned to the vocab tile size" in result["message"]


@pytest.mark.skipif(torch.version.hip is None, reason="requires a ROCm PyTorch build")
class TestRocmNativeCrossTP:
    """The ROCm production path must preserve the same TP contract as reference."""

    @staticmethod
    def _require_native() -> None:
        from rl_engine.runtime.registry import _rocm_vocab_logprob_native_available

        if not _rocm_vocab_logprob_native_available():
            pytest.skip("requires the compiled ROCm logprob extension")

    @_requires_gpus(2)
    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
    def test_tp2_native_matches_tp1_and_repeat(self, dtype_name):
        self._require_native()
        results = _run_nccl_scenario(2, dtype_name=dtype_name, backend_kind="rocm")
        TestCrossTPBitwise._assert_matches_tp1(results)

    @_requires_gpus(4)
    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
    def test_tp4_native_matches_tp1_and_repeat(self, dtype_name):
        self._require_native()
        results = _run_nccl_scenario(4, dtype_name=dtype_name, backend_kind="rocm")
        TestCrossTPBitwise._assert_matches_tp1(results)


def _triton_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _triton_available(), reason="requires Triton on a CUDA/ROCm GPU")
class TestTritonNativeCrossTP:
    """The Triton production path must preserve the same TP contract as reference."""

    @_requires_gpus(2)
    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
    def test_tp2_native_matches_tp1_and_repeat(self, dtype_name):
        results = _run_nccl_scenario(2, dtype_name=dtype_name, backend_kind="triton")
        TestCrossTPBitwise._assert_matches_tp1(results)

    @_requires_gpus(4)
    @pytest.mark.parametrize("dtype_name", ["fp32", "bf16"])
    def test_tp4_native_matches_tp1_and_repeat(self, dtype_name):
        results = _run_nccl_scenario(4, dtype_name=dtype_name, backend_kind="triton")
        TestCrossTPBitwise._assert_matches_tp1(results)
