"""ROCm WS2 vocab-parallel logprob backend.

The reference operator owns the contract: TP transport, fixed global tile-order
merge, target ownership, masking, entropy, and autograd semantics.  This backend
keeps all of that and replaces the two large per-shard passes with HIP kernels
from ``csrc/hip/hip_deterministic_logp_kernel.hip`` (compiled only for ROCm; the
shared ``csrc/deterministic_logp_kernel.cu`` keeps the SM90-tuned CUDA path):

* ``hip_deterministic_logp_tile_stats`` computes the per-row, per-tile FP32
  ``(max, sumexp)`` partials straight from the BF16/FP16/FP32 shard.  The kernel
  converts each element exactly, filters padding columns itself, and reduces
  every tile with a fixed tree, so no FP32 copy of the logits is materialized.
* ``hip_deterministic_logp_backward`` produces ``grad_logits`` for the selected
  logprob and LSE outputs in one fused pass from the saved input shard.

``apply`` keeps the fused ROCm forward/backward local while reusing the shared
contract validation, rank-ordered transport, and fixed tile merge helpers.
``apply_with_entropy`` keeps the inherited reference autograd path because the
entropy gradient needs the full probability tensor anyway.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import torch

from rl_engine.kernels.logprob_contract import LogprobContract, LogprobContractError
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
    VocabParallelLogprobOp,
    _merge_tile_partials,
    _tile_size,
    _validate_invocation,
)

BACKEND_ID = "rocm-vocab-parallel-logp-ws2"

# These tensors are immutable metadata.  Vime's logprob requests normally use
# an all-active mask and repeat the same token count/sharding on every forward;
# recreating them from Python tuples showed up as synchronous ``aten::to`` /
# ``aten::copy_`` activity in the ROCm trace.  Keep the cache bounded because
# sequence packing can expose many token counts over a long run.
_METADATA_CACHE_LIMIT = 32
_ACTIVE_MASK_CACHE: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
_SHARD_START_CACHE: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
_PREFLIGHT_FINGERPRINT_CACHE: OrderedDict[tuple[Any, ...], torch.Tensor] = OrderedDict()
_VERIFIED_PREFLIGHT_CACHE: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_LOGP_GATHER_CACHE: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, list[torch.Tensor]]] = (
    OrderedDict()
)


def _device_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def _cached_active_mask(
    contract: LogprobContract, device: torch.device
) -> tuple[torch.Tensor, bool]:
    values = contract.mask.active_mask
    all_active = all(values)
    # The all-active case is by far the common path; avoid hashing/copying the
    # complete mask tuple for that case.
    signature: Any = ("all", len(values)) if all_active else ("mask", values)
    key = (_device_key(device), signature)
    cached = _ACTIVE_MASK_CACHE.get(key)
    if cached is None:
        cached = (
            torch.ones((len(values),), dtype=torch.bool, device=device)
            if all_active
            else torch.tensor(values, dtype=torch.bool, device=device)
        )
        _ACTIVE_MASK_CACHE[key] = cached
        if len(_ACTIVE_MASK_CACHE) > _METADATA_CACHE_LIMIT:
            _ACTIVE_MASK_CACHE.popitem(last=False)
    else:
        _ACTIVE_MASK_CACHE.move_to_end(key)
    return cached, all_active


def _cached_shard_starts(bounds: tuple[tuple[int, int], ...], device: torch.device) -> torch.Tensor:
    key = (_device_key(device), bounds)
    cached = _SHARD_START_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([start for start, _ in bounds], dtype=torch.long, device=device)
        _SHARD_START_CACHE[key] = cached
        if len(_SHARD_START_CACHE) > _METADATA_CACHE_LIMIT:
            _SHARD_START_CACHE.popitem(last=False)
    else:
        _SHARD_START_CACHE.move_to_end(key)
    return cached


def _cached_logp_gather_buffers(
    *,
    rows: int,
    max_tiles: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    key = (
        _device_key(device),
        int(world_size),
        int(rows),
        int(max_tiles),
    )
    cached = _LOGP_GATHER_CACHE.get(key)
    if cached is None:
        width = 2 * max_tiles + 1
        local = torch.empty((rows, width), dtype=torch.float32, device=device)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        cached = (local, gathered)
        _LOGP_GATHER_CACHE[key] = cached
        if len(_LOGP_GATHER_CACHE) > _METADATA_CACHE_LIMIT:
            _LOGP_GATHER_CACHE.popitem(last=False)
    else:
        _LOGP_GATHER_CACHE.move_to_end(key)
    return cached


def _cached_preflight_fingerprint(
    contract: LogprobContract,
    num_vocab_tiles: int,
    device: torch.device,
) -> tuple[torch.Tensor, bytes]:
    payload = (
        f"{contract.cross_rank_fingerprint()}:{BACKEND_ID}:" f"{int(num_vocab_tiles)}:deterministic"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    key = (_device_key(device), digest)
    cached = _PREFLIGHT_FINGERPRINT_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(tuple(digest), dtype=torch.uint8, device=device)
        _PREFLIGHT_FINGERPRINT_CACHE[key] = cached
        if len(_PREFLIGHT_FINGERPRINT_CACHE) > _METADATA_CACHE_LIMIT:
            _PREFLIGHT_FINGERPRINT_CACHE.popitem(last=False)
    else:
        _PREFLIGHT_FINGERPRINT_CACHE.move_to_end(key)
    return cached, digest


def _preflight_cross_rank_agreement_device(
    contract: LogprobContract,
    tp_group: Any,
    num_vocab_tiles: int,
    device: torch.device,
    *,
    cache_result: bool = False,
) -> None:
    """Check the immutable contract without object collectives or host readback."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise LogprobContractError("vocab-parallel logprob requires initialized torch.distributed")
    world = torch.distributed.get_world_size(group=tp_group)
    local, digest = _cached_preflight_fingerprint(
        contract,
        num_vocab_tiles,
        device,
    )
    cache_key = (
        id(tp_group),
        _device_key(device),
        digest,
    )
    if cache_result and cache_key in _VERIFIED_PREFLIGHT_CACHE:
        _VERIFIED_PREFLIGHT_CACHE.move_to_end(cache_key)
        return
    gathered = torch.empty(
        (world, local.numel()),
        dtype=local.dtype,
        device=device,
    )
    torch.distributed.all_gather_into_tensor(gathered, local, group=tp_group)
    agreement = torch.all(gathered == local.unsqueeze(0))
    if agreement.is_cuda:
        torch._assert_async(
            agreement,
            "cross-rank preflight failed: TP ranks disagree on the logprob contract",
        )
    elif not bool(agreement):
        raise LogprobContractError(
            "cross-rank preflight failed: TP ranks disagree on the logprob contract"
        )
    if cache_result:
        _VERIFIED_PREFLIGHT_CACHE[cache_key] = None
        if len(_VERIFIED_PREFLIGHT_CACHE) > _METADATA_CACHE_LIMIT:
            _VERIFIED_PREFLIGHT_CACHE.popitem(last=False)


def _validate_active_targets_device(
    target_1d: torch.Tensor,
    active_mask: torch.Tensor,
    real_vocab_size: int,
) -> None:
    valid = torch.all(~active_mask | ((target_1d >= 0) & (target_1d < real_vocab_size)))
    if valid.is_cuda:
        torch._assert_async(
            valid,
            f"active target_ids must lie in the real vocabulary [0, {real_vocab_size})",
        )
    elif not bool(valid):
        raise LogprobContractError(
            f"active target_ids must lie in the real vocabulary [0, {real_vocab_size})"
        )


def _gather_logp_partials(
    local_m: torch.Tensor,
    local_s: torch.Tensor,
    z_masked: torch.Tensor,
    safe_target: torch.Tensor,
    contract: LogprobContract,
    tp_group: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather tile statistics and the selected logit with one TP collective."""

    sharding = contract.sharding
    n = z_masked.shape[0]
    start = sharding.local_vocab_start
    local_vocab = sharding.local_vocab_size
    local_idx = (safe_target - start).clamp(0, max(local_vocab - 1, 0))
    owns = (safe_target >= start) & (safe_target < sharding.local_vocab_end)
    rows = torch.arange(n, device=z_masked.device)
    local_contrib = torch.where(
        owns,
        z_masked[rows, local_idx],
        torch.zeros_like(safe_target, dtype=z_masked.dtype),
    ).contiguous()

    if sharding.tp_world_size == 1:
        return local_m.contiguous(), local_s.contiguous(), local_contrib.float()

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise LogprobContractError("vocab-parallel logprob requires initialized torch.distributed")
    tile_size = sharding.local_vocab_size // local_m.shape[1]
    tile_counts = [(end - start) // tile_size for start, end in sharding.vocab_shard_bounds]
    max_tiles = max(tile_counts)
    packed, gathered = _cached_logp_gather_buffers(
        rows=n,
        max_tiles=max_tiles,
        world_size=sharding.tp_world_size,
        device=z_masked.device,
    )
    local_tiles = local_m.shape[1]
    packed[:, :local_tiles].copy_(local_m)
    packed[:, max_tiles : max_tiles + local_tiles].copy_(local_s)
    packed[:, -1].copy_(local_contrib)
    # vLLM's ROCm full-graph path captures list-form all_gather, while
    # all_gather_into_tensor remains an eager _allgather_base_ call per token.
    torch.distributed.all_gather(gathered, packed, group=tp_group)

    m_parts = [gathered[rank][:, : tile_counts[rank]] for rank in range(sharding.tp_world_size)]
    s_parts = [
        gathered[rank][:, max_tiles : max_tiles + tile_counts[rank]]
        for rank in range(sharding.tp_world_size)
    ]
    m_all = torch.cat(m_parts, dim=1).contiguous()
    s_all = torch.cat(s_parts, dim=1).contiguous()

    starts = _cached_shard_starts(sharding.vocab_shard_bounds, safe_target.device)
    owner = torch.bucketize(safe_target, starts, right=True) - 1
    stacked = torch.stack(gathered, dim=0)
    target_logit = stacked[owner, rows, -1]
    return m_all, s_all, target_logit


def _native_backward_available() -> bool:
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    except ImportError:
        return False
    return bool(
        _EXT_AVAILABLE
        and hasattr(_C, "hip_deterministic_logp_tile_stats")
        and hasattr(_C, "hip_deterministic_logp_backward")
    )


class _HipKernels:
    """``VocabParallelLogprobKernels`` over the ROCm extension symbols."""

    @staticmethod
    def tile_stats(shard, vocab_start, real_vocab, num_tiles):
        from rl_engine.kernels.ops.base import _C

        tile_max, tile_sum = _C.hip_deterministic_logp_tile_stats(
            shard, vocab_start, real_vocab, num_tiles
        )
        return tile_max, tile_sum

    @staticmethod
    def backward(
        shard, lse, coef_logp, coef_lse, target_local, vocab_start, real_vocab, has_lse_grad
    ):
        from rl_engine.kernels.ops.base import _C

        return _C.hip_deterministic_logp_backward(
            shard, lse, coef_logp, coef_lse, target_local, vocab_start, real_vocab, has_lse_grad
        )


class _RocmVocabParallelLogprobFunction(torch.autograd.Function):
    """ROCm tile statistics and backward with the shared WS2 merge contract."""

    @staticmethod
    def forward(ctx, local_logits, target_1d, active_mask, contract, tp_group, tile, all_active):
        sharding = contract.sharding
        shard = local_logits.contiguous()
        local_tiles = sharding.local_vocab_size // tile
        if local_tiles <= 0:
            raise RuntimeError("native tile stats require at least one local vocab tile")
        local_m, local_s = _HipKernels.tile_stats(
            shard,
            sharding.local_vocab_start,
            sharding.real_vocab_size,
            local_tiles,
        )
        safe_target = (
            target_1d
            if all_active
            else torch.where(active_mask, target_1d, torch.zeros_like(target_1d))
        )
        m_all, s_all, target_logit = _gather_logp_partials(
            local_m.contiguous(),
            local_s.contiguous(),
            shard,
            safe_target,
            contract,
            tp_group,
        )
        lse = _merge_tile_partials(m_all, s_all)
        selected_logp = (
            target_logit - lse
            if all_active
            else torch.where(active_mask, target_logit - lse, torch.zeros_like(lse))
        )

        ctx.save_for_backward(shard, lse, safe_target, active_mask)
        ctx.local_vocab_start = sharding.local_vocab_start
        ctx.real_vocab_size = sharding.real_vocab_size
        ctx.all_active = all_active
        ctx.set_materialize_grads(False)
        return selected_logp, lse

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        if not ctx.needs_input_grad[0] or (grad_logp is None and grad_lse is None):
            return None, None, None, None, None, None
        shard, lse, safe_target, active_mask = ctx.saved_tensors
        rows, local_vocab = shard.shape
        start = ctx.local_vocab_start
        if grad_logp is not None:
            coef_logp = (
                grad_logp.float().contiguous()
                if ctx.all_active
                else torch.where(active_mask, grad_logp, torch.zeros_like(grad_logp))
                .float()
                .contiguous()
            )
            owns = (safe_target >= start) & (safe_target < start + local_vocab)
            target_local = torch.where(
                owns if ctx.all_active else owns & active_mask,
                safe_target - start,
                torch.full_like(safe_target, -1),
            ).contiguous()
        else:
            coef_logp = lse.new_zeros((rows,))
            target_local = torch.full((rows,), -1, dtype=torch.long, device=shard.device)
        has_lse_grad = grad_lse is not None
        coef_lse = grad_lse.float().contiguous() if has_lse_grad else lse.new_zeros((rows,))
        grad = _HipKernels.backward(
            shard,
            lse.contiguous(),
            coef_logp,
            coef_lse,
            target_local,
            start,
            ctx.real_vocab_size,
            has_lse_grad,
        )
        return grad, None, None, None, None, None, None


def _apply_with_kernels(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    contract: LogprobContract,
    tp_group: Any,
    num_vocab_tiles: int,
    validate: bool,
    targets_validated: bool,
    cache_preflight: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(contract, LogprobContract):
        raise LogprobContractError("contract must be a LogprobContract")
    tile = _tile_size(contract, num_vocab_tiles)
    _validate_invocation(local_logits, target_ids, contract, tp_group)

    target_1d = target_ids.reshape(-1).to(device=local_logits.device, dtype=torch.long)
    active_mask, all_active = _cached_active_mask(contract, local_logits.device)
    if validate and not targets_validated:
        _validate_active_targets_device(target_1d, active_mask, contract.sharding.real_vocab_size)
    if validate:
        if contract.sharding.tp_world_size > 1:
            _preflight_cross_rank_agreement_device(
                contract,
                tp_group,
                num_vocab_tiles,
                local_logits.device,
                cache_result=cache_preflight,
            )

    selected_logp, lse = _RocmVocabParallelLogprobFunction.apply(
        local_logits, target_1d, active_mask, contract, tp_group, tile, all_active
    )
    if validate:
        finite = torch.all(torch.isfinite(lse) | ~active_mask)
        message = (
            "non-finite logsumexp on an active row: logits over the real "
            "vocabulary must be finite for every active token"
        )
        if finite.is_cuda:
            torch._assert_async(finite, message)
        elif not bool(finite):
            raise LogprobContractError(message)
    return selected_logp, lse


class RocmVocabParallelLogprobOp(VocabParallelLogprobOp):
    """Contract-preserving ROCm implementation with HIP local reductions."""

    op_class = "logprob"
    is_batch_invariant = True
    backend_id = BACKEND_ID

    def apply(
        self,
        local_logits: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        contract: LogprobContract,
        tp_group: Any = None,
        num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES,
        validate: bool = True,
        deterministic: bool = True,
        targets_validated: bool = False,
        cache_preflight: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not deterministic:
            return super().apply(
                local_logits,
                target_ids,
                contract=contract,
                tp_group=tp_group,
                num_vocab_tiles=num_vocab_tiles,
                validate=validate,
                deterministic=False,
            )
        if not _native_backward_available():
            raise RuntimeError(
                f"{BACKEND_ID} requires rl_engine._C built with a ROCm toolchain "
                "(hip_deterministic_logp_* symbols are missing); it does not fall back"
            )
        return _apply_with_kernels(
            local_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
            targets_validated=targets_validated,
            cache_preflight=cache_preflight,
        )


__all__ = ["BACKEND_ID", "RocmVocabParallelLogprobOp"]
