"""CUDA WS2 vocab-parallel logprob backend.

The reference operator owns the contract: TP transport, fixed global tile-order
merge, target ownership, masking, entropy, and autograd semantics.  This backend
keeps all of that and replaces the large per-shard tile scan with the SM90-tuned
CUDA kernel in ``csrc/cuda/logprob/deterministic_logp_kernel.cu``:

* ``deterministic_logp_tile_stats`` computes the per-row, per-tile FP32
  ``(max, sumexp)`` partials straight from the FP16/BF16/FP32 shard.  It filters
  padding columns itself and reduces every tile with a fixed warp/block tree, so
  the partials do not depend on the tile's position in the global order.

* ``deterministic_logp_backward`` produces ``grad_logits`` for the selected
  logprob and LSE outputs in one fused pass from the saved input shard.

Both run through the shared :func:`apply_with_kernels` path, which reads the
stored BF16/FP16/FP32 shard directly instead of materializing an FP32 copy of
it.  ``apply_with_entropy`` keeps the shared autograd path (with the CUDA tile
kernel) because the entropy gradient needs the full probability tensor anyway.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.contracts.operators.logprob import LogprobContract
from rl_engine.reference.logprob.vocab_parallel_logp import (
    DEFAULT_NUM_VOCAB_TILES,
    VocabParallelLogprobOp,
    apply_with_kernels,
)

BACKEND_ID = "cuda-vocab-parallel-logp-ws2"


def native_tile_stats_available() -> bool:
    """True when a CUDA-built ``rl_engine._C`` exposes both fused kernels."""

    if torch.version.hip is not None:
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    except ImportError:
        return False
    return bool(
        _EXT_AVAILABLE
        and hasattr(_C, "deterministic_logp_tile_stats")
        and hasattr(_C, "deterministic_logp_backward")
    )


def _native_cuda_tile_stats(
    z_masked: torch.Tensor,
    tile: int,
    *,
    vocab_start: int,
    real_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA counterpart of ``_native_rocm_tile_stats``.

    TP transport and the global merge deliberately stay in Python so the issue
    #241 reduction contract is identical across CUDA, ROCm, and the reference.
    """

    if torch.version.hip is not None or not z_masked.is_cuda:
        raise RuntimeError("CUDA native tile stats require a CUDA tensor on a CUDA build")
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        if not _EXT_AVAILABLE or not hasattr(_C, "deterministic_logp_tile_stats"):
            raise RuntimeError(
                "CUDA vocab-parallel logprob native extension is unavailable; "
                "build rl_engine._C with a CUDA toolchain"
            )
        local_tiles = z_masked.shape[1] // tile
        if local_tiles <= 0:
            raise RuntimeError("CUDA native tile stats require at least one local vocab tile")
        return tuple(
            tensor.contiguous()
            for tensor in _C.deterministic_logp_tile_stats(
                z_masked,
                int(vocab_start),
                int(real_vocab_size),
                int(local_tiles),
            )
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("CUDA vocab-parallel logprob native extension is unavailable") from exc


class _CudaKernels:
    """``VocabParallelLogprobKernels`` over the CUDA extension symbols."""

    @staticmethod
    def tile_stats(shard, vocab_start, real_vocab, num_tiles):
        from rl_engine.backends.extension import _C

        tile_max, tile_sum = _C.deterministic_logp_tile_stats(
            shard, vocab_start, real_vocab, num_tiles
        )
        return tile_max, tile_sum

    @staticmethod
    def backward(
        shard, lse, coef_logp, coef_lse, target_local, vocab_start, real_vocab, has_lse_grad
    ):
        from rl_engine.backends.extension import _C

        return _C.deterministic_logp_backward(
            shard, lse, coef_logp, coef_lse, target_local, vocab_start, real_vocab, has_lse_grad
        )


class CudaVocabParallelLogprobOp(VocabParallelLogprobOp):
    """Contract-preserving CUDA implementation with fused local reductions."""

    op_class = "logprob"
    is_batch_invariant = True
    backend_id = BACKEND_ID
    # Used by apply_with_entropy, which keeps the shared autograd path.
    # staticmethod: the base reads ``self.use_native_tile_stats`` and calls it
    # with the _native_rocm_tile_stats signature, so it must not bind ``self``.
    use_native_tile_stats = staticmethod(_native_cuda_tile_stats)

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
        if not native_tile_stats_available():
            raise RuntimeError(
                f"{BACKEND_ID} requires rl_engine._C built with a CUDA toolchain "
                "(deterministic_logp_* symbols are missing); it does not fall back"
            )
        return apply_with_kernels(
            local_logits,
            target_ids,
            contract=contract,
            tp_group=tp_group,
            num_vocab_tiles=num_vocab_tiles,
            validate=validate,
            kernels=_CudaKernels,
        )


__all__ = ["BACKEND_ID", "CudaVocabParallelLogprobOp", "native_tile_stats_available"]
