# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict Nemotron-H LM-head selected-token logprob on SM90.

Forward reuses the frozen ``fused_linear_logp_sm90`` contract (BN=64, BK=32,
N_SPLIT=64, IEEE expf). The backward contract added here is:

* dlogits are produced by ``_C.fused_linear_logp_sm90_strict_dlogits``, which
  recomputes the forward kernel's own FP32 logits and applies IEEE ``expf``,
  the forward temperature rounding points, and one BF16 cast;
* ``grad_hidden = dlogits @ W`` and ``grad_weight = dlogits^T @ hidden`` run
  through the strict deterministic GEMM (no split-K, FP32 accumulation, one
  cast in the GEMM epilogue);
* ``grad_hidden`` is chunked only along rows and ``grad_weight`` only along
  vocab, so neither output is ever summed across chunks. Chunk sizes are
  therefore bit-neutral and only bound the peak dlogits memory.
* the SM90 det-GEMM tree covers at most ``SM90_DET_GEMM_MAX_K`` reduction
  elements, so on that backend each GEMM is split along its reduction axis
  (vocab for ``grad_hidden``, tokens for ``grad_weight``) at fixed absolute
  boundaries; the BF16 partials are added in ascending order in FP32 and cast
  once. The boundaries do not depend on the memory chunks, so chunking stays
  bit-neutral and ``grad_hidden`` rows stay batch invariant.
"""
from __future__ import annotations

from typing import Optional

import torch

from rl_engine.backends.cuda.embedding.embedding import SM90EmbeddingOp
from rl_engine.backends.cuda.gemm.det_gemm import (
    det_gemm_backend,
    det_gemm_backend_id,
    det_gemm_fallback_reason,
    det_gemm_linear_input_gradient,
    det_gemm_linear_weight_gradient,
)
from rl_engine.backends.cuda.logprob.linear_logp import (
    SM90_LINEAR_LOGP_CONTRACT_VERSION,
    sm90_deterministic_linear_logp,
)
from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.nemotron_h.lm_head_contract import (
    CHECKPOINT_ID,
    CHECKPOINT_REVISION,
    LM_HEAD_LOGPROB_CONTRACT_VERSION,
    NEMOTRON_H_LM_HEAD_SPEC,
    NemotronHGeometryError,
    validate_embedding_geometry,
    validate_lm_head_logprob_geometry,
)
from rl_engine.ops.autograd.backward_runtime import record_backward

# Mirrors of ``BM`` / ``BN`` in csrc/cuda/logprob/fused_linear_logp_sm90.cu. Backward
# slices start on these tile boundaries so every recomputed logit occupies the same
# intra-tile MMA position it had in the forward kernel.
SM90_ROW_TILE = 256
SM90_VOCAB_TILE = 64
# Peak number of BF16 dlogits elements materialized at once (512 MiB).
DLOGITS_BUDGET_ELEMS = 1 << 28
# Mirror of ``BK << TREE_DEPTH`` in csrc/cuda/gemm/det_gemm_kernel.cu. Longer
# reductions leave the SM90 tile path for the scalar fallback, which faults on
# the 131072-wide vocab reduction of ``grad_hidden``.
SM90_DET_GEMM_MAX_K = 32 << 10

_STRICT_SYMBOLS = ("fused_linear_logp_sm90", "fused_linear_logp_sm90_strict_dlogits")
_DLOGITS_KERNEL_ID = "rl_engine._C.fused_linear_logp_sm90_strict_dlogits"


def missing_strict_symbols() -> list[str]:
    if not _EXT_AVAILABLE or _C is None:
        return list(_STRICT_SYMBOLS)
    return [name for name in _STRICT_SYMBOLS if not hasattr(_C, name)]


def _require_strict_symbols() -> None:
    missing = missing_strict_symbols()
    if missing:
        raise RuntimeError(
            "Nemotron-H strict lm_head_logprob requires the SM90 extension symbols "
            f"{missing}; rebuild with KERNEL_ALIGN_FORCE_SM90=1 (no fallback in strict mode)"
        )


def _aligned_chunk(budget_elems: int, other_extent: int, tile: int, limit: int) -> int:
    chunk = (budget_elems // max(other_extent, 1)) // tile * tile
    return min(max(chunk, tile), limit)


def _strict_dlogits(
    hidden_2d: torch.Tensor,
    weight_slice: torch.Tensor,
    target_1d: torch.Tensor,
    grad_1d: torch.Tensor,
    lse_1d: torch.Tensor,
    temperature: Optional[torch.Tensor],
    vocab_start: int,
    real_vocab_size: int,
) -> torch.Tensor:
    return _C.fused_linear_logp_sm90_strict_dlogits(
        hidden_2d,
        weight_slice,
        target_1d,
        grad_1d,
        lse_1d,
        temperature,
        int(vocab_start),
        int(real_vocab_size),
    )


def _reduction_segments(extent: int) -> list[tuple[int, int]]:
    if det_gemm_backend() != "sm90" or extent <= SM90_DET_GEMM_MAX_K:
        return [(0, extent)]
    return [
        (start, min(start + SM90_DET_GEMM_MAX_K, extent))
        for start in range(0, extent, SM90_DET_GEMM_MAX_K)
    ]


def _sum_partials(partials: list[torch.Tensor]) -> torch.Tensor:
    if len(partials) == 1:
        return partials[0]
    total = partials[0].float()
    for partial in partials[1:]:
        total.add_(partial.float())
    return total.to(partials[0].dtype)


def _grad_hidden(dlogits: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _sum_partials(
        [
            det_gemm_linear_input_gradient(dlogits[:, start:end], weight[start:end])
            for start, end in _reduction_segments(weight.size(0))
        ]
    )


def _grad_weight(hidden_2d: torch.Tensor, dlogits: torch.Tensor) -> torch.Tensor:
    return _sum_partials(
        [
            det_gemm_linear_weight_gradient(hidden_2d[start:end], dlogits[start:end])
            for start, end in _reduction_segments(hidden_2d.size(0))
        ]
    )


def strict_lm_head_logp_backward(
    grad_logp: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    target_1d: torch.Tensor,
    lse_1d: torch.Tensor,
    temperature: Optional[torch.Tensor],
    *,
    real_vocab_size: int = -1,
    compute_grad_hidden: bool = True,
    compute_grad_weight: bool = True,
    budget_elems: int = DLOGITS_BUDGET_ELEMS,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(grad_hidden [N, D], grad_weight [V, D])`` in the input dtype."""

    _require_strict_symbols()
    rows, vocab = hidden_2d.size(0), weight.size(0)
    grad_1d = grad_logp.reshape(-1).to(torch.float32).contiguous()
    if rows * vocab <= budget_elems:
        dlogits = _strict_dlogits(
            hidden_2d, weight, target_1d, grad_1d, lse_1d, temperature, 0, real_vocab_size
        )
        grad_hidden = _grad_hidden(dlogits, weight) if compute_grad_hidden else None
        grad_weight = _grad_weight(hidden_2d, dlogits) if compute_grad_weight else None
        return grad_hidden, grad_weight

    grad_hidden = None
    if compute_grad_hidden:
        grad_hidden = torch.empty_like(hidden_2d)
        step = _aligned_chunk(budget_elems, vocab, SM90_ROW_TILE, rows)
        for start in range(0, rows, step):
            end = min(start + step, rows)
            temp = None if temperature is None else temperature[start:end]
            dlogits = _strict_dlogits(
                hidden_2d[start:end],
                weight,
                target_1d[start:end],
                grad_1d[start:end],
                lse_1d[start:end],
                temp,
                0,
                real_vocab_size,
            )
            grad_hidden[start:end] = _grad_hidden(dlogits, weight)

    grad_weight = None
    if compute_grad_weight:
        grad_weight = torch.empty_like(weight)
        step = _aligned_chunk(budget_elems, rows, SM90_VOCAB_TILE, vocab)
        for start in range(0, vocab, step):
            end = min(start + step, vocab)
            dlogits = _strict_dlogits(
                hidden_2d,
                weight[start:end],
                target_1d,
                grad_1d,
                lse_1d,
                temperature,
                start,
                real_vocab_size,
            )
            grad_weight[start:end] = _grad_weight(hidden_2d, dlogits)
    return grad_hidden, grad_weight


class _NemotronHStrictLMHeadLogpAutograd(torch.autograd.Function):
    """Strict forward + strict backward; ``lse`` is returned but not differentiable."""

    @staticmethod
    def forward(ctx, hidden, weight, target_ids, temperature, real_vocab_size):
        lead_shape = hidden.shape[:-1]
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = weight.contiguous()
        target_1d = target_ids.reshape(-1).to(device=hidden.device, dtype=torch.long).contiguous()
        logp, lse = _C.fused_linear_logp_sm90(
            hidden_2d, weight, target_1d, None, temperature, False, int(real_vocab_size)
        )
        ctx.save_for_backward(
            hidden_2d,
            weight,
            target_1d,
            lse,
            temperature if temperature is not None else hidden_2d.new_empty(0),
        )
        ctx.lead_shape = lead_shape
        ctx.real_vocab_size = int(real_vocab_size)
        lse_out = lse.reshape(lead_shape)
        ctx.mark_non_differentiable(lse_out)
        return logp.reshape(lead_shape), lse_out

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        del grad_lse
        hidden_2d, weight, target_1d, lse, temperature = ctx.saved_tensors
        if grad_logp is None:
            return None, None, None, None, None
        grad_hidden, grad_weight = strict_lm_head_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            target_1d,
            lse,
            temperature if temperature.numel() else None,
            real_vocab_size=ctx.real_vocab_size,
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
        )
        record_backward(
            "nemotron_h_lm_head_logprob",
            kernel_id=f"{_DLOGITS_KERNEL_ID}+{det_gemm_backend_id()}",
            impl="cuda_strict_dlogits_det_gemm",
            family="cuda",
        )
        if grad_hidden is not None:
            grad_hidden = grad_hidden.reshape(*ctx.lead_shape, hidden_2d.size(1))
        return grad_hidden, grad_weight, None, None, None


def _is_hopper(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 9


def _require_hopper(tensor: torch.Tensor) -> None:
    if not _is_hopper(tensor.device):
        raise NemotronHGeometryError(
            "Nemotron-H lm_head_logprob strict contract: inputs must live on an SM90 CUDA device"
        )


def _per_token_temperature(
    temperature: Optional[torch.Tensor], rows: int, device: torch.device
) -> Optional[torch.Tensor]:
    if temperature is None:
        return None
    temp = temperature.to(device=device).reshape(-1)
    if temp.numel() == 1:
        temp = temp.expand(rows)
    temp = temp.contiguous()
    if rows and bool((temp <= 0).any().item()):
        raise NemotronHGeometryError(
            "Nemotron-H lm_head_logprob strict contract: temperature must be strictly positive"
        )
    return temp


def _require_target_range(target_ids: torch.Tensor) -> None:
    if target_ids.numel() == 0:
        return
    vocab = NEMOTRON_H_LM_HEAD_SPEC.vocab_size
    if bool(((target_ids < 0) | (target_ids >= vocab)).any().item()):
        raise NemotronHGeometryError(
            f"Nemotron-H lm_head_logprob strict contract: target_ids must be in [0, {vocab})"
        )


class NemotronHLMHeadLogprobSM90Op:
    """Strict SM90 embedding lookup and LM-head selected-token logprob for Nemotron-H.

    * ``embed`` is the untied, unscaled input embedding (bitwise row copy).
    * ``logprob`` is the training entry: FP32 ``(logp, lse)`` with the strict
      backward of this module.
    * ``logprob_with_logits`` is the rollout entry: identical ``logp``/``lse``
      bytes plus the unscaled FP32 logits for the sampler (inference only).

    Every entry fails closed outside the pinned checkpoint geometry.
    """

    op_class = "logprob"
    is_batch_invariant = True
    contract_version = LM_HEAD_LOGPROB_CONTRACT_VERSION

    def __init__(self) -> None:
        _require_strict_symbols()
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise RuntimeError("NemotronHLMHeadLogprobSM90Op requires an SM90 (Hopper) device")
        self._embedding = SM90EmbeddingOp()

    def embed(self, token_ids: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        validate_embedding_geometry(token_ids, embed_weight)
        _require_hopper(embed_weight)
        return self._embedding.forward(token_ids, embed_weight)

    def _prepare(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        temperature: Optional[torch.Tensor],
        embed_weight: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        validate_lm_head_logprob_geometry(
            hidden,
            lm_head_weight,
            target_ids,
            embed_weight=embed_weight,
            temperature=temperature,
        )
        _require_hopper(hidden)
        _require_target_range(target_ids)
        rows = hidden.numel() // hidden.size(-1)
        return _per_token_temperature(temperature, rows, hidden.device)

    def logprob(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        temperature: Optional[torch.Tensor] = None,
        embed_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temp = self._prepare(hidden, lm_head_weight, target_ids, temperature, embed_weight)
        return _NemotronHStrictLMHeadLogpAutograd.apply(
            hidden, lm_head_weight, target_ids, temp, -1
        )

    def logprob_with_logits(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        temperature: Optional[torch.Tensor] = None,
        embed_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad):
            raise RuntimeError("logprob_with_logits is the inference-only rollout entry")
        temp = self._prepare(hidden, lm_head_weight, target_ids, temperature, embed_weight)
        return sm90_deterministic_linear_logp(
            hidden, lm_head_weight, target_ids, temperature=temp, return_logits=True
        )

    def forward(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Operator-checker entry: selected-token logp only."""

        validate_lm_head_logprob_geometry(hidden, lm_head_weight, target_ids, bias=bias)
        logp, _lse = self.logprob(hidden, lm_head_weight, target_ids)
        return logp

    __call__ = forward

    @staticmethod
    def provenance() -> dict[str, object]:
        fallback_reason = det_gemm_fallback_reason()
        return {
            "contract_version": LM_HEAD_LOGPROB_CONTRACT_VERSION,
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "forward_contract_version": SM90_LINEAR_LOGP_CONTRACT_VERSION,
            "forward_kernel_id": "rl_engine._C.fused_linear_logp_sm90",
            "dlogits_kernel_id": _DLOGITS_KERNEL_ID,
            "embedding_kernel_id": "rl_engine._C.embedding_sm90_forward",
            "det_gemm_backend_id": det_gemm_backend_id(),
            "det_gemm_fallback_reason": fallback_reason,
            "dlogits_budget_elems": DLOGITS_BUDGET_ELEMS,
            "fallback": False,
        }
