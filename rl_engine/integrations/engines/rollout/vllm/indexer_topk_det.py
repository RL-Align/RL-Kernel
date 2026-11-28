# SPDX-License-Identifier: Apache-2.0
"""Rollout DSV4 indexer Top-K on the same deterministic rule as the training actor.

vLLM picks aiter or the ``_C`` top-k kernels per shape, with different (and not
guaranteed stable) tie handling; exact-0.0 ReLU scores then diverge from the actor.
This binds the shared stable-sort Top-K (lower column wins ties, relative ascending,
-1 padded) into both the prefill and decode call sites and forces those branches so
the ``_C`` kernels are never used.
"""

import importlib

from rl_engine.backends.rocm.attention.deterministic_indexer_topk import (
    PROFILE,
    deterministic_topk_decode,
    deterministic_topk_prefill,
)

_SPARSE = "vllm.v1.attention.ops.rocm_aiter_mla_sparse"
_STATE = {"bound": False, "prefill": False, "decode": False}


def _mark(kind, logits, topk):
    if not _STATE[kind]:
        _STATE[kind] = True
        print("RLK_ROLLOUT_DETERMINISTIC_INDEXER_TOPK_EXECUTED", kind, tuple(logits.shape), topk, PROFILE, flush=True)


def _prefill(logits, row_starts, row_ends, indices, topk_tokens):
    # Writes row-relative indices directly; the call site has no further localize.
    deterministic_topk_prefill(logits, row_starts, row_ends, indices, topk_tokens)
    _mark("prefill", logits, topk_tokens)


def _decode(logits, next_n, seq_lens, topk_indices, topk_tokens, *, stable=False, values=None):
    if values is not None:
        raise ValueError("deterministic indexer Top-K does not return values")
    deterministic_topk_decode(logits, next_n, seq_lens, topk_indices, topk_tokens)
    _mark("decode", logits, topk_tokens)


def bind_rollout_deterministic_indexer_topk():
    """Idempotent, process-wide. Must run before the model's first forward."""
    if _STATE["bound"]:
        return
    sparse = importlib.import_module(_SPARSE)
    from vllm._aiter_ops import rocm_aiter_ops

    original_supported = rocm_aiter_ops.is_indexer_top_k_supported

    def supported(*, is_prefill, compress_ratio, num_rows, max_valid_seq_len=None, topk_tokens=None):
        if compress_ratio >= 1:
            return True
        return original_supported(
            is_prefill=is_prefill,
            compress_ratio=compress_ratio,
            num_rows=num_rows,
            max_valid_seq_len=max_valid_seq_len,
            topk_tokens=topk_tokens,
        )

    # The sparse module calls these through module / class attributes (function-local
    # `from vllm._aiter_ops import rocm_aiter_ops`), so patching them here is enough.
    sparse._launch_aiter_top_k_per_row_prefill = _prefill
    rocm_aiter_ops.indexer_top_k_decode = staticmethod(_decode)
    rocm_aiter_ops.is_indexer_top_k_supported = staticmethod(supported)
    rocm_aiter_ops.dsv4_indexer_prefers_native_top_k = staticmethod(lambda **_: False)
    _STATE["bound"] = True
    print("RLK_ROLLOUT_DETERMINISTIC_INDEXER_TOPK_BOUND", PROFILE, flush=True)
