# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant schedule for the pinned vLLM C128 two-stage KV compressor.

vLLM picks the stage-1 column split (num_splits -> constexpr HEAD_TILE) from the total
token count of the call, which changes Triton's softmax/sum reduction layout and so the
fp32 result by ULPs. We pin 16 splits (HEAD_TILE=32) for every call and send decode
tokens through the same two-stage kernel, so results do not depend on batch composition.
Used identically by the training actor and the vLLM rollout.
"""

import functools
import importlib
import inspect
import sys

_MODULE = "vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache"
_NUM_SPLITS = 16
_STATE = {"bound": False, "executed": False}


def _rebind_aliases(original, replacement, name):
    # Other modules may have imported the function by name before binding.
    for module in list(sys.modules.values()):
        try:
            if getattr(module, name, None) is original:
                setattr(module, name, replacement)
        except Exception:  # noqa: BLE001
            continue


def _make_pick(m):
    def pick(num_actual, compress_ratio, head_dim):
        if compress_ratio != 128 or head_dim != 512 or head_dim % _NUM_SPLITS != 0:
            raise RuntimeError(
                f"pinned C128 schedule supports compress_ratio=128, head_dim=512 only, "
                f"got {compress_ratio}, {head_dim}"
            )
        return _NUM_SPLITS

    return pick


def _make_dispatch(original):
    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        if args:
            raise TypeError("compress_norm_rope_store_two_stage_triton expects keyword arguments")
        # Two-stage kernel handles all tokens; decode tokens are not sent to single-pass.
        kwargs["num_decode_tokens"] = 0
        scratch = kwargs.get("compress_scratch")
        if scratch is None or scratch.shape[0] < kwargs["num_actual"] or scratch.shape[1] < kwargs["head_dim"]:
            raise RuntimeError("compress_scratch smaller than [num_actual, head_dim]")
        result = original(**kwargs)
        if not _STATE["executed"]:
            print("RLK_C128_SPLITS_PINNED_EXECUTED", _NUM_SPLITS, kwargs["num_actual"], flush=True)
            _STATE["executed"] = True
        return result

    dispatch._rlk_pinned = True
    return dispatch


def pin_c128_compressor_schedule():
    """Idempotent, process-wide. Fails closed on any unsupported vLLM source."""
    if _STATE["bound"]:
        return
    m = importlib.import_module(_MODULE)
    for name in (
        "_pick_compress_num_splits",
        "compress_norm_rope_store_two_stage_triton",
        "_launch_two_stage_sparse_attn_compressor",
    ):
        if not hasattr(m, name):
            raise RuntimeError(f"unsupported vLLM: {name} not found in {_MODULE}")
    # The launcher must look the picker up as a module global at call time.
    if "_pick_compress_num_splits(" not in inspect.getsource(m._launch_two_stage_sparse_attn_compressor):
        raise RuntimeError("unsupported vLLM: launcher does not call _pick_compress_num_splits")
    original_pick = m._pick_compress_num_splits
    original_dispatch = m.compress_norm_rope_store_two_stage_triton
    pick = _make_pick(m)
    dispatch = _make_dispatch(original_dispatch)
    m._pick_compress_num_splits = pick
    m.compress_norm_rope_store_two_stage_triton = dispatch
    _rebind_aliases(original_pick, pick, "_pick_compress_num_splits")
    _rebind_aliases(original_dispatch, dispatch, "compress_norm_rope_store_two_stage_triton")
    # A not-yet-imported compressor.py picks up the patched module attribute on import.
    _STATE["originals"] = (original_pick, original_dispatch)  # for tests
    _STATE["bound"] = True
    print("RLK_C128_SPLITS_PINNED", _NUM_SPLITS, __file__, flush=True)
