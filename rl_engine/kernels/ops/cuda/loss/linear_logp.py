# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from rl_engine.distributed.collectives import collective_for_group
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.matmul.det_gemm import (
    det_gemm_linear,
    det_gemm_linear_input_gradient,
    det_gemm_linear_weight_gradient,
)
from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
    _assert_global_targets_async,
    _deterministic_tp_all_reduce_,
    _merge_tp_local_logp,
    _require_distributed_initialized,
    _validate_even_tp_vocab_partition_local,
    _validate_global_targets,
    _validate_tp_targets_enabled,
    _validate_tp_vocab_partition_cached,
    chunked_linear_logp_backward,
    should_use_tensor_parallel_linear_logp,
    tensor_parallel_linear_logp,
    tensor_parallel_linear_logp_backward,
)
from rl_engine.utils.logger import logger

# Hidden-dim slice the SM90 forward streams per TMA load; D must be a multiple of
# it (mirrors `constexpr int BK` in csrc/cuda/fused_linear_logp_sm90.cu).
_SM90_BK = 32
_SM90_TP_PATH_LOGGED = False
_SM90_FUSED_BWD_PATH_LOGGED = False
_SM90_SAVE_PROBS_BF16_PATH_LOGGED = False
_SM90_SAVE_PROBS_BF16_SKIP_LOGGED = False
_SM90_FUSED_TILE_BF16_PATH_LOGGED = False

# Frozen reduction contract carried by the SM90 fused kernel. Any change to a
# bit-relevant constant (vocab tile width, K slab, split policy, reduction
# trees, transcendental choice, temperature/clamp placement) must bump this.
SM90_LINEAR_LOGP_CONTRACT_VERSION = "cuda-fused-linear-logp-sm90-contract-v1"
# Mirror of ``constexpr int N_SPLIT_CONTRACT`` in
# ``csrc/cuda/fused_linear_logp_sm90.cu``: the vocab-split count is a function
# of V only, never of batch shape, occupancy, or SM count.
SM90_N_SPLIT_CONTRACT = 64


def _env_disabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _fused_bwd_precision() -> str:
    return os.getenv("RL_KERNEL_LINEAR_LOGP_FUSED_BWD_PRECISION", "auto").strip().lower()


def _sm90_fused_backward_tf32_enabled() -> bool:
    return _fused_bwd_precision() != "fp32" and not _env_disabled(
        "RL_KERNEL_LINEAR_LOGP_FUSED_BWD_TF32"
    )


def _sm90_fused_backward_available() -> bool:
    return (
        _EXT_AVAILABLE
        and hasattr(_C, "fused_linear_logp_sm90_backward")
        and not _env_disabled("RL_KERNEL_LINEAR_LOGP_FUSED_BACKWARD")
    )


def _sm90_global_target_forward_available() -> bool:
    return _EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90_global_target")


def _sm90_save_probs_bf16_available() -> bool:
    return (
        _EXT_AVAILABLE
        and _env_enabled("RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16")
        and hasattr(_C, "linear_logp_probs_bf16_forward")
        and hasattr(_C, "linear_logp_probs_bf16_to_dlogits_")
    )


def _sm90_save_probs_bf16_tp_available() -> bool:
    return (
        _sm90_save_probs_bf16_available()
        and hasattr(_C, "linear_logp_local_probs_bf16_forward")
        and hasattr(_C, "linear_logp_local_probs_bf16_to_dlogits_")
    )


def _sm90_fused_tile_full_enabled() -> bool:
    value = os.getenv("RL_KERNEL_LINEAR_LOGP_FUSED_TILE_BWD_FULL", "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _sm90_fused_tile_bf16_forward_available(*, tp_path: bool) -> bool:
    if not (_EXT_AVAILABLE and _sm90_fused_tile_full_enabled()):
        return False
    if not _sm90_fused_backward_available():
        return False
    if not (
        hasattr(_C, "linear_logp_bf16_forward")
        and hasattr(_C, "linear_logp_logits_bf16_to_dlogits")
    ):
        return False
    if tp_path and not hasattr(_C, "linear_logp_local_bf16_forward"):
        return False
    return True


def _fused_tile_vocab_tile(vocab: int) -> int:
    raw = os.getenv("RL_KERNEL_LINEAR_LOGP_STREAMING_BWD_VOCAB_TILE")
    if raw is None or raw.strip() == "":
        return max(1, vocab)
    try:
        tile = int(raw)
    except ValueError:
        tile = vocab
    return max(1, min(tile, vocab))


def _log_sm90_save_probs_bf16_skip_once(
    *, tp_path: bool, details: dict[str, Any] | None = None, **conditions: bool
) -> None:
    global _SM90_SAVE_PROBS_BF16_SKIP_LOGGED
    if _SM90_SAVE_PROBS_BF16_SKIP_LOGGED or not _env_enabled(
        "RL_KERNEL_LINEAR_LOGP_SAVE_PROBS_BF16"
    ):
        return
    failed = [name for name, ok in conditions.items() if not ok]
    logger.info(
        "save-probs bf16 %slinear_logp fast path not selected; failed=%s conditions=%s details=%s",
        "tensor-parallel " if tp_path else "",
        failed,
        conditions,
        details,
    )
    _SM90_SAVE_PROBS_BF16_SKIP_LOGGED = True


def _sm90_linear_logp_backward(
    grad_logp: torch.Tensor,
    hidden_2d: torch.Tensor,
    weight: torch.Tensor,
    bias_t: torch.Tensor,
    target_1d: torch.Tensor,
    lse: torch.Tensor,
    *,
    has_bias: bool,
    lead_shape,
    hidden_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    bias_dtype,
    vocab_start_index: int = 0,
    tp_group: Any = None,
    compute_grad_hidden: bool = True,
    compute_grad_weight: bool = True,
    compute_grad_bias: bool = True,
):
    global _SM90_FUSED_BWD_PATH_LOGGED
    if not (compute_grad_hidden or compute_grad_weight or (has_bias and compute_grad_bias)):
        return None, None, None

    if _sm90_fused_backward_available():
        if not _SM90_FUSED_BWD_PATH_LOGGED:
            logger.info("Using fused_linear_logp_sm90_backward fast path.")
            _SM90_FUSED_BWD_PATH_LOGGED = True
        prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        enable_tf32 = _sm90_fused_backward_tf32_enabled()
        if enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        try:
            grad_hidden_t, grad_weight_t, grad_bias_t = _C.fused_linear_logp_sm90_backward(
                grad_logp,
                hidden_2d,
                weight,
                target_1d,
                lse,
                bias_t if has_bias else None,
                int(vocab_start_index),
                bool(compute_grad_hidden),
                bool(compute_grad_weight),
                bool(has_bias and compute_grad_bias),
                bool(tp_group is not None),
            )
        finally:
            if enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
        grad_hidden = None
        if compute_grad_hidden:
            if tp_group is not None:
                dist = _require_distributed_initialized()
                dist.all_reduce(grad_hidden_t, op=dist.ReduceOp.SUM, group=tp_group)
            grad_hidden = grad_hidden_t.to(hidden_dtype).reshape(
                (*tuple(lead_shape), hidden_2d.size(1))
            )
        grad_weight = grad_weight_t.to(weight_dtype) if compute_grad_weight else None
        grad_bias = grad_bias_t.to(bias_dtype) if has_bias and compute_grad_bias else None
        return grad_hidden, grad_weight, grad_bias

    if tp_group is not None:
        return tensor_parallel_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            lse,
            has_bias=has_bias,
            lead_shape=lead_shape,
            hidden_dtype=hidden_dtype,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            vocab_start_index=vocab_start_index,
            tp_group=tp_group,
            compute_grad_hidden=compute_grad_hidden,
            compute_grad_weight=compute_grad_weight,
            compute_grad_bias=compute_grad_bias,
        )
    return chunked_linear_logp_backward(
        grad_logp,
        hidden_2d,
        weight,
        target_1d,
        bias_t,
        has_bias=has_bias,
        lead_shape=lead_shape,
        hidden_dtype=hidden_dtype,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        compute_grad_hidden=compute_grad_hidden,
        compute_grad_weight=compute_grad_weight,
        compute_grad_bias=compute_grad_bias,
    )


class _LinearLogpSaveProbsBF16Function(torch.autograd.Function):
    """bf16 path that stores softmax probs instead of log-probs.

    It relies on cuBLAS for the linear GEMMs and uses the CUDA helper only for
    the softmax/probability transforms. When hidden needs gradients, backward
    also computes ``grad_hidden = dlogits @ weight``.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        logp, probs = _C.linear_logp_probs_bf16_forward(logits, target_1d, 0)
        ctx.save_for_backward(probs, hidden_2d, weight, target_1d)
        ctx.lead_shape = hidden.shape[:-1]
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        probs, hidden_2d, weight, target_1d = ctx.saved_tensors
        grad_hidden = None
        grad_weight = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            _C.linear_logp_probs_bf16_to_dlogits_(probs, target_1d, grad_logp, 0)
        if ctx.needs_input_grad[1]:
            grad_weight = probs.t().matmul(hidden_2d)
        if ctx.needs_input_grad[0]:
            grad_hidden = probs.matmul(weight).reshape((*tuple(ctx.lead_shape), weight.size(1)))
        return grad_hidden, grad_weight, None


class _TensorParallelLinearLogpSaveProbsBF16Function(torch.autograd.Function):
    """TP bf16 path that saves local softmax probs.

    Each rank materializes only its local vocab shard logits/probs. The forward
    merges local ``target_logit`` and ``lse`` through the shared TP fast merge;
    backward converts local probs to global dlogits using the saved local/global
    lse, computes the local shard's ``grad_weight`` with cuBLAS, and all-reduces
    ``grad_hidden`` when hidden participates in training.
    """

    @staticmethod
    def forward(
        ctx, hidden, lm_head_weight, target_ids, vocab_start_index, global_vocab_size, tp_group
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        if _validate_tp_targets_enabled():
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        local_target_logit, local_lse, probs = _C.linear_logp_local_probs_bf16_forward(
            logits,
            target_1d,
            vocab_start_index,
        )
        log_prob, global_lse = _merge_tp_local_logp(
            local_lse,
            local_target_logit,
            tp_group=tp_group,
        )

        ctx.save_for_backward(probs, hidden_2d, weight, target_1d, local_lse, global_lse)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        probs, hidden_2d, weight, target_1d, local_lse, global_lse = ctx.saved_tensors
        grad_hidden = None
        grad_weight = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            _C.linear_logp_local_probs_bf16_to_dlogits_(
                probs,
                target_1d,
                grad_logp,
                local_lse,
                global_lse,
                ctx.vocab_start_index,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = probs.t().matmul(hidden_2d)
        if ctx.needs_input_grad[0]:
            grad_hidden_2d = probs.matmul(weight)
            dist = _require_distributed_initialized()
            dist.all_reduce(grad_hidden_2d, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            grad_hidden = grad_hidden_2d.reshape((*tuple(ctx.lead_shape), weight.size(1)))
        return grad_hidden, grad_weight, None, None, None, None


class _LinearLogpFusedTileBF16Function(torch.autograd.Function):
    """cuBLAS bf16 logits forward + tiled backward without saving probs."""

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        logp, lse = _C.linear_logp_bf16_forward(logits, target_1d, 0)
        ctx.save_for_backward(logits, hidden_2d, weight, target_1d, lse)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        logits, hidden_2d, weight, target_1d, lse = ctx.saved_tensors
        compute_grad_hidden = ctx.needs_input_grad[0]
        compute_grad_weight = ctx.needs_input_grad[1]
        grad_hidden_2d = None
        grad_weight = None
        if compute_grad_hidden or compute_grad_weight:
            n_tokens, vocab = logits.shape
            tile_v = _fused_tile_vocab_tile(vocab)
            use_inplace_logits = tile_v >= vocab
            dlogits_workspace = None
            if not use_inplace_logits:
                dlogits_workspace = torch.empty(
                    (n_tokens, min(tile_v, vocab)),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            grad_hidden_initialized = False
            if compute_grad_hidden:
                grad_hidden_2d = torch.empty_like(hidden_2d)
            if compute_grad_weight:
                grad_weight = torch.empty_like(weight)
            for v0 in range(0, vocab, tile_v):
                vc = min(tile_v, vocab - v0)
                logits_tile = logits.narrow(1, v0, vc)
                if use_inplace_logits:
                    dlogits_tile = logits_tile
                else:
                    dlogits_tile = (
                        dlogits_workspace
                        if vc == dlogits_workspace.size(1)
                        else torch.empty((n_tokens, vc), device=logits.device, dtype=logits.dtype)
                    )
                _C.linear_logp_logits_bf16_to_dlogits(
                    logits_tile,
                    dlogits_tile,
                    target_1d,
                    grad_logp,
                    lse,
                    v0,
                )
                weight_tile = weight.narrow(0, v0, vc)
                if compute_grad_hidden:
                    if not grad_hidden_initialized:
                        torch.mm(dlogits_tile, weight_tile, out=grad_hidden_2d)
                        grad_hidden_initialized = True
                    else:
                        torch.addmm(
                            grad_hidden_2d,
                            dlogits_tile,
                            weight_tile,
                            out=grad_hidden_2d,
                        )
                if compute_grad_weight:
                    torch.mm(
                        dlogits_tile.transpose(0, 1),
                        hidden_2d,
                        out=grad_weight.narrow(0, v0, vc),
                    )
        grad_hidden = (
            None
            if grad_hidden_2d is None
            else grad_hidden_2d.to(ctx.hidden_dtype).reshape(
                (*tuple(ctx.lead_shape), hidden_2d.size(1))
            )
        )
        if grad_weight is not None:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None


class _TensorParallelLinearLogpFusedTileBF16Function(torch.autograd.Function):
    """TP cuBLAS bf16 logits forward + tiled local backward."""

    @staticmethod
    def forward(
        ctx, hidden, lm_head_weight, target_ids, vocab_start_index, global_vocab_size, tp_group
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        if _validate_tp_targets_enabled():
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        logits = torch.nn.functional.linear(hidden_2d, weight, None)
        local_target_logit, local_lse = _C.linear_logp_local_bf16_forward(
            logits,
            target_1d,
            vocab_start_index,
        )
        log_prob, global_lse = _merge_tp_local_logp(
            local_lse,
            local_target_logit,
            tp_group=tp_group,
        )

        ctx.save_for_backward(logits, hidden_2d, weight, target_1d, global_lse)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        logits, hidden_2d, weight, target_1d, global_lse = ctx.saved_tensors
        compute_grad_hidden = ctx.needs_input_grad[0]
        compute_grad_weight = ctx.needs_input_grad[1]
        grad_hidden_2d = None
        grad_weight = None
        if compute_grad_hidden or compute_grad_weight:
            n_tokens, vocab = logits.shape
            tile_v = _fused_tile_vocab_tile(vocab)
            use_inplace_logits = tile_v >= vocab
            dlogits_workspace = None
            if not use_inplace_logits:
                dlogits_workspace = torch.empty(
                    (n_tokens, min(tile_v, vocab)),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            grad_hidden_initialized = False
            if compute_grad_hidden:
                grad_hidden_2d = torch.empty_like(hidden_2d)
            if compute_grad_weight:
                grad_weight = torch.empty_like(weight)
            for v0 in range(0, vocab, tile_v):
                vc = min(tile_v, vocab - v0)
                logits_tile = logits.narrow(1, v0, vc)
                if use_inplace_logits:
                    dlogits_tile = logits_tile
                else:
                    dlogits_tile = (
                        dlogits_workspace
                        if vc == dlogits_workspace.size(1)
                        else torch.empty((n_tokens, vc), device=logits.device, dtype=logits.dtype)
                    )
                _C.linear_logp_logits_bf16_to_dlogits(
                    logits_tile,
                    dlogits_tile,
                    target_1d,
                    grad_logp,
                    global_lse,
                    ctx.vocab_start_index + v0,
                )
                weight_tile = weight.narrow(0, v0, vc)
                if compute_grad_hidden:
                    if not grad_hidden_initialized:
                        torch.mm(dlogits_tile, weight_tile, out=grad_hidden_2d)
                        grad_hidden_initialized = True
                    else:
                        torch.addmm(
                            grad_hidden_2d,
                            dlogits_tile,
                            weight_tile,
                            out=grad_hidden_2d,
                        )
                if compute_grad_weight:
                    torch.mm(
                        dlogits_tile.transpose(0, 1),
                        hidden_2d,
                        out=grad_weight.narrow(0, v0, vc),
                    )
        grad_hidden = None
        if grad_hidden_2d is not None:
            dist = _require_distributed_initialized()
            dist.all_reduce(grad_hidden_2d, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            grad_hidden = grad_hidden_2d.to(ctx.hidden_dtype).reshape(
                (*tuple(ctx.lead_shape), hidden_2d.size(1))
            )
        if grad_weight is not None:
            grad_weight = grad_weight.to(ctx.weight_dtype)
        return grad_hidden, grad_weight, None, None, None, None


def _sm90_support_details(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> dict[str, Any]:
    """Per-condition diagnostics for the bf16 TMA+MMA forward."""
    same_device = hidden.device == lm_head_weight.device
    cc_major = None
    cc_minor = None
    if hidden.is_cuda:
        try:
            cc_major, cc_minor = torch.cuda.get_device_capability(hidden.device)
        except Exception:
            pass
    return {
        "hidden_is_cuda": hidden.is_cuda,
        "weight_is_cuda": lm_head_weight.is_cuda,
        "same_device": same_device,
        "cc_major_9": cc_major == 9,
        "hidden_bf16": hidden.dtype == torch.bfloat16,
        "weight_bf16": lm_head_weight.dtype == torch.bfloat16,
        "hidden_dim_multiple_32": hidden.size(-1) % _SM90_BK == 0,
        "hidden_shape": tuple(hidden.shape),
        "hidden_dtype": str(hidden.dtype),
        "hidden_device": str(hidden.device),
        "hidden_requires_grad": bool(hidden.requires_grad),
        "weight_shape": tuple(lm_head_weight.shape),
        "weight_dtype": str(lm_head_weight.dtype),
        "weight_device": str(lm_head_weight.device),
        "weight_requires_grad": bool(lm_head_weight.requires_grad),
        "device_capability": None if cc_major is None else (cc_major, cc_minor),
    }


def _sm90_supported(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> bool:
    """Whether the bf16 TMA+MMA forward can run these inputs directly."""
    details = _sm90_support_details(hidden, lm_head_weight)
    return all(
        bool(details[name])
        for name in (
            "hidden_is_cuda",
            "weight_is_cuda",
            "same_device",
            "cc_major_9",
            "hidden_bf16",
            "weight_bf16",
            "hidden_dim_multiple_32",
        )
    )


def _fallback_op():
    """Portable op for inputs the SM90 forward cannot take (fp32/fp16, or a hidden
    dim not divisible by the kernel's K slice). Prefers Triton, else native."""
    try:
        from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

        return TritonLinearLogpOp()
    except Exception:  # pragma: no cover - Triton missing
        from rl_engine.kernels.ops.pytorch.loss.linear_logp import NativeLinearLogpOp

        return NativeLinearLogpOp()


class _FusedLinearLogpSM90Function(torch.autograd.Function):
    """SM90 TMA+WGMMA fused forward + CUDA fused backward when available.

    The forward calls the compiled ``_C.fused_linear_logp_sm90`` kernel (logits
    never materialized). The fast backward is a CUDA extension path that
    materializes local dlogits once and uses GEMMs for the linear gradients,
    falling back to the deterministic chunked path when unavailable.
    """

    @staticmethod
    def forward(ctx, hidden, lm_head_weight, bias, target_ids):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.int32).contiguous()
        )
        logp, lse = _C.fused_linear_logp_sm90(hidden_2d, weight, target_1d, bias)

        ctx.save_for_backward(
            hidden_2d,
            weight,
            bias if bias is not None else hidden_2d,
            target_1d,
            lse,
        )
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d, lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _sm90_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            lse,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.needs_input_grad[2],
        )
        # Inputs: hidden, lm_head_weight, bias, target_ids.
        return grad_hidden, grad_weight, grad_bias, None


class _FusedTensorParallelLinearLogpSM90Function(torch.autograd.Function):
    """SM90 local-shard forward with tensor-parallel logsumexp reduction.

    Each rank runs the fused SM90 kernel over its local vocab shard to get local
    log-sum-exp and the owned target logit. TP ranks then merge those states into
    the global selected log-prob. Backward intentionally reuses the existing
    the global selected log-prob. Backward uses the CUDA fused backward fast path
    when available and otherwise falls back to the portable TP helper.
    """

    @staticmethod
    def forward(
        ctx,
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        vocab_start_index,
        global_vocab_size,
        tp_group,
    ):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = lm_head_weight.contiguous()
        target_1d = (
            target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
        )
        bias_t = bias.contiguous() if bias is not None else hidden_2d
        vocab_start_index = int(vocab_start_index)
        global_vocab_size = _validate_tp_vocab_partition_cached(
            tp_group=tp_group,
            device=hidden_2d.device,
            vocab_start_index=vocab_start_index,
            local_vocab_size=weight.size(0),
            global_vocab_size=global_vocab_size,
        )
        if _validate_tp_targets_enabled():
            _validate_global_targets(target_1d, global_vocab_size, tp_group)

        if _sm90_global_target_forward_available():
            local_target_logit, local_lse = _C.fused_linear_logp_sm90_global_target(
                hidden_2d,
                weight,
                target_1d,
                bias,
                vocab_start_index,
            )
        else:
            local_vocab = weight.size(0)
            local_target = target_1d - vocab_start_index
            owns_target = (local_target >= 0) & (local_target < local_vocab)
            kernel_target = torch.where(
                local_target >= 0, local_target, torch.zeros_like(local_target)
            )
            kernel_target = torch.where(
                kernel_target < local_vocab, kernel_target, torch.zeros_like(kernel_target)
            )
            kernel_target = kernel_target.to(torch.int32).contiguous()

            local_logp, local_lse = _C.fused_linear_logp_sm90(
                hidden_2d, weight, kernel_target, bias
            )
            local_target_logit = torch.where(
                owns_target, local_logp + local_lse, torch.zeros_like(local_lse)
            )
        log_prob, global_lse = _merge_tp_local_logp(
            local_lse,
            local_target_logit,
            tp_group=tp_group,
        )

        ctx.save_for_backward(hidden_2d, weight, bias_t, target_1d, global_lse)
        ctx.has_bias = bias is not None
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = lm_head_weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.vocab_start_index = vocab_start_index
        ctx.tp_group = tp_group
        return log_prob.reshape(hidden.shape[:-1])

    @staticmethod
    def backward(ctx, grad_logp):
        hidden_2d, weight, bias_t, target_1d, global_lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _sm90_linear_logp_backward(
            grad_logp,
            hidden_2d,
            weight,
            bias_t,
            target_1d,
            global_lse,
            has_bias=ctx.has_bias,
            lead_shape=ctx.lead_shape,
            hidden_dtype=ctx.hidden_dtype,
            weight_dtype=ctx.weight_dtype,
            bias_dtype=ctx.bias_dtype,
            vocab_start_index=ctx.vocab_start_index,
            tp_group=ctx.tp_group,
            compute_grad_hidden=ctx.needs_input_grad[0],
            compute_grad_weight=ctx.needs_input_grad[1],
            compute_grad_bias=ctx.needs_input_grad[2],
        )
        return grad_hidden, grad_weight, grad_bias, None, None, None, None


def _sm90_tensor_parallel_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    tp_group: Any,
    vocab_start_index: int,
    global_vocab_size: Optional[int],
) -> torch.Tensor:
    global _SM90_TP_PATH_LOGGED
    if not _SM90_TP_PATH_LOGGED:
        logger.info("Using fused_linear_logp_sm90 tensor-parallel local-shard path.")
        _SM90_TP_PATH_LOGGED = True
    return _FusedTensorParallelLinearLogpSM90Function.apply(
        hidden,
        lm_head_weight,
        bias,
        target_ids,
        int(vocab_start_index),
        None if global_vocab_size is None else int(global_vocab_size),
        tp_group,
    )


class FusedLinearLogpSM90Op:
    """SM90 (Hopper) TMA+WGMMA fused linear log-prob.

    Computes ``log_softmax(hidden @ W^T + b)[target]`` without materializing the
    ``[N, V]`` logits. Requires the extension built with ``KERNEL_ALIGN_FORCE_SM90=1``
    on an SM90 device; bfloat16 hidden/weight only.
    """

    def __init__(self) -> None:
        if not _EXT_AVAILABLE or not hasattr(_C, "fused_linear_logp_sm90"):
            raise RuntimeError(
                "fused_linear_logp_sm90 is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C.fused_linear_logp_sm90 kernel.")

    def __call__(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self.apply(
            hidden,
            lm_head_weight,
            target_ids,
            bias,
            tp_group=tp_group,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
        )

    def apply(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        *,
        tp_group: Any = None,
        vocab_start_index: int = 0,
        global_vocab_size: Optional[int] = None,
    ) -> torch.Tensor:
        global _SM90_SAVE_PROBS_BF16_PATH_LOGGED, _SM90_FUSED_TILE_BF16_PATH_LOGGED
        if lm_head_weight.size(-1) != hidden.size(-1):
            raise ValueError(
                f"hidden dim {hidden.size(-1)} must match lm_head_weight dim "
                f"{lm_head_weight.size(-1)}"
            )
        if lm_head_weight.device != hidden.device:
            raise ValueError(
                f"lm_head_weight device {lm_head_weight.device} must match hidden "
                f"device {hidden.device}"
            )
        n_tokens = hidden.numel() // hidden.size(-1)
        if target_ids.numel() != n_tokens:
            raise ValueError(
                f"target_ids must have one id per token: expected {n_tokens}, "
                f"got {target_ids.numel()}"
            )
        if bias is not None:
            if bias.numel() != lm_head_weight.size(0):
                raise ValueError(
                    f"bias must have V={lm_head_weight.size(0)} elements, got {bias.numel()}"
                )
            if bias.device != hidden.device:
                raise ValueError(
                    f"bias device {bias.device} must match hidden device {hidden.device}"
                )
        if should_use_tensor_parallel_linear_logp(
            tp_group,
            int(vocab_start_index),
            global_vocab_size,
            lm_head_weight.size(0),
        ):
            save_probs_tp_available = _sm90_save_probs_bf16_tp_available()
            sm90_supported = _sm90_supported(hidden, lm_head_weight)
            bias_none = bias is None
            hidden_no_grad = not hidden.requires_grad
            hidden_requires_grad = hidden.requires_grad
            weight_requires_grad = lm_head_weight.requires_grad
            any_grad_required = hidden_requires_grad or weight_requires_grad
            output_only_grad = hidden_no_grad and weight_requires_grad
            if save_probs_tp_available and sm90_supported and bias_none and output_only_grad:
                if not _SM90_SAVE_PROBS_BF16_PATH_LOGGED:
                    logger.info(
                        "Using save-probs bf16 output-only tensor-parallel linear_logp fast path."
                    )
                    _SM90_SAVE_PROBS_BF16_PATH_LOGGED = True
                return _TensorParallelLinearLogpSaveProbsBF16Function.apply(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    int(vocab_start_index),
                    None if global_vocab_size is None else int(global_vocab_size),
                    tp_group,
                )
            _log_sm90_save_probs_bf16_skip_once(
                tp_path=True,
                details=_sm90_support_details(hidden, lm_head_weight),
                save_probs_tp_available=save_probs_tp_available,
                sm90_supported=sm90_supported,
                bias_none=bias_none,
                hidden_no_grad=hidden_no_grad,
                hidden_requires_grad=hidden_requires_grad,
                weight_requires_grad=weight_requires_grad,
                any_grad_required=any_grad_required,
                output_only_grad=output_only_grad,
            )
            fused_tile_forward_available = _sm90_fused_tile_bf16_forward_available(tp_path=True)
            if fused_tile_forward_available and sm90_supported and bias_none and any_grad_required:
                if not _SM90_FUSED_TILE_BF16_PATH_LOGGED:
                    logger.info(
                        "Using fused-tile bf16 %stensor-parallel linear_logp fast path.",
                        "output-only " if hidden_no_grad else "full-gradient ",
                    )
                    _SM90_FUSED_TILE_BF16_PATH_LOGGED = True
                return _TensorParallelLinearLogpFusedTileBF16Function.apply(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    int(vocab_start_index),
                    None if global_vocab_size is None else int(global_vocab_size),
                    tp_group,
                )
            if sm90_supported:
                return _sm90_tensor_parallel_linear_logp(
                    hidden,
                    lm_head_weight,
                    target_ids,
                    bias,
                    tp_group=tp_group,
                    vocab_start_index=vocab_start_index,
                    global_vocab_size=global_vocab_size,
                )
            return tensor_parallel_linear_logp(
                hidden,
                lm_head_weight,
                target_ids,
                bias,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                global_vocab_size=global_vocab_size,
            )
        if not _sm90_supported(hidden, lm_head_weight):
            return _fallback_op()(hidden, lm_head_weight, target_ids, bias)
        vocab = lm_head_weight.size(0)
        if bool(((target_ids < 0) | (target_ids >= vocab)).any()):
            t_min, t_max = int(target_ids.min()), int(target_ids.max())
            raise ValueError(
                f"target_ids out of range: expected [0, {vocab - 1}], got [{t_min}, {t_max}]. "
                "Mask or filter padding / ignore-index values (e.g. -100) before this op."
            )
        save_probs_available = _sm90_save_probs_bf16_available()
        bias_none = bias is None
        hidden_no_grad = not hidden.requires_grad
        hidden_requires_grad = hidden.requires_grad
        weight_requires_grad = lm_head_weight.requires_grad
        any_grad_required = hidden_requires_grad or weight_requires_grad
        if save_probs_available and bias_none and any_grad_required:
            if not _SM90_SAVE_PROBS_BF16_PATH_LOGGED:
                logger.info(
                    "Using save-probs bf16 %slinear_logp fast path.",
                    "output-only " if hidden_no_grad else "full-gradient ",
                )
                _SM90_SAVE_PROBS_BF16_PATH_LOGGED = True
            return _LinearLogpSaveProbsBF16Function.apply(hidden, lm_head_weight, target_ids)
        _log_sm90_save_probs_bf16_skip_once(
            tp_path=False,
            details=_sm90_support_details(hidden, lm_head_weight),
            save_probs_available=save_probs_available,
            bias_none=bias_none,
            hidden_no_grad=hidden_no_grad,
            hidden_requires_grad=hidden_requires_grad,
            weight_requires_grad=weight_requires_grad,
            any_grad_required=any_grad_required,
        )
        fused_tile_forward_available = _sm90_fused_tile_bf16_forward_available(tp_path=False)
        if fused_tile_forward_available and bias_none and any_grad_required:
            if not _SM90_FUSED_TILE_BF16_PATH_LOGGED:
                logger.info(
                    "Using fused-tile bf16 %slinear_logp fast path.",
                    "output-only " if hidden_no_grad else "full-gradient ",
                )
                _SM90_FUSED_TILE_BF16_PATH_LOGGED = True
            return _LinearLogpFusedTileBF16Function.apply(hidden, lm_head_weight, target_ids)
        return _FusedLinearLogpSM90Function.apply(hidden, lm_head_weight, bias, target_ids)


class _StrictLinearLogpAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, target, temperature, real_vocab_size):
        hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
        weight = weight.contiguous()
        target = target.reshape(-1).to(device=hidden.device, dtype=torch.long).contiguous()
        bias_arg = bias.contiguous() if bias is not None else None
        logp, lse = _C.fused_linear_logp_sm90(
            hidden_2d, weight, target, bias_arg, temperature, False, int(real_vocab_size)
        )
        ctx.save_for_backward(
            hidden_2d,
            weight,
            bias_arg if bias_arg is not None else hidden_2d.new_empty(0),
            target,
            lse,
            temperature if temperature is not None else hidden_2d.new_empty(0),
        )
        ctx.has_bias = bias_arg is not None
        ctx.real_vocab_size = int(real_vocab_size)
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype = hidden.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return logp.reshape(ctx.lead_shape), lse.reshape(ctx.lead_shape)

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        hidden, weight, bias, target, lse, temperature = ctx.saved_tensors
        if grad_logp is None and grad_lse is None:
            return (None, None, None, None, None, None)
        temp = temperature if temperature.numel() else None
        with torch.no_grad():
            # Keep the backward GEMMs dtype-consistent. The fused forward
            # contract is BF16 input/output-weight, while its log-sum-exp
            # statistics and the autograd upstream are FP32.
            hidden_f = hidden.float() if hidden.dtype != torch.float32 else hidden
            weight_f = weight.float() if weight.dtype != torch.float32 else weight
            bias_f = (
                bias.float()
                if ctx.has_bias and bias.dtype != torch.float32
                else (bias if ctx.has_bias else None)
            )
            logits = torch.nn.functional.linear(hidden_f, weight_f, bias_f)
            if temp is not None:
                logits = logits / temp.reshape(-1, 1)
            if ctx.real_vocab_size >= 0 and ctx.real_vocab_size < logits.size(1):
                columns = torch.arange(logits.size(1), device=logits.device)
                logits = logits.masked_fill(
                    columns[None, :] >= ctx.real_vocab_size,
                    float("-inf"),
                )
            probs = torch.exp(logits - lse.reshape(-1, 1).float())
            logp_grad = (
                torch.zeros_like(lse) if grad_logp is None else grad_logp.reshape(-1).float()
            )
            lse_grad = torch.zeros_like(lse) if grad_lse is None else grad_lse.reshape(-1).float()
            # d(logp) / d(logits) = onehot(target) - softmax(logits), while
            # d(lse) / d(logits) = softmax(logits).
            dlogits = probs * (lse_grad - logp_grad).reshape(-1, 1)
            rows = torch.arange(target.numel(), device=target.device)
            dlogits[rows, target] += logp_grad
            if temp is not None:
                dlogits = dlogits / temp.reshape(-1, 1)
            grad_hidden = dlogits.matmul(weight_f)
            grad_weight = dlogits.transpose(0, 1).matmul(hidden_f)
            grad_bias = dlogits.sum(0) if ctx.has_bias else None
        return (
            grad_hidden.reshape((*tuple(ctx.lead_shape), weight.size(1))).to(ctx.hidden_dtype),
            grad_weight.to(ctx.weight_dtype),
            None if grad_bias is None else grad_bias.to(ctx.bias_dtype),
            None,
            None,
            None,
        )


STRICT_LINEAR_LOGP_CONTRACT_VERSION = "cuda-det-gemm-linear-logp-sm90-contract-v2"
_STRICT_LOGP_CANONICAL_TP_ENV = "RL_KERNEL_STRICT_CANONICAL_TP"
_STRICT_LOGP_CANONICAL_VOCAB_ENV = "RL_KERNEL_STRICT_CANONICAL_VOCAB_SIZE"


def _strict_logp_vocab_summaries(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start: int,
    global_vocab: int,
    tp_group: Any,
    top_p_args: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute summaries in canonical virtual-TP shards.

    A TP2 training rank, for example, emits two summaries when paired with a
    TP4 rollout.  Both sides then merge the same four contiguous vocab shards
    in the same order, while the TP4/CP2 reference path remains one summary per
    rank with no extra launches.
    """

    dist = _require_distributed_initialized()
    physical_tp = dist.get_world_size(group=tp_group)
    physical_rank = dist.get_rank(group=tp_group)
    raw_canonical_tp = os.getenv(_STRICT_LOGP_CANONICAL_TP_ENV, "").strip()
    canonical_tp = physical_tp if not raw_canonical_tp else int(raw_canonical_tp)
    if canonical_tp < physical_tp or canonical_tp % physical_tp:
        raise ValueError(
            f"{_STRICT_LOGP_CANONICAL_TP_ENV}={canonical_tp} must be a positive "
            f"multiple of physical TP={physical_tp}"
        )
    raw_canonical_vocab = os.getenv(_STRICT_LOGP_CANONICAL_VOCAB_ENV, "").strip()
    canonical_global_vocab = (
        global_vocab if not raw_canonical_vocab else int(raw_canonical_vocab)
    )
    if canonical_global_vocab < global_vocab:
        raise ValueError(
            f"{_STRICT_LOGP_CANONICAL_VOCAB_ENV}={canonical_global_vocab} must be "
            f"at least the physical padded vocab {global_vocab}"
        )
    if canonical_global_vocab % canonical_tp:
        raise ValueError(
            f"canonical vocab {canonical_global_vocab} must divide evenly across "
            f"canonical TP {canonical_tp}"
        )
    summaries_per_rank = canonical_tp // physical_tp
    canonical_vocab = canonical_global_vocab // canonical_tp
    canonical_local_vocab = summaries_per_rank * canonical_vocab
    layout_matches = (
        global_vocab == canonical_global_vocab
        and logits.size(1) == canonical_local_vocab
        and int(vocab_start) == physical_rank * canonical_local_vocab
    )
    summary_source = logits
    summary_source_start = int(vocab_start)
    if not layout_matches:
        if physical_tp == 1:
            global_logits = logits
        else:
            collective = collective_for_group(
                tp_group,
                min_size_bytes=logits.numel() * logits.element_size(),
                device=logits.device,
            )
            if collective is None:
                raise RuntimeError(
                    "canonical TP logp redistribution requires an RL-Kernel collective"
                )
            gathered = collective.all_gather(
                logits,
                validate_signature=False,
            ).reshape(physical_tp, logits.size(0), logits.size(1))
            global_logits = gathered.permute(1, 0, 2).reshape(logits.size(0), -1)
        if global_logits.size(1) != global_vocab:
            raise RuntimeError(
                "physical TP logits do not reconstruct the declared padded vocab: "
                f"{global_logits.size(1)} != {global_vocab}"
            )
        if canonical_global_vocab > global_vocab:
            global_logits = torch.nn.functional.pad(
                global_logits,
                (0, canonical_global_vocab - global_vocab),
                value=float("-inf"),
            )
        summary_source_start = physical_rank * canonical_local_vocab
        summary_source = global_logits.narrow(
            1,
            summary_source_start,
            canonical_local_vocab,
        ).contiguous()

    local_targets = []
    local_lses = []
    for summary in range(summaries_per_rank):
        begin = summary * canonical_vocab
        end = begin + canonical_vocab
        summary_logits = summary_source[:, begin:end].contiguous()
        summary_vocab_start = summary_source_start + begin
        if top_p_args is None:
            local_target, local_lse = _C.linear_logp_local_bf16_forward(
                summary_logits,
                target,
                summary_vocab_start,
            )
        else:
            replay_ids, replay_logprobs, temperature = top_p_args
            local_target, local_lse = _C.linear_logp_top_p_local_bf16_forward(
                summary_logits,
                target,
                replay_ids,
                replay_logprobs,
                temperature,
                summary_vocab_start,
            )
        local_targets.append(local_target)
        local_lses.append(local_lse)

    if summaries_per_rank == 1:
        return local_targets[0], local_lses[0]
    return torch.stack(local_targets, dim=0), torch.stack(local_lses, dim=0)


def _strict_tp_run(
    hidden,
    weight,
    bias,
    target,
    *,
    vocab_start,
    global_vocab,
    real_vocab,
    temperature,
    tp_group,
):
    required = (
        "linear_logp_local_bf16_forward",
        "linear_logp_logits_bf16_to_dlogits",
    )
    missing = [n for n in required if not (_EXT_AVAILABLE and hasattr(_C, n))]
    if missing:
        raise RuntimeError(f"strict TP linear_logp missing CUDA symbols: {missing}")
    hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
    weight = weight.contiguous()
    target = target.reshape(-1).to(device=hidden.device, dtype=torch.long).contiguous()
    if hidden_2d.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("strict TP linear_logp requires BF16 hidden and weight")
    global_vocab = _validate_even_tp_vocab_partition_local(
        tp_group=tp_group,
        vocab_start_index=int(vocab_start),
        local_vocab_size=weight.size(0),
        global_vocab_size=int(global_vocab),
    )
    if not 0 < int(real_vocab) <= global_vocab:
        raise ValueError(f"invalid real_vocab_size={real_vocab} for padded vocab={global_vocab}")
    if _validate_tp_targets_enabled():
        # Opt-in diagnostics retain the synchronized cross-rank checks. The
        # cached contiguous [0, V) partition plus a valid target already prove
        # unique ownership during normal execution.
        _validate_global_targets(target, int(real_vocab), tp_group)
        dist = _require_distributed_initialized()
        owners = ((target >= int(vocab_start)) & (target < int(vocab_start) + weight.size(0))).to(
            torch.int32
        )
        dist.all_reduce(owners, op=dist.ReduceOp.SUM, group=tp_group)
        if bool((owners != 1).any().item()):
            raise ValueError("each selected target must have exactly one TP LM-head owner")
    else:
        _assert_global_targets_async(target, int(real_vocab))
    logits = det_gemm_linear(hidden_2d, weight)
    temp = hidden_2d.new_empty(0, dtype=torch.float32)
    if bias is not None:
        logits = (logits.float() + bias.float().reshape(1, -1)).to(torch.bfloat16)
    if temperature is not None:
        temp = temperature.to(device=hidden.device, dtype=torch.float32).reshape(-1)
        if temp.numel() == 1:
            temp = temp.expand(hidden_2d.size(0)).contiguous()
        if temp.numel() != hidden_2d.size(0):
            raise ValueError("temperature must be positive and scalar or per-token")
        torch._assert_async(
            (temp > 0).all(),
            "temperature must be positive and scalar or per-token",
        )
        logits = (logits.float() / temp.reshape(-1, 1)).to(torch.bfloat16)
    local_real = max(0, min(weight.size(0), int(real_vocab) - int(vocab_start)))
    if local_real < weight.size(0):
        logits = logits.clone()
        logits[:, local_real:] = float("-inf")
    logits = logits.contiguous()
    local_target, local_lse = _strict_logp_vocab_summaries(
        logits,
        target,
        vocab_start=int(vocab_start),
        global_vocab=global_vocab,
        tp_group=tp_group,
    )
    logp, lse = _merge_tp_local_logp(local_lse, local_target, tp_group=tp_group)
    return logp, lse, hidden_2d, weight, target, logits, temp


def _strict_logp_from_local_logits_tp_run(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    global_vocab_size: int,
    real_vocab_size: int = -1,
    temperature: Optional[torch.Tensor] = None,
    tp_group: Any,
):
    """Raw strict TP logp over a previously materialized deterministic LM head."""

    required = "linear_logp_local_bf16_forward"
    if not (_EXT_AVAILABLE and hasattr(_C, required)):
        raise RuntimeError(f"strict TP logits logp missing CUDA symbol: {required}")
    if local_logits.ndim != 2 or local_logits.dtype != torch.bfloat16:
        raise TypeError("strict TP logits logp requires 2-D BF16 local logits")
    if not local_logits.is_cuda or not local_logits.is_contiguous():
        raise ValueError("strict TP logits logp requires contiguous CUDA logits")

    vocab_start = int(vocab_start_index)
    global_vocab = _validate_even_tp_vocab_partition_local(
        tp_group=tp_group,
        vocab_start_index=vocab_start,
        local_vocab_size=local_logits.size(1),
        global_vocab_size=int(global_vocab_size),
    )
    real_vocab = global_vocab if int(real_vocab_size) < 0 else int(real_vocab_size)
    if not 0 < real_vocab <= global_vocab:
        raise ValueError(f"invalid real_vocab_size={real_vocab} for padded vocab={global_vocab}")
    target = target_ids.reshape(-1).to(device=local_logits.device, dtype=torch.long).contiguous()
    if target.numel() != local_logits.size(0):
        raise ValueError("target_ids must contain one id per local-logits row")
    _assert_global_targets_async(target, real_vocab)

    logits = local_logits
    temp = local_logits.new_empty(0, dtype=torch.float32)
    if temperature is not None:
        temp = temperature.to(device=logits.device, dtype=torch.float32).reshape(-1)
        if temp.numel() == 1:
            temp = temp.expand(logits.size(0)).contiguous()
        if temp.numel() != logits.size(0):
            raise ValueError("temperature must be positive and scalar or per-token")
        torch._assert_async(
            (temp > 0).all(),
            "temperature must be positive and scalar or per-token",
        )
        logits = (logits.float() / temp.reshape(-1, 1)).to(torch.bfloat16)
    local_real = max(0, min(logits.size(1), real_vocab - vocab_start))
    if local_real < logits.size(1):
        logits = logits.clone()
        logits[:, local_real:] = float("-inf")
    logits = logits.contiguous()
    local_target, local_lse = _strict_logp_vocab_summaries(
        logits,
        target,
        vocab_start=vocab_start,
        global_vocab=global_vocab,
        tp_group=tp_group,
    )
    logp, lse = _merge_tp_local_logp(local_lse, local_target, tp_group=tp_group)
    lead_shape = target_ids.shape
    return logp, lse, target, logits, temp, lead_shape


class _StrictLocalLogitsTPAutograd(torch.autograd.Function):
    """Differentiate strict selected-logp through a reused TP LM-head result."""

    @staticmethod
    def forward(
        ctx,
        local_logits,
        target_ids,
        vocab_start_index,
        global_vocab_size,
        real_vocab_size,
        temperature,
        tp_group,
    ):
        logp, lse, target, logits, temp, lead_shape = _strict_logp_from_local_logits_tp_run(
            local_logits,
            target_ids,
            vocab_start_index=vocab_start_index,
            global_vocab_size=global_vocab_size,
            real_vocab_size=real_vocab_size,
            temperature=temperature,
            tp_group=tp_group,
        )
        ctx.save_for_backward(target, logits, lse, temp)
        ctx.vocab_start = int(vocab_start_index)
        ctx.input_shape = local_logits.shape
        ctx.set_materialize_grads(False)
        return logp.reshape(lead_shape), lse.reshape(lead_shape)

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        target, logits, lse, temp = ctx.saved_tensors
        if grad_logp is None and grad_lse is None:
            return (None, None, None, None, None, None, None)
        logp_grad = torch.zeros_like(lse) if grad_logp is None else grad_logp.reshape(-1).float()
        dlogits = torch.empty_like(logits)
        _C.linear_logp_logits_bf16_to_dlogits(
            logits, dlogits, target, logp_grad, lse, ctx.vocab_start
        )
        if grad_lse is not None:
            probs = torch.exp(logits.float() - lse.reshape(-1, 1))
            dlogits = (dlogits.float() + probs * grad_lse.reshape(-1, 1).float()).to(torch.bfloat16)
        if temp.numel():
            dlogits = (dlogits.float() / temp.reshape(-1, 1)).to(torch.bfloat16)
        return dlogits.reshape(ctx.input_shape).contiguous(), None, None, None, None, None, None


def sm90_deterministic_logp_from_local_logits_tp(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    global_vocab_size: int,
    real_vocab_size: int = -1,
    temperature: Optional[torch.Tensor] = None,
    tp_group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge a deterministic TP LM-head result without recomputing its GEMM."""

    if torch.is_grad_enabled() and local_logits.requires_grad:
        return _StrictLocalLogitsTPAutograd.apply(
            local_logits,
            target_ids,
            int(vocab_start_index),
            int(global_vocab_size),
            int(real_vocab_size),
            temperature,
            tp_group,
        )
    logp, lse, _target, _logits, _temp, lead_shape = _strict_logp_from_local_logits_tp_run(
        local_logits,
        target_ids,
        vocab_start_index=vocab_start_index,
        global_vocab_size=global_vocab_size,
        real_vocab_size=real_vocab_size,
        temperature=temperature,
        tp_group=tp_group,
    )
    return logp.reshape(lead_shape), lse.reshape(lead_shape)


def sm90_deterministic_top_p_logp_from_local_logits_tp(
    local_logits: torch.Tensor,
    target_ids: torch.Tensor,
    replay_ids: torch.Tensor,
    replay_logprobs: torch.Tensor,
    *,
    vocab_start_index: int,
    global_vocab_size: int,
    real_vocab_size: int = -1,
    temperature: Optional[torch.Tensor] = None,
    tp_group: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score a compact top-p replay set with the dense kernel's reduction order.

    This is deliberately inference-only: training retains the dense masked
    logits so its existing autograd contract is unchanged.
    """

    required = "linear_logp_top_p_local_bf16_forward"
    if not (_EXT_AVAILABLE and hasattr(_C, required)):
        raise RuntimeError(f"strict TP top-p logits logp missing CUDA symbol: {required}")
    if torch.is_grad_enabled() and local_logits.requires_grad:
        raise RuntimeError("compact top-p local-logits logp is inference-only")
    if local_logits.ndim != 2 or local_logits.dtype != torch.bfloat16:
        raise TypeError("strict TP top-p logits logp requires 2-D BF16 local logits")
    if not local_logits.is_cuda or not local_logits.is_contiguous():
        raise ValueError("strict TP top-p logits logp requires contiguous CUDA logits")
    if replay_ids.ndim != 2 or replay_logprobs.shape != replay_ids.shape:
        raise ValueError("top-p replay ids/logprobs must be aligned 2-D tensors")
    if replay_ids.size(0) != local_logits.size(0):
        raise ValueError("top-p replay rows must align with local logits")
    if replay_ids.size(1) > 64:
        raise ValueError("top-p replay width exceeds the compact kernel limit of 64")

    vocab_start = int(vocab_start_index)
    global_vocab = _validate_even_tp_vocab_partition_local(
        tp_group=tp_group,
        vocab_start_index=vocab_start,
        local_vocab_size=local_logits.size(1),
        global_vocab_size=int(global_vocab_size),
    )
    real_vocab = global_vocab if int(real_vocab_size) < 0 else int(real_vocab_size)
    if not 0 < real_vocab <= global_vocab:
        raise ValueError(f"invalid real_vocab_size={real_vocab} for padded vocab={global_vocab}")
    target = target_ids.reshape(-1).to(device=local_logits.device, dtype=torch.long).contiguous()
    if target.numel() != local_logits.size(0):
        raise ValueError("target_ids must contain one id per local-logits row")
    _assert_global_targets_async(target, real_vocab)

    if temperature is None:
        temp = torch.ones(
            local_logits.size(0), device=local_logits.device, dtype=torch.float32
        )
    else:
        temp = temperature.to(device=local_logits.device, dtype=torch.float32).reshape(-1)
        if temp.numel() == 1:
            temp = temp.expand(local_logits.size(0)).contiguous()
        if temp.numel() != local_logits.size(0):
            raise ValueError("temperature must be positive and scalar or per-token")
        torch._assert_async((temp > 0).all(), "temperature must be positive")

    ids = replay_ids.to(device=local_logits.device, dtype=torch.int32).contiguous()
    values = replay_logprobs.to(device=local_logits.device, dtype=torch.float32).contiguous()
    local_target, local_lse = _strict_logp_vocab_summaries(
        local_logits,
        target,
        vocab_start=vocab_start,
        global_vocab=global_vocab,
        tp_group=tp_group,
        top_p_args=(ids, values, temp),
    )
    logp, lse = _merge_tp_local_logp(local_lse, local_target, tp_group=tp_group)
    return logp.reshape(target_ids.shape), lse.reshape(target_ids.shape)


class _StrictTensorParallelLinearLogpAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        bias,
        target,
        vocab_start,
        global_vocab,
        real_vocab,
        temperature,
        tp_group,
    ):
        logp, lse, hidden_2d, weight, target, logits, temp = _strict_tp_run(
            hidden,
            weight,
            bias,
            target,
            vocab_start=vocab_start,
            global_vocab=global_vocab,
            real_vocab=real_vocab,
            temperature=temperature,
            tp_group=tp_group,
        )
        ctx.save_for_backward(hidden_2d, weight, target, logits, lse, temp)
        ctx.vocab_start = int(vocab_start)
        ctx.tp_group = tp_group
        ctx.lead_shape = hidden.shape[:-1]
        ctx.hidden_dtype, ctx.weight_dtype = hidden.dtype, weight.dtype
        ctx.has_bias = bias is not None
        ctx.bias_dtype = None if bias is None else bias.dtype
        return logp.reshape(ctx.lead_shape), lse.reshape(ctx.lead_shape)

    @staticmethod
    def backward(ctx, grad_logp, grad_lse):
        hidden, weight, target, logits, lse, temp = ctx.saved_tensors
        if grad_logp is None and grad_lse is None:
            return (None, None, None, None, None, None, None, None, None)
        logp_grad = torch.zeros_like(lse) if grad_logp is None else grad_logp.reshape(-1).float()
        dlogits = torch.empty_like(logits)
        _C.linear_logp_logits_bf16_to_dlogits(
            logits, dlogits, target, logp_grad, lse, ctx.vocab_start
        )
        if grad_lse is not None:
            probs = torch.exp(logits.float() - lse.reshape(-1, 1))
            dlogits = (dlogits.float() + probs * grad_lse.reshape(-1, 1).float()).to(torch.bfloat16)
        if temp.numel():
            dlogits = (dlogits.float() / temp.reshape(-1, 1)).to(torch.bfloat16).contiguous()
        else:
            dlogits = dlogits.contiguous()
        grad_hidden = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_hidden = det_gemm_linear_input_gradient(dlogits, weight)
            _deterministic_tp_all_reduce_(grad_hidden, ctx.tp_group)
            grad_hidden = grad_hidden.reshape(ctx.lead_shape + (weight.size(1),)).to(
                ctx.hidden_dtype
            )
        if ctx.needs_input_grad[1]:
            grad_weight = det_gemm_linear_weight_gradient(hidden, dlogits).to(ctx.weight_dtype)
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = dlogits.float().sum(dim=0).to(ctx.bias_dtype)
        return grad_hidden, grad_weight, grad_bias, None, None, None, None, None, None


def sm90_deterministic_linear_logp(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    temperature: Optional[torch.Tensor] = None,
    return_logits: bool = False,
    real_vocab_size: int = -1,
):
    """Direct forward entry into the frozen-contract SM90 fused linear_logp.

    This is the strict-mode boundary shared by the training and rollout sides:
    both engines call this function with byte-identical ``hidden`` /
    ``lm_head_weight`` / ``target_ids`` / ``temperature`` and the same contract
    version, and receive bitwise-identical FP32 outputs.

    * ``temperature`` (float32, one value per token, strictly positive) is
      applied in the kernel epilogue to the streaming-softmax stats and the
      selected logit — the same rounding point on every caller.
    * ``return_logits=True`` additionally stores the unscaled FP32 logits row
      for the sampler (serving entry) without changing the ``logp``/``lse``
      bytes (trainer entry keeps ``return_logits=False``).
    * ``real_vocab_size`` normalizes padded training vocab (e.g. 152064) onto
      the real serving vocab (e.g. 151936): lanes at or beyond it contribute
      exactly ``-inf``/``0`` to the LSE on every rank. ``-1`` disables masking.
    """
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    if not (_EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90")):
        raise RuntimeError(
            "fused_linear_logp_sm90 is not compiled into the extension. Rebuild with "
            "KERNEL_ALIGN_FORCE_SM90=1 on an SM90 (Hopper) device."
        )
    if hidden.dim() < 1:
        raise ValueError("hidden must have at least one dimension")
    lead_shape = hidden.shape[:-1]
    hidden_2d = hidden.reshape(-1, hidden.size(-1)).contiguous()
    weight = lm_head_weight.contiguous()
    target_1d = target_ids.reshape(-1).to(device=hidden_2d.device, dtype=torch.long).contiguous()
    if target_1d.numel() != hidden_2d.size(0):
        raise ValueError("target_ids must have one id per hidden row")
    real_vocab = weight.size(0) if int(real_vocab_size) < 0 else int(real_vocab_size)
    if not 0 < real_vocab <= weight.size(0):
        raise ValueError(f"real_vocab_size must be in [1, {weight.size(0)}], got {real_vocab_size}")
    if bool(((target_1d < 0) | (target_1d >= real_vocab)).any().item()):
        target_min = int(target_1d.min().item())
        target_max = int(target_1d.max().item())
        raise ValueError(
            f"target_ids must be in the real vocabulary [0, {real_vocab}), "
            f"got [{target_min}, {target_max}]"
        )
    bias_arg = bias.contiguous() if bias is not None else None
    temp_arg = None
    if temperature is not None:
        temp_arg = temperature.to(device=hidden_2d.device, dtype=torch.float32).reshape(-1)
        if temp_arg.numel() == 1:
            temp_arg = temp_arg.expand(hidden_2d.size(0)).contiguous()
        if temp_arg.numel() != hidden_2d.size(0) or bool((temp_arg <= 0).any().item()):
            raise ValueError("temperature must be positive and scalar or per-token")
    if (
        torch.is_grad_enabled()
        and (
            hidden.requires_grad
            or lm_head_weight.requires_grad
            or (bias is not None and bias.requires_grad)
        )
        and not return_logits
    ):
        return _StrictLinearLogpAutograd.apply(
            hidden, lm_head_weight, bias, target_ids, temp_arg, int(real_vocab_size)
        )
    out = _C.fused_linear_logp_sm90(
        hidden_2d,
        weight,
        target_1d,
        bias_arg,
        temp_arg,
        return_logits,
        int(real_vocab_size),
    )
    logp = out[0].reshape(lead_shape)
    lse = out[1].reshape(lead_shape)
    if return_logits:
        logits = out[2].reshape(*lead_shape, weight.size(0))
        return logp, lse, logits
    return logp, lse


def sm90_deterministic_linear_logp_tp(
    hidden: torch.Tensor,
    lm_head_weight_shard: torch.Tensor,
    target_ids: torch.Tensor,
    bias_shard: Optional[torch.Tensor] = None,
    *,
    vocab_start_index: int,
    global_vocab_size: Optional[int] = None,
    real_vocab_size: int = -1,
    temperature: Optional[torch.Tensor] = None,
    tp_group: Any = None,
):
    """Strict TP selected logprob; the high-performance fused path is unchanged."""
    if tp_group is None:
        raise ValueError("strict TP linear_logp requires a TP process group")
    if global_vocab_size is None:
        dist = _require_distributed_initialized()
        global_vocab_size = lm_head_weight_shard.size(0) * dist.get_world_size(tp_group)
    real_vocab = int(global_vocab_size) if int(real_vocab_size) < 0 else int(real_vocab_size)
    if torch.is_grad_enabled() and (
        hidden.requires_grad
        or lm_head_weight_shard.requires_grad
        or (bias_shard is not None and bias_shard.requires_grad)
    ):
        return _StrictTensorParallelLinearLogpAutograd.apply(
            hidden,
            lm_head_weight_shard,
            bias_shard,
            target_ids,
            int(vocab_start_index),
            int(global_vocab_size),
            real_vocab,
            temperature,
            tp_group,
        )
    logp, lse, _hidden, _weight, _target, _logits, _temp = _strict_tp_run(
        hidden,
        lm_head_weight_shard,
        bias_shard,
        target_ids,
        vocab_start=int(vocab_start_index),
        global_vocab=int(global_vocab_size),
        real_vocab=real_vocab,
        temperature=temperature,
        tp_group=tp_group,
    )
    return logp.reshape(hidden.shape[:-1]), lse.reshape(hidden.shape[:-1])
