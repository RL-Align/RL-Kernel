# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from typing import Any, Callable

import torch

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    SplitKVMode,
    SplitKVSpec,
)
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.attention.deterministic_attn import (
    DeterministicAttentionCoreResult,
    RLKernelDeterministicAttentionCore,
)
from rl_engine.utils.logger import logger

_FA4_API_SOURCE = "flash_attn.cute.interface"
_FA4_COMPAT_API_SOURCE = "vllm.vllm_flash_attn.cute.interface"
_FA4_API_SOURCES = (_FA4_API_SOURCE, _FA4_COMPAT_API_SOURCE)
_FA4_REQUIRED_PARAMETERS = frozenset(
    {"softmax_scale", "causal", "num_splits", "pack_gqa", "deterministic", "return_lse"}
)
_FA4_PAGED_REQUIRED_PARAMETERS = _FA4_REQUIRED_PARAMETERS | frozenset(
    {"page_table", "seqused_k", "max_seqlen_k"}
)


class StrictFlashAttentionUnavailable(RuntimeError):
    """Raised when the exact FlashAttention production contract is unavailable."""


def _load_fa4_cute_op() -> tuple[Callable[..., Any], str, str]:
    errors: list[str] = []
    for api_source in _FA4_API_SOURCES:
        try:
            module = importlib.import_module(api_source)
            op = getattr(module, "flash_attn_func")
        except (AttributeError, ImportError, OSError, RuntimeError) as exc:
            errors.append(f"{api_source}: {exc}")
            continue

        package_name = "vllm" if api_source == _FA4_COMPAT_API_SOURCE else "flash-attn"
        try:
            package_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            package_version = "unknown"
        return op, package_version, api_source

    raise StrictFlashAttentionUnavailable(
        "strict CUDA Attention requires a FlashAttention CuTe API; tried "
        + ", ".join(_FA4_API_SOURCES)
        + (f" ({'; '.join(errors)})" if errors else "")
    )


class StrictFlashAttention4Core:
    """Shared CUDA production core with the reduction-affecting FA4 knobs fixed."""

    core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID
    backend_id = "flash_attention_4.cute"
    api_source = _FA4_API_SOURCE
    merge_order = "global_block_index"
    accum_dtype = "fp32"
    downcast_at = "final_write"
    fallback = False
    native_attention_arithmetic = True
    production_ready = True
    reference_only = False
    num_splits = 1
    deterministic_backward = True

    def __init__(
        self,
        *,
        split_kv: SplitKVSpec | None = None,
        _op: Callable[..., Any] | None = None,
        _paged_op: Callable[..., Any] | None = None,
        _package_version: str | None = None,
    ) -> None:
        requested = SplitKVSpec.disabled() if split_kv is None else split_kv
        if not isinstance(requested, SplitKVSpec):
            raise TypeError("split_kv must be a SplitKVSpec")
        if requested.mode is not SplitKVMode.DISABLED:
            raise ValueError("strict FA4 Attention requires Split-KV to be disabled")
        if _op is None:
            op, package_version, api_source = _load_fa4_cute_op()
            paged_op = getattr(importlib.import_module(api_source), "flash_attn_varlen_func", None)
        else:
            op = _op
            paged_op = _paged_op
            package_version = "test-double" if _package_version is None else _package_version
            api_source = _FA4_API_SOURCE
        self._validate_api(op)
        if paged_op is not None:
            self._validate_paged_api(paged_op)
        self.split_kv = requested
        self.package_version = package_version
        self.api_source = api_source
        self._op = op
        self._paged_op = paged_op

    @classmethod
    def precompile_training(
        cls,
        *,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        sequence_length: int = 512,
    ) -> None:
        """Compile strict training FA4 forward/backward before timing.

        FA4 CuTe compiles its forward kernel on the first invocation and its
        deterministic backward kernel on the first autograd backward.  A
        one-step RL workload would otherwise charge both compilations to
        Vime's ``actor_train`` timer.  These isolated tensors exercise the
        same Qwen-style GQA and multi-block shape class without touching model
        tensors, RNG state, or distributed collectives.
        """
        if torch.version.hip is not None:
            raise StrictFlashAttentionUnavailable("FA4 CUDA precompile is unavailable on ROCm")
        if not torch.cuda.is_available():
            raise StrictFlashAttentionUnavailable(
                "FA4 CUDA precompile requires an available CUDA device"
            )
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict FA4 training precompile requires FP16 or BF16")
        if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
            raise ValueError("Q/KV head counts must be positive and GQA-compatible")
        if head_dim <= 0 or sequence_length <= 0:
            raise ValueError("head_dim and sequence_length must be positive")

        target = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        if target.type != "cuda":
            raise ValueError("strict FA4 training precompile requires a CUDA device")

        core = cls()
        # Zeros deliberately avoid consuming RNG state.  The tensors are
        # independent from the model and only populate FA4's process-local
        # JIT caches.
        q = torch.zeros(
            (1, sequence_length, q_heads, head_dim),
            dtype=dtype,
            device=target,
            requires_grad=True,
        )
        k = torch.zeros(
            (1, sequence_length, kv_heads, head_dim),
            dtype=dtype,
            device=target,
            requires_grad=True,
        )
        v = torch.zeros_like(k, requires_grad=True)
        positions = torch.arange(sequence_length, dtype=torch.int64, device=target).expand(1, -1)
        with torch.enable_grad():
            result = core.forward_bshd_with_lse(
                q,
                k,
                v,
                causal=True,
                scale=head_dim**-0.5,
                query_position_ids=positions,
                key_position_ids=positions,
                output_dtype=dtype,
            )
            result.out.sum().backward()
        torch.cuda.synchronize(target)
        del result, q, k, v, positions, core

    @staticmethod
    def _validate_api(op: Callable[..., Any]) -> None:
        try:
            parameters = inspect.signature(op).parameters
        except (TypeError, ValueError) as exc:
            raise StrictFlashAttentionUnavailable(
                "cannot inspect the FlashAttention CuTe API signature"
            ) from exc
        missing = sorted(_FA4_REQUIRED_PARAMETERS.difference(parameters))
        if missing:
            raise StrictFlashAttentionUnavailable(
                "FlashAttention CuTe API is missing strict controls: " + ", ".join(missing)
            )

    @staticmethod
    def _validate_paged_api(op: Callable[..., Any]) -> None:
        try:
            parameters = inspect.signature(op).parameters
        except (TypeError, ValueError) as exc:
            raise StrictFlashAttentionUnavailable(
                "cannot inspect the FlashAttention CuTe paged API signature"
            ) from exc
        missing = sorted(_FA4_PAGED_REQUIRED_PARAMETERS.difference(parameters))
        if missing:
            raise StrictFlashAttentionUnavailable(
                "FlashAttention CuTe paged API is missing strict controls: " + ", ".join(missing)
            )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs: Any,
    ) -> DeterministicAttentionCoreResult:
        return self.forward_with_lse(q, k, v, **kwargs)

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        key_position_ids: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        self._validate_inputs(q, k, v, key_padding_mask)
        RLKernelDeterministicAttentionCore._validate_positions(
            q,
            k,
            causal=causal,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype != q.dtype:
            raise ValueError("strict Attention output_dtype must match the Q/K/V input dtype")

        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        result = self.forward_bshd_with_lse(
            q_fa,
            k_fa,
            v_fa,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
            output_dtype=resolved_dtype,
        )
        return DeterministicAttentionCoreResult(
            out=result.out.transpose(1, 2).contiguous(),
            lse=result.lse,
            provenance=result.provenance,
        )

    def forward_bshd_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: float | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        key_position_ids: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> DeterministicAttentionCoreResult:
        """Run strict FA4 on tensors already laid out as [B, S, H, D]."""
        self._validate_bshd_inputs(q, k, v, key_padding_mask)
        RLKernelDeterministicAttentionCore._validate_positions(
            q.transpose(1, 2),
            k.transpose(1, 2),
            causal=causal,
            query_position_ids=query_position_ids,
            key_position_ids=key_position_ids,
        )
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype != q.dtype:
            raise ValueError("strict Attention output_dtype must match the Q/K/V input dtype")

        result = self._op(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            num_splits=self.num_splits,
            pack_gqa=q.size(2) > k.size(2),
            deterministic=self.deterministic_backward,
            return_lse=True,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise StrictFlashAttentionUnavailable(
                "FlashAttention CuTe must return exactly (out, lse) when return_lse=True"
            )
        out_fa, lse = result
        if not isinstance(out_fa, torch.Tensor) or not isinstance(lse, torch.Tensor):
            raise StrictFlashAttentionUnavailable("FlashAttention CuTe returned non-tensor output")
        expected_out_shape = q.shape
        expected_lse_shape = (q.size(0), q.size(2), q.size(1))
        if tuple(out_fa.shape) != expected_out_shape:
            raise StrictFlashAttentionUnavailable(
                f"FlashAttention output must have shape {tuple(expected_out_shape)}"
            )
        if tuple(lse.shape) != expected_lse_shape:
            raise StrictFlashAttentionUnavailable(
                f"FlashAttention LSE must have shape {expected_lse_shape}"
            )
        if out_fa.dtype != resolved_dtype:
            raise StrictFlashAttentionUnavailable(
                "FlashAttention output dtype does not match the requested output dtype"
            )
        if lse.dtype != torch.float32:
            raise StrictFlashAttentionUnavailable(
                "FlashAttention attention-domain LSE must be FP32"
            )

        return DeterministicAttentionCoreResult(
            out=out_fa,
            lse=lse.contiguous(),
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": self.backend_id,
                "fa_api_source": self.api_source,
                "fa_package_version": self.package_version,
                "num_splits": self.num_splits,
                "deterministic_backward": self.deterministic_backward,
                "dropout_p": 0.0,
                "split_kv": self.split_kv.resolve(k.size(1), backend=self.backend_id).to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "fallback": self.fallback,
                "fallback_reason": None,
                "native_attention_arithmetic": self.native_attention_arithmetic,
                "production_ready": self.production_ready,
                "reference_only": self.reference_only,
            },
        )

    def forward_paged_bshd_with_lse(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        scale: float | None = None,
        output_dtype: torch.dtype | None = None,
        out: torch.Tensor | None = None,
    ) -> DeterministicAttentionCoreResult:
        """Run the strict FA4 schedule directly over a paged KV cache."""

        if self._paged_op is None:
            raise StrictFlashAttentionUnavailable(
                "strict paged Attention requires flash_attn_varlen_func"
            )
        self._validate_paged_bshd_inputs(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
        )
        resolved_dtype = q.dtype if output_dtype is None else output_dtype
        if resolved_dtype != q.dtype:
            raise ValueError("strict Attention output_dtype must match the Q/K/V input dtype")
        if out is not None:
            if out.shape != q.shape:
                raise ValueError("paged Attention out must have the same shape as q")
            if out.dtype != resolved_dtype or out.device != q.device:
                raise ValueError("paged Attention out must match the requested dtype and device")
            if not out.is_contiguous():
                raise ValueError("paged Attention out must be contiguous")

        paged_kwargs = {
            "page_table": page_table,
            "seqused_k": seqused_k,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": scale,
            "causal": False,
            "num_splits": self.num_splits,
            "pack_gqa": q.size(2) > k_cache.size(2),
            "deterministic": self.deterministic_backward,
            "return_lse": True,
        }
        if out is not None:
            paged_kwargs["out"] = out
        out_fa, lse = self._paged_op(
            q,
            k_cache,
            v_cache,
            **paged_kwargs,
        )
        expected_lse_shape = (q.size(0), q.size(2), q.size(1))
        if not isinstance(out_fa, torch.Tensor) or tuple(out_fa.shape) != tuple(q.shape):
            raise StrictFlashAttentionUnavailable(
                f"paged FlashAttention output must have shape {tuple(q.shape)}"
            )
        if not isinstance(lse, torch.Tensor) or tuple(lse.shape) != expected_lse_shape:
            raise StrictFlashAttentionUnavailable(
                f"paged FlashAttention LSE must have shape {expected_lse_shape}"
            )
        if out_fa.dtype != resolved_dtype or lse.dtype != torch.float32:
            raise StrictFlashAttentionUnavailable(
                "paged FlashAttention returned an incompatible output dtype"
            )
        if out is not None and out_fa.data_ptr() != out.data_ptr():
            raise StrictFlashAttentionUnavailable(
                "paged FlashAttention did not write to the requested output buffer"
            )
        return DeterministicAttentionCoreResult(
            out=out_fa,
            lse=lse.contiguous(),
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": f"{self.backend_id}.paged",
                "fa_api_source": self.api_source,
                "fa_package_version": self.package_version,
                "num_splits": self.num_splits,
                "deterministic_backward": self.deterministic_backward,
                "dropout_p": 0.0,
                "split_kv": self.split_kv.resolve(max_seqlen_k, backend=self.backend_id).to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "kv_layout": "paged_direct",
                "output_buffer_reused": out is not None,
                "fallback": False,
                "fallback_reason": None,
                "native_attention_arithmetic": True,
                "production_ready": True,
                "reference_only": False,
            },
        )

    @staticmethod
    def _validate_paged_bshd_inputs(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> None:
        if q.ndim != 4 or q.size(1) != 1:
            raise ValueError("paged q must use [B, 1, H, D]")
        if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
            raise ValueError("paged k/v must use [pages, page_size, H, D]")
        if q.size(2) % k_cache.size(2) != 0 or q.size(3) != k_cache.size(3):
            raise ValueError("paged q/k head counts or head dimensions are incompatible")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict paged Attention supports FP16/BF16 only")
        if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
            raise ValueError("paged q/k/v must share one dtype")
        if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
            raise ValueError("strict paged Attention requires CUDA tensors")
        if not (q.device == k_cache.device == v_cache.device):
            raise ValueError("paged q/k/v must be on one CUDA device")
        if page_table.shape[0] != q.size(0) or page_table.dtype != torch.int32:
            raise ValueError("page_table must be int32 with one row per query")
        if seqused_k.shape != (q.size(0),) or seqused_k.dtype != torch.int32:
            raise ValueError("seqused_k must be int32 with one length per query")
        if page_table.device != q.device or seqused_k.device != q.device:
            raise ValueError("paged Attention metadata must be on the Q device")
        if max_seqlen_k <= 0 or max_seqlen_k > page_table.size(1) * k_cache.size(1):
            raise ValueError("max_seqlen_k exceeds the page table capacity")

    @staticmethod
    def _validate_bshd_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> None:
        if key_padding_mask is not None:
            raise ValueError(
                "strict FA4 core does not accept padding masks; materialize each logical row"
            )
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4-D [B, S, H, D]")
        if q.size(0) < 1:
            raise ValueError("strict FA4 core requires at least one logical batch row")
        if k.size(0) != q.size(0) or v.size(0) != q.size(0):
            raise ValueError("q/k/v batch sizes must match")
        if k.shape != v.shape or q.size(3) != k.size(3):
            raise ValueError("k/v shapes and q/k/v head dimensions must match")
        if q.size(2) % k.size(2) != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict FA4 core supports FP16/BF16 only")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must share one dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("strict FA4 core requires CUDA tensors")
        if not (q.device == k.device == v.device):
            raise ValueError("q/k/v must be on one CUDA device")

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> None:
        if key_padding_mask is not None:
            raise ValueError(
                "strict FA4 core does not accept padding masks; materialize each logical row"
            )
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4-D [B, H, S, D]")
        if q.size(0) < 1:
            raise ValueError("strict FA4 core requires at least one logical batch row")
        if k.size(0) != q.size(0) or v.size(0) != q.size(0):
            raise ValueError("q/k/v batch sizes must match")
        if k.shape != v.shape or q.size(3) != k.size(3):
            raise ValueError("k/v shapes and q/k/v head dimensions must match")
        if q.size(1) % k.size(1) != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict FA4 core supports FP16/BF16 only")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must share one dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("strict FA4 core requires CUDA tensors")
        if not (q.device == k.device == v.device):
            raise ValueError("q/k/v must be on one CUDA device")


class FlashAttentionOp:
    """
    Standard FlashAttention wrapper for CUDA.
    Demonstrates the reference structure for adding new operator families.
    """

    def __init__(self):
        if not _EXT_AVAILABLE:
            raise RuntimeError(
                "Core binary extension is unavailable. FlashAttention cannot be initialized."
            )

        try:
            from flash_attn import flash_attn_func

            self.op = flash_attn_func
            logger.info("Successfully linked to external flash_attn library.")
        except ImportError:
            if hasattr(_C, "flash_attn_forward"):
                self.op = _C.flash_attn_forward
                logger.info("Successfully linked to RL-Kernel _C.flash_attn_forward.")
            else:
                raise RuntimeError(
                    "Neither external flash_attn nor _C.flash_attn_forward is available."
                ) from None

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Standard attention forward pass.
        Args:
            q: (batch, seqlen, nheads, headdim)
            k: (batch, seqlen, nheads_k, headdim)
            v: (batch, seqlen, nheads_k, headdim)
        """
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "FlashAttention requires FP16 or BF16"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        return self.op(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
