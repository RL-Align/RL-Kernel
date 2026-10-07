# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    SplitKVExecutionPlan,
    SplitKVMode,
    SplitKVSpec,
)
from rl_engine.kernels.ops.cuda.attention.deterministic_attn import (
    DeterministicAttentionCoreResult,
    RLKernelDeterministicAttentionCore,
)
from rl_engine.utils.logger import logger

_MAX_TESTED_ROCM_TRITON_HEAD_DIM = 512
_AITER_API_SOURCE = "aiter.ops.mha"
_AITER_OP_NAMESPACE = "aiter"

# AITER wraps its kernels in a JIT loader whose Python signature is
# ``(*args, **kwargs)``, so ``inspect.signature`` cannot see the contract the
# way it can for the FA4 CuTe API. The names live in the registered Torch
# schema instead, and the calls below are positional, so the order is part of
# what has to hold: an upstream insertion would silently reinterpret every
# argument after it. These tuples are the exact positional prefix each call
# site assumes.
_AITER_FWD_POSITIONAL_CONTRACT = (
    "q",
    "k",
    "v",
    "dropout_p",
    "softmax_scale",
    "is_causal",
    "window_size_left",
    "window_size_right",
    "sink_size",
    "return_softmax_lse",
    "return_dropout_randval",
)
_AITER_BWD_POSITIONAL_CONTRACT = (
    "dout",
    "q",
    "k",
    "v",
    "out",
    "softmax_lse",
    "dropout_p",
    "softmax_scale",
    "is_causal",
    "window_size_left",
    "window_size_right",
    "deterministic",
)
# Passed by keyword, so only presence matters.
_AITER_BWD_REQUIRED_KEYWORDS = frozenset({"rng_state"})
_AITER_BATCH_PREFILL_POSITIONAL_CONTRACT = (
    "q",
    "k",
    "v",
    "cu_seqlens_q",
    "kv_indptr",
    "kv_page_indices",
    "max_seqlen_q",
    "max_seqlen_k",
    "dropout_p",
    "softmax_scale",
    "logits_soft_cap",
    "zero_tensors",
    "is_causal",
    "window_size_left",
    "window_size_right",
    "sink_size",
    "return_softmax_lse",
    "return_dropout_randval",
)
_AITER_BATCH_PREFILL_REQUIRED_KEYWORDS = frozenset({"block_table", "seqlen_k"})

# Stable dispatch identity for the strict ROCm attention core.  Kept at module
# scope so contract-aware dispatch and the Vime adapter name one constant
# instead of duplicating the string.
BACKEND_ID = "aiter.rocm.ck_dense_mha"
AITER_PAGED_KERNEL_ID = "aiter_mha_batch_prefill_non_split_ck"
TRITON_CHUNKED_FLASH_ATTENTION_ID = "rlkernel.rocm.triton_chunked_flash_attention.v1"
_ATTENTION_BACKEND_ENV = "RL_KERNEL_ROCM_ATTENTION_BACKEND"


def _requested_attention_backend() -> str:
    """Return ``ck`` (AITER/CK entry points) or ``triton`` (chunked flash contract)."""

    value = os.environ.get(_ATTENTION_BACKEND_ENV, "ck").strip().lower()
    value = {"aiter": "ck", "chunked_flash": "triton"}.get(value, value)
    if value not in {"ck", "triton"}:
        raise RuntimeError(f"{_ATTENTION_BACKEND_ENV} must be 'ck' or 'triton', got {value!r}")
    return value


class StrictRocmAttentionUnavailable(RuntimeError):
    """Raised when the exact AITER CK strict contract is unavailable."""


def _aiter_schema_argument_names(op_name: str) -> tuple[str, ...]:
    """Return the registered Torch schema argument names for one AITER op."""

    namespace = getattr(torch.ops, _AITER_OP_NAMESPACE, None)
    if namespace is None:
        raise StrictRocmAttentionUnavailable(
            f"the '{_AITER_OP_NAMESPACE}' Torch operator namespace is not registered"
        )
    try:
        overload = getattr(namespace, op_name).default
        arguments = overload._schema.arguments
    except (AttributeError, RuntimeError) as exc:
        raise StrictRocmAttentionUnavailable(
            f"cannot read the Torch schema for {_AITER_OP_NAMESPACE}::{op_name}"
        ) from exc
    return tuple(argument.name for argument in arguments)


def _validate_aiter_schema(
    op_name: str,
    positional_contract: tuple[str, ...],
    *,
    required_keywords: frozenset[str] = frozenset(),
) -> None:
    """Fail closed unless AITER still accepts what the call sites pass.

    The strict calls are positional, so a renamed *or reordered* argument
    changes their meaning without changing their shape. Checking the ordered
    prefix catches both, which name-presence alone would not.
    """

    names = _aiter_schema_argument_names(op_name)
    prefix = names[: len(positional_contract)]
    if prefix != positional_contract:
        raise StrictRocmAttentionUnavailable(
            f"AITER {op_name} positional contract changed: strict ROCm Attention "
            f"passes {positional_contract} but the schema declares {prefix}"
        )
    missing = sorted(required_keywords.difference(names))
    if missing:
        raise StrictRocmAttentionUnavailable(
            f"AITER {op_name} is missing strict controls: " + ", ".join(missing)
        )


def _load_aiter_ck_ops() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], str]:
    try:
        module = importlib.import_module(_AITER_API_SOURCE)
        mha_fwd = module.mha_fwd
        mha_bwd = module.mha_bwd
        mha_batch_prefill = module.mha_batch_prefill
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        raise StrictRocmAttentionUnavailable(
            "strict ROCm Attention requires AITER dense backward and batch-prefill CK ops"
        ) from exc
    if not all(callable(op) for op in (mha_fwd, mha_bwd, mha_batch_prefill)):
        raise StrictRocmAttentionUnavailable("AITER CK MHA entry points are not callable")
    _validate_aiter_schema("mha_fwd", _AITER_FWD_POSITIONAL_CONTRACT)
    _validate_aiter_schema(
        "mha_bwd",
        _AITER_BWD_POSITIONAL_CONTRACT,
        required_keywords=_AITER_BWD_REQUIRED_KEYWORDS,
    )
    _validate_aiter_schema(
        "mha_batch_prefill",
        _AITER_BATCH_PREFILL_POSITIONAL_CONTRACT,
        required_keywords=_AITER_BATCH_PREFILL_REQUIRED_KEYWORDS,
    )
    module_file = inspect.getsourcefile(module)
    if not module_file:
        raise StrictRocmAttentionUnavailable("cannot fingerprint the AITER MHA source module")
    source_sha256 = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
    return mha_fwd, mha_bwd, mha_batch_prefill, source_sha256


class _AiterCKAttentionFn(Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        mha_fwd: Callable[..., Any],
        mha_bwd: Callable[..., Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        result = mha_fwd(
            q_fa,
            k_fa,
            v_fa,
            0.0,
            float(scale),
            bool(causal),
            -1,
            -1,
            0,
            True,
            False,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise StrictRocmAttentionUnavailable(
                "AITER mha_fwd must return (out, lse, dropout_mask, rng_state)"
            )
        out_fa, lse, _dropout_mask, rng_state = result
        if not all(isinstance(item, torch.Tensor) for item in (out_fa, lse, rng_state)):
            raise StrictRocmAttentionUnavailable("AITER mha_fwd returned non-tensor state")
        ctx.save_for_backward(q_fa, k_fa, v_fa, out_fa, lse, rng_state)
        ctx.causal = bool(causal)
        ctx.scale = float(scale)
        ctx.mha_bwd = mha_bwd
        ctx.mark_non_differentiable(lse)
        return out_fa.transpose(1, 2).contiguous(), lse.contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        q_fa, k_fa, v_fa, out_fa, lse, rng_state = ctx.saved_tensors
        grad_out_fa = grad_out.transpose(1, 2).contiguous()
        result = ctx.mha_bwd(
            grad_out_fa,
            q_fa,
            k_fa,
            v_fa,
            out_fa,
            lse,
            0.0,
            ctx.scale,
            ctx.causal,
            -1,
            -1,
            True,
            rng_state=rng_state,
        )
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise StrictRocmAttentionUnavailable("AITER mha_bwd must return dQ/dK/dV")
        dq, dk, dv = result[:3]
        return (
            dq.transpose(1, 2).contiguous(),
            dk.transpose(1, 2).contiguous(),
            dv.transpose(1, 2).contiguous(),
            None,
            None,
            None,
            None,
        )


class _AiterCKPagedAttentionFn(Function):
    """Use one non-Split-K paged CK schedule for train forward and decode."""

    page_size = 16

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        scale: float,
        mha_batch_prefill: Callable[..., Any],
        mha_bwd: Callable[..., Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, q_heads, q_len, head_dim = q.shape
        kv_heads, kv_len = k.size(1), k.size(2)
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        page_count = (kv_len + _AiterCKPagedAttentionFn.page_size - 1) // (
            _AiterCKPagedAttentionFn.page_size
        )
        padded_kv_len = page_count * _AiterCKPagedAttentionFn.page_size
        if padded_kv_len == kv_len:
            k_cache = k_fa.reshape(
                batch * page_count,
                _AiterCKPagedAttentionFn.page_size,
                kv_heads,
                head_dim,
            )
            v_cache = v_fa.reshape_as(k_cache)
        else:
            padded_shape = (batch, padded_kv_len, kv_heads, head_dim)
            k_padded = torch.zeros(padded_shape, dtype=k.dtype, device=k.device)
            v_padded = torch.zeros_like(k_padded)
            k_padded[:, :kv_len].copy_(k_fa)
            v_padded[:, :kv_len].copy_(v_fa)
            k_cache = k_padded.reshape(
                batch * page_count,
                _AiterCKPagedAttentionFn.page_size,
                kv_heads,
                head_dim,
            )
            v_cache = v_padded.reshape_as(k_cache)

        block_table = torch.arange(batch * page_count, dtype=torch.int32, device=q.device).reshape(
            batch, page_count
        )
        # CK fetches page IDs for a complete 128-token tile before masking.
        # Page 0 contains valid, initialized training KV for the unused columns.
        guard_columns = (-page_count) % (128 // _AiterCKPagedAttentionFn.page_size)
        if guard_columns:
            block_table = torch.nn.functional.pad(block_table, (0, guard_columns), value=0)
        cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * q_len
        kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * page_count
        seqlen_k = torch.full((batch,), kv_len, dtype=torch.int32, device=q.device)
        result = mha_batch_prefill(
            q_fa.reshape(batch * q_len, q_heads, head_dim),
            k_cache,
            v_cache,
            cu_seqlens_q,
            kv_indptr,
            block_table.reshape(-1),
            q_len,
            kv_len,
            0.0,
            float(scale),
            0.0,
            False,
            bool(causal),
            -1,
            -1,
            0,
            True,
            False,
            block_table=block_table,
            seqlen_k=seqlen_k,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise StrictRocmAttentionUnavailable(
                "AITER mha_batch_prefill must return (out, lse, dropout_mask, rng_state)"
            )
        out_flat, lse_flat, _dropout_mask, rng_state = result
        if not all(isinstance(item, torch.Tensor) for item in (out_flat, lse_flat, rng_state)):
            raise StrictRocmAttentionUnavailable("AITER paged CK returned non-tensor state")
        expected_out = (batch * q_len, q_heads, head_dim)
        if tuple(out_flat.shape) != expected_out:
            raise StrictRocmAttentionUnavailable("AITER paged CK output shape changed")
        if tuple(lse_flat.shape) != (q_heads, batch * q_len) or lse_flat.dtype != torch.float32:
            raise StrictRocmAttentionUnavailable("AITER paged CK must export [H,total_q] FP32 LSE")

        out_fa = out_flat.reshape(batch, q_len, q_heads, head_dim)
        lse = lse_flat.reshape(q_heads, batch, q_len).permute(1, 0, 2).contiguous()
        ctx.save_for_backward(q_fa, k_fa, v_fa, out_fa, lse, rng_state)
        ctx.causal = bool(causal)
        ctx.scale = float(scale)
        ctx.mha_bwd = mha_bwd
        ctx.mark_non_differentiable(lse)
        return out_fa.transpose(1, 2).contiguous(), lse

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: torch.Tensor, grad_lse: torch.Tensor):
        q_fa, k_fa, v_fa, out_fa, lse, rng_state = ctx.saved_tensors
        grad_out_fa = grad_out.transpose(1, 2).contiguous()
        result = ctx.mha_bwd(
            grad_out_fa,
            q_fa,
            k_fa,
            v_fa,
            out_fa,
            lse,
            0.0,
            ctx.scale,
            ctx.causal,
            -1,
            -1,
            True,
            rng_state=rng_state,
        )
        if not isinstance(result, (tuple, list)) or len(result) < 3:
            raise StrictRocmAttentionUnavailable("AITER mha_bwd must return dQ/dK/dV")
        dq, dk, dv = result[:3]
        return (
            dq.transpose(1, 2).contiguous(),
            dk.transpose(1, 2).contiguous(),
            dv.transpose(1, 2).contiguous(),
            None,
            None,
            None,
            None,
        )


class StrictRocmAiterCKAttentionCore:
    """Shared ROCm production core using the non-Split-K AITER CK dense MHA."""

    core_id = STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_ROCM_SCHEDULE_ID
    backend_id = BACKEND_ID
    api_source = _AITER_API_SOURCE
    merge_order = "global_block_index"
    accum_dtype = "fp32"
    downcast_at = "final_write"
    fallback = False
    native_attention_arithmetic = True
    production_ready = True
    reference_only = False
    num_splits = 1
    split_kv_control = "dense_non_split_api"
    deterministic_backward = True

    def __init__(
        self,
        *,
        split_kv: SplitKVSpec | None = None,
        _mha_fwd: Callable[..., Any] | None = None,
        _mha_bwd: Callable[..., Any] | None = None,
        _mha_batch_prefill: Callable[..., Any] | None = None,
        _source_sha256: str | None = None,
    ) -> None:
        requested = SplitKVSpec.disabled() if split_kv is None else split_kv
        if not isinstance(requested, SplitKVSpec):
            raise TypeError("split_kv must be a SplitKVSpec")
        if requested.mode is not SplitKVMode.DISABLED:
            raise ValueError("strict AITER CK Attention requires Split-KV to be disabled")
        if (_mha_fwd is None) != (_mha_bwd is None):
            raise ValueError("test injection requires both AITER forward and backward callables")
        mha_batch_prefill: Callable[..., Any] | None
        if _mha_fwd is None:
            mha_fwd, mha_bwd, mha_batch_prefill, source_sha256 = _load_aiter_ck_ops()
        else:
            assert _mha_bwd is not None
            mha_fwd = _mha_fwd
            mha_bwd = _mha_bwd
            mha_batch_prefill = _mha_batch_prefill
            source_sha256 = "test-double" if _source_sha256 is None else _source_sha256
        self._fixed_paged_tile = 0
        self._attention_backend = "ck"
        fixed_tile = os.environ.get("RL_KERNEL_ROCM_FIXED_PAGED_TILE", "0")
        requested_backend = _requested_attention_backend()
        if _mha_fwd is None and requested_backend == "triton":
            from rl_engine.kernels.ops.triton.attention import chunked_flash_attn

            self._attention_backend = "triton"
            mha_batch_prefill = chunked_flash_attn.triton_paged_prefill
            source_digest = hashlib.sha256(source_sha256.encode())
            source_path = inspect.getsourcefile(chunked_flash_attn)
            if source_path is None:
                raise StrictRocmAttentionUnavailable("Triton attention source file is unavailable")
            source_digest.update(Path(source_path).read_bytes())
            source_digest.update(chunked_flash_attn.CHUNKED_FLASH_ATTENTION_CONTRACT_ID.encode())
            source_sha256 = source_digest.hexdigest()
        elif _mha_fwd is None and fixed_tile != "0":
            if fixed_tile not in ("64", "128"):
                raise ValueError("RL_KERNEL_ROCM_FIXED_PAGED_TILE must be 0, 64 or 128")
            from .fixed_paged_ck import fixed_paged_prefill

            self._fixed_paged_tile = int(fixed_tile)
            mha_batch_prefill = partial(fixed_paged_prefill, tile_m=self._fixed_paged_tile)
            source_digest = hashlib.sha256(source_sha256.encode())
            source_path = inspect.getsourcefile(fixed_paged_prefill)
            if source_path is None:
                raise StrictRocmAttentionUnavailable(
                    "fixed paged attention source file is unavailable"
                )
            source_digest.update(Path(source_path).read_bytes())
            source_digest.update(
                (
                    Path(__file__).resolve().parents[5] / "csrc/rocm/attention/fixed_paged_ck.hip"
                ).read_bytes()
            )
            source_digest.update(fixed_tile.encode())
            source_sha256 = source_digest.hexdigest()
        if not callable(mha_fwd) or not callable(mha_bwd):
            raise StrictRocmAttentionUnavailable("AITER CK MHA entry points are not callable")
        self.split_kv = requested
        self.source_sha256 = source_sha256
        self._mha_fwd = mha_fwd
        self._mha_bwd = mha_bwd
        self._mha_batch_prefill = mha_batch_prefill
        self.supports_paged_schedule = callable(mha_batch_prefill)
        # The Triton contract serves decode, extend and prefill rows from one
        # launch; the CK entry point needs one query row per decode request.
        self.supports_mixed_paged_batches = self._attention_backend == "triton"
        self._device_description_cache: tuple[torch.device, tuple[str, str]] | None = None
        self._split_kv_plan_cache: (
            tuple[tuple[SplitKVSpec, int, str], SplitKVExecutionPlan] | None
        ) = None

    @property
    def attention_backend(self) -> str:
        return self._attention_backend

    @property
    def paged_entrypoint_id(self) -> str:
        if self._attention_backend == "triton":
            return TRITON_CHUNKED_FLASH_ATTENTION_ID
        if self._fixed_paged_tile:
            return f"rl_kernel_fixed_paged_ck_m{self._fixed_paged_tile}"
        return "mha_batch_prefill"

    @property
    def paged_kernel_id(self) -> str:
        if self._attention_backend == "triton":
            return TRITON_CHUNKED_FLASH_ATTENTION_ID
        return AITER_PAGED_KERNEL_ID

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
        resolved_scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
        if self._mha_batch_prefill is None:
            out, lse = _AiterCKAttentionFn.apply(
                q,
                k,
                v,
                bool(causal),
                resolved_scale,
                self._mha_fwd,
                self._mha_bwd,
            )
            forward_entrypoint = "mha_fwd"
            kv_layout = "dense_bshd"
        else:
            out, lse = _AiterCKPagedAttentionFn.apply(
                q,
                k,
                v,
                bool(causal),
                resolved_scale,
                self._mha_batch_prefill,
                self._mha_bwd,
            )
            forward_entrypoint = self.paged_entrypoint_id
            kv_layout = "sequential_linear_pages"
        expected_lse_shape = (q.size(0), q.size(1), q.size(2))
        if out.shape != q.shape or out.dtype != resolved_dtype:
            raise StrictRocmAttentionUnavailable("AITER CK output shape/dtype changed")
        if tuple(lse.shape) != expected_lse_shape or lse.dtype != torch.float32:
            raise StrictRocmAttentionUnavailable("AITER CK must export [B,H,Sq] FP32 LSE")
        gpu_name, gpu_arch = self._device_description(q.device)
        split_kv_plan = self._resolve_split_kv_plan(k.size(2))
        return DeterministicAttentionCoreResult(
            out=out,
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": self.backend_id,
                "platform": "rocm",
                "torch_version": torch.__version__,
                "rocm_version": torch.version.hip,
                "gpu_name": gpu_name,
                "gpu_arch": gpu_arch,
                "aiter_api_source": self.api_source,
                "aiter_source_sha256": self.source_sha256,
                "num_splits": self.num_splits,
                "split_kv_control": self.split_kv_control,
                "deterministic_backward": self.deterministic_backward,
                "forward_entrypoint": forward_entrypoint,
                "kv_layout": kv_layout,
                "dense_kv_materialized": False,
                "dropout_p": 0.0,
                "split_kv": split_kv_plan.to_dict(),
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

    def forward_paged_varlen_with_lse(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        kv_indptr: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float | None,
        out: torch.Tensor | None = None,
        return_lse: bool = True,
    ) -> DeterministicAttentionCoreResult:
        """Run one graph-safe CK launch over packed queries and paged KV."""

        if self._mha_batch_prefill is None:
            raise StrictRocmAttentionUnavailable("AITER paged CK entry point is unavailable")
        if q.ndim != 3:
            raise ValueError("packed paged Q must use [tokens, heads, head_dim]")
        if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
            raise ValueError("paged K/V must use [pages, page_size, heads, head_dim]")
        if q.size(1) % k_cache.size(2) or q.size(2) != k_cache.size(3):
            raise ValueError("paged Q/K head counts or dimensions are incompatible")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict paged Attention supports FP16/BF16 only")
        if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
            raise ValueError("paged Q/K/V must share one dtype")
        if not (q.device == k_cache.device == v_cache.device):
            raise ValueError("paged Q/K/V must share one ROCm device")
        batch = page_table.size(0)
        if page_table.ndim != 2 or seqused_k.shape != (batch,):
            raise ValueError("paged metadata must carry one row per sequence")
        if cu_seqlens_q.shape != (batch + 1,) or kv_indptr.shape != (batch + 1,):
            raise ValueError("paged indptr tensors must carry batch + 1 entries")
        if any(
            tensor.device != q.device for tensor in (page_table, seqused_k, cu_seqlens_q, kv_indptr)
        ):
            raise ValueError("paged metadata must be on the Q device")
        if not q.is_contiguous() or not page_table.is_contiguous():
            raise ValueError("packed Q and page_table must be contiguous")
        if max_seqlen_q <= 0 or max_seqlen_k <= 0:
            raise ValueError("paged maximum sequence lengths must be positive")
        if max_seqlen_k > page_table.size(1) * k_cache.size(1):
            raise ValueError("max_seqlen_k exceeds the page table capacity")
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k_cache, v_cache)
        ):
            raise RuntimeError("direct paged Attention is inference-only")
        if out is not None:
            if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
                raise ValueError("direct paged output must match packed Q")
            if not out.is_contiguous():
                raise ValueError("direct paged output must be contiguous")

        resolved_scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
        result = self._mha_batch_prefill(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            kv_indptr,
            page_table.reshape(-1),
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            resolved_scale,
            0.0,
            False,
            bool(causal),
            -1,
            -1,
            0,
            bool(return_lse),
            False,
            out=out,
            block_table=page_table,
            seqlen_k=seqused_k,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise StrictRocmAttentionUnavailable(
                "AITER mha_batch_prefill must return (out, lse, dropout_mask, rng_state)"
            )
        result_out, lse, _dropout_mask, _rng_state = result
        if tuple(result_out.shape) != tuple(q.shape) or result_out.dtype != q.dtype:
            raise StrictRocmAttentionUnavailable("AITER paged CK output shape/dtype changed")
        if out is not None and result_out.data_ptr() != out.data_ptr():
            raise StrictRocmAttentionUnavailable("AITER paged CK ignored the output buffer")
        if return_lse:
            if tuple(lse.shape) != (q.size(1), q.size(0)) or lse.dtype != torch.float32:
                raise StrictRocmAttentionUnavailable(
                    "AITER paged CK must export [heads,total_q] FP32 LSE"
                )
        else:
            lse = torch.empty((0,), dtype=torch.float32, device=q.device)
        gpu_name, gpu_arch = self._device_description(q.device)
        split_kv_plan = self._resolve_split_kv_plan(max_seqlen_k)
        return DeterministicAttentionCoreResult(
            out=result_out,
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": self.backend_id,
                "platform": "rocm",
                "torch_version": torch.__version__,
                "rocm_version": torch.version.hip,
                "gpu_name": gpu_name,
                "gpu_arch": gpu_arch,
                "aiter_api_source": self.api_source,
                "aiter_source_sha256": self.source_sha256,
                "forward_entrypoint": self.paged_entrypoint_id,
                "paged_kernel": self.paged_kernel_id,
                "kv_layout": "vllm_linear_paged",
                "dense_kv_materialized": False,
                "num_splits": self.num_splits,
                "split_kv_control": "batch_prefill_non_split_ck",
                "deterministic_backward": False,
                "dropout_p": 0.0,
                "split_kv": split_kv_plan.to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "direct_output": out is not None,
                "packed_query_count": batch,
                "packed_query_tokens": q.size(0),
                "fallback": self.fallback,
                "fallback_reason": None,
                "native_attention_arithmetic": self.native_attention_arithmetic,
                "production_ready": self.production_ready,
                "reference_only": self.reference_only,
            },
        )

    def forward_paged_with_lse(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        kv_indptr: torch.Tensor,
        max_seqlen_k: int,
        causal: bool,
        scale: float | None,
        out: torch.Tensor | None = None,
        return_lse: bool = True,
    ) -> DeterministicAttentionCoreResult:
        self._validate_paged_inputs(q, k_cache, v_cache, page_table, seqused_k)
        if q.size(2) != 1:
            raise ValueError("direct paged decode requires one query token per row")
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q, k_cache, v_cache)
        ):
            raise RuntimeError("direct paged decode is inference-only")
        resolved_scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else float(scale)
        q_flat = q.transpose(1, 2).reshape(q.size(0), q.size(1), q.size(3))
        if not q_flat.is_contiguous():
            q_flat = q_flat.contiguous()
        out_flat = None
        if out is not None:
            if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
                raise ValueError("direct paged output must match Q shape, dtype, and device")
            out_flat = out.transpose(1, 2).reshape_as(q_flat)
            if not out_flat.is_contiguous():
                raise ValueError("direct paged output must expose a contiguous AITER view")
        flat_result = self.forward_paged_varlen_with_lse(
            q_flat,
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            scale=resolved_scale,
            out=out_flat,
            return_lse=return_lse,
        )
        result_flat = flat_result.out
        if return_lse:
            lse = flat_result.lse.transpose(0, 1).unsqueeze(-1).contiguous()
        else:
            lse = torch.empty((0,), dtype=torch.float32, device=q.device)
        return DeterministicAttentionCoreResult(
            out=out if out is not None else result_flat.unsqueeze(1).transpose(1, 2).contiguous(),
            lse=lse,
            provenance=dict(flat_result.provenance),
        )

    @staticmethod
    def _validate_paged_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
    ) -> None:
        if torch.version.hip is None:
            raise StrictRocmAttentionUnavailable("strict AITER CK core requires ROCm PyTorch")
        if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
            raise ValueError("paged q/k/v layouts must be BHSD and page-linear BSHD")
        if q.size(1) % k.size(2) or q.size(3) != k.size(3):
            raise ValueError("paged Q/K head counts or dimensions are incompatible")
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or k.dtype != q.dtype
            or v.dtype != q.dtype
        ):
            raise ValueError("paged q/k/v must share FP16 or BF16 dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda) or not (q.device == k.device == v.device):
            raise ValueError("paged q/k/v must share one ROCm device")
        if page_table.shape[0] != q.size(0) or seqused_k.shape != (q.size(0),):
            raise ValueError("paged metadata must carry one row per query")
        if page_table.device != q.device or seqused_k.device != q.device:
            raise ValueError("paged metadata must be on the Q device")

    def forward_bshd_with_lse(
        self,
        q_bshd: torch.Tensor,
        k_bshd: torch.Tensor,
        v_bshd: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        out: torch.Tensor | None = None,
    ) -> DeterministicAttentionCoreResult:
        """Run the identical CK forward arithmetic on already packed BSHD tensors.

        Decode has no backward pass.  Keeping paged gather output in AITER's
        native layout removes the BHSD round trip, while the same dense
        non-Split-K entry point and arguments preserve the strict arithmetic.
        ``out`` uses the public BHSD shape; Sq=1 makes its BSHD transpose a
        contiguous view suitable for AITER's output argument.
        """

        self._validate_bshd_inputs(q_bshd, k_bshd, v_bshd)
        if torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (q_bshd, k_bshd, v_bshd)
        ):
            raise RuntimeError("strict BSHD decode entry point is inference-only")
        resolved_scale = 1.0 / math.sqrt(q_bshd.size(-1)) if scale is None else float(scale)
        out_bshd = None
        if out is not None:
            expected_out = (
                q_bshd.size(0),
                q_bshd.size(2),
                q_bshd.size(1),
                q_bshd.size(3),
            )
            if out.shape != expected_out or out.dtype != q_bshd.dtype:
                raise ValueError("strict BSHD decode output buffer has the wrong shape or dtype")
            out_bshd = out.transpose(1, 2)
            if not out_bshd.is_contiguous():
                raise ValueError("strict BSHD decode output must expose a contiguous AITER view")

        if self._fixed_paged_tile or self._attention_backend == "triton":
            # Keep the same arithmetic if a caller already materialized BSHD
            # inputs (e.g. a mixed prefill/decode fallback).
            fixed_out, fixed_lse = _AiterCKPagedAttentionFn.apply(
                q_bshd.transpose(1, 2),
                k_bshd.transpose(1, 2),
                v_bshd.transpose(1, 2),
                bool(causal),
                resolved_scale,
                self._mha_batch_prefill,
                self._mha_bwd,
            )
            if out is not None:
                out.copy_(fixed_out)
                fixed_out = out
            return DeterministicAttentionCoreResult(
                out=fixed_out,
                lse=fixed_lse,
                provenance={
                    "actual_backend": self.backend_id,
                    "forward_entrypoint": self.paged_entrypoint_id,
                    "aiter_source_sha256": self.source_sha256,
                    "dense_kv_materialized": True,
                    "fallback": False,
                },
            )

        result = self._mha_fwd(
            q_bshd,
            k_bshd,
            v_bshd,
            0.0,
            resolved_scale,
            bool(causal),
            -1,
            -1,
            0,
            True,
            False,
            out=out_bshd,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise StrictRocmAttentionUnavailable(
                "AITER mha_fwd must return (out, lse, dropout_mask, rng_state)"
            )
        result_bshd, lse, _dropout_mask, _rng_state = result
        if not isinstance(result_bshd, torch.Tensor) or not isinstance(lse, torch.Tensor):
            raise StrictRocmAttentionUnavailable("AITER mha_fwd returned non-tensor output")
        if result_bshd.shape != q_bshd.shape or result_bshd.dtype != q_bshd.dtype:
            raise StrictRocmAttentionUnavailable("AITER CK output shape/dtype changed")
        expected_lse_shape = (q_bshd.size(0), q_bshd.size(2), q_bshd.size(1))
        if tuple(lse.shape) != expected_lse_shape or lse.dtype != torch.float32:
            raise StrictRocmAttentionUnavailable("AITER CK must export [B,H,Sq] FP32 LSE")
        if out_bshd is not None and result_bshd.data_ptr() != out_bshd.data_ptr():
            raise StrictRocmAttentionUnavailable("AITER CK ignored the strict decode output buffer")

        gpu_name, gpu_arch = self._device_description(q_bshd.device)
        split_kv_plan = self._resolve_split_kv_plan(k_bshd.size(1))
        result_out = out if out is not None else result_bshd.transpose(1, 2).contiguous()
        return DeterministicAttentionCoreResult(
            out=result_out,
            lse=lse.contiguous(),
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": self.backend_id,
                "platform": "rocm",
                "torch_version": torch.__version__,
                "rocm_version": torch.version.hip,
                "gpu_name": gpu_name,
                "gpu_arch": gpu_arch,
                "aiter_api_source": self.api_source,
                "aiter_source_sha256": self.source_sha256,
                "num_splits": self.num_splits,
                "split_kv_control": self.split_kv_control,
                "deterministic_backward": False,
                "dropout_p": 0.0,
                "split_kv": split_kv_plan.to_dict(),
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "input_layout": "bshd_direct",
                "direct_output": out is not None,
                "fallback": self.fallback,
                "fallback_reason": None,
                "native_attention_arithmetic": self.native_attention_arithmetic,
                "production_ready": self.production_ready,
                "reference_only": self.reference_only,
            },
        )

    def _device_description(self, device: torch.device) -> tuple[str, str]:
        cached = self._device_description_cache
        if cached is not None and cached[0] == device:
            return cached[1]
        properties = torch.cuda.get_device_properties(device)
        description = (properties.name, getattr(properties, "gcnArchName", "unknown"))
        self._device_description_cache = (device, description)
        return description

    def _resolve_split_kv_plan(self, total_kv_tokens: int) -> SplitKVExecutionPlan:
        key = (self.split_kv, total_kv_tokens, self.backend_id)
        cached = self._split_kv_plan_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        plan = self.split_kv.resolve(total_kv_tokens, backend=self.backend_id)
        self._split_kv_plan_cache = (key, plan)
        return plan

    @staticmethod
    def _validate_bshd_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if torch.version.hip is None:
            raise StrictRocmAttentionUnavailable("strict AITER CK core requires ROCm PyTorch")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4-D [B,S,H,D]")
        if q.size(0) != k.size(0) or q.size(0) != v.size(0):
            raise ValueError("q/k/v batch sizes must match")
        if k.shape != v.shape or q.size(-1) != k.size(-1):
            raise ValueError("k/v shapes and q/k/v head dimensions must match")
        if q.size(2) % k.size(2) != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")
        if q.size(-1) > 256 or q.size(-1) % 8:
            raise ValueError("AITER CK requires head_dim <= 256 and divisible by 8")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict AITER CK core supports FP16/BF16 only")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must share one dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("strict AITER CK core requires ROCm GPU tensors")
        if not (q.device == k.device == v.device):
            raise ValueError("q/k/v must be on one ROCm device")
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            raise ValueError("strict BSHD decode inputs must be contiguous")

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> None:
        if torch.version.hip is None:
            raise StrictRocmAttentionUnavailable("strict AITER CK core requires ROCm PyTorch")
        if key_padding_mask is not None:
            raise ValueError("strict AITER CK core materializes each unpadded logical row")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q/k/v must be 4-D [B,H,S,D]")
        if q.size(0) != 1 or k.size(0) != 1 or v.size(0) != 1:
            raise ValueError("strict AITER CK core executes one logical batch row at a time")
        if k.shape != v.shape or q.size(-1) != k.size(-1):
            raise ValueError("k/v shapes and q/k/v head dimensions must match")
        if q.size(1) % k.size(1) != 0:
            raise ValueError("Q heads must be divisible by KV heads for GQA")
        if q.size(-1) > 256 or q.size(-1) % 8:
            raise ValueError("AITER CK requires head_dim <= 256 and divisible by 8")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict AITER CK core supports FP16/BF16 only")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must share one dtype")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("strict AITER CK core requires ROCm GPU tensors")
        if not (q.device == k.device == v.device):
            raise ValueError("q/k/v must be on one ROCm device")


def _select_flash_attn_backend() -> str:
    """Select the installed FlashAttention ROCm backend."""
    return "triton"


class RocmFlashAttentionOp:
    """
    Standard FlashAttention wrapper for ROCm.
    Demonstrates the reference structure for adding new operator families.
    """

    def __init__(self):
        if torch.version.hip is None:
            raise RuntimeError("RocmFlashAttentionOp requires a ROCm PyTorch build.")

        backend = _select_flash_attn_backend()
        if backend == "triton":
            # flash-attn selects the ROCm CK/Triton backend at import time.
            os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
        try:
            from flash_attn import flash_attn_func

            self.op = flash_attn_func
            logger.info("Successfully linked to external flash_attn library (%s backend).", backend)
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "ROCm FlashAttention requires a ROCm-compatible flash-attn installation. "
                "See docs/getting_started/installation.md#rocm-backend."
            ) from exc

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
        valid_dtypes = (torch.float16, torch.bfloat16)
        if (
            q.dtype not in valid_dtypes
            or k.dtype not in valid_dtypes
            or v.dtype not in valid_dtypes
        ):
            raise TypeError("FlashAttention requires FP16 or BF16 for q/k/v")
        # PyTorch uses the CUDA device API for both CUDA and ROCm tensors.
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("Inputs must be on a CUDA/ROCm GPU device")
        if not (q.device == k.device == v.device):
            raise ValueError("q, k, and v must be on the same device")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "q, k, and v must be rank-4 tensors: (batch, seqlen, nheads, head_dim)"
            )

        head_dim = q.shape[-1]
        if head_dim == 0:
            raise ValueError("head_dim must be positive")
        if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
            raise ValueError("q, k, and v must have the same head_dim")
        if head_dim > _MAX_TESTED_ROCM_TRITON_HEAD_DIM:
            raise NotImplementedError(
                "RL-Kernel's ROCm FlashAttention wrapper currently supports "
                f"head_dim <= {_MAX_TESTED_ROCM_TRITON_HEAD_DIM}; got {head_dim}"
            )

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        return self.op(q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
