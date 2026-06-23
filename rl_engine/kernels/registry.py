# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib
import os
from enum import Enum, EnumMeta
from typing import Any, Dict, Optional, Set, Type

from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


class _KernelEnumMeta(EnumMeta):
    """Metaclass to provide enhanced error messaging for backend lookups."""

    def __getitem__(cls, name: str):
        try:
            return super().__getitem__(name)
        except KeyError as e:
            valid_ops = ", ".join(cls.__members__.keys())
            raise ValueError(f"Operator '{name}' not found. Supported backends: {valid_ops}") from e


class OpBackend(Enum, metaclass=_KernelEnumMeta):
    # NVIDIA optimized stack
    FLASH_ATTN = "rl_engine.kernels.ops.cuda.attention.flash_attn.FlashAttentionOp"
    FLASHINFER = "rl_engine.kernels.ops.cuda.flashinfer.FlashInferOp"

    # TMA-accelerated LogP for SM90+ (Warp Specialization)
    CUDA_FUSED_LOGP_SM90 = "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpSM90Op"
    CUDA_FUSED_LOGP_GENERIC = "rl_engine.kernels.ops.cuda.loss.logp.FusedLogpGenericOp"

    # AMD ROCm optimized stack
    ROCM_AITER = "rl_engine.kernels.ops.rocm.aiter.AiterOp"
    ROCM_CK = "rl_engine.kernels.ops.rocm.composable_kernel.CKOp"
    ROCM_FLASH_ATTN = "rl_engine.kernels.ops.rocm.attention.flash_attn.RocmFlashAttentionOp"

    # GRPO loss (group reward normalization + clipped surrogate + KL)
    TRITON_GRPO_LOSS = "rl_engine.kernels.ops.triton.loss.grpo_loss.TritonGRPOLossOp"
    PYTORCH_GRPO_LOSS = "rl_engine.kernels.ops.pytorch.loss.grpo_loss.NativeGRPOLossOp"

    # Fused linear log-prob (hidden @ W^T -> selected-token logp, no [N, V] logits)
    CUDA_FUSED_LINEAR_LOGP_SM90 = (
        "rl_engine.kernels.ops.cuda.loss.linear_logp.FusedLinearLogpSM90Op"
    )
    TRITON_LINEAR_LOGP = "rl_engine.kernels.ops.triton.loss.linear_logp.TritonLinearLogpOp"
    PYTORCH_LINEAR_LOGP = "rl_engine.kernels.ops.pytorch.loss.linear_logp.NativeLinearLogpOp"
    # Fused policy-ratio + KL-penalty front-end (PPO/GRPO), logits -> (ratio, kl)
    TRITON_RATIO_KL = "rl_engine.kernels.ops.triton.loss.ratio_kl.TritonRatioKLOp"
    PYTORCH_RATIO_KL = "rl_engine.kernels.ops.pytorch.loss.ratio_kl.NativeRatioKLOp"

    # Generic fallback
    TRITON_GENERIC = "rl_engine.kernels.ops.triton.generic.TritonOp"
    PYTORCH_ATTN = "rl_engine.kernels.ops.pytorch.attention.NativeAttentionOp"
    PYTORCH_NATIVE = "rl_engine.kernels.ops.pytorch.loss.logp.NativeLogpOp"


class KernelRegistry:
    """
    Central dispatcher for high-performance kernels.
    Handles dynamic routing between ROCm and CUDA backends at runtime.
    """

    def __init__(self):
        self._instance_cache: Dict[str, Any] = {}
        self._failed_backends: Set[str] = set()

        self._priority_map = {
            "cuda": {
                "logp": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.FLASHINFER,
                    OpBackend.TRITON_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_indexed": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_online": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "logp_online_indexed": [
                    OpBackend.CUDA_FUSED_LOGP_GENERIC,
                    OpBackend.PYTORCH_NATIVE,
                ],
                "attn": [OpBackend.FLASH_ATTN, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_ATTN],
                "grpo_loss": [OpBackend.TRITON_GRPO_LOSS, OpBackend.PYTORCH_GRPO_LOSS],
                "linear_logp": [OpBackend.TRITON_LINEAR_LOGP, OpBackend.PYTORCH_LINEAR_LOGP],
                "ratio_kl": [OpBackend.TRITON_RATIO_KL, OpBackend.PYTORCH_RATIO_KL],
                # Default dispatch logic for new operators
            },
            "rocm": {
                "logp": [OpBackend.ROCM_AITER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [
                    OpBackend.ROCM_FLASH_ATTN,
                    OpBackend.PYTORCH_ATTN,
                    OpBackend.TRITON_GENERIC,
                ],
                "grpo_loss": [OpBackend.TRITON_GRPO_LOSS, OpBackend.PYTORCH_GRPO_LOSS],
                "linear_logp": [OpBackend.TRITON_LINEAR_LOGP, OpBackend.PYTORCH_LINEAR_LOGP],
                "ratio_kl": [OpBackend.TRITON_RATIO_KL, OpBackend.PYTORCH_RATIO_KL],
            },
            "cpu": {
                "logp": [OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.PYTORCH_ATTN],
                "grpo_loss": [OpBackend.PYTORCH_GRPO_LOSS],
                "linear_logp": [OpBackend.PYTORCH_LINEAR_LOGP],
                "ratio_kl": [OpBackend.PYTORCH_RATIO_KL],
            },
        }
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")
        self._adjust_priority_for_hardware()
        self._adjust_priority_from_env()

    def _adjust_priority_from_env(self):
        rocm_attn_backend = os.getenv("RL_KERNEL_ROCM_ATTN_BACKEND", "").strip().lower()
        if rocm_attn_backend in {"flash_attn", "flash-attn", "flash_attention"}:
            self._priority_map["rocm"]["attn"] = [
                OpBackend.ROCM_FLASH_ATTN,
                OpBackend.PYTORCH_ATTN,
                OpBackend.TRITON_GENERIC,
            ]
        elif rocm_attn_backend in {"native", "pytorch", "sdpa"}:
            self._priority_map["rocm"]["attn"] = [
                OpBackend.PYTORCH_ATTN,
                OpBackend.ROCM_FLASH_ATTN,
                OpBackend.TRITON_GENERIC,
            ]
        elif rocm_attn_backend and rocm_attn_backend not in {"native", "pytorch", "sdpa"}:
            logger.warning(
                "Unknown RL_KERNEL_ROCM_ATTN_BACKEND=%s; using default ROCm attention priority.",
                rocm_attn_backend,
            )

    def _adjust_priority_for_hardware(self):
        """Prioritize the fused TMA LogP kernel only when it is compiled into the
        extension and the device is TMA-capable (SM90/100/120)."""
        if device_ctx.device_type != "cuda":
            return
        try:
            import torch

            from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

            cc_major, cc_minor = torch.cuda.get_device_capability()
            cc = cc_major * 10 + cc_minor
            tma_compiled = _EXT_AVAILABLE and hasattr(_C, "fused_logp_sm90")

            if tma_compiled and cc_major in (9, 10, 12):
                logger.info(
                    f"Detected TMA-capable architecture (SM{cc}); "
                    "prioritizing fused TMA LogP kernel."
                )
                logp_list = self._priority_map["cuda"]["logp"]
                if OpBackend.CUDA_FUSED_LOGP_SM90 not in logp_list:
                    logp_list.insert(0, OpBackend.CUDA_FUSED_LOGP_SM90)

            # The fused linear-logp SM90 kernel uses TMA bulk-tensor copies built
            # for sm_90a -- gate strictly on cc_major == 9 (Hopper), not >= 9.
            linear_logp_compiled = _EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90")
            if linear_logp_compiled and cc_major == 9:
                ll_list = self._priority_map["cuda"]["linear_logp"]
                if OpBackend.CUDA_FUSED_LINEAR_LOGP_SM90 not in ll_list:
                    ll_list.insert(0, OpBackend.CUDA_FUSED_LINEAR_LOGP_SM90)
            elif cc >= 90:
                logger.debug(
                    f"SM{cc}: fused TMA LogP kernel not compiled into _C; "
                    "using generic fused kernel."
                )
        except Exception as e:
            logger.warning(f"Failed to probe device capability: {e}")

    def get_op(self, op_type: str) -> Any:
        """Core distribution logic: Automatically select the best operator
        based on hardware and priority.
        """
        if device_ctx.is_rocm:
            platform = "rocm"
        elif device_ctx.device_type == "cuda":
            platform = "cuda"
        else:
            platform = "cpu"
        candidates = self._priority_map.get(platform, {}).get(op_type, [OpBackend.PYTORCH_NATIVE])

        for backend in candidates:
            if backend.name in self._instance_cache:
                return self._instance_cache[backend.name]

            if backend.name in self._failed_backends:
                continue

            op_class = self._load_backend(backend)
            if op_class:
                try:
                    op_instance = op_class()
                    self._instance_cache[backend.name] = op_instance
                    return op_instance
                except Exception as e:
                    logger.error(f"Failed to instantiate {backend.name}: {e}")
                    self._failed_backends.add(backend.name)
            else:
                self._failed_backends.add(backend.name)

        raise RuntimeError(f"No functional backend found for {op_type} on {platform}")

    def _load_backend(self, backend: OpBackend) -> Optional[Type]:
        """Dynamic loading technique: Import modules only when needed
        and check environment dependencies.
        """
        module_path, class_name = backend.value.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            missing_module = str(e.name) if hasattr(e, "name") else ""
            is_missing_backend = missing_module and (
                missing_module == module_path or module_path.startswith(missing_module)
            )
            if missing_module and "rl_engine" in missing_module and not is_missing_backend:
                logger.critical(f"Internal wrapper implementation bug in '{module_path}': {e}")
                raise e
            logger.warning(f"Backend {backend.name} unavailable: {e}. Falling back...")
            return None


kernel_registry = KernelRegistry()
