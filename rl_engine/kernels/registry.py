# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import importlib
from enum import Enum, EnumMeta
from typing import Optional, Dict, Any, Type, Set
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
    FLASH_ATTN = "rl_engine.kernels.cuda.flash_attn.FlashAttentionOp"
    FLASHINFER = "rl_engine.kernels.cuda.flashinfer.FlashInferOp"

    # AMD ROCm optimized stack
    ROCM_AITER = "rl_engine.kernels.rocm.aiter.AiterOp"
    ROCM_CK = "rl_engine.kernels.rocm.composable_kernel.CKOp"

    # Generic fallback
    TRITON_GENERIC = "rl_engine.kernels.triton.generic.TritonOp"
    PYTORCH_NATIVE = "rl_engine.kernels.ops.pytorch.NativeOp"


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
                "logp": [OpBackend.FLASHINFER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.FLASH_ATTN, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
            },
            "rocm": {
                "logp": [OpBackend.ROCM_AITER, OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.TRITON_GENERIC, OpBackend.PYTORCH_NATIVE],
            },
            "cpu": {
                "logp": [OpBackend.PYTORCH_NATIVE],
                "attn": [OpBackend.PYTORCH_NATIVE],
            },
        }
        logger.info(f"KernelRegistry initialized for {device_ctx.device_type}")

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
