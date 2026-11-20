# SPDX-License-Identifier: Apache-2.0
"""Freeze upstream dispatch metadata; do not implement numerical kernels."""

from functools import lru_cache
import hashlib
import importlib
from pathlib import Path

import torch

from rl_engine.contracts.operators.indexer_projection import (
    INDEXER_BACKEND_ID,
    FixedGemmPlan,
    IndexerProjectionError,
)


class AiterIndexerProjectionBackend:
    backend_id = INDEXER_BACKEND_ID
    device_type = "cuda"

    def __init__(self):
        if torch.version.hip is None or not torch.cuda.is_available():
            raise IndexerProjectionError("ROCm GPU is required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        if arch != "gfx942":
            raise IndexerProjectionError(f"unvalidated architecture: {arch}")
        names = (
            "aiter.ops.triton.gemm.basic.gemm_a16w16",
            "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16",
            "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale",
            "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale",
            "vllm.model_executor.layers.quantization.utils.fp8_utils",
            "vllm.model_executor.layers.quantization.utils.w8a8_utils",
        )
        try:
            modules = [importlib.import_module(name) for name in names]
            self._gemm16 = modules[0].gemm_a16w16
            self._gemm8 = modules[2].gemm_a8w8_blockscale
            self._quantize = modules[4].per_token_group_quant_fp8
            self.normalize_checkpoint = modules[5].normalize_e4m3fn_to_e4m3fnuz
        except (ImportError, AttributeError) as exc:
            raise IndexerProjectionError("required aiter/vLLM ABI is missing; no fallback") from exc
        self.provenance = {
            "backend": self.backend_id,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": importlib.import_module("triton").__version__,
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "sources": {
                name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in zip(names, modules, strict=True)
            },
        }

    @staticmethod
    @lru_cache(maxsize=128)
    def plan(k: int, n: int, role: str, precision: str = "bf16") -> FixedGemmPlan:
        if k <= 0 or n <= 0:
            raise IndexerProjectionError("a nonempty GEMM plan requires positive K,N")
        # Freeze the validated gfx942 aiter recipe, including instruction shape.
        # No runtime-M lookup or environment-dependent autotuning is permitted.
        config = {
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "kpack": 2 if precision == "block_fp8_ue8m0" else 1,
        }
        if precision == "block_fp8_ue8m0":
            config.update(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, NUM_KSPLIT=1)
        else:
            narrow = role == "weights" and n <= 64 and k >= 1024
            splits = min(8, max(1, k // 512)) if narrow else 1
            chunk = ((k + splits - 1) // splits + 63) // 64 * 64
            splits = (k + chunk - 1) // chunk
            config.update(
                BLOCK_SIZE_M=16 if narrow else 32,
                BLOCK_SIZE_N=64 if narrow else 128,
                BLOCK_SIZE_K=64,
                NUM_KSPLIT=splits,
                SPLITK_BLOCK_SIZE=chunk,
            )
        return FixedGemmPlan(role, k, n, precision, tuple(sorted(config.items())))

    def linear(self, x, weight, *, output_dtype, role):
        if x.shape[0] == 0:
            return torch.empty((0, weight.shape[0]), device=x.device, dtype=output_dtype)
        plan = self.plan(x.shape[1], weight.shape[0], role)
        return self._gemm16(x, weight, dtype=output_dtype, config=plan.config)

    def linear_fp8(self, x, packed, *, output_dtype):
        if not x.shape[0]:
            return torch.empty((0, packed.data.shape[0]), device=x.device, dtype=output_dtype)
        # FNUZ is an explicit gfx942 deployment format, not a hardware-default guess.
        qx, scales = self.quantize_activation(x)
        plan = self.plan(x.shape[1], packed.data.shape[0], "q", "block_fp8_ue8m0")
        return self._gemm8(
            qx,
            packed.data,
            scales,
            packed.scales,
            dtype=output_dtype,
            config=plan.config,
        )

    def quantize_activation(self, x):
        return self._quantize(
            x,
            128,
            dtype=torch.float8_e4m3fnuz,
            use_ue8m0=True,
            column_major_scales=False,
            tma_aligned_scales=False,
        )

    def input_gradient(self, grad, weight):
        # The library accepts weight strides and transposes this argument
        # internally, so reuse the original RHS without a full weight copy.
        return self.linear(
            grad,
            weight.t(),
            output_dtype=torch.bfloat16,
            role="dx",
        )

    def weight_gradient(self, rows, grads):
        return self.linear(
            grads.t().contiguous(),
            rows.t(),
            output_dtype=torch.bfloat16,
            role="dw",
        )
