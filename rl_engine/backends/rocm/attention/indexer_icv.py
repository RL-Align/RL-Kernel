# SPDX-License-Identifier: Apache-2.0
"""Reuse aiter decode/GEMMs. No new ICV, unpacking or reduction kernel."""

import importlib

import torch

from rl_engine.contracts.operators.indexer_icv import ICV_BACKEND_ID, IndexerICVError, source_sha256


class AiterIndexerICVBackend:
    backend_id = ICV_BACKEND_ID
    device_type = "cuda"

    def __init__(self):
        if torch.version.hip is None or not torch.cuda.is_available():
            raise IndexerICVError("ICV requires a ROCm GPU")
        self.device = torch.device("cuda", torch.cuda.current_device())
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0]
        if arch != "gfx942":
            raise IndexerICVError(f"unvalidated architecture {arch}")
        names = (
            "aiter.ops.triton.moe.quant_moe",
            "aiter.ops.triton._triton_kernels.moe.quant_moe",
            "aiter.ops.triton.gemm.basic.gemm_a16w16",
            "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16",
            "aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale",
            "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_per_token_scale",
            "vllm.model_executor.layers.quantization.utils.w8a8_utils",
        )
        try:
            modules = [importlib.import_module(name) for name in names]
            self.upcast = modules[0].upcast_from_mxfp
            self.gemm16 = modules[2].gemm_a16w16
            self.gemm8 = modules[4].gemm_a8w8_per_token_scale
            self.normalize_fp8 = modules[6].normalize_e4m3fn_to_e4m3fnuz
        except (ImportError, AttributeError) as exc:
            raise IndexerICVError("required aiter/vLLM numerical ABI unavailable") from exc
        self.provenance = {
            "backend": self.backend_id,
            "arch": arch,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": importlib.import_module("triton").__version__,
            "adapter_sha256": source_sha256(__file__, label=__name__),
            "sources": {
                name: source_sha256(getattr(module, "__file__", None), label=name)
                for name, module in zip(names, modules, strict=True)
            },
        }

    @staticmethod
    def plan(contract):
        # The library's FP8 wrapper lowers BLOCK_K for D<64, using D only.
        # Neither actual query rows nor key rows select arithmetic/tuning.
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg",
            "NUM_KSPLIT": 1,
            "kpack": 1,
            "SPLITK_BLOCK_SIZE": (contract.head_dim + 63) // 64 * 64,
        }

    def prepare(self, data, scales, contract):
        """Map the *quantized* values to the admitted execution representation."""
        if contract.precision == "mxfp4":
            if data.numel() == 0:
                return torch.empty(
                    (*data.shape[:-1], contract.head_dim),
                    dtype=torch.bfloat16,
                    device=data.device,
                ), None
            # E2M1 times the admitted powers of two is exactly representable in
            # BF16: this is explicit MXFP4 emulation, not re-quantized BF16 data.
            decoded = self.upcast(data, scales, dtype=torch.bfloat16, axis=-1)
            return decoded[..., : contract.head_dim].contiguous(), None
        if contract.fp8_encoding == "e4m3fn":
            # The native normalizer mutates negative-zero codes; protect inputs.
            values, adjusted, _ = self.normalize_fp8(data.clone(), scales)
            return values, adjusted
        return data, scales

    def dot(self, q, k, q_scale, k_scale, contract):
        m, h, d = q.shape
        if m == 0 or k.shape[0] == 0:
            return torch.empty((m, h, k.shape[0]), dtype=torch.float32, device=q.device)
        flat = q.reshape(m * h, d)
        config = self.plan(contract)
        if contract.precision == "mxfp4":
            out = self.gemm16(flat, k, dtype=torch.float32, config=config)
        else:
            out = self.gemm8(
                flat,
                k,
                q_scale.reshape(m * h, 1),
                k_scale,
                dtype=torch.float32,
                config=config,
            )
        return out.view(m, h, k.shape[0])
