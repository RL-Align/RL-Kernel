# SPDX-License-Identifier: Apache-2.0
"""Reuse task 19, aiter native Hadamard/MXFP4 and vLLM FP8; no new GPU kernel."""

import hashlib
import importlib
import json

import torch

from rl_engine.backends.rocm.attention.kv_compressor_c4 import AiterC4Backend
from rl_engine.contracts.operators.indexer_compressor_c4 import INDEXER_C4_BACKEND_ID
from rl_engine.contracts.operators.kv_compressor_c4 import C4Error, source_hash


class AiterIndexerC4Backend(AiterC4Backend):
    backend_id = INDEXER_C4_BACKEND_ID

    def __init__(self, *, megatron_backward_path=None):
        super().__init__(megatron_backward_path=megatron_backward_path)
        try:
            quant = importlib.import_module("aiter.ops.quant")
            fp8 = importlib.import_module("vllm.model_executor.layers.quantization.utils.fp8_utils")
            self.rotate_activation = quant.rotate_activation
            self.quant_mxfp4 = quant.per_1x32_f4_quant_triton
            self.quant_fp8 = fp8.per_token_group_quant_fp8
        except (ImportError, AttributeError) as exc:
            raise C4Error("native aiter rotation/quantization or vLLM FP8 ABI unavailable") from exc
        modules = {
            "aiter_quant": quant,
            "aiter_fp4_utils": importlib.import_module("aiter.utility.fp4_utils"),
            "vllm_fp8": fp8,
            "aiter_tensor_shim": importlib.import_module("aiter.ops.flydsl.kernels.tensor_shim"),
        }
        quant_impl = modules["aiter_fp4_utils"].dynamic_mxfp4_quant
        modules["mxfp4_implementation"] = importlib.import_module(quant_impl.__module__)
        # Track the native rotation binary as well as Python glue. Priming the
        # actual native entrypoint happens before deriving the fingerprint.
        dummy = torch.zeros((1, 128), dtype=torch.bfloat16, device=self.device)
        self._rotate(dummy)
        native = importlib.import_module("aiter.jit.module_dsv4_rotate_quant")
        modules["native_rotation"] = native
        self.provenance["indexer_sources"] = {
            label: source_hash(getattr(module, "__file__", None))
            for label, module in modules.items()
        }
        self.provenance["indexer_bridge"] = source_hash(__file__)
        self.provenance["rotation"] = "aiter.rotate_activation, no torch.matmul fallback"
        self.provenance["rotation_rows"] = "pad both buffers to 8 rows on gfx942, trim logical rows"
        self.provenance["provider_sha256"] = hashlib.sha256(
            json.dumps(self.provenance, sort_keys=True).encode()
        ).hexdigest()

    def _rotate(self, values):
        if not values.numel():
            return torch.empty_like(values)
        # This native D=128 kernel executes 8 rows per wave. The installed
        # provider's tail buffer extent is a full wave, so NEVER pass shorter
        # physical input/output allocations. Padding only changes independent
        # row launch geometry, not any within-row reduction.
        rows = values.shape[0]
        physical_rows = (rows + 7) // 8 * 8
        if physical_rows == rows:
            source = values.contiguous()
        else:
            source = torch.zeros((physical_rows, 128), dtype=values.dtype, device=values.device)
            source[:rows].copy_(values)
        destination = torch.empty_like(source)
        self.rotate_activation(destination, source)
        return destination[:rows]

    def forward(self, bound, ape, norm, cos, sin, contract):
        # Parent executes exactly the same validated C4 overlap and position
        # handling, now with the separately admitted head_dim=128 profile.
        compressed = super().forward(bound, ape, norm, cos, sin, contract)
        return self._rotate(compressed)

    def quantize(self, values, contract):
        if not values.shape[0]:
            return (
                torch.empty(
                    (0, 64 if contract.precision == "mxfp4" else 128),
                    dtype=torch.uint8 if contract.precision == "mxfp4" else torch.float8_e4m3fnuz,
                    device=values.device,
                ),
                torch.empty(
                    (0, 4 if contract.precision == "mxfp4" else 1),
                    dtype=torch.uint8 if contract.precision == "mxfp4" else torch.float32,
                    device=values.device,
                ),
            )
        if contract.precision == "mxfp4":
            data, scale = self.quant_mxfp4(values.contiguous(), shuffle=False)
            return data.view(torch.uint8), scale.view(torch.uint8)
        return self.quant_fp8(
            values.contiguous(),
            128,
            dtype=torch.float8_e4m3fnuz,
            use_ue8m0=True,
            column_major_scales=False,
        )

    def vjp(self, bound, ape, norm, cos, sin, grad, contract):
        # H is symmetric; reuse the same native transform for its transpose.
        # The BF16 dY boundary is explicit and is tested separately from the
        # continuous FP64 mathematical reference.
        unrotated_grad = self._rotate(grad.to(torch.bfloat16))
        return super().vjp(bound, ape, norm, cos, sin, unrotated_grad, contract)
