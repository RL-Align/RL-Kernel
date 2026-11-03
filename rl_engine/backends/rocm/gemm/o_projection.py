# SPDX-License-Identifier: Apache-2.0
"""Only dispatch/adaptation: aiter RoPE and grouped GEMM, shared item-13 GEMMs."""

import hashlib
import importlib
from pathlib import Path

import torch

from rl_engine.backends.rocm.gemm.indexer_projection import AiterIndexerProjectionBackend
from rl_engine.contracts.operators.o_projection import O_PROJECTION_BACKEND, OProjectionError


class AiterOProjectionBackend(AiterIndexerProjectionBackend):
    backend_id = O_PROJECTION_BACKEND

    # No runtime-token-count autotuning, split-K or atomic reduction.
    GROUPED_CONFIG = (
        ("BLOCK_SIZE_M", 32),
        ("BLOCK_SIZE_N", 128),
        ("BLOCK_SIZE_K", 64),
        ("GROUP_SIZE_M", 1),
        ("num_warps", 4),
        ("num_stages", 2),
        ("matrix_instr_nonkdim", 16),
        ("kpack", 1),
    )

    def __init__(self):
        super().__init__()
        names = (
            "aiter.ops.rope",
            "aiter.ops.triton.gemm.batched.batched_gemm_bf16",
            "aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_bf16",
            "aiter.jit.module_rope_1c_cached_bwd",
            "aiter.jit.module_rope_1c_cached_fwd",
        )
        try:
            modules = [importlib.import_module(name) for name in names]
            self._inverse = modules[0].rope_cached_bwd
            self._adjoint = modules[0].rope_cached_fwd
            self._grouped = modules[1].batched_gemm_bf16
        except (ImportError, AttributeError) as exc:
            raise OProjectionError("required grouped GEMM/RoPE ABI missing; no fallback") from exc
        self.provenance["sources"].update(
            {
                name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in zip(names, modules, strict=True)
            }
        )
        self.provenance["grouped_config"] = self.GROUPED_CONFIG
        self.provenance["o_projection_adapter_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()

    def inverse_rope(self, x, cos, sin):
        if not x.shape[0]:
            return x.clone()
        return self._inverse(x[:, None], cos, sin, 1, True, True).squeeze(1)

    def inverse_rope_gradient(self, grad, cos, sin):
        if not grad.shape[0]:
            return grad.clone()
        return self._adjoint(grad[:, None].contiguous(), cos, sin, 1, True, True).squeeze(1)

    def grouped(self, x, weight):
        if not x.shape[1]:
            return torch.empty(
                (x.shape[0], 0, weight.shape[1]), dtype=torch.bfloat16, device=x.device
            )
        return self._grouped(x, weight, config=dict(self.GROUPED_CONFIG))
