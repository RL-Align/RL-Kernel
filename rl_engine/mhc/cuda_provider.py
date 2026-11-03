from __future__ import annotations
from typing import Any

from rl_engine.kernels.ops.cuda.norm.rmsnorm_residual import (
    cuda_rmsnorm_residual_bwd,
    cuda_rmsnorm_residual_fwd,
)
from rl_engine.mhc.provider import ReferenceProvider


class CudaMHCProvider(ReferenceProvider):

    name = "cuda-rmsnorm-residual"

    def capabilities(self):
        capabilities = super().capabilities()
        capabilities.update(
            {
                "backend": "cuda-rmsnorm-residual",
                "devices": ["cuda"],
                "implemented_operators": ["rmsnorm_residual"],
            }
        )
        return capabilities

    def provenance(self):
        provenance = super().provenance()
        provenance.update(
            {
                "requested_backend": self.name,
                "actual_backend": "cuda-rmsnorm-residual+oracle-rest",
            }
        )
        return provenance

    rmsnorm_residual_fwd = staticmethod(cuda_rmsnorm_residual_fwd)
    rmsnorm_residual_bwd = staticmethod(cuda_rmsnorm_residual_bwd)
