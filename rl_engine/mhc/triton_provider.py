from __future__ import annotations

from rl_engine.kernels.ops.triton.rmsnorm_residual_triton import (
    triton_rmsnorm_residual_bwd,
    triton_rmsnorm_residual_fwd,
)
from rl_engine.mhc.provider import ReferenceProvider


class TritonMHCProvider(ReferenceProvider):
    """P1-5 Triton implementation with oracle fallbacks for the other P1 ops."""

    name = "triton-rmsnorm-residual"

    def capabilities(self):
        capabilities = super().capabilities()
        capabilities.update(
            {
                "backend": "triton-rmsnorm-residual",
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
                "actual_backend": "triton-rmsnorm-residual+oracle-rest",
            }
        )
        return provenance

    rmsnorm_residual_fwd = staticmethod(triton_rmsnorm_residual_fwd)
    rmsnorm_residual_bwd = staticmethod(triton_rmsnorm_residual_bwd)
