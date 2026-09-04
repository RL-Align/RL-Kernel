# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P1 provider interface (``P1-D4``) plus reference and fail-closed stub.

A provider implements the six WS1 operators. Sub-issue owners (D1-D6)
subclass :class:`ReferenceProvider` and override only the methods their PR
delivers; every other method stays on the oracle, so each PR can run the full
acceptance command independently and land without waiting on the others.

Fail-closed contract (issue #2): a provider must raise on an input it does not
support instead of silently falling back to another implementation, and
``provenance()`` must report the backend that actually ran. When an external
core (TE / Megatron) fails strict bytes, the dispatcher switches to the
RL-Kernel core through :meth:`capabilities` -- never silently.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import torch

from rl_engine.mhc import oracle
from rl_engine.mhc.contract import ORACLE_PROFILE, LayerContract, ResidualBatch


@runtime_checkable
class MHCProvider(Protocol):
    """The six P1 WS1 operators. See :mod:`rl_engine.mhc.oracle` for semantics."""

    name: str
    numeric_profile: str

    def capabilities(self) -> dict[str, Any]: ...

    def provenance(self) -> dict[str, Any]: ...

    def hc_split_sinkhorn_fwd(
        self, h: torch.Tensor, contract: LayerContract | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def hc_split_sinkhorn_bwd(
        self,
        dpre: torch.Tensor,
        dpost: torch.Tensor,
        dc: torch.Tensor,
        saved: dict[str, Any],
    ) -> torch.Tensor: ...

    def fp32_gemm_rms_fwd(
        self, x_flat: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def fp32_gemm_rms_bwd(
        self,
        dp: torch.Tensor,
        dr: torch.Tensor,
        x_flat: torch.Tensor,
        weight: torch.Tensor,
        saved: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def mhc_post_fwd(
        self, r_old: torch.Tensor, y: torch.Tensor, c: torch.Tensor, post: torch.Tensor
    ) -> torch.Tensor: ...

    def mhc_post_bwd(
        self,
        dr_new: torch.Tensor,
        r_old: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
        post: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def h_aggregate_fwd(self, pre: torch.Tensor, r_old: torch.Tensor) -> torch.Tensor: ...

    def h_aggregate_bwd(
        self, dh: torch.Tensor, pre: torch.Tensor, r_old: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def rmsnorm_residual_fwd(
        self, x: torch.Tensor, gamma: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def rmsnorm_residual_bwd(
        self,
        dy: torch.Tensor,
        d_residual: torch.Tensor,
        x: torch.Tensor,
        gamma: torch.Tensor,
        saved: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def fixed_k_gemm_fwd(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor: ...

    def fixed_k_gemm_bwd(
        self, dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def mhc_pre_fwd(
        self, batch: ResidualBatch, ops: Any = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def mhc_pre_bwd(
        self,
        dhidden: torch.Tensor,
        dpost: torch.Tensor,
        dc: torch.Tensor,
        batch: ResidualBatch,
        saved: dict[str, Any],
        ops: Any = None,
    ) -> dict[str, torch.Tensor | None]: ...

    def mhc_pre_rmsnorm_fused_fwd(
        self, batch: ResidualBatch, ops: Any = None
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]: ...


class ReferenceProvider:
    """Binds the FP32 oracle. Always passes acceptance; defines the golden bytes."""

    name = "reference"
    numeric_profile = ORACLE_PROFILE

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "pytorch-oracle",
            "geometry": ["one-row", "packed"],
            "devices": ["cpu", "cuda"],
            "fusion_modes": list(oracle.SUPPORTED_FUSION),
            "trainability": list(oracle.SUPPORTED_TRAINABILITY),
            "placements": ["replicated"],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
        }

    hc_split_sinkhorn_fwd = staticmethod(oracle.hc_split_sinkhorn_fwd)
    hc_split_sinkhorn_bwd = staticmethod(oracle.hc_split_sinkhorn_bwd)
    fp32_gemm_rms_fwd = staticmethod(oracle.fp32_gemm_rms_fwd)
    fp32_gemm_rms_bwd = staticmethod(oracle.fp32_gemm_rms_bwd)
    mhc_post_fwd = staticmethod(oracle.mhc_post_fwd)
    mhc_post_bwd = staticmethod(oracle.mhc_post_bwd)
    h_aggregate_fwd = staticmethod(oracle.h_aggregate_fwd)
    h_aggregate_bwd = staticmethod(oracle.h_aggregate_bwd)
    rmsnorm_residual_fwd = staticmethod(oracle.rmsnorm_residual_fwd)
    rmsnorm_residual_bwd = staticmethod(oracle.rmsnorm_residual_bwd)
    fixed_k_gemm_fwd = staticmethod(oracle.fixed_k_gemm_fwd)
    fixed_k_gemm_bwd = staticmethod(oracle.fixed_k_gemm_bwd)
    mhc_pre_fwd = staticmethod(oracle.mhc_pre_fwd)
    mhc_pre_bwd = staticmethod(oracle.mhc_pre_bwd)
    mhc_pre_rmsnorm_fused_fwd = staticmethod(oracle.mhc_pre_rmsnorm_fused_fwd)


class StubProvider(ReferenceProvider):
    """Fail-closed placeholder: every operator raises until a backend claims it.

    Deliberately NOT a fallback to the oracle -- issue #2 forbids silent
    fallback, so an unimplemented operator must be loud.
    """

    name = "stub"
    numeric_profile = "unimplemented"

    @staticmethod
    def _todo(task: str) -> NotImplementedError:
        return NotImplementedError(
            f"P1 operator not implemented; claim it on {task} "
            "(fail-closed: no silent fallback to the oracle)"
        )

    def hc_split_sinkhorn_fwd(self, h, contract=None):
        raise self._todo("P1-1 (#14)")

    def hc_split_sinkhorn_bwd(self, dpre, dpost, dc, saved):
        raise self._todo("P1-1 (#14)")

    def fp32_gemm_rms_fwd(self, x_flat, weight, eps):
        raise self._todo("P1-2 (#15)")

    def fp32_gemm_rms_bwd(self, dp, dr, x_flat, weight, saved):
        raise self._todo("P1-2 (#15)")

    def mhc_post_fwd(self, r_old, y, c, post):
        raise self._todo("P1-3 (#16)")

    def mhc_post_bwd(self, dr_new, r_old, y, c, post):
        raise self._todo("P1-3 (#16)")

    def h_aggregate_fwd(self, pre, r_old):
        raise self._todo("P1-4 (#17)")

    def h_aggregate_bwd(self, dh, pre, r_old):
        raise self._todo("P1-4 (#17)")

    def rmsnorm_residual_fwd(self, x, gamma, eps):
        raise self._todo("P1-5 (#18)")

    def rmsnorm_residual_bwd(self, dy, d_residual, x, gamma, saved):
        raise self._todo("P1-5 (#18)")

    def fixed_k_gemm_fwd(self, x, w):
        raise self._todo("P1-2 (#15)")

    def fixed_k_gemm_bwd(self, dy, x, w):
        raise self._todo("P1-2 (#15)")


def resolve_provider(spec: str) -> MHCProvider:
    """Instantiate a provider from ``"module.path:ClassName"`` (or an alias)."""
    aliases = {
        "reference": "rl_engine.mhc.provider:ReferenceProvider",
        "stub": "rl_engine.mhc.provider:StubProvider",
    }
    spec = aliases.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"provider spec {spec!r} must look like 'module.path:ClassName'")
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    instance = cls()
    if not isinstance(instance, MHCProvider):
        raise TypeError(f"{spec} does not implement the MHCProvider protocol")
    return instance


def check_capability(provider: MHCProvider, contract: LayerContract) -> None:
    """Fail closed when the contract asks for something the provider lacks.

    This is the hook ``P1-D4`` uses at the TE/Megatron/Miles dispatch point:
    an unsupported fusion mode, trainability mode or placement must raise
    here, never degrade into a different-but-similar implementation.
    """
    caps = provider.capabilities()
    for key, label, want in (
        ("fusion_modes", "fusion mode", contract.fusion_mode),
        ("trainability", "trainability mode", contract.trainability),
        ("placements", "placement", contract.placement),
    ):
        supported = caps.get(key)
        if supported is not None and want not in supported:
            raise NotImplementedError(
                f"provider {provider.name!r} does not support {label} {want!r}; "
                f"supported: {supported} (fail-closed)"
            )


__all__ = [
    "MHCProvider",
    "ReferenceProvider",
    "StubProvider",
    "check_capability",
    "resolve_provider",
]
