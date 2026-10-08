# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""P5 provider interface (P5-6, issue #65) plus reference and stub implementations.

A provider implements the five WS1 operators. Sub-issue owners (D1-D6)
subclass :class:`ReferenceProvider` and override only the methods their PR
delivers; every other method stays on the oracle, so each PR can run the full
acceptance command independently.

Fail-closed contract: a provider must raise on unsupported input instead of
silently falling back to another implementation, and ``provenance()`` must
report the backend that actually ran.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import torch

from rl_engine.moe import oracle
from rl_engine.moe.contract import ORACLE_PROFILE, SharedBatch
from rl_engine.moe.mx_format import MXTensor


@runtime_checkable
class ExpertProvider(Protocol):
    """The five P5 WS1 operators. See ``oracle`` for the frozen semantics."""

    name: str
    numeric_profile: str

    def capabilities(self) -> dict[str, Any]: ...

    def provenance(self) -> dict[str, Any]: ...

    def mxfp8_act_quant_fwd(self, x: torch.Tensor) -> MXTensor: ...

    def mxfp8_act_quant_bwd(self, dy: torch.Tensor) -> torch.Tensor: ...

    def mxfp8_mxfp4_grouped_gemm_fwd(
        self, a: MXTensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor: ...

    def mxfp8_mxfp4_grouped_gemm_bwd(
        self, dy: torch.Tensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor: ...

    def shared_grouped_lora_delta_fwd(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def shared_grouped_lora_delta_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        u_bf16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def clamp_swiglu_weighted_fwd(
        self, gate: torch.Tensor, up: torch.Tensor, p_s: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def clamp_swiglu_weighted_bwd(
        self, dh: torch.Tensor, saved: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[torch.Tensor, dict[str, Any]]: ...

    def shared_expert_mlp_bwd(
        self, dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> torch.Tensor: ...


class ReferenceProvider:
    """Binds the FP32 oracle. Always passes acceptance; defines the golden bytes."""

    name = "reference"
    numeric_profile = ORACLE_PROFILE

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "pytorch-oracle",
            "geometry": ["one-row", "packed"],
            "devices": ["cpu", "cuda"],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "torch_version": torch.__version__,
        }

    mxfp8_act_quant_fwd = staticmethod(oracle.mxfp8_act_quant_fwd)
    mxfp8_act_quant_bwd = staticmethod(oracle.mxfp8_act_quant_bwd)
    mxfp8_mxfp4_grouped_gemm_fwd = staticmethod(oracle.mxfp8_mxfp4_grouped_gemm_fwd)
    mxfp8_mxfp4_grouped_gemm_bwd = staticmethod(oracle.mxfp8_mxfp4_grouped_gemm_bwd)
    shared_grouped_lora_delta_fwd = staticmethod(oracle.shared_grouped_lora_delta_fwd)
    shared_grouped_lora_delta_bwd = staticmethod(oracle.shared_grouped_lora_delta_bwd)
    clamp_swiglu_weighted_fwd = staticmethod(oracle.clamp_swiglu_weighted_fwd)
    clamp_swiglu_weighted_bwd = staticmethod(oracle.clamp_swiglu_weighted_bwd)
    shared_expert_mlp_fwd = staticmethod(oracle.shared_expert_mlp_fwd)
    shared_expert_mlp_bwd = staticmethod(oracle.shared_expert_mlp_bwd)


class StubProvider(ReferenceProvider):
    """Fail-closed placeholder: every operator raises until a backend claims it.

    This is deliberately NOT a fallback to the oracle — P5-6 (#65) forbids
    silent fallback, so an unimplemented operator must be loud.
    """

    name = "stub"
    numeric_profile = "unimplemented"

    @staticmethod
    def _todo(issue: str) -> NotImplementedError:
        return NotImplementedError(
            f"P5 operator not implemented; claim it on issue {issue} "
            "(fail-closed: no silent fallback to the oracle)"
        )

    def mxfp8_act_quant_fwd(self, x: torch.Tensor) -> MXTensor:
        raise self._todo("P5-1 (#60)")

    def mxfp8_act_quant_bwd(self, dy: torch.Tensor) -> torch.Tensor:
        raise self._todo("P5-1 (#60)")

    def mxfp8_mxfp4_grouped_gemm_fwd(
        self, a: MXTensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        raise self._todo("P5-4 (#61)")

    def mxfp8_mxfp4_grouped_gemm_bwd(
        self, dy: torch.Tensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        raise self._todo("P5-4 (#61)")

    def shared_grouped_lora_delta_fwd(
        self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise self._todo("P5-3 (#62)")

    def shared_grouped_lora_delta_bwd(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        alpha: float,
        u_bf16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise self._todo("P5-3 (#62)")

    def clamp_swiglu_weighted_fwd(
        self, gate: torch.Tensor, up: torch.Tensor, p_s: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise self._todo("P5-2 (#63)")

    def clamp_swiglu_weighted_bwd(
        self, dh: torch.Tensor, saved: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raise self._todo("P5-2 (#63)")

    def shared_expert_mlp_fwd(self, batch: SharedBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        raise self._todo("P5-5 (#64)")

    def shared_expert_mlp_bwd(
        self, dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
    ) -> torch.Tensor:
        raise self._todo("P5-5 (#64)")


def resolve_provider(spec: str) -> ExpertProvider:
    """Instantiate a provider from ``"module.path:ClassName"`` (or an alias)."""
    aliases = {
        "reference": "rl_engine.moe.provider:ReferenceProvider",
        "stub": "rl_engine.moe.provider:StubProvider",
        # Kernel backends live in rl_engine.moe.backends; this module keeps
        # only the protocol, the oracle-backed reference, and the stub.
        #
        # The production fused kernel (backends.sm90_fused_mlp) is deliberately
        # absent: it fuses the whole routed MLP behind one call rather than
        # implementing the nine per-operator hooks, so it is not an
        # ExpertProvider and check_p5.py cannot drive it.
        "cuda": "rl_engine.moe.backends.grouped_gemm:CudaP5GemmProvider",
    }
    spec = aliases.get(spec, spec)
    if ":" not in spec:
        raise ValueError(f"provider spec {spec!r} must look like 'module.path:ClassName'")
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    instance = cls()
    if not isinstance(instance, ExpertProvider):
        raise TypeError(f"{spec} does not implement the ExpertProvider protocol")
    return instance
