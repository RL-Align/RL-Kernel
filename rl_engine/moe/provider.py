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


class NativeP5Provider(ReferenceProvider):
    """Fail-closed adapter for the P5 CUDA ABI skeleton.

    The native entry points currently expose the documented pointer/shape
    contract but intentionally raise until strict kernels are implemented.
    This adapter must never silently call the Python oracle.
    """

    name = "native-p5"
    numeric_profile = "p5-strict-skeleton-unimplemented"

    @staticmethod
    def _extension():
        try:
            from rl_engine import _C
        except ImportError as exc:
            raise RuntimeError("P5 native extension is not built") from exc
        return _C

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "cuda-p5-skeleton",
            "geometry": ["one-row"],
            "devices": ["cuda"],
            "implemented": [],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "numeric_profile": self.numeric_profile,
            "implementation": "interface-skeleton",
        }

    def mxfp8_act_quant_fwd(self, x: torch.Tensor) -> MXTensor:
        codes, scales = self._extension().moe_mxfp8_act_quant_forward(x)
        return MXTensor(codes, scales, "e4m3", tuple(x.shape))

    def mxfp8_mxfp4_grouped_gemm_fwd(
        self, a: MXTensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        return self._extension().moe_mxfp8_mxfp4_grouped_gemm_forward(
            a.codes, a.scales, w.codes, w.scales, expert_offsets
        )

    def mxfp8_mxfp4_grouped_gemm_bwd(
        self, dy: torch.Tensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        return self._extension().moe_mxfp8_mxfp4_grouped_gemm_backward(
            dy, w.codes, w.scales, expert_offsets
        )

    def mxfp8_act_quant_bwd(self, dy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("P5-1 native backward is not part of this skeleton")


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


class CudaP5GemmProvider(ReferenceProvider):
    """Strict CUDA GEMM backend for P5-4 only.

    Overrides the MXFP8xMXFP4 grouped GEMM fwd/bwd with the hand-written strict
    FFMA kernel (bit-exact vs the FP32 oracle). Every other operator stays on
    the oracle. Fail-closed: raises if the native extension is not built.
    """

    name = "cuda-p5-gemm"
    numeric_profile = "p5-strict-ffma-v1"

    @staticmethod
    def _extension():
        try:
            from rl_engine import _C
        except ImportError as exc:
            raise RuntimeError("P5 CUDA extension is not built") from exc
        return _C

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "cuda-p5-strict",
            "geometry": ["one-row"],
            "devices": ["cuda"],
            "implemented": [
                "mxfp8_mxfp4_grouped_gemm_fwd",
                "mxfp8_mxfp4_grouped_gemm_bwd",
            ],
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "requested_backend": self.name,
            "actual_backend": "cuda-strict",
            "numeric_profile": self.numeric_profile,  # "p5-strict-ffma-v1"
            "geometry": "one-row-unpadded",
            "tile": None,  # no tiling in the strict one-row path
            "split_k": 1,  # no split-K
            "kernel_fingerprint": "mxfp8-mxfp4-grouped-gemm-strict-v1",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    def mxfp8_mxfp4_grouped_gemm_fwd(
        self, a: MXTensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        return self._extension().moe_mxfp8_mxfp4_grouped_gemm_forward(
            a.codes, a.scales, w.codes, w.scales, expert_offsets
        )

    def mxfp8_mxfp4_grouped_gemm_bwd(
        self, dy: torch.Tensor, w: MXTensor, expert_offsets: torch.Tensor
    ) -> torch.Tensor:
        return self._extension().moe_mxfp8_mxfp4_grouped_gemm_backward(
            dy, w.codes, w.scales, expert_offsets
        )


def resolve_provider(spec: str) -> ExpertProvider:
    """Instantiate a provider from ``"module.path:ClassName"`` (or an alias)."""
    aliases = {
        "reference": "rl_engine.moe.provider:ReferenceProvider",
        "stub": "rl_engine.moe.provider:StubProvider",
        "cuda": "rl_engine.moe.provider:CudaP5GemmProvider",
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
