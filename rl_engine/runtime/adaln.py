# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""AdaLN execution policy; arithmetic stays in the existing backend operators."""

from __future__ import annotations

import sys
from typing import Any

import torch

SHARED_PROFILE = "adaln-shared-fp32-tree-v1"
SELECT01_PROFILE = "adaln-select01-framework-v1"
INDEXED_PROFILE = "adaln-indexed-fp32-tree-v1"
_BACKENDS = {
    "rl_engine.reference.norm.adaln_modulation": ("NativeAdaLNModulationOp", "pytorch"),
    "rl_engine.backends.cuda.norm.adaln_modulation": ("CudaAdaLNModulationOp", "cuda"),
    "rl_engine.backends.shared.triton.norm.adaln_modulation": ("TritonAdaLNModulationOp", "triton"),
}


def adaln_backend(op: Any) -> str | None:
    cls = type(op)
    entry = _BACKENDS.get(cls.__module__)
    if entry and getattr(sys.modules[cls.__module__], entry[0], None) is cls:
        return entry[1]
    return None


class AdaLNDispatch:
    """A resolution-local trace, updated only after successful operator execution.

    Strict enforces the declared arithmetic, not hardware qualification. CUDA and
    Triton share the profile; Native on GPU is excluded pending its replay audit.
    """

    def __init__(self, implementation, trace, *, hidden, strict=False):
        self.implementation = implementation
        self.trace = trace
        self.hidden = hidden
        self.strict = strict
        self.arithmetic_profile = trace["arithmetic_profile"]
        trace.update(
            actual_backend=None,
            observation=None,
            execution_dtype=None,
            output_dtype=None,
            execution_device=None,
            binary_path=None,
            binary_sha256=None,
            hardware_qualified=None,
            profiler_observed=False,
            semantic_version="adaln-v1",
            strict=strict,
        )
        self._resolution_trace = trace.copy()

    def __call__(self, x, modulation, *, eps=1e-6, modulate_index=None):
        return self.forward(x, modulation, eps=eps, modulate_index=modulate_index)

    def forward(self, x, modulation, *, eps=1e-6, modulate_index=None):
        self.trace.clear()
        self.trace.update(self._resolution_trace)
        backend = adaln_backend(self.implementation)
        indexed = modulate_index is not None
        profile = SELECT01_PROFILE if indexed else SHARED_PROFILE
        if self.strict:
            if indexed:
                raise RuntimeError("strict AdaLN rejects select01 framework backward")
            if backend is None or (backend == "pytorch" and x.device.type != "cpu"):
                raise RuntimeError("strict AdaLN backend has no declared replay profile")
            if backend in {"cuda", "triton"} and (
                x.device.type != "cuda" or torch.version.hip is not None
            ):
                raise RuntimeError("strict AdaLN GPU profile requires NVIDIA CUDA")
            if self.arithmetic_profile != profile:
                raise RuntimeError("strict AdaLN arithmetic profile mismatch")
            if x.ndim != 3 or x.shape[-1] != self.hidden:
                raise ValueError("strict AdaLN input hidden differs from resolved hidden")
        if indexed:
            implementation = self.implementation
            if backend in {"cuda", "triton"}:
                from rl_engine.backends.cuda.norm.adaln_modulation import _indexed_slice_supported

                if (
                    backend == "cuda"
                    and getattr(implementation, "indexed_available", False)
                    and _indexed_slice_supported(x, modulate_index, eps)
                ):
                    profile = INDEXED_PROFILE
                else:
                    from rl_engine.reference.norm.adaln_modulation import NativeAdaLNModulationOp

                    implementation = NativeAdaLNModulationOp()
                    backend = "pytorch"
            outputs = implementation(x, modulation, eps=eps, modulate_index=modulate_index)
        else:
            outputs = self.implementation(x, modulation, eps=eps)
        self.trace.update(
            actual_backend=backend,
            observation="successful_forward",
            arithmetic_profile=profile,
            execution_dtype=str(x.dtype).removeprefix("torch."),
            output_dtype=str(outputs[0].dtype).removeprefix("torch."),
            execution_device=str(x.device),
        )
        if indexed and profile == SELECT01_PROFILE:
            self.trace.update(
                fallback=True,
                fallback_reason=(
                    "indexed_cuda_slice_unsupported"
                    if self._resolution_trace["arithmetic_profile"] == INDEXED_PROFILE
                    else "modulate_index_requires_select01"
                ),
                reduction_order="pairwise_lower_upper_H; framework_select01_backward",
                kernel_fingerprint="adaln_modulation_select01_native_v1",
            )
        elif indexed:
            self.trace.update(
                fallback=False,
                fallback_reason=None,
                reduction_order="pairwise_lower_upper_H; ascending_selected_token_S_per_bank",
                kernel_fingerprint="adaln_indexed_cuda_v1",
            )
        elif self._resolution_trace["arithmetic_profile"] == INDEXED_PROFILE:
            self.trace.update(
                reduction_order="pairwise_lower_upper_H; ascending_logical_token_S",
                kernel_fingerprint=f"adaln_modulation_v1:{self.trace['selected_backend']}",
            )
        if backend == "cuda":
            from rl_engine.backends.extension import _C

            self.trace["binary_path"] = getattr(_C, "__file__", None)
        return outputs
