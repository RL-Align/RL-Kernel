# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Registry and semantic-entry tests for ``h3_sigma_schedule`` (RFC #420).

Covers the public selection paths on any machine: legacy ``get_op`` serving
the PyTorch reference on every platform (the Triton backend is opt-in only and
never a legacy candidate), semantic descriptor registration, session resolve /
instantiate / provenance for both backends, and the CPU-pinned scalar-rounding
helper. The device-path parity (the default reference path producing the same
bytes as the directly constructed Triton class) is asserted on hardware in
tests/backends/cuda/test_h3_sigma_schedule_cuda.py.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from rl_engine.runtime.registry import KernelRegistry, OpBackend, kernel_registry
from rl_engine.runtime.semantic_registry import (
    OperatorInstantiationError,
    OperatorRequirements,
    OperatorResolutionError,
)

_BACKEND_MODULE = importlib.import_module(
    "rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule"
)
_REFERENCE_PATH = "rl_engine.reference.minimax_h3.sigma_schedule.NativeH3SigmaScheduleOp"


def test_get_op_cpu_selects_reference_and_builds_pinned_grid():
    op = kernel_registry.get_op("sigma_schedule_h3", device="cpu")
    assert f"{type(op).__module__}.{type(op).__name__}" == _REFERENCE_PATH

    schedule = op(4, shift=3.0, device="cpu")
    assert [f"{v:08x}" for v in schedule.sigmas.view(torch.int32).tolist()] == [
        "3f800000",
        "3f5b6db6",
        "3f199999",
        "00000000",
    ]
    assert schedule.num_model_evals == 3


def test_get_op_cuda_defaults_to_reference_regardless_of_triton(monkeypatch):
    # Default backend change per review (2026-10-25): the CUDA candidate list
    # carries only the PyTorch reference (CPU build + host-to-device copy) —
    # the device-side Triton build measured 2.5-3.2x slower at typical H3
    # sizes and nothing requires on-device construction. Triton is opt-in via
    # the semantic path or direct construction, so whether it is importable
    # must not influence the legacy selection in either direction.
    for triton_importable in (False, True):
        monkeypatch.setattr(_BACKEND_MODULE, "_TRITON_AVAILABLE", triton_importable)
        registry = KernelRegistry()

        assert registry._priority_map["cuda"]["sigma_schedule_h3"] == [
            OpBackend.PYTORCH_H3_SIGMA_SCHEDULE
        ]
        op = registry.get_op("sigma_schedule_h3", device="cuda")
        assert f"{type(op).__module__}.{type(op).__name__}" == _REFERENCE_PATH


def test_cuda_backend_construction_fails_loudly_without_triton(monkeypatch):
    monkeypatch.setattr(_BACKEND_MODULE, "_TRITON_AVAILABLE", False)

    with pytest.raises(ImportError, match="triton is required"):
        _BACKEND_MODULE.H3SigmaScheduleCudaOp()


def test_shift_scalars_pinned_to_cpu_under_meta_default():
    # Old behavior: the factory calls followed the ambient default device and
    # failed under a meta default (RuntimeError on .item()).
    with torch.device("meta"):
        shift_f32, shift_m1_f32 = _BACKEND_MODULE._shift_scalars(3.0)
    assert shift_f32 == 3.0
    assert shift_m1_f32 == 2.0


def test_shift_scalars_keep_fp64_subtract_then_round_semantics():
    # Pinned artifact of the upstream scalar handling: (1e-10 - 1.0) is
    # subtracted in fp64 first and only then rounded to fp32, landing on
    # exactly -1.0 (this feeds the division-by-zero +inf first sigma).
    _, shift_m1_f32 = _BACKEND_MODULE._shift_scalars(1e-10)
    assert shift_m1_f32 == -1.0


def test_cuda_backend_construction_rejected_on_hip_build(monkeypatch):
    # The kernels emit inline PTX div.rn.f32 and cannot run on AMD: a HIP
    # build must be rejected AT CONSTRUCTION (long before any Triton
    # compilation), with the same ImportError contract the registry's
    # candidate loop already handles for missing Triton. Simulated without
    # any AMD hardware by monkeypatching torch.version.hip.
    monkeypatch.setattr(torch.version, "hip", "6.2")

    with pytest.raises(ImportError, match="NVIDIA-only.*inline.*PTX"):
        _BACKEND_MODULE.H3SigmaScheduleCudaOp()


def test_get_op_on_hip_build_serves_reference(monkeypatch):
    # Same simulation, end to end: ``device="cuda"`` resolves to the rocm
    # platform, whose list has always been reference-only, and the NVIDIA-only
    # Triton class is not a legacy candidate anywhere — the caller gets the
    # reference with no error and nothing silently misattributed, exactly as
    # on an NVIDIA build.
    monkeypatch.setattr(torch.version, "hip", "6.2")
    registry = KernelRegistry()

    op = registry.get_op("sigma_schedule_h3", device="cuda")

    assert f"{type(op).__module__}.{type(op).__name__}" == _REFERENCE_PATH


def test_triton_backend_is_not_a_legacy_candidate_on_any_platform():
    # Opt-in only: the Triton class is reachable through the semantic path
    # (requested_backend="rlkernel.h3_sigma_schedule.triton") or direct
    # construction, never through a legacy ``get_op`` candidate list.
    registry = KernelRegistry()

    for platform in ("cuda", "rocm", "musa", "cpu", "npu"):
        candidates = registry._priority_map[platform]["sigma_schedule_h3"]
        assert candidates == [OpBackend.PYTORCH_H3_SIGMA_SCHEDULE], platform


def test_semantic_descriptors_declared_truthfully():
    reference = kernel_registry.semantic.backend_descriptor(
        "sigma_schedule_h3", "rlkernel.h3_sigma_schedule.reference"
    )
    triton = kernel_registry.semantic.backend_descriptor(
        "sigma_schedule_h3", "rlkernel.h3_sigma_schedule.triton"
    )
    assert reference is not None and triton is not None
    assert reference.supported_devices == frozenset({"cpu", "cuda", "rocm"})
    # NVIDIA-only backend (inline PTX); HIP/ROCm stays on the reference.
    assert triton.supported_devices == frozenset({"cuda"})
    assert reference.supported_dtypes == triton.supported_dtypes == frozenset({"float32"})
    assert reference.implementation_reference == _REFERENCE_PATH


def test_semantic_reference_path_resolves_instantiates_and_reports_provenance():
    session = kernel_registry.semantic.session()
    resolution = session.resolve(
        semantic_op="sigma_schedule_h3",
        requested_backend="rlkernel.h3_sigma_schedule.reference",
        target="rollout",
        requirements=OperatorRequirements(device="cpu", dtype="float32"),
    )
    assert resolution.trace.status == "resolved"
    assert resolution.trace.concrete_backend == "rlkernel.h3_sigma_schedule.reference"
    assert resolution.trace.implementation_reference == _REFERENCE_PATH

    instance = session.instantiate(resolution)
    assert type(instance).__name__ == "NativeH3SigmaScheduleOp"
    assert instance(4, shift=3.0, device="cpu").num_model_evals == 3

    provenance = session.instance_provenance(resolution, instance)
    assert provenance.semantic_op == "sigma_schedule_h3"
    assert provenance.backend_id == "rlkernel.h3_sigma_schedule.reference"
    assert provenance.concrete_implementation == _REFERENCE_PATH
    assert provenance.factory_reference == _REFERENCE_PATH


def test_semantic_triton_backend_fails_explicitly_when_unavailable(monkeypatch):
    # Resolution matches the declared capability (device "cuda"); the missing
    # runtime dependency then surfaces as an explicit instantiation error —
    # never as a silent swap to another backend's arithmetic.
    monkeypatch.setattr(_BACKEND_MODULE, "_TRITON_AVAILABLE", False)
    session = kernel_registry.semantic.session()
    resolution = session.resolve(
        semantic_op="sigma_schedule_h3",
        requested_backend="rlkernel.h3_sigma_schedule.triton",
        target="rollout",
        requirements=OperatorRequirements(device="cuda", dtype="float32"),
    )
    with pytest.raises(OperatorInstantiationError, match="triton is required"):
        session.instantiate(resolution)


def test_semantic_unknown_backend_is_rejected():
    session = kernel_registry.semantic.session()
    with pytest.raises(OperatorResolutionError):
        session.resolve(
            semantic_op="sigma_schedule_h3",
            requested_backend="rlkernel.h3_sigma_schedule.does-not-exist",
            target="rollout",
            requirements=OperatorRequirements(device="cpu", dtype="float32"),
        )
