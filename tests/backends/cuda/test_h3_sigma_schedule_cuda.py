# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Device tests for the Triton NVIDIA CUDA h3_sigma_schedule backend (RFC #420).

The acceptance oracle is the pinned CPU fixture generated from the upstream
diffusers scheduler (see tests/reference/minimax_h3/). GPU results are copied
to the host only for comparison; no value is recomputed on the CPU.

The backend is NVIDIA-only (its kernels emit inline PTX ``div.rn.f32``);
the suite runs on NVIDIA CUDA hardware and skips on HIP/ROCm builds, where
``H3SigmaScheduleCudaOp`` refuses construction and the registry serves the
PyTorch reference instead.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA device required: the Triton path is NVIDIA-only "
    "(skipped without a GPU and on HIP/ROCm builds)",
)

_FIXTURE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "reference"
    / "minimax_h3"
    / "h3_sigma_schedule_upstream_fixture.json"
)
_EXPECTED_CASES = {(2, 3.0), (2, 12.0), (4, 3.0), (4, 12.0), (25, 3.0), (25, 12.0)}


def _cuda_op():
    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )

    return H3SigmaScheduleCudaOp()


def _int32_bits(tensor: torch.Tensor) -> list[int]:
    return tensor.detach().contiguous().view(torch.int32).tolist()


def _fixture_bits(bits: list[str]) -> list[int]:
    return [int(value, 16) for value in bits]


def test_matches_pinned_cpu_fixture_bitwise():
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["upstream"]["blob_sha1"] == "ca69ae6a2f6aae5bb68498216316a71eb07c1f7b"

    actual_cases = [(case["num_inference_steps"], case["shift"]) for case in fixture["cases"]]
    assert set(actual_cases) == _EXPECTED_CASES

    op = _cuda_op()
    for case in fixture["cases"]:
        schedule = op(case["num_inference_steps"], shift=case["shift"])

        assert schedule.sigmas.is_cuda
        assert schedule.sigmas.dtype is torch.float32
        assert tuple(schedule.sigmas.shape) == (case["sigmas_count"],)
        assert _int32_bits(schedule.sigmas.cpu()) == _fixture_bits(case["sigmas_bits"])

        assert schedule.timesteps.is_cuda
        assert schedule.timesteps.dtype is torch.float32
        assert tuple(schedule.timesteps.shape) == (case["timesteps_count"],)
        expected = _fixture_bits(case["timesteps_bits"])
        assert _int32_bits(schedule.timesteps.cpu()) == expected

        assert schedule.num_model_evals == case["num_model_evals"]


@pytest.mark.parametrize(
    ("num_inference_steps", "shift"),
    [
        # N=50, shift=12 covers the fusion-sensitive elements (e.g. index 16)
        # where a contracted FMA and separate mul/add roundings differ in the
        # CPU simulation; the launch disables fp fusion and this comparison is
        # the hardware verdict.
        (50, 12.0),
        (50, 3.0),
        (1000, 12.0),
    ],
)
def test_matches_live_cpu_reference_bitwise(num_inference_steps, shift):
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(num_inference_steps, shift=shift)
    actual = _cuda_op()(num_inference_steps, shift=shift)

    assert _int32_bits(actual.sigmas.cpu()) == _int32_bits(expected.sigmas)
    assert _int32_bits(actual.timesteps.cpu()) == _int32_bits(expected.timesteps)
    assert actual.num_model_evals == expected.num_model_evals


def test_dense_grid_dedup_matches_cpu_reference():
    # shift=12 creates fp32 collisions near sigma=1 at this density: the grid
    # collapses by ~4%, so the compaction path is genuinely exercised.
    n = 1_000_000
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(n, shift=12.0)
    actual = _cuda_op()(n, shift=12.0)

    assert actual.sigmas.numel() < n
    assert actual.sigmas.numel() == expected.sigmas.numel()
    assert _int32_bits(actual.sigmas.cpu()) == _int32_bits(expected.sigmas)
    assert _int32_bits(actual.timesteps.cpu()) == _int32_bits(expected.timesteps)
    assert actual.num_model_evals == expected.num_model_evals


def test_reference_with_cuda_device_preserves_fixture_bytes():
    # Path-B oracle (the benchmark's no-GPU-backend baseline): same CPU build,
    # then the operator moves sigmas/timesteps to the GPU. The move must
    # preserve the pinned bytes — this also guards against a later
    # "optimization" building the grid directly on the device inside the
    # reference, which would silently change its arithmetic.
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    op = NativeH3SigmaScheduleOp()
    for case in fixture["cases"]:
        schedule = op(case["num_inference_steps"], shift=case["shift"], device="cuda")

        assert schedule.sigmas.is_cuda and schedule.timesteps.is_cuda
        assert _int32_bits(schedule.sigmas.cpu()) == _fixture_bits(case["sigmas_bits"])
        assert _int32_bits(schedule.timesteps.cpu()) == _fixture_bits(case["timesteps_bits"])
        assert schedule.num_model_evals == case["num_model_evals"]


@pytest.mark.parametrize("shift", [3.0, 12.0])
@pytest.mark.parametrize("num_inference_steps", [2, 4, 25])
def test_grid_invariants_on_device(num_inference_steps, shift):
    schedule = _cuda_op()(num_inference_steps, shift=shift)
    sigmas = schedule.sigmas
    timesteps = schedule.timesteps

    assert sigmas.is_cuda and timesteps.is_cuda
    assert sigmas.dtype is torch.float32 and timesteps.dtype is torch.float32
    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())
    assert bool((timesteps[1:] > timesteps[:-1]).all())
    assert timesteps[0].item() == 0.0
    assert schedule.num_model_evals == sigmas.numel() - 1 == timesteps.numel()


@pytest.mark.parametrize("num_inference_steps", [1, 0, -4])
def test_rejects_num_inference_steps_below_two(num_inference_steps):
    with pytest.raises(ValueError, match="num_inference_steps"):
        _cuda_op()(num_inference_steps, shift=12.0)


# --- Edge-consistency regression (independent GPU audit 2026-10-16) -------
#
# These four cases come from the independent audit's counterexamples, each
# confirmed against the AST-executed pinned upstream set_timesteps
# (upstream-counterexamples.json): the CPU reference matches upstream on all
# of them, so the device path must match the reference bitwise. Landmark
# bits/lengths from the upstream execution are asserted directly so the
# oracle does not depend on the local reference staying correct.


def _assert_bitwise_match(reference, actual):
    assert tuple(actual.sigmas.shape) == tuple(reference.sigmas.shape)
    assert tuple(actual.timesteps.shape) == tuple(reference.timesteps.shape)
    assert actual.num_model_evals == reference.num_model_evals
    assert _int32_bits(actual.sigmas.cpu()) == _int32_bits(reference.sigmas)
    assert _int32_bits(actual.timesteps.cpu()) == _int32_bits(reference.timesteps)


def test_subnormal_shift_1e_45_matches_reference_upstream_parity():
    # fl32(1e-45) rounds UP to the minimum subnormal 2^-149 (bit pattern 1):
    # upstream grid = [+inf, subnormal quotients..., 0.0] with N=25 collapsing
    # to 10 rows. A flush-to-zero device multiply turns the numerator into 0
    # and produced [NaN, 0.0] (length 2) before the fix.
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(25, shift=1e-45)
    actual = _cuda_op()(25, shift=1e-45)

    _assert_bitwise_match(expected, actual)
    # Landmarks from the pinned upstream execution:
    assert expected.sigmas.numel() == 10
    assert _int32_bits(actual.sigmas.cpu())[0] == 0x7F800000  # +inf


def test_underflow_shift_1e_46_matches_reference_upstream_parity():
    # fl32(1e-46) underflows to +0.0: element 0 is 0/+0, an invalid operation
    # whose torch-CPU (x86) result is the DEFAULT QUIET NaN WITH SIGN BIT SET
    # (0xffc00000); the rest of the grid is +0.0, so upstream keeps 2 rows.
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(25, shift=1e-46)
    actual = _cuda_op()(25, shift=1e-46)

    _assert_bitwise_match(expected, actual)
    assert expected.sigmas.numel() == 2
    assert _int32_bits(actual.sigmas.cpu())[0] == -4194304  # 0xffc00000, signed


def test_positive_nan_shift_propagates_positive_quiet_nan():
    # shift=+NaN: torch CPU propagates the (quieted) input NaN bits through
    # the whole grid -> 0x7fc00000 everywhere, sign included.
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(25, shift=float("nan"))
    actual = _cuda_op()(25, shift=float("nan"))

    _assert_bitwise_match(expected, actual)
    assert set(_int32_bits(actual.sigmas.cpu())) == {0x7FC00000}


def test_negative_nan_shift_propagates_negative_quiet_nan():
    # shift=-NaN: same propagation, but the SIGN BIT travels with the payload
    # -> 0xffc00000 everywhere (the fixed positive fill was wrong here).
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(25, shift=-float("nan"))
    actual = _cuda_op()(25, shift=-float("nan"))

    _assert_bitwise_match(expected, actual)
    assert set(_int32_bits(actual.sigmas.cpu())) == {-4194304}  # 0xffc00000, signed


_EDGE_SHIFTS = [
    float("nan"),  # all-NaN grid; IEEE != keeps every row (matches upstream)
    float("inf"),  # all-NaN grid via fp32 scalar overflow semantics
    1e-10,  # +inf first sigma: (1e-10 - 1.0) rounds to exactly -1.0 in fp32
    1e39,  # fp32-overflowing shift -> NaN grid
    3.5e38,  # just above float32 max -> NaN grid
    1e30,  # extreme finite shift collapses the grid to [1.0, 0.0]
]


@pytest.mark.parametrize("shift", _EDGE_SHIFTS)
def test_edge_shifts_match_reference_upstream_parity(shift):
    # contract-audit.md §2 proved the CPU reference bit-identical to the
    # actually-executed upstream set_timesteps on exactly these inputs, so
    # the device path must reproduce the reference bytes, NaN grids included.
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(8, shift=shift)
    actual = _cuda_op()(8, shift=shift)

    assert actual.num_model_evals == expected.num_model_evals
    assert tuple(actual.sigmas.shape) == tuple(expected.sigmas.shape)
    assert tuple(actual.timesteps.shape) == tuple(expected.timesteps.shape)
    assert _int32_bits(actual.sigmas.cpu()) == _int32_bits(expected.sigmas)
    assert _int32_bits(actual.timesteps.cpu()) == _int32_bits(expected.timesteps)


@pytest.mark.parametrize(("num_inference_steps", "truncated_to"), [(2.9, 2), (3.1, 3)])
def test_float_num_steps_truncate_like_reference(num_inference_steps, truncated_to):
    # Upstream parity: int() truncation, device side must agree with both the
    # reference call and the truncated integer grid.
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    expected = NativeH3SigmaScheduleOp()(num_inference_steps, shift=12.0)
    integer_grid = NativeH3SigmaScheduleOp()(truncated_to, shift=12.0)
    actual = _cuda_op()(num_inference_steps, shift=12.0)

    assert _int32_bits(expected.sigmas) == _int32_bits(integer_grid.sigmas)
    assert _int32_bits(actual.sigmas.cpu()) == _int32_bits(expected.sigmas)
    assert _int32_bits(actual.timesteps.cpu()) == _int32_bits(expected.timesteps)
    assert actual.num_model_evals == expected.num_model_evals


@pytest.mark.parametrize("shift", [0.0, -1.5, float("-inf")])
def test_rejects_non_positive_shift(shift):
    with pytest.raises(ValueError, match="positive"):
        _cuda_op()(8, shift=shift)


def test_rejects_cpu_device_instead_of_silent_fallback():
    with pytest.raises(ValueError, match="NVIDIA CUDA device"):
        _cuda_op()(25, shift=12.0, device="cpu")


def test_repeated_and_interleaved_calls_are_stable():
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    cuda_op = _cuda_op()
    cpu_op = NativeH3SigmaScheduleOp()

    first = cuda_op(25, shift=12.0)
    cpu_op(25, shift=3.0)  # interleaved CPU call must not pollute device state
    second = cuda_op(25, shift=3.0)
    third = cuda_op(25, shift=12.0)

    assert _int32_bits(first.sigmas.cpu()) == _int32_bits(third.sigmas.cpu())
    assert _int32_bits(first.timesteps.cpu()) == _int32_bits(third.timesteps.cpu())
    assert _int32_bits(second.timesteps.cpu()) != _int32_bits(first.timesteps.cpu())


def test_backend_identity_is_device_path_not_cpu_fallback():
    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )
    from rl_engine.runtime.registry import OpBackend

    # The registry lists the NVIDIA Triton implementation ahead of the PyTorch
    # reference for device "cuda"; outputs above prove the device path ran.
    assert OpBackend.CUDA_H3_SIGMA_SCHEDULE.value.endswith(
        "model_specific.minimax_h3.sigma_schedule.H3SigmaScheduleCudaOp"
    )
    assert OpBackend.PYTORCH_H3_SIGMA_SCHEDULE.value.endswith(
        "reference.minimax_h3.sigma_schedule.NativeH3SigmaScheduleOp"
    )

    schedule = H3SigmaScheduleCudaOp()(8, shift=3.0)
    assert schedule.sigmas.device.type == "cuda"
    assert torch.version.hip is None  # NVIDIA-only backend


def test_public_entry_selects_device_backend_and_reports_provenance():
    # The registry must hand out the Triton class itself (device identity is
    # the instance type, not just `.is_cuda`, which a CPU-build-then-copy
    # would also have), and the semantic path must report the same backend.
    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )
    from rl_engine.runtime.registry import kernel_registry
    from rl_engine.runtime.semantic_registry import OperatorRequirements

    op = kernel_registry.get_op("sigma_schedule_h3", device="cuda")
    assert isinstance(op, H3SigmaScheduleCudaOp)
    assert op(25, shift=12.0).sigmas.is_cuda

    session = kernel_registry.semantic.session()
    resolution = session.resolve(
        semantic_op="sigma_schedule_h3",
        requested_backend="rlkernel.h3_sigma_schedule.triton",
        target="rollout",
        requirements=OperatorRequirements(device="cuda", dtype="float32"),
    )
    instance = session.instantiate(resolution)
    assert isinstance(instance, H3SigmaScheduleCudaOp)
    provenance = session.instance_provenance(resolution, instance)
    assert provenance.backend_id == "rlkernel.h3_sigma_schedule.triton"
    assert provenance.concrete_implementation.endswith("H3SigmaScheduleCudaOp")
    assert torch.version.hip is None  # NVIDIA-only backend


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="multi-GPU device-guard test needs 2 devices"
)
def test_target_device_guard_with_mismatched_current_device():
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    other = "cuda:1" if torch.cuda.current_device() == 0 else "cuda:0"
    expected = NativeH3SigmaScheduleOp()(25, shift=12.0)
    with torch.cuda.device(0 if other == "cuda:1" else 1):
        # Current device deliberately differs from the requested target.
        schedule = _cuda_op()(25, shift=12.0, device=other)

    assert schedule.sigmas.device == torch.device(other)
    assert _int32_bits(schedule.sigmas.cpu()) == _int32_bits(expected.sigmas)
    assert _int32_bits(schedule.timesteps.cpu()) == _int32_bits(expected.timesteps)
