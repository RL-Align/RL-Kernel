# SPDX-License-Identifier: Apache-2.0
import math
import struct
from fractions import Fraction

import pytest
import torch

from rl_engine.ops.norm.reproducible import integer_square_bins, norm_from_bins


def reference_bins(values):
    bins = [0] * 770
    for value in values:
        (bits,) = struct.unpack("I", struct.pack("f", value))
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7FFFFF
        if exponent == 255:
            bins[768 + (mantissa == 0)] = 1
            continue
        if exponent:
            mantissa |= 1 << 23
        square = mantissa * mantissa
        for limb in range(3):
            bins[3 * exponent + limb] += (square >> (16 * limb)) & 65535
    return bins


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.0, -0.0],
        [3.0, -4.0],
        [2.0**-149, -(2.0**-126)],
        [2.0**127, -(2.0**100), 2.0**-149],
        [1.0, 2.0**-12, -(2.0**-24)] * 101,
    ],
)
def test_norm_matches_exact_rational_square_sum(values):
    exact = sum((Fraction(value) ** 2 for value in values), Fraction())
    assert norm_from_bins(reference_bins(values)) == math.sqrt(float(exact))


def test_nonfinite():
    assert math.isnan(norm_from_bins(reference_bins([float("nan"), float("inf")])))
    assert norm_from_bins(reference_bins([-float("inf")])) == float("inf")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required")
def test_accelerator_bins_preserve_exact_sum_under_repartition():
    generator = torch.Generator(device="cuda").manual_seed(71)
    values = torch.randn(32769, generator=generator, device="cuda")
    values[:4] = torch.tensor([2.0**-149, -(2.0**-126), 2.0**127, 0.0], device="cuda")
    full = integer_square_bins([values])
    partitioned = integer_square_bins(list(values.split(733)))
    assert torch.equal(full, partitioned)
    assert full.cpu().tolist() == reference_bins(values.cpu().tolist())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two accelerators required")
def test_accelerator_device_context_is_restored():
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    values = torch.tensor([3.0, -4.0], device=f"cuda:{other}")
    assert norm_from_bins(integer_square_bins([values]).cpu().tolist()) == 5.0
    assert torch.cuda.current_device() == current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="accelerator required")
@pytest.mark.parametrize("length", [1, 31, 32, 33, 257, 32771])
def test_accelerator_warp_bins_with_mixed_exponents_and_nonfinite(length):
    generator = torch.Generator(device="cuda").manual_seed(length)
    values = torch.randn(length, generator=generator, device="cuda")
    exponents = torch.randint(-120, 120, (length,), generator=generator, device="cuda")
    values *= torch.pow(2.0, exponents)
    if length >= 33:
        values[:16] = 0.125
        values[16:21] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, 2.0**-149], device="cuda"
        )
    actual = integer_square_bins([values]).cpu().tolist()
    assert actual == reference_bins(values.cpu().tolist())


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def test_nvrtc_candidates_find_cuda13_wheel_layout(tmp_path):
    from rl_engine.ops.norm.reproducible import _nvrtc_candidates

    wheel = _touch(tmp_path / "cu13" / "lib" / "libnvrtc.so.13")
    candidates = _nvrtc_candidates(tmp_path, "13.0")
    assert candidates[0] == wheel
    assert candidates.index("libnvrtc.so.13") < candidates.index("libnvrtc.so.12")


def test_nvrtc_candidates_try_the_matching_major_before_other_wheels(tmp_path):
    """A leftover CUDA 12 wheel must not win over the CUDA 13 system soname."""
    from rl_engine.ops.norm.reproducible import _nvrtc_candidates

    stale = _touch(tmp_path / "cu12" / "lib" / "libnvrtc.so.12")
    stale_legacy = _touch(tmp_path / "cuda_nvrtc" / "lib" / "libnvrtc.so.12")
    candidates = _nvrtc_candidates(tmp_path, "13.0")
    assert candidates[0] == "libnvrtc.so.13"
    assert candidates.index("libnvrtc.so.13") < candidates.index(stale)
    assert candidates.index("libnvrtc.so.13") < candidates.index(stale_legacy)


def test_nvrtc_candidates_find_cuda12_wheel_layout(tmp_path):
    from rl_engine.ops.norm.reproducible import _nvrtc_candidates

    wheel = _touch(tmp_path / "cuda_nvrtc" / "lib" / "libnvrtc.so.12")
    candidates = _nvrtc_candidates(tmp_path, "12.8")
    assert candidates[0] == wheel
    assert candidates.index("libnvrtc.so.12") < candidates.index("libnvrtc.so.13")
    assert len(candidates) == len(set(candidates))


def test_load_nvrtc_reports_every_attempt(monkeypatch):
    from rl_engine.ops.norm import reproducible

    monkeypatch.setattr(reproducible, "_nvrtc_candidates", lambda: ["a.so", "b.so"])
    with pytest.raises(OSError, match="(?s)a.so.*b.so"):
        reproducible._load_nvrtc()
