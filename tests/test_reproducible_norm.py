# SPDX-License-Identifier: Apache-2.0
import math
import struct
from fractions import Fraction

import pytest
import torch

from rl_engine.integrations.reproducible_norm import integer_square_bins, norm_from_bins


def reference_bins(values):
    bins = [0] * 770
    for value in values:
        bits, = struct.unpack('I', struct.pack('f', value))
        exponent = (bits >> 23) & 255
        mantissa = bits & 0x7fffff
        if exponent == 255:
            bins[768 + (mantissa == 0)] = 1
            continue
        if exponent:
            mantissa |= 1 << 23
        square = mantissa * mantissa
        for limb in range(3):
            bins[3 * exponent + limb] += (square >> (16 * limb)) & 65535
    return bins


@pytest.mark.parametrize('values', [
    [], [0., -0.], [3., -4.], [2.**-149, -2.**-126],
    [2.**127, -2.**100, 2.**-149],
    [1., 2.**-12, -2.**-24] * 101,
])
def test_norm_matches_exact_rational_square_sum(values):
    exact = sum((Fraction(value)**2 for value in values), Fraction())
    assert norm_from_bins(reference_bins(values)) == math.sqrt(float(exact))


def test_nonfinite():
    assert math.isnan(norm_from_bins(reference_bins([float('nan'), float('inf')])))
    assert norm_from_bins(reference_bins([-float('inf')])) == float('inf')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='accelerator required')
def test_accelerator_bins_preserve_exact_sum_under_repartition():
    generator = torch.Generator(device='cuda').manual_seed(71)
    values = torch.randn(32769, generator=generator, device='cuda')
    values[:4] = torch.tensor([2.**-149, -2.**-126, 2.**127, 0.], device='cuda')
    full = integer_square_bins([values])
    partitioned = integer_square_bins(list(values.split(733)))
    assert torch.equal(full, partitioned)
    assert full.cpu().tolist() == reference_bins(values.cpu().tolist())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='two accelerators required')
def test_accelerator_device_context_is_restored():
    current = torch.cuda.current_device()
    other = (current + 1) % torch.cuda.device_count()
    values = torch.tensor([3., -4.], device=f'cuda:{other}')
    assert norm_from_bins(integer_square_bins([values]).cpu().tolist()) == 5.
    assert torch.cuda.current_device() == current
