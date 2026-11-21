# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden-value tests for the P5 MX codecs (P5-1 (#60) contract)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe import mx_format as mx


def test_e4m3_golden_codes() -> None:
    vals = [448.0, 464.0, 500.0, 17.0, 18.0, 19.0, -464.0, 2**-9, 2**-10, 1.5 * 2**-9, 0.0]
    want = [0x7E, 0x7E, 0x7E, 0x58, 0x59, 0x5A, 0xFE, 0x01, 0x00, 0x02, 0x00]
    codes = mx.e4m3_encode(torch.tensor(vals, dtype=torch.float32))
    assert codes.tolist() == want


def test_e4m3_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        mx.e4m3_encode(torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        mx.e4m3_encode(torch.tensor([float("inf")]))


def test_e4m3_roundtrip_all_finite_codes() -> None:
    codes = torch.arange(256, dtype=torch.uint8)
    finite = (codes & 0x7F) != 0x7F  # exclude NaN codes
    decoded = mx.e4m3_decode(codes[finite])
    re_encoded = mx.e4m3_encode(decoded)
    assert torch.equal(re_encoded, codes[finite])


def test_e8m0_scale_recipe() -> None:
    amax = torch.tensor([1.0, 448.0, 0.0, 2.0**-10])
    codes = mx.e8m0_scale_from_amax(amax, "e4m3")
    # floor(log2(amax)) - 8, bias 127; amax==0 -> 127
    assert codes.tolist() == [127 - 8, 127, 127, 127 - 10 - 8]
    codes4 = mx.e8m0_scale_from_amax(torch.tensor([1.0]), "e2m1")
    assert codes4.tolist() == [127 - 2]
    with pytest.raises(ValueError):
        mx.e8m0_decode(torch.tensor([255], dtype=torch.uint8))


def test_e2m1_tie_to_even_and_roundtrip() -> None:
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert mx.e2m1_encode(ties).tolist() == [0, 2, 2, 4, 4, 6, 6]
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for sign in (1.0, -1.0):
        codes = mx.e2m1_encode(values * sign)
        assert torch.equal(mx.e2m1_decode(codes), values * sign)
    # saturation
    assert mx.e2m1_encode(torch.tensor([100.0, -100.0])).tolist() == [7, 15]


def test_nibble_pack_roundtrip() -> None:
    g = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 16, (4, 32), generator=g, dtype=torch.uint8)
    assert torch.equal(mx.unpack_nibbles(mx.pack_nibbles(codes)), codes)
    # low nibble first
    packed = mx.pack_nibbles(torch.tensor([[0x1, 0x2]], dtype=torch.uint8))
    assert packed.tolist() == [[0x21]]


def test_mx_quantize_row_invariant() -> None:
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(8, 64, generator=g)).to(torch.bfloat16)
    for fmt in ("e4m3", "e2m1"):
        full = mx.mx_quantize(x, fmt)
        one = mx.mx_quantize(x[3:4], fmt)
        assert torch.equal(full.codes[3:4], one.codes)
        assert torch.equal(full.scales[3:4], one.scales)


def test_mx_quantize_error_bounds_and_validation() -> None:
    g = torch.Generator().manual_seed(2)
    x = torch.randn(4, 64, generator=g).to(torch.bfloat16)
    d8 = mx.mx_dequantize(mx.mx_quantize(x, "e4m3"))
    d4 = mx.mx_dequantize(mx.mx_quantize(x, "e2m1"))
    scale = x.float().abs().max()
    # The OCP floor(log2) recipe saturates the top (448,512)*scale band, so the
    # worst error at block amax is 12.5% for e4m3 (25% for e2m1) plus rounding.
    assert (d8 - x.float()).abs().max() / scale < 0.13
    assert (d4 - x.float()).abs().max() / scale < 0.30
    with pytest.raises(ValueError):
        mx.mx_quantize(torch.randn(4, 33), "e4m3")
    with pytest.raises(ValueError):
        mx.mx_quantize(torch.full((1, 32), float("inf")), "e4m3")
