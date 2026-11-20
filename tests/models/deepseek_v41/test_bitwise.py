# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.models.deepseek_v41.bitwise import compare_bits


def test_signed_zero_and_first_bit_pattern():
    r = compare_bits(
        torch.tensor([[0.0, -0.0]]), torch.zeros(1, 2), token_dims=1, layer=1, operator="gate"
    )
    assert r["mismatched_elements"] == r["mismatched_tokens"] == 1
    assert r["first_mismatch"] == {
        "flat_index": 1,
        "index": [0, 1],
        "actual_bits": "0x80000000",
        "expected_bits": "0x00000000",
    }


def test_nan_payloads_are_compared_without_float_equality():
    a = torch.tensor([0x7FC00001, 0x7FC00002], dtype=torch.int32).view(torch.float32)
    assert compare_bits(a, a.clone())["mismatched_elements"] == 0
    assert compare_bits(a, a.flip(0))["mismatched_elements"] == 2


@pytest.mark.parametrize(
    "dtype", [torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2]
)
def test_packed_representations(dtype):
    a = torch.tensor([0x80, 0x81, 0xFF, 0], dtype=torch.uint8).view(dtype)
    b = a.view(torch.uint8).clone()
    b[2] = 0xFE
    r = compare_bits(a, b.view(dtype))
    assert r["compared_elements"] == 4  # FP4 counts packed storage elements, not 8 nibbles
    assert r["mismatched_elements"] == 1
    assert r["first_mismatch"]["actual_bits"] == "0xff"


def test_noncontiguous_scalar_and_empty():
    a = torch.arange(12).reshape(3, 4).t()
    assert compare_bits(a, a.clone())["status"] == "PASS"
    assert compare_bits(torch.tensor(0.0), torch.tensor(-0.0))["mismatched_elements"] == 1
    r = compare_bits(torch.empty(0, 4), torch.empty(0, 4), token_dims=1)
    assert r["compared_elements"] == r["mismatched_elements"] == r["total_tokens"] == 0


def test_different_dtype_is_an_error():
    with pytest.raises(ValueError):
        compare_bits(torch.ones(2), torch.ones(2, dtype=torch.bfloat16))
