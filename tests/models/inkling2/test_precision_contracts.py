# SPDX-License-Identifier: Apache-2.0
"""INK-04, INK-05, INK-30, INK-31, INK-52, INK-67 and INK-69 contracts."""

import pytest
import torch

from rl_engine.models.inkling2.precision_contracts import (
    PrecisionContractError,
    QuantBackendCapability,
    QuantBackendMatrix,
    compare_checkpoint_fingerprints,
    dequantize_per_tensor,
    fingerprint_checkpoint,
    quantize_per_tensor,
    quantized_linear,
    require_checkpoint_match,
)


def test_checkpoint_fingerprint_is_order_independent_and_detects_bytes():
    state = {"b": torch.tensor([2.0, 3.0]), "a": torch.tensor([1.0])}
    expected = fingerprint_checkpoint(state, source="miles", revision="r1")
    actual = fingerprint_checkpoint(
        {"a": state["a"], "b": state["b"]}, source="miles", revision="r1"
    )
    assert expected.identity_sha256 == actual.identity_sha256
    require_checkpoint_match(expected, actual)

    changed = fingerprint_checkpoint(
        {"a": state["a"], "b": torch.tensor([2.0, 4.0])}, source="miles", revision="r1"
    )
    assert compare_checkpoint_fingerprints(expected, changed) == ("tensor fingerprint differs: b",)
    with pytest.raises(PrecisionContractError, match="identity mismatch"):
        require_checkpoint_match(expected, changed)


def test_checkpoint_fingerprint_serializes_shape_dtype_and_revision():
    fingerprint = fingerprint_checkpoint(
        {"weight": torch.ones(2, 3, dtype=torch.bfloat16)}, source="hf", revision="abc"
    )
    payload = fingerprint.to_dict()
    assert payload["tensors"][0]["shape"] == [2, 3]
    assert payload["tensors"][0]["dtype"] == "torch.bfloat16"
    assert payload["identity_sha256"] == fingerprint.identity_sha256


def test_quant_backend_matrix_fails_closed_for_unavailable_runtime():
    matrix = QuantBackendMatrix(
        (
            QuantBackendCapability("cuda", "torch", "int8_symmetric", True, True, True),
            QuantBackendCapability("rocm", "aiter", "mxfp8", False, True, False, "MI35x gate"),
        )
    )
    assert (
        matrix.require(platform="cuda", format="int8_symmetric", purpose="training").backend
        == "torch"
    )
    with pytest.raises(RuntimeError, match="MI35x gate"):
        matrix.require(platform="rocm", format="mxfp8", purpose="inference")
    with pytest.raises(PrecisionContractError, match="purpose"):
        matrix.require(platform="cuda", format="int8_symmetric", purpose="serve")


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_int8_round_trip_and_quantized_linear(device):
    torch.manual_seed(52)
    x = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
    weight = torch.randn(6, 8, device=device, dtype=torch.bfloat16)
    quantized = quantize_per_tensor(x)
    restored = dequantize_per_tensor(quantized, dtype=torch.bfloat16)
    assert restored.shape == x.shape
    assert restored.dtype == torch.bfloat16
    output = quantized_linear(x, weight)
    reference = torch.nn.functional.linear(x.float(), weight.float()).bfloat16()
    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, atol=0.35, rtol=0.25)


def test_float8_capability_is_explicitly_gated():
    x = torch.tensor([1.0, -2.0, 0.5])
    if getattr(torch, "float8_e4m3fn", None) is None:
        with pytest.raises(RuntimeError, match="float8"):
            quantize_per_tensor(x, format="float8_e4m3fn")
        return
    quantized = quantize_per_tensor(x, format="float8_e4m3fn")
    restored = dequantize_per_tensor(quantized, dtype=torch.float32)
    assert restored.shape == x.shape
    assert torch.isfinite(restored).all()
