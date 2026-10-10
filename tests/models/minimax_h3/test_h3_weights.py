# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Extracted H3 weights must retain the pinned tensor contents."""

from __future__ import annotations

import hashlib

import pytest
import torch

from rl_engine.validation.models import h3_weights


@pytest.mark.parametrize(
    "dtype,raw_bytes",
    [
        (torch.float32, bytes.fromhex("000000000000803f")),
        (torch.bfloat16, bytes.fromhex("0000803f")),
    ],
)
def test_tensor_checksum_hashes_raw_bytes(dtype, raw_bytes):
    """Hash the exact FP32 and BF16 bit patterns without numerical conversion."""

    tensor = torch.tensor([0.0, 1.0], dtype=dtype)
    assert h3_weights.sha256_tensor(tensor) == hashlib.sha256(raw_bytes).hexdigest()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mismatch", [None, "dtype", "shape", "sha256"])
@pytest.mark.parametrize("names", [None, ["bias"]])
def test_loading_rejects_modified_weight_contents(tmp_path, monkeypatch, dtype, mismatch, names):
    """Reject altered shape, dtype, or bytes even outside the requested tensor subset."""

    from safetensors.torch import save_file

    tensors = {
        "weight": torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=dtype),
        "bias": torch.tensor([0.0, 1.0], dtype=dtype),
    }
    manifest = {
        "tensors": {
            name: {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "sha256": h3_weights.sha256_tensor(tensor),
            }
            for name, tensor in tensors.items()
        }
    }
    if mismatch == "dtype":
        tensors["weight"] = tensors["weight"].to(torch.float64)
    elif mismatch == "shape":
        tensors["weight"] = tensors["weight"].reshape(4)
    elif mismatch == "sha256":
        tensors["weight"][0, 0] = 1.0
    # Tensor checksums do not depend on serializer metadata or tensor key order.
    save_file(
        dict(reversed(list(tensors.items()))),
        str(tmp_path / h3_weights.EXTRACTED_FILE),
        metadata={"revision": "a different serializer metadata value"},
    )
    monkeypatch.setenv(h3_weights.WEIGHTS_ENV, str(tmp_path))
    monkeypatch.setattr(h3_weights, "load_h3_manifest", lambda: manifest)

    if mismatch:
        with pytest.raises(ValueError, match=f"weight: {mismatch} .* != manifest"):
            h3_weights.load_h3_conditioning_weights("cpu", names=names)
    else:
        actual = h3_weights.load_h3_conditioning_weights("cpu", names=names)
        assert list(actual) == (names if names is not None else list(tensors))
        for name, tensor in actual.items():
            assert torch.equal(tensor, tensors[name])
