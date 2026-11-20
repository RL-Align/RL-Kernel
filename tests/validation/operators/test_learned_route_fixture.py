# SPDX-License-Identifier: Apache-2.0
"""Frozen external bias identity and bounded HTTP payload failures."""

import copy
from types import SimpleNamespace

import pytest
import torch

from rl_engine.ops.moe.learned_route import tensor_sha256
from rl_engine.validation.operators import learned_route_fixture as loader
from tools.weights import prepare_learned_route_bias as download


def fixture():
    bias = torch.zeros(256)
    return {
        "bias": bias,
        "metadata": {
            "model": loader.CHECKPOINT_MODEL,
            "revision": loader.CHECKPOINT_REVISION,
            "absolute_layer": 3,
            "tensor": "layers.3.ffn.gate.bias",
            "shard": "model-00005-of-00048.safetensors",
            "sha256": tensor_sha256(bias),
            "tensor_metadata": {"dtype": "F32", "shape": [256], "data_offsets": [0, 1024]},
        },
    }


def test_loader_checks_actual_bytes_against_independent_frozen_digest(monkeypatch, tmp_path):
    payload = fixture()
    monkeypatch.setattr(loader, "CHECKPOINT_BIAS_SHA256", payload["metadata"]["sha256"])
    path = tmp_path / "bias.pt"
    torch.save(payload, path)
    bias, metadata = loader.load_checkpoint_bias(path)
    assert torch.equal(bias, payload["bias"]) and metadata == payload["metadata"]
    payload["bias"][0] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="actual bytes"):
        loader.load_checkpoint_bias(path)
    # Replacing payload AND its claimed hash cannot change the frozen identity.
    payload["metadata"]["sha256"] = tensor_sha256(payload["bias"])
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        loader.load_checkpoint_bias(path)


@pytest.mark.parametrize("key", ["model", "revision", "absolute_layer", "tensor", "shard"])
def test_loader_rejects_other_identity(monkeypatch, tmp_path, key):
    payload = fixture()
    monkeypatch.setattr(loader, "CHECKPOINT_BIAS_SHA256", payload["metadata"]["sha256"])
    payload["metadata"][key] = "other"
    path = tmp_path / "bias.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        loader.load_checkpoint_bias(path)


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "BF16"},
        {"shape": [128]},
        {"data_offsets": [-1, 1023]},
        {"data_offsets": [False, 1024]},
        {"data_offsets": [0, 1025]},
        {"data_offsets": [0, 10**12]},
        {"data_offsets": [2, 1]},
        {"data_offsets": []},
    ],
)
def test_bad_header_rejected_before_tensor_fetch(change):
    header = copy.deepcopy(fixture()["metadata"]["tensor_metadata"])
    header.update(change)
    with pytest.raises(ValueError):
        download.validate_header(header)


@pytest.mark.parametrize("start,size", [(-1, 8), (0, 0), (0, 10_000_001), (False, 8)])
def test_invalid_range_never_opens_network(monkeypatch, start, size):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid range must not open network")

    monkeypatch.setattr(download, "urlopen", forbidden)
    with pytest.raises(ValueError):
        download.read_range("https://example.invalid/shard", start, size, "test")


@pytest.mark.parametrize("status,header", [(200, ""), (206, "bytes 0-7/100")])
def test_ignored_or_wrong_range_never_reads_response(monkeypatch, status, header):
    class Response:
        def __enter__(self):
            return SimpleNamespace(status=status, headers={"Content-Range": header})

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="HTTP byte range"):
        download.read_range("https://example.invalid/shard", 8, 4, "test")
