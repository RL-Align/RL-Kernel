# SPDX-License-Identifier: Apache-2.0
"""CPU-safe frozen-table identity and bounded checkpoint-range validation."""

from unittest.mock import Mock

import pytest
import torch

from rl_engine.contracts.operators.hash_route import CHECKPOINT_TABLE_SHA256, MODEL, REVISION
from rl_engine.ops.moe.hash_route import tensor_sha256
from rl_engine.validation.operators.hash_route_fixture import load_checkpoint_table
from tools.weights import prepare_hash_route_tables as download


@pytest.mark.parametrize(
    "offsets",
    [[-1, 6205439], [0, 8_000_000_000], [0, 6205441], [False, 6205440], [0.0, 6205440], [0], []],
)
def test_bad_header_ranges_rejected_before_downloading(offsets):
    with pytest.raises(ValueError):
        download.validate_header({"dtype": "I64", "shape": [129280, 6], "data_offsets": offsets})


@pytest.mark.parametrize(
    "dtype,shape", [("I32", [129280, 6]), ("I64", [6, 129280]), ("I64", [129280, 5])]
)
def test_wrong_checkpoint_layout_rejected(dtype, shape):
    with pytest.raises(ValueError, match="I64"):
        download.validate_header({"dtype": dtype, "shape": shape, "data_offsets": [0, 6205440]})


@pytest.mark.parametrize("start,size", [(-1, 8), (0, 0), (0, 10_000_001), (False, 8)])
def test_bad_range_never_opens_connection(monkeypatch, start, size):
    connection = Mock()
    monkeypatch.setattr(download, "urlopen", connection)
    with pytest.raises(ValueError, match="bounded positive size"):
        download.read_range("https://example.test/shard", start, size, "test")
    connection.assert_not_called()


@pytest.mark.parametrize(
    "status,header,data",
    [(200, "", b"abcd"), (206, "bytes 1-4/100", b"abcd"), (206, "bytes 0-3/100", b"abc")],
)
def test_unexpected_response_rejected_without_full_shard(monkeypatch, status, header, data):
    response = Mock(status=status, headers={"Content-Range": header})
    response.read.return_value = data
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: context)
    with pytest.raises(ValueError):
        download.read_range("https://example.test/shard", 0, 4, "test")
    if status != 206 or header != "bytes 0-3/100":
        response.read.assert_not_called()


@pytest.fixture
def fixture(monkeypatch):
    table = torch.zeros(129280, 6, dtype=torch.int64)
    digest = tensor_sha256(table)
    monkeypatch.setitem(CHECKPOINT_TABLE_SHA256, 0, digest)
    return {
        "table": table,
        "metadata": {
            "model": MODEL,
            "revision": REVISION,
            "absolute_layer": 0,
            "tensor": "layers.0.ffn.gate.tid2eid",
            "shard": "model-00002-of-00048.safetensors",
            "sha256": digest,
            "tensor_metadata": {"dtype": "I64", "shape": [129280, 6], "data_offsets": [0, 6205440]},
        },
    }


def test_fixture_identity_and_actual_bytes_match(monkeypatch, fixture):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: fixture)
    table, metadata = load_checkpoint_table("unused.pt", 0)
    assert table is fixture["table"] and metadata["absolute_layer"] == 0


@pytest.mark.parametrize(
    "field", ["model", "revision", "absolute_layer", "tensor", "shard", "sha256"]
)
def test_wrong_frozen_identity_rejected(monkeypatch, fixture, field):
    fixture["metadata"][field] = "wrong"
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: fixture)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_checkpoint_table("unused.pt", 0)


def test_corrupt_bytes_rejected_even_with_valid_metadata(monkeypatch, fixture):
    fixture["table"][0, 0] = 1
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: fixture)
    with pytest.raises(ValueError, match="actual bytes differ"):
        load_checkpoint_table("unused.pt", 0)


def test_replacement_with_self_computed_checksum_rejected(monkeypatch):
    table = torch.zeros(129280, 6, dtype=torch.int64)
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "absolute_layer": 0,
        "tensor": "layers.0.ffn.gate.tid2eid",
        "shard": "model-00002-of-00048.safetensors",
        "sha256": tensor_sha256(table),
    }
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: {"table": table, "metadata": metadata}
    )
    with pytest.raises(ValueError, match="sha256 identity"):
        load_checkpoint_table("unused.pt", 0)
