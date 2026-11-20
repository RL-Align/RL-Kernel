# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for bounded downloads and independent frozen-fixture checks."""

import copy
import io
import json
import struct
from unittest.mock import Mock

import pytest
import torch

from rl_engine.validation.operators import router_gate_fixture as contract
from tools.validation.operators import check_router_gate as replay
from tools.weights import prepare_router_gate_weights as download


def tensor_metadata(offsets=None):
    return {
        "dtype": "BF16",
        "shape": [256, 4096],
        "data_offsets": [0, contract.WEIGHT_BYTES] if offsets is None else offsets,
    }


@pytest.mark.parametrize(
    "offsets",
    [
        [-1, contract.WEIGHT_BYTES - 1],
        [0, -1],
        [1, 0],
        [0, 8_000_000_000],
        [0, contract.WEIGHT_BYTES - 1],
        [False, contract.WEIGHT_BYTES],
        [0.0, float(contract.WEIGHT_BYTES)],
        ["0", contract.WEIGHT_BYTES],
        [],
        [0],
    ],
)
def test_bad_offsets_rejected_before_tensor_download(monkeypatch, tmp_path, offsets):
    calls = mock_download(monkeypatch, tmp_path, tensor_metadata(offsets))
    with pytest.raises(ValueError, match="data_offsets|byte span"):
        download.main()
    assert len(calls) == 2  # Only the header size and bounded header were read.
    assert not (tmp_path / "fixture.pt").exists()


def mock_download(monkeypatch, tmp_path, tensor, data=None):
    header = json.dumps({contract.TENSOR: tensor}).encode()
    index = {"weight_map": {contract.TENSOR: contract.SHARD}}
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: io.StringIO(json.dumps(index)))
    calls = []

    def read(url, start, size):
        calls.append((start, size))
        if url.endswith("?range=header-size"):
            return struct.pack("<Q", len(header))
        if url.endswith("?range=header"):
            return header
        assert size == contract.WEIGHT_BYTES
        return bytes(size) if data is None else data

    monkeypatch.setattr(download, "read_range", read)
    monkeypatch.setattr(
        "sys.argv", ["prepare_router_gate_weights", "--output", str(tmp_path / "fixture.pt")]
    )
    return calls


def test_wrong_payload_rejected_before_overwriting_output(monkeypatch, tmp_path):
    mock_download(monkeypatch, tmp_path, tensor_metadata())
    output = tmp_path / "fixture.pt"
    output.write_bytes(b"existing fixture")
    with pytest.raises(ValueError, match="SHA256"):
        download.main()
    assert output.read_bytes() == b"existing fixture"


@pytest.mark.parametrize(
    "start,size", [(-1, 8), (0, 0), (0, -1), (False, 8), (0, 10_000_001), (0, 8.0)]
)
def test_invalid_range_never_opens_a_connection(monkeypatch, start, size):
    connect = Mock()
    monkeypatch.setattr(download, "urlopen", connect)
    with pytest.raises(ValueError, match="HTTP range"):
        download.read_range("https://example.test/weights", start, size)
    connect.assert_not_called()


@pytest.mark.parametrize(
    "status,header,data,error",
    [
        (200, "", b"abcd", "support HTTP byte ranges"),
        (206, "bytes 0-3/100", b"abcd", "unexpected checkpoint range"),
        (206, "bytes 8-11/100", b"abc", "truncated checkpoint range"),
        (206, "bytes 8-11/100", b"abcde", "truncated checkpoint range"),
        (206, "bytes 8-11/100", b"abcd", None),
    ],
)
def test_http_range_response_validation(monkeypatch, status, header, data, error):
    response = Mock(status=status, headers={"Content-Range": header})
    response.read.return_value = data
    connection = Mock()
    connection.__enter__ = Mock(return_value=response)
    connection.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: connection)
    if error:
        with pytest.raises(RuntimeError, match=error):
            download.read_range("https://example.test/weights", 8, 4)
        if status != 206 or header != "bytes 8-11/100":
            response.read.assert_not_called()
    else:
        assert download.read_range("https://example.test/weights", 8, 4) == data


@pytest.fixture
def synthetic_fixture(monkeypatch):
    # The positive CPU fixture has a deliberately substituted trust anchor.
    # Real checkpoint acceptance is exercised separately with its frozen SHA256.
    weight = torch.zeros(contract.SHAPE, dtype=torch.bfloat16)
    digest = contract.checksum(weight)
    monkeypatch.setattr(contract, "WEIGHT_SHA256", digest)
    return {
        "weight": weight,
        "metadata": {
            "model": contract.MODEL,
            "revision": contract.REVISION,
            "tensor": contract.TENSOR,
            "shard": contract.SHARD,
            "weight_sha256": digest,
            "tensor_metadata": tensor_metadata(),
        },
    }


def test_valid_fixture(synthetic_fixture):
    assert contract.validate_gate_fixture(synthetic_fixture) is synthetic_fixture["weight"]


@pytest.mark.parametrize("field", ["model", "revision", "tensor", "shard", "weight_sha256"])
def test_fixture_identity_checked_independently(synthetic_fixture, field):
    synthetic_fixture["metadata"][field] = "wrong identity"
    with pytest.raises(ValueError, match=field):
        contract.validate_gate_fixture(synthetic_fixture)


def test_replacement_with_its_own_checksum_is_rejected():
    weight = torch.zeros(contract.SHAPE, dtype=torch.bfloat16)
    fixture = {
        "weight": weight,
        "metadata": {
            "model": contract.MODEL,
            "revision": contract.REVISION,
            "tensor": contract.TENSOR,
            "shard": contract.SHARD,
            "weight_sha256": contract.checksum(weight),
            "tensor_metadata": tensor_metadata(),
        },
    }
    with pytest.raises(ValueError, match="weight_sha256"):
        contract.validate_gate_fixture(fixture)


def test_modified_bytes_with_frozen_metadata_are_rejected(synthetic_fixture):
    synthetic_fixture["weight"][0, 0] = 1
    with pytest.raises(ValueError, match="weight SHA256"):
        contract.validate_gate_fixture(synthetic_fixture)


@pytest.mark.parametrize(
    "dtype,shape", [(torch.float32, contract.SHAPE), (torch.bfloat16, (4096, 256))]
)
def test_actual_weight_layout_checked(synthetic_fixture, dtype, shape):
    synthetic_fixture["weight"] = torch.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match="weight must be BF16"):
        contract.validate_gate_fixture(synthetic_fixture)


def test_invalid_replay_is_rejected_before_gpu_selection(monkeypatch, tmp_path, synthetic_fixture):
    fixture = copy.deepcopy(synthetic_fixture)
    fixture["metadata"]["revision"] = "different checkpoint"
    path = tmp_path / "fixture.pt"
    torch.save(fixture, path)
    select_gpu = Mock()
    monkeypatch.setattr(torch.cuda, "set_device", select_gpu)
    monkeypatch.setattr(
        "sys.argv",
        ["check_router_gate", "--weights", str(path), "--output", str(tmp_path / "report.json")],
    )
    with pytest.raises(ValueError, match="revision"):
        replay.main()
    select_gpu.assert_not_called()
    assert not (tmp_path / "report.json").exists()
