# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Range-download safety checks, without checkpoint network calls."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "h3_fixture_under_test",
    Path(__file__).resolve().parents[1] / "scripts/prepare_h3_down_fixture.py",
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class Response:
    def __init__(self, status, content_range, data):
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.data = data
        self.read_calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, count):
        self.read_calls.append(count)
        return self.data[:count]


def test_range_download_limits_read_and_preserves_requested_bytes():
    response = Response(206, "bytes 10-13/100", b"abcd")

    def opener(request, timeout):
        assert request.get_header("Range") == "bytes=10-13"
        assert timeout == 120
        return response

    assert fixture.read_range("https://example.test/shard", 10, 13, opener=opener) == b"abcd"
    assert response.read_calls == [5]


@pytest.mark.parametrize(
    "status,content_range,data,error",
    [
        (200, "", b"full shard", "refusing full shard"),
        (206, "bytes 0-3/100", b"abcd", "Content-Range"),
        (206, "bytes 10-13/100", b"abc", "length mismatch"),
        (206, "bytes 10-13/100", b"abcde", "length mismatch"),
    ],
)
def test_range_download_rejects_invalid_ranges(status, content_range, data, error):
    response = Response(status, content_range, data)
    with pytest.raises(RuntimeError, match=error):
        fixture.read_range("https://example.test/shard", 10, 13, opener=lambda *a, **k: response)
    if status == 200:
        assert response.read_calls == []


def test_tensor_extent_rejects_wrong_shape_dtype_or_byte_count():
    entry = {"dtype": "BF16", "shape": [5376, 14336], "data_offsets": [1024, 154141696]}
    assert fixture.tensor_extent(entry, 512) == (1544, 154142216)
    for changed in ({"dtype": "F32"}, {"shape": [14336, 5376]}, {"data_offsets": [0, 2]}):
        with pytest.raises(ValueError):
            fixture.tensor_extent(entry | changed, 512)
