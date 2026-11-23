# SPDX-License-Identifier: Apache-2.0
"""h3-row-mean-v1: fixed-order FP32 row sums, independent of the batch."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytest.importorskip("triton")

from rl_engine.backends.shared.triton.reduction import row_mean  # noqa: E402

# Row lengths around the chunk and block edges, plus one H3 video latent (24 x 32 x 48 x 84).
LENGTHS = [1, 7, 1023, 1024, 1025, 8191, 8192, 8193, 3 * 8192 + 5, 24 * 32 * 48 * 84]


def _rows(b, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, k, generator=g) * torch.logspace(-3, 3, b)[:, None]).float().cuda()


@pytest.mark.parametrize("k", LENGTHS)
def test_sums_agree_with_float64(k):
    x = _rows(3, k, k)
    got = row_mean.row_sums(x).double()
    ref = x.double().sum(dim=1)
    scale = x.double().abs().sum(dim=1)
    assert ((got - ref).abs() <= 1e-6 * scale + 1e-30).all()


@pytest.mark.parametrize("k", [1025, 8193, 24 * 32 * 48 * 84])
def test_a_row_is_the_same_bytes_alone_and_in_any_batch(k):
    x = _rows(8, k, 3)
    alone = [row_mean.row_sums(x[i : i + 1].contiguous()) for i in range(8)]
    for b in (2, 3, 5, 8):
        for start in range(0, 8 - b + 1):
            out = row_mean.row_sums(x[start : start + b].contiguous())
            for j in range(b):
                assert torch.equal(
                    out[j : j + 1].view(torch.int32), alone[start + j].view(torch.int32)
                )


def test_repeated_calls_give_the_same_bytes():
    x = _rows(4, 3 * 8192 + 5, 9)
    first = row_mean.row_sums(x)
    for _ in range(3):
        assert torch.equal(row_mean.row_sums(x).view(torch.int32), first.view(torch.int32))


def test_a_strided_batch_view_matches_contiguous_rows():
    x = _rows(6, 9000, 4)
    view = x[::2]  # rows with stride 2 * K, unit stride inside a row
    assert torch.equal(row_mean.row_sums(view), row_mean.row_sums(view.contiguous()))


def test_refuses_what_it_does_not_implement():
    with pytest.raises(TypeError):
        row_mean.row_sums(torch.ones(2, 3, device="cuda", dtype=torch.float64))
    with pytest.raises(TypeError):
        row_mean.row_sums(torch.ones(2, 3))
    with pytest.raises(ValueError):
        row_mean.row_sums(torch.ones(3, 2, device="cuda").t())
    with pytest.raises(ValueError):
        row_mean.row_sums(torch.ones(2, 0, device="cuda"))
    assert row_mean.row_sums(torch.ones(0, 4, device="cuda")).shape == (0,)


@pytest.mark.parametrize("offset", [0, 1, 3, 4, 8])
def test_the_bytes_do_not_depend_on_alignment(offset):
    """Rows at offsets and strides with other alignments sum to the same bytes."""

    k = 9000
    base = _rows(1, 4 * k + 64, 5)[0]
    row = base[offset : offset + k].reshape(1, k)
    copy = row.clone()
    wide = torch.empty(2, 2 * k + 16, device="cuda")
    wide[1, :k] = row[0]
    assert torch.equal(
        row_mean.row_sums(row).view(torch.int32), row_mean.row_sums(copy).view(torch.int32)
    )
    assert torch.equal(
        row_mean.row_sums(wide[1:, :k]).view(torch.int32), row_mean.row_sums(copy).view(torch.int32)
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_a_tensor_on_another_device_than_the_current_one():
    """The launches follow the input's device, not the current one (whose CUDA context may not
    even exist yet: run in a fresh process that has only touched ``cuda:1``)."""

    import subprocess
    import sys

    code = """
import torch
from rl_engine.backends.shared.triton.reduction import row_mean
g = torch.Generator().manual_seed(11)
x = torch.randn(3, 8193, generator=g).to("cuda:1")
with torch.cuda.device(1):
    want = row_mean.row_sums(x)
with torch.cuda.device(0):
    got = row_mean.row_sums(x)
assert got.device == x.device, got.device
assert torch.equal(got.view(torch.int32), want.view(torch.int32))
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
