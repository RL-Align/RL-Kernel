# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA stable_topk_per_row: byte-equal to the golden and batch-invariant."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.topk import stable_topk_per_row


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "stable_topk_per_row_out")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with stable_topk required")


def _cuda(*args, **kwargs):
    from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda

    return stable_topk_per_row_cuda(*args, **kwargs)


def _ranges(pairs):
    t = torch.tensor(pairs, dtype=torch.int32, device="cuda")
    return t[:, 0].contiguous(), t[:, 1].contiguous()


def _logits(kind: str, rows: int, width: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "randn":
        x = torch.randn(rows, width, generator=gen)
    elif kind == "few_values":  # large tie groups
        x = torch.randint(-3, 4, (rows, width), generator=gen).float()
    elif kind == "relu":  # indexer-like: most scores exactly zero
        x = torch.relu(torch.randn(rows, width, generator=gen) - 1.5)
    elif kind == "constant":
        x = torch.full((rows, width), 0.25)
    elif kind == "special":
        pool = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), 1e-45, -1e-45, 3.0e38]
        )
        x = pool[torch.randint(0, len(pool), (rows, width), generator=gen)]
        bits = x.view(torch.int32)
        nan = torch.rand(rows, width, generator=gen) < 0.02
        bits[nan] = torch.where(
            torch.rand(int(nan.sum()), generator=gen) < 0.5,
            torch.tensor(0x7FC00001, dtype=torch.int32),
            torch.tensor(-0x003FFFFF, dtype=torch.int32),  # 0xFFC00001
        )
    else:
        raise ValueError(kind)
    return x.cuda()


def _check(logits, starts, ends, top_k):
    """Every block shape (and the automatic choice) must write the golden bytes."""
    want = stable_topk_per_row(logits, starts, ends, top_k)
    for threads in (0, 256, 512, 1024):
        got = _cuda(logits, starts, ends, top_k, threads=threads)
        assert got.dtype == torch.int32 and got.shape == want.shape
        mismatched = (got != want).any(dim=1).nonzero().flatten().tolist()
        assert not mismatched, f"threads={threads}: rows {mismatched[:8]} differ"
    return got


@pytest.mark.parametrize("kind", ["randn", "few_values", "relu", "constant", "special"])
@pytest.mark.parametrize("top_k", [1, 64, 512, 2048])
def test_matches_golden(kind, top_k):
    width = 5000
    lengths = [0, 1, top_k - 1, top_k, top_k + 1, 2 * top_k + 3, width - 37, width]
    pairs = []
    for i, n in enumerate(lengths):
        n = max(0, min(n, width))
        start = (i * 37) % (width - n + 1)
        pairs.append((start, start + n))
    _check(_logits(kind, len(pairs), width, seed=top_k), *_ranges(pairs), top_k)


@pytest.mark.parametrize("width", [4095, 4096, 4097, 65536, 262144])
def test_wide_rows(width):
    # vLLM ..._has_no_packed_width_limit, plus random and tie-heavy rows.
    logits = torch.cat(
        [
            torch.zeros(1, width, device="cuda"),
            _logits("randn", 1, width, 3),
            _logits("relu", 1, width, 4),
        ]
    )
    starts, ends = _ranges([(17, min(width, 529)), (0, width), (width // 3, width)])
    got = _check(logits, starts, ends, 512)
    assert got[0, :64].tolist() == list(range(511, 447, -1))


def test_tie_cutoff_spans_many_chunks():
    # The threshold tie group is far larger than one 512-thread chunk and only
    # part of it is kept, so the cutoff scan has to walk several chunks.
    width = 20000
    logits = torch.zeros(3, width, device="cuda")
    logits[0, ::7] = 1.0  # 2858 values above the tie
    logits[1, :300] = 2.0  # cutoff deep inside the zeros
    logits[2, 19000:] = -1.0  # ties at the low end of the row
    starts, ends = _ranges([(0, width)] * 3)
    _check(logits, starts, ends, 2048)
    _check(logits, starts, ends, 512)


def test_batch_invariance_and_run_to_run():
    width, top_k = 8192, 512
    logits = _logits("relu", 64, width, 5)
    pairs = [(i * 61 % 1000, width - (i * 13 % 2000)) for i in range(64)]
    starts, ends = _ranges(pairs)
    full = _cuda(logits, starts, ends, top_k)
    assert torch.equal(_cuda(logits, starts, ends, top_k), full)
    for r in (0, 1, 31, 63):
        s, e = pairs[r]
        alone = _cuda(logits[r : r + 1, s:e].clone(), *_ranges([(0, e - s)]), top_k)
        assert torch.equal(alone[0], full[r]), r
        window = _cuda(logits[r : r + 3], starts[r : r + 3], ends[r : r + 3], top_k)
        assert torch.equal(window, full[r : r + 3]), r


def test_strided_logits_and_output_view():
    base = _logits("few_values", 3000, 16, 6)
    logits = base.t()  # stride1 = 16
    starts, ends = _ranges([(0, 3000)] * 16)
    want = stable_topk_per_row(logits.contiguous(), starts, ends, 256)
    buffer = torch.full((16, 300), 7, dtype=torch.int32, device="cuda")
    view = buffer[:, :256]
    assert _cuda(logits, starts, ends, 256, out=view) is view
    assert torch.equal(view, want)
    assert torch.all(buffer[:, 256:] == 7)


def test_empty_batch_and_validation():
    empty = torch.empty(0, 100, device="cuda")
    starts, ends = _ranges([(0, 1)])
    assert _cuda(empty, starts[:0], ends[:0], 8).shape == (0, 8)
    logits = torch.zeros(1, 100, device="cuda")
    with pytest.raises(RuntimeError):
        _cuda(logits.double(), starts, ends, 8)
    with pytest.raises(RuntimeError):
        _cuda(logits, starts.long(), ends, 8)
    with pytest.raises(ValueError):
        _cuda(logits, starts, ends, 4096)
    with pytest.raises(RuntimeError):
        _cuda(logits, starts, ends, 8, out=torch.empty(1, 8, device="cuda", dtype=torch.int64))
    with pytest.raises(RuntimeError):
        _cuda(logits, starts, ends, 8, threads=128)
    torch.cuda.synchronize()
