# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Direct native calls reject malformed pointers and layouts before launching CUDA work."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.cuda_only,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

TILE_NAMES = ("sorted_pos", "tile_begin", "tile_end", "seg_first_tile")


@pytest.fixture
def native():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not all(
        hasattr(_C, name) for name in ("h3_rmsnorm_forward", "h3_rmsnorm_backward")
    ):
        pytest.skip("rl_engine._C lacks h3_rmsnorm_*")
    return _C


@pytest.fixture
def inputs(native):
    x = torch.randn(4, 8, device="cuda")
    weight = torch.ones(8, device=x.device)
    _, rstd = native.h3_rmsnorm_forward(x, weight, 1e-5)
    shift, scale = torch.zeros(2, 16, device=x.device).chunk(2, dim=1)
    return {
        "grad": torch.ones_like(x),
        "x": x,
        "weight": weight,
        "rstd": rstd,
        "shift": shift,
        "scale": scale,
        "index": torch.tensor([0, 1], dtype=torch.int64, device=x.device),
        "sorted_pos": torch.tensor([0, 2, 1, 3], dtype=torch.int64, device=x.device),
        "tile_begin": torch.tensor([0, 2], dtype=torch.int64, device=x.device),
        "tile_end": torch.tensor([2, 4], dtype=torch.int64, device=x.device),
        "seg_first_tile": torch.tensor([0, 1, 2], dtype=torch.int64, device=x.device),
    }


def _call(native, inputs, entrypoint):
    if entrypoint == "forward":
        return native.h3_rmsnorm_forward(
            inputs["x"],
            inputs["weight"],
            1e-5,
            inputs["shift"],
            inputs["scale"],
            inputs["index"],
        )
    return native.h3_rmsnorm_backward(**inputs)


def _malformed(tensor, defect):
    if defect == "cpu":
        return tensor.cpu()
    if defect == "dtype":
        dtype = torch.float16 if tensor.dtype == torch.float32 else torch.int32
        return tensor.to(dtype)
    if defect == "rank":
        return tensor.unsqueeze(0)
    if defect == "stride":
        return torch.stack((tensor, tensor), dim=1)[:, 0]
    if defect == "length":
        return tensor[:-1]
    raise ValueError(f"unknown defect: {defect}")


def test_valid_plain_and_modulated_backward(native, inputs):
    plain = {name: inputs[name] for name in ("grad", "x", "weight", "rstd")}
    assert len(native.h3_rmsnorm_backward(**plain)) == 2
    dx, dweight, dshift, dscale = native.h3_rmsnorm_backward(**inputs)
    assert dx.shape == inputs["x"].shape
    assert dweight.shape == inputs["weight"].shape
    assert dscale.shape == inputs["scale"].shape
    torch.testing.assert_close(dshift, torch.full_like(inputs["shift"], 2.0))
    torch.cuda.synchronize(inputs["x"].device)


@pytest.mark.parametrize("entrypoint", ["forward", "backward"])
def test_rejects_empty_row_index(native, inputs, entrypoint):
    inputs["index"] = inputs["index"][:0]
    with pytest.raises(RuntimeError, match="row index must be a non-empty"):
        _call(native, inputs, entrypoint)


@pytest.mark.parametrize("entrypoint", ["forward", "backward"])
def test_rejects_zero_hidden_size(native, inputs, entrypoint):
    inputs["x"] = inputs["x"][:, :0].contiguous()
    inputs["weight"] = inputs["weight"][:0]
    with pytest.raises(RuntimeError, match="at least one column"):
        _call(native, inputs, entrypoint)


@pytest.mark.parametrize("defect", ["cpu", "dtype", "rank", "stride", "length"])
@pytest.mark.parametrize("mode", ["plain", "modulated"])
def test_rejects_invalid_rstd(native, inputs, defect, mode):
    inputs["rstd"] = _malformed(inputs["rstd"], defect)
    if mode == "plain":
        inputs = {name: inputs[name] for name in ("grad", "x", "weight", "rstd")}
    with pytest.raises(RuntimeError, match="rstd must be contiguous float32"):
        native.h3_rmsnorm_backward(**inputs)


@pytest.mark.parametrize("name", TILE_NAMES)
@pytest.mark.parametrize("defect", ["cpu", "dtype", "rank", "stride", "length"])
def test_rejects_invalid_segment_tiles(native, inputs, name, defect):
    inputs[name] = _malformed(inputs[name], defect)
    message = "tile_begin and tile_end" if defect == "length" and name.startswith("tile_") else name
    with pytest.raises(RuntimeError, match=message):
        native.h3_rmsnorm_backward(**inputs)


@pytest.mark.parametrize("name", TILE_NAMES)
def test_rejects_missing_segment_tiles(native, inputs, name):
    inputs[name] = None
    with pytest.raises(RuntimeError, match="needs the sorted segment tiles"):
        native.h3_rmsnorm_backward(**inputs)


@pytest.mark.parametrize("name", TILE_NAMES)
def test_plain_backward_rejects_segment_tiles(native, inputs, name):
    plain = {key: inputs[key] for key in ("grad", "x", "weight", "rstd", name)}
    with pytest.raises(RuntimeError, match="sorted segment tiles require modulation"):
        native.h3_rmsnorm_backward(**plain)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize(
    ("entrypoint", "name"),
    [
        ("forward", "weight"),
        ("forward", "index"),
        ("backward", "weight"),
        ("backward", "index"),
        ("backward", "rstd"),
        ("backward", "sorted_pos"),
        ("backward", "tile_begin"),
        ("backward", "tile_end"),
        ("backward", "seg_first_tile"),
    ],
)
def test_rejects_tensor_on_another_cuda_device(native, inputs, entrypoint, name):
    other_device = (inputs["x"].device.index + 1) % torch.cuda.device_count()
    inputs[name] = inputs[name].to(f"cuda:{other_device}")
    message = "row index" if name == "index" else name
    with pytest.raises(RuntimeError, match=message):
        _call(native, inputs, entrypoint)


@pytest.mark.parametrize("entrypoint", ["forward", "backward"])
@pytest.mark.parametrize("bad_row", [-1, 2], ids=["negative", "past-table"])
def test_modulation_row_outside_the_table_asserts_in_the_kernel(native, entrypoint, bad_row):
    """check_range=False callers get a device-side assert, not a read past the shift/scale table."""

    probe = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine.backends.extension import _C

        entrypoint, bad_row = sys.argv[1], int(sys.argv[2])
        x = torch.randn(4, 8, device="cuda")
        weight = torch.ones(8, device="cuda")
        _, rstd = _C.h3_rmsnorm_forward(x, weight, 1e-5)
        shift, scale = torch.zeros(2, 16, device="cuda").chunk(2, dim=1)  # a table of two rows
        index = torch.tensor([0, bad_row], dtype=torch.int64, device="cuda")
        if entrypoint == "forward":
            _C.h3_rmsnorm_forward(x, weight, 1e-5, shift, scale, index)
        else:
            _C.h3_rmsnorm_backward(
                grad=torch.ones_like(x), x=x, weight=weight, rstd=rstd, shift=shift, scale=scale,
                index=index,
                sorted_pos=torch.tensor([0, 2, 1, 3], dtype=torch.int64, device="cuda"),
                tile_begin=torch.tensor([0, 2], dtype=torch.int64, device="cuda"),
                tile_end=torch.tensor([2, 4], dtype=torch.int64, device="cuda"),
                seg_first_tile=torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda"),
            )
        torch.cuda.synchronize()
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, entrypoint, str(bad_row)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0, "the kernel accepted a modulation row outside the table"
    assert (
        "device-side assert triggered" in completed.stderr
    ), f"expected a CUDA bounds assertion, got:\n{completed.stdout}\n{completed.stderr}"


def test_rejects_modulation_rows_that_overlap(native, inputs):
    """A table of rows needs a row stride of at least N; an overlapping view is not one."""

    table = torch.arange(32, dtype=torch.float32, device="cuda")
    inputs["shift"] = inputs["scale"] = table.as_strided((2, 8), (4, 1))
    with pytest.raises(RuntimeError, match="row stride 4 is smaller than N=8"):
        _call(native, inputs, "forward")


def test_a_view_of_a_wider_table_is_read_by_its_row_pitch(native, inputs):
    """``table.chunk(2, dim=-1)`` has row stride 2N; the kernel must index rows by it."""

    g = torch.Generator(device="cpu").manual_seed(0)
    table = torch.randn(2, 16, generator=g).cuda()
    views = table.chunk(2, dim=-1)
    copies = tuple(v.contiguous() for v in views)
    assert views[0].stride(0) == 16 and copies[0].stride(0) == 8
    out = []
    for shift, scale in (views, copies):
        inputs["shift"], inputs["scale"] = shift, scale
        out.append(_call(native, inputs, "forward")[0])
    assert torch.equal(out[0], out[1])


def _backward_in_child(mutation: str) -> subprocess.CompletedProcess:
    probe = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine.backends.extension import _C

        mutation = sys.argv[1]
        x = torch.randn(4, 8, device="cuda")
        weight = torch.ones(8, device="cuda")
        _, rstd = _C.h3_rmsnorm_forward(x, weight, 1e-5)
        shift, scale = torch.zeros(2, 16, device="cuda").chunk(2, dim=1)
        args = dict(
            grad=torch.ones_like(x), x=x, weight=weight, rstd=rstd, shift=shift, scale=scale,
            index=torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda"),
            sorted_pos=torch.tensor([0, 2, 1, 3], dtype=torch.int64, device="cuda"),
            tile_begin=torch.tensor([0, 2], dtype=torch.int64, device="cuda"),
            tile_end=torch.tensor([2, 4], dtype=torch.int64, device="cuda"),
            seg_first_tile=torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda"),
        )
        if mutation == "tile_end_past_rows":
            args["tile_end"] = torch.tensor([2, 9], dtype=torch.int64, device="cuda")
        elif mutation == "position_past_rows":
            args["sorted_pos"] = torch.tensor([0, 2, 1, 4], dtype=torch.int64, device="cuda")
        elif mutation == "segments_past_tiles":
            args["seg_first_tile"] = torch.tensor([0, 1, 5], dtype=torch.int64, device="cuda")
        _C.h3_rmsnorm_backward(**args)
        torch.cuda.synchronize()
        """
    )
    return subprocess.run(
        [sys.executable, "-c", probe, mutation],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_the_backward_stops_on_tile_metadata_that_does_not_describe_the_rows(native):
    control = _backward_in_child("none")
    assert control.returncode == 0, control.stderr
    for mutation in ("tile_end_past_rows", "position_past_rows", "segments_past_tiles"):
        done = _backward_in_child(mutation)
        assert done.returncode != 0, mutation
        assert "device-side assert triggered" in done.stderr, (mutation, done.stderr)
