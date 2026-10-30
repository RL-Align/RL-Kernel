# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise H3 validation at the native entrypoints, bypassing Python guards."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def native_extension():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    names = (
        "h3_rmsnorm_forward",
        "h3_rmsnorm_backward",
        "h3_gate_residual_forward",
        "h3_gate_residual_backward",
    )
    if not _EXT_AVAILABLE or not all(hasattr(_C, name) for name in names):
        pytest.skip("rl_engine._C lacks native H3 entrypoints")
    return _C


@pytest.fixture
def native_case():
    device = "cuda:0"
    x = torch.arange(16, device=device, dtype=torch.float32).view(2, 8) / 10
    return {
        "x": x,
        "weight": torch.ones(8, device=device),
        "shift": torch.zeros(2, 8, device=device),
        "scale": torch.zeros(2, 8, device=device),
        "gate": torch.ones(2, 8, device=device),
        "grad": torch.ones_like(x),
        "rstd": torch.rsqrt(x.square().mean(-1) + 1e-5),
        "index": torch.arange(2, device=device, dtype=torch.int64),
        "sorted_pos": torch.arange(2, device=device, dtype=torch.int64),
        "tile_begin": torch.arange(2, device=device, dtype=torch.int64),
        "tile_end": torch.arange(1, 3, device=device, dtype=torch.int64),
        "seg_first_tile": torch.arange(3, device=device, dtype=torch.int64),
    }


def _native_call(extension, operation, boundary, case):
    x, index = case["x"], case["index"]
    tiles = tuple(case[name] for name in ("sorted_pos", "tile_begin", "tile_end", "seg_first_tile"))
    if operation == "rmsnorm":
        modulation = (case["shift"], case["scale"], index)
        if boundary == "forward":
            return extension.h3_rmsnorm_forward(x, case["weight"], 1e-5, *modulation)
        return extension.h3_rmsnorm_backward(
            case["grad"], x, case["weight"], case["rstd"], *modulation, *tiles
        )
    if boundary == "forward":
        return extension.h3_gate_residual_forward(x, x, case["gate"], index)
    return extension.h3_gate_residual_backward(case["grad"], x, case["gate"], index, *tiles)


def _other_cuda(tensor):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    return tensor.to("cuda:1")


@pytest.mark.parametrize("operation", ["rmsnorm", "gate_residual"])
@pytest.mark.parametrize("boundary", ["forward", "backward"])
@pytest.mark.parametrize("index_case", ["empty", "cpu", "other_cuda"])
def test_native_rejects_invalid_row_index(
    native_extension, native_case, operation, boundary, index_case
):
    index = native_case["index"]
    if index_case == "empty":
        native_case["index"] = index[:0]
    elif index_case == "cpu":
        native_case["index"] = index.cpu()
    else:
        native_case["index"] = _other_cuda(index)
    with pytest.raises(
        RuntimeError, match="index must be a non-empty contiguous int64 tensor on cuda:0"
    ):
        _native_call(native_extension, operation, boundary, native_case)


@pytest.mark.parametrize("operation", ["rmsnorm", "gate_residual"])
@pytest.mark.parametrize("metadata", ["sorted_pos", "tile_begin", "tile_end", "seg_first_tile"])
@pytest.mark.parametrize("device", ["cpu", "other_cuda"])
def test_native_rejects_mixed_device_tiles(
    native_extension, native_case, operation, metadata, device
):
    tensor = native_case[metadata]
    native_case[metadata] = tensor.cpu() if device == "cpu" else _other_cuda(tensor)
    with pytest.raises(
        RuntimeError, match="tile metadata must be contiguous int64 tensors on cuda:0"
    ):
        _native_call(native_extension, operation, "backward", native_case)


@pytest.mark.parametrize("operation", ["rmsnorm", "gate_residual"])
def test_native_accepts_same_device_inputs(native_extension, native_case, operation):
    x = native_case["x"].detach().clone().requires_grad_(True)
    if operation == "rmsnorm":
        weight = native_case["weight"].detach().clone().requires_grad_(True)
        shift = native_case["shift"].detach().clone().requires_grad_(True)
        scale = native_case["scale"].detach().clone().requires_grad_(True)
        norm = torch.nn.functional.rms_norm(x, (8,), weight, 1e-5)
        ref = norm * (1 + scale) + shift
        ref.backward(native_case["grad"])
        expected_grads = (x.grad, weight.grad, shift.grad, scale.grad)
        got, _ = _native_call(native_extension, operation, "forward", native_case)
    else:
        gate = native_case["gate"].detach().clone().requires_grad_(True)
        ref = native_case["x"] + gate * x
        ref.backward(native_case["grad"])
        expected_grads = (x.grad, gate.grad)
        got = _native_call(native_extension, operation, "forward", native_case)
    torch.testing.assert_close(got, ref)
    actual_grads = _native_call(native_extension, operation, "backward", native_case)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected)


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


@pytest.mark.cuda_only
def test_valid_plain_and_modulated_backward(native, inputs):
    plain = {name: inputs[name] for name in ("grad", "x", "weight", "rstd")}
    assert len(native.h3_rmsnorm_backward(**plain)) == 2
    dx, dweight, dshift, dscale = native.h3_rmsnorm_backward(**inputs)
    assert dx.shape == inputs["x"].shape
    assert dweight.shape == inputs["weight"].shape
    assert dscale.shape == inputs["scale"].shape
    torch.testing.assert_close(dshift, torch.full_like(inputs["shift"], 2.0))
    torch.cuda.synchronize(inputs["x"].device)


@pytest.mark.cuda_only
@pytest.mark.parametrize("entrypoint", ["forward", "backward"])
def test_rejects_empty_row_index(native, inputs, entrypoint):
    inputs["index"] = inputs["index"][:0]
    with pytest.raises(RuntimeError, match="row index must be a non-empty"):
        _call(native, inputs, entrypoint)


@pytest.mark.cuda_only
@pytest.mark.parametrize("entrypoint", ["forward", "backward"])
def test_rejects_zero_hidden_size(native, inputs, entrypoint):
    inputs["x"] = inputs["x"][:, :0].contiguous()
    inputs["weight"] = inputs["weight"][:0]
    with pytest.raises(RuntimeError, match="at least one column"):
        _call(native, inputs, entrypoint)


@pytest.mark.cuda_only
@pytest.mark.parametrize("defect", ["cpu", "dtype", "rank", "stride", "length"])
@pytest.mark.parametrize("mode", ["plain", "modulated"])
def test_rejects_invalid_rstd(native, inputs, defect, mode):
    inputs["rstd"] = _malformed(inputs["rstd"], defect)
    if mode == "plain":
        inputs = {name: inputs[name] for name in ("grad", "x", "weight", "rstd")}
    with pytest.raises(RuntimeError, match="rstd must be contiguous float32"):
        native.h3_rmsnorm_backward(**inputs)


@pytest.mark.cuda_only
@pytest.mark.parametrize("name", TILE_NAMES)
@pytest.mark.parametrize("defect", ["cpu", "dtype", "rank", "stride", "length"])
def test_rejects_invalid_segment_tiles(native, inputs, name, defect):
    inputs[name] = _malformed(inputs[name], defect)
    message = "tile_begin and tile_end" if defect == "length" and name.startswith("tile_") else name
    with pytest.raises(RuntimeError, match=message):
        native.h3_rmsnorm_backward(**inputs)


@pytest.mark.cuda_only
@pytest.mark.parametrize("name", TILE_NAMES)
def test_rejects_missing_segment_tiles(native, inputs, name):
    inputs[name] = None
    with pytest.raises(RuntimeError, match="needs the sorted segment tiles"):
        native.h3_rmsnorm_backward(**inputs)


@pytest.mark.cuda_only
@pytest.mark.parametrize("name", TILE_NAMES)
def test_plain_backward_rejects_segment_tiles(native, inputs, name):
    plain = {key: inputs[key] for key in ("grad", "x", "weight", "rstd", name)}
    with pytest.raises(RuntimeError, match="sorted segment tiles require modulation"):
        native.h3_rmsnorm_backward(**plain)


@pytest.mark.cuda_only
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


@pytest.mark.parametrize(
    "operation", ["gate_forward", "gate_dy", "norm_forward", "norm_dx", "norm_partials"]
)
@pytest.mark.parametrize("bad_index", [-1, 2])
def test_native_table_bounds_fail_in_isolated_process(native_extension, operation, bad_index):
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine import _C
        op, bad = sys.argv[1], int(sys.argv[2])
        x = torch.ones(2, 8, device="cuda")
        w = torch.ones(8, device="cuda")
        table = torch.ones(2, 8, device="cuda")
        index = torch.tensor([bad, 0], device="cuda")
        rstd = torch.ones(2, device="cuda")
        rows = torch.arange(2, device="cuda")
        begin = torch.tensor([0], device="cuda")
        end = torch.tensor([2], device="cuda")
        try:
            if op == "gate_forward":
                _C.h3_gate_residual_forward(x, x, table, index)
            elif op == "gate_dy":
                _C.h3_gate_residual_backward_dy(x, table, index)
            elif op == "norm_forward":
                _C.h3_rmsnorm_forward(x, w, 1e-5, table, table, index)
            elif op == "norm_dx":
                _C.h3_rmsnorm_backward_dx(x, x, w, rstd, table, table, index)
            else:
                _C.h3_rmsnorm_backward_partials(
                    x, x, w, rstd, table, table, index, rows, begin, end, rows, begin, end)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "device-side assert" in str(exc):
                print("BOUNDS_ASSERT_CONFIRMED")
                sys.exit(0)
            raise
        raise AssertionError("invalid index reached native table lookup without an assertion")
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, operation, str(bad_index)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BOUNDS_ASSERT_CONFIRMED" in result.stdout


@pytest.mark.parametrize("metadata", ["sorted_pos", "tile_begin", "tile_end", "seg_first_tile"])
def test_row_gather_rejects_metadata_on_other_device(native_extension, metadata):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    grad = torch.ones(6, 2, 8, device="cuda:0")
    tensors = dict(
        sorted_pos=torch.tensor([0, 1], device="cuda:0"),
        tile_begin=torch.tensor([0], device="cuda:0"),
        tile_end=torch.tensor([2], device="cuda:0"),
        seg_first_tile=torch.tensor([0, 1], device="cuda:0"),
    )
    tensors[metadata] = tensors[metadata].to("cuda:1")
    with pytest.raises(RuntimeError, match="grad device"):
        native_extension.h3_adaln_row_gather_backward(grad, **tensors, out_dtype=torch.float32)


@pytest.mark.parametrize("metadata", ["sorted_pos", "tile_begin", "tile_end", "seg_first_tile"])
@pytest.mark.parametrize("bound", ["negative", "past_end"])
def test_row_gather_metadata_bounds_in_isolated_process(native_extension, metadata, bound):
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        import torch
        from rl_engine import _C
        name, bound = sys.argv[1:]
        grad = torch.ones(6, 2, 8, device="cuda")
        metadata = dict(sorted_pos=torch.tensor([0, 1], device="cuda"),
                        tile_begin=torch.tensor([0], device="cuda"),
                        tile_end=torch.tensor([2], device="cuda"),
                        seg_first_tile=torch.tensor([0, 1], device="cuda"))
        metadata[name][0] = -1 if bound == "negative" else 3
        try:
            _C.h3_adaln_row_gather_backward(grad, **metadata, out_dtype=torch.float32)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "device-side assert" in str(exc):
                print("BOUNDS_ASSERT_CONFIRMED")
                sys.exit(0)
            raise
        raise AssertionError("invalid metadata was not rejected")
    """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, metadata, bound], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BOUNDS_ASSERT_CONFIRMED" in result.stdout


def test_rmsnorm_rejects_modulation_rows_that_overlap(native_extension, native_case):
    """A table of rows needs a row stride of at least N; an overlapping view is not one."""

    table = torch.arange(32, dtype=torch.float32, device="cuda")
    overlapping = table.as_strided((2, 8), (4, 1))
    with pytest.raises(RuntimeError, match="row stride 4 is smaller than N=8"):
        native_extension.h3_rmsnorm_forward(
            native_case["x"], native_case["weight"], 1e-5, overlapping, overlapping,
            native_case["index"],
        )  # fmt: skip


def test_rmsnorm_reads_a_view_of_a_wider_table_by_its_row_pitch(native_extension, native_case):
    """``table.chunk(2, dim=-1)`` has row stride 2N; the kernel must index rows by it."""

    g = torch.Generator(device="cpu").manual_seed(0)
    views = torch.randn(2, 16, generator=g).cuda().chunk(2, dim=-1)
    copies = tuple(v.contiguous() for v in views)
    assert views[0].stride(0) == 16 and copies[0].stride(0) == 8
    out = [
        native_extension.h3_rmsnorm_forward(
            native_case["x"], native_case["weight"], 1e-5, shift, scale, native_case["index"]
        )[0]
        for shift, scale in (views, copies)
    ]
    assert torch.equal(out[0], out[1])


_BACKWARD_CHILD = """
import sys
import torch
from rl_engine.backends.extension import _C

kind, mutation = sys.argv[1:]
x = torch.randn(4, 8, device="cuda")
weight = torch.ones(8, device="cuda")
gate = torch.ones(2, 8, device="cuda")
index = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
i64 = dict(dtype=torch.int64, device="cuda")
sorted_pos = torch.tensor([0, 2, 1, 3], **i64)
begin, end = torch.tensor([0, 2], **i64), torch.tensor([2, 4], **i64)
first = torch.tensor([0, 1, 2], **i64)
if mutation == "tile_end_past_rows":
    end = torch.tensor([2, 9], **i64)
elif mutation == "position_past_rows":
    sorted_pos = torch.tensor([0, 2, 1, 4], **i64)
elif mutation == "segments_past_tiles":
    first = torch.tensor([0, 1, 5], **i64)
elif mutation == "index_past_table":
    index = torch.tensor([0, 1, 0, 7], **i64)
elif mutation == "index_negative":
    index = torch.tensor([0, 1, 0, -1], **i64)
if kind == "rmsnorm_backward":
    _, rstd = _C.h3_rmsnorm_forward(x, weight, 1e-5)
    shift, scale = torch.zeros(2, 16, device="cuda").chunk(2, dim=1)
    _C.h3_rmsnorm_backward(torch.ones_like(x), x, weight, rstd, shift, scale, index, sorted_pos,
                           begin, end, first)
elif kind == "gate_backward":
    _C.h3_gate_residual_backward(torch.ones_like(x), x, gate, index, sorted_pos, begin, end, first)
elif kind == "gate_partials":  # sequence parallel: explicit row list and tiles
    _C.h3_gate_grad_partials(torch.ones_like(x), x, sorted_pos, begin, end)
elif kind == "gate_fold":
    _C.h3_gate_grad_fold(torch.ones(2, 8, device="cuda"), first, torch.float32)
elif kind == "rmsnorm_partials":
    _, rstd = _C.h3_rmsnorm_forward(x, weight, 1e-5)
    _C.h3_rmsnorm_backward_partials(torch.ones_like(x), x, weight, rstd, None, None, None,
                                    sorted_pos, begin, end, None, None, None)
elif kind == "rmsnorm_fold":
    _C.h3_rmsnorm_fold_partials(torch.ones(2, 8, device="cuda"), weight,
                                torch.ones(2, 16, device="cuda"), first)
elif kind == "rmsnorm_partials_modulated":
    # called without the dx entry point having run: the partials check the index values themselves
    _, rstd = _C.h3_rmsnorm_forward(x, weight, 1e-5)
    shift, scale = torch.zeros(2, 16, device="cuda").chunk(2, dim=1)
    _C.h3_rmsnorm_backward_partials(torch.ones_like(x), x, weight, rstd, shift, scale, index,
                                    sorted_pos, begin, end, sorted_pos, begin, end)
torch.cuda.synchronize()
"""

_METADATA_CASES = {
    "rmsnorm_backward": ("tile_end_past_rows", "position_past_rows", "segments_past_tiles"),
    "gate_backward": ("tile_end_past_rows", "position_past_rows", "segments_past_tiles"),
    "gate_partials": ("tile_end_past_rows", "position_past_rows"),
    "gate_fold": ("segments_past_tiles",),
    "rmsnorm_partials": ("tile_end_past_rows", "position_past_rows"),
    "rmsnorm_partials_modulated": ("index_past_table", "index_negative"),
    "rmsnorm_fold": ("segments_past_tiles",),
}


@pytest.mark.parametrize("kind", sorted(_METADATA_CASES))
def test_segment_metadata_that_does_not_describe_the_rows_ends_in_a_device_assert(
    native_extension, kind
):
    """Every entry point that reads caller-built tile lists checks them before it reads."""

    import os
    import subprocess
    import sys

    if not hasattr(native_extension, "h3_gate_grad_partials"):
        pytest.skip("rl_engine._C lacks the H3 SP entry points")

    def run(mutation):
        return subprocess.run(
            [sys.executable, "-c", _BACKWARD_CHILD, kind, mutation],
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )

    control = run("none")
    assert control.returncode == 0, control.stderr
    for mutation in _METADATA_CASES[kind]:
        done = run(mutation)
        assert done.returncode != 0, (kind, mutation)
        assert "device-side assert triggered" in done.stderr, (kind, mutation, done.stderr)


@pytest.mark.parametrize("entry", ["backward_dx", "backward_partials"])
@pytest.mark.parametrize("shape", ["strided", "expanded"])
def test_sequence_parallel_norm_entries_reject_a_non_dense_rstd(
    native_extension, native_case, shape, entry
):
    """``rstd`` is read as a dense (M,) array: a strided or stride-0 view must be rejected."""

    name = f"h3_rmsnorm_{entry}"
    if not hasattr(native_extension, name):
        pytest.skip(f"rl_engine._C lacks {name}")
    case = dict(native_case)
    rows = case["x"].shape[0]
    if shape == "strided":
        bad = torch.stack([case["rstd"], case["rstd"]], dim=1)[:, 0]
    else:
        bad = torch.zeros(1, device=case["x"].device).expand(rows)
    assert bad.numel() == rows and not bad.is_contiguous()
    modulation = (case["shift"], case["scale"], case["index"])
    tiles = (case["sorted_pos"], case["tile_begin"], case["tile_end"])
    with pytest.raises(RuntimeError, match="rstd must be contiguous"):
        if entry == "backward_dx":
            native_extension.h3_rmsnorm_backward_dx(
                case["grad"], case["x"], case["weight"], bad, *modulation
            )
        else:
            native_extension.h3_rmsnorm_backward_partials(
                case["grad"], case["x"], case["weight"], bad, *modulation, *tiles, *tiles
            )
