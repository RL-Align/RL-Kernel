# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise H3 validation at the native entrypoints, bypassing Python guards."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture
def native_extension():
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    names = (
        "h3_adaln_row_gather_backward",
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
    if operation == "gather":
        return extension.h3_adaln_row_gather_backward(
            case["grad"].unsqueeze(0), *tiles, case["grad"].dtype
        )
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


@pytest.mark.parametrize("operation", ["rmsnorm", "gate_residual", "gather"])
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


@pytest.mark.parametrize("operation", ["rmsnorm", "gate_residual"])
@pytest.mark.parametrize("boundary", ["forward", "backward"])
@pytest.mark.parametrize("bad_index", [-1, 2])
def test_native_rejects_out_of_range_index_in_subprocess(
    native_extension, operation, boundary, bad_index
):
    # A device assertion poisons the CUDA context, so isolate each invalid launch.
    import os
    import subprocess
    import sys

    code = f"""
import torch
from rl_engine import _C
x = torch.ones(2, 8, device='cuda')
w = torch.ones(8, device='cuda')
index = torch.tensor([0, {bad_index}], device='cuda', dtype=torch.int64)
pos = torch.arange(2, device='cuda', dtype=torch.int64)
ends = pos + 1
seg = torch.arange(3, device='cuda', dtype=torch.int64)
if {operation!r} == 'gate_residual':
    if {boundary!r} == 'forward':
        _C.h3_gate_residual_forward(x, x, x, index)
    else:
        _C.h3_gate_residual_backward(x, x, x, index, pos, pos, ends, seg)
else:
    if {boundary!r} == 'forward':
        _C.h3_rmsnorm_forward(x, w, 1e-5, x, x, index)
    else:
        rstd = torch.ones(2, device='cuda')
        _C.h3_rmsnorm_backward(x, x, w, rstd, x, x, index, pos, pos, ends, seg)
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
    )
    assert result.returncode != 0
    assert "device-side assert" in result.stderr
