# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import rl_engine.ops.autograd.vjp_fp32 as vjp_fp32
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32

_ROOT = Path(__file__).resolve().parents[3]
_CUDA_SOURCE = _ROOT / "csrc/cuda/norm/rmsnorm.cu"
_OPS_SOURCE = _ROOT / "csrc/bindings/ops.cpp"
_RMSNORM_BACKENDS = (
    _ROOT / "rl_engine/backends/cuda/norm/rmsnorm.py",
    _ROOT / "rl_engine/backends/shared/triton/norm/rmsnorm.py",
)


def _left_fold_reference(rows: torch.Tensor) -> torch.Tensor:
    output_shape = tuple(rows.shape[1:])
    columns = 1
    for extent in output_shape:
        columns *= extent
    flat = rows.reshape(rows.shape[0], columns).float()
    acc = torch.zeros(columns, device=rows.device, dtype=torch.float32)
    for row in range(flat.shape[0]):
        acc = acc + flat[row]
    return acc.reshape(output_shape)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (32, 128),
        (1024, 128),
        (37, 127),
        (3, 2, 5),
        (4,),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_reduce_rows_fp32_matches_ascending_left_fold_on_cpu(shape, dtype):
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32)
    rows = ((values.remainder(29) - 14) / 8).reshape(shape).to(dtype)

    reduced = reduce_rows_fp32(rows)

    assert reduced.dtype == torch.float32
    assert torch.equal(reduced, _left_fold_reference(rows))


def test_reduce_rows_fp32_pins_non_associative_order():
    rows = torch.tensor([[1.0e20], [1.0], [-1.0e20], [1.0]], dtype=torch.float32)

    reduced = reduce_rows_fp32(rows)

    assert torch.equal(reduced, torch.tensor([1.0], dtype=torch.float32))


def test_reduce_rows_fp32_accepts_noncontiguous_rows():
    rows = torch.arange(7 * 37, dtype=torch.float32).reshape(7, 37).transpose(0, 1)
    assert not rows.is_contiguous()

    assert torch.equal(reduce_rows_fp32(rows), _left_fold_reference(rows))


@pytest.mark.parametrize("shape", [(0, 128), (0, 2, 3), (0,), (4, 0)])
def test_reduce_rows_fp32_empty_boundaries(shape):
    rows = torch.empty(shape, dtype=torch.bfloat16)

    reduced = reduce_rows_fp32(rows)

    assert reduced.shape == rows.shape[1:]
    assert reduced.dtype == torch.float32
    assert torch.equal(reduced, torch.zeros(rows.shape[1:], dtype=torch.float32))


def test_reduce_rows_fp32_rejects_scalar_input():
    with pytest.raises(ValueError, match="at least one dimension"):
        reduce_rows_fp32(torch.tensor(1.0))


def test_reduce_rows_fp32_keeps_cpu_fallback_when_extension_is_present(monkeypatch):
    class _UnexpectedExtensionCall:
        def reduce_rows_fp32_left_fold(self, rows):
            raise AssertionError(f"CPU rows reached the CUDA extension: {rows.shape}")

    monkeypatch.setattr(vjp_fp32, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(vjp_fp32, "_C", _UnexpectedExtensionCall())
    rows = torch.arange(33 * 5, dtype=torch.float32).reshape(33, 5)

    assert torch.equal(reduce_rows_fp32(rows), _left_fold_reference(rows))


def test_cuda_left_fold_is_one_ordered_kernel_launch():
    source = _CUDA_SOURCE.read_text(encoding="utf-8")
    kernel_start = source.index("__global__ void reduce_rows_fp32_left_fold_kernel")
    launcher_start = source.index("void reduce_rows_fp32_left_fold_cuda")
    kernel = source[kernel_start:launcher_start]
    launcher = source[launcher_start:]

    assert "for (int64_t row = 0; row < row_count; ++row)" in kernel
    assert "#pragma unroll 1" in kernel
    assert "__fadd_rn(acc, rows[row * columns + column])" in kernel
    assert "float acc = 0.0f;" in kernel
    assert "output[column] = acc;" in kernel
    assert "tl.sum" not in kernel
    assert "cub::" not in kernel.lower()
    assert launcher.count("reduce_rows_fp32_left_fold_kernel<<<") == 1


def test_cuda_left_fold_binding_has_no_multilaunch_reduction():
    source = _OPS_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse((_ROOT / "rl_engine/ops/autograd/vjp_fp32.py").read_text())
    reduce_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "reduce_rows_fp32"
    )

    extension_calls = [
        node
        for node in ast.walk(reduce_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reduce_rows_fp32_left_fold"
    ]
    assert len(extension_calls) == 1
    assert source.count('m.def(\n        "reduce_rows_fp32_left_fold"') == 1
    assert "if (rows.size(1) != 0)" in source


def test_cuda_and_triton_rmsnorm_share_the_left_fold_entrypoint():
    for path in _RMSNORM_BACKENDS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reduce_rows_fp32"
        ]
        assert len(calls) == 1, path


_HAS_CUDA_LEFT_FOLD = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and vjp_fp32._EXT_AVAILABLE
    and hasattr(vjp_fp32._C, "reduce_rows_fp32_left_fold")
)


@pytest.mark.skipif(not _HAS_CUDA_LEFT_FOLD, reason="CUDA left-fold extension is unavailable")
@pytest.mark.parametrize("shape", [(32, 128), (1024, 128), (37, 127), (0, 128), (4, 0)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_reduce_rows_fp32_is_bitwise_equal_to_original_row_loop(shape, dtype):
    generator = torch.Generator().manual_seed(17)
    rows = torch.randn(shape, generator=generator, dtype=torch.float32).to(
        device="cuda", dtype=dtype
    )

    expected = _left_fold_reference(rows)
    reduced = reduce_rows_fp32(rows)

    assert torch.equal(reduced, expected)
