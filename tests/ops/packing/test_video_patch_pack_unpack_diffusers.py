# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare H3 pack and its inverse with the pinned Diffusers implementation."""

import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import torch

from rl_engine.reference.packing.video_patch_pack_unpack import NativeVideoPatchPackUnpackOp

# Load the repository file explicitly: an installed package named benchmarks
# would otherwise hide this repository's namespace directory.
_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks/operators/packing/benchmark_video_patch_pack_unpack.py"
)
_SPEC = importlib.util.spec_from_file_location("_video_patch_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)

SHAPES = (
    (2, 24, 3, 6, 10),
    (1, 24, 32, 48, 84),
    (1, 24, 37, 48, 84),  # Default 124 frames map to T=37 under the pinned VAE rules.
    (1, 24, 72, 84, 48),  # Ten-second sample: 243 aligned frames in portrait layout.
    (1, 24, 102, 48, 84),  # Longest valid 345-frame sample within the 15-second limit.
    (1, 24, 1, 48, 84),  # Single-frame condition; its access size differs from generation.
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().cpu().view(torch.uint8),
        expected.detach().contiguous().cpu().view(torch.uint8),
    )


@pytest.fixture(scope="module")
def diffusers_op():
    try:
        installed = version("diffusers")
    except PackageNotFoundError:
        pytest.skip(f"Optional reference requires diffusers=={_BENCHMARK.DIFFUSERS_VERSION}")
    if installed != _BENCHMARK.DIFFUSERS_VERSION:
        pytest.skip(
            f"Reference requires diffusers=={_BENCHMARK.DIFFUSERS_VERSION}; got {installed}"
        )
    return _BENCHMARK.DiffusersVideoPatch()


@pytest.fixture(params=("cpu", "cuda"))
def op(request):
    device = request.param
    if device == "cuda" and (not torch.cuda.is_available() or torch.version.hip is not None):
        pytest.skip("NVIDIA CUDA GPU required for explicit Diffusers comparison")
    return NativeVideoPatchPackUnpackOp(), device


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_against_diffusers_pack_and_gradient(diffusers_op, op, shape, dtype):
    kernel, device = op
    generator = torch.Generator().manual_seed(420)
    values = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    upstream_x = values.detach().requires_grad_()
    kernel_x = values.detach().requires_grad_()

    upstream = diffusers_op.pack(upstream_x)
    actual = kernel.pack(kernel_x)
    assert_bits(actual, upstream)
    assert_bits(kernel.unpack(upstream.reshape(actual.shape), shape), values)

    grad_rows = torch.randn(upstream.shape, generator=generator).to(device=device, dtype=dtype)
    upstream_dx = torch.autograd.grad(upstream, upstream_x, grad_rows)[0]
    kernel_dx = torch.autograd.grad(actual, kernel_x, grad_rows.reshape(actual.shape))[0]
    assert_bits(kernel_dx, upstream_dx)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_against_diffusers_unpack_and_gradient(diffusers_op, op, shape, dtype):
    kernel, device = op
    b, _, t, h, w = shape
    values = torch.randn(b, t * (h // 2) * (w // 2), 96, device=device, dtype=dtype)
    upstream_x = values.detach().requires_grad_()
    kernel_x = values.detach().requires_grad_()
    upstream = diffusers_op.unpack(upstream_x, shape)
    actual = kernel.unpack(kernel_x, shape)
    assert_bits(actual, upstream)
    grad = torch.randn_like(actual)
    assert_bits(
        torch.autograd.grad(actual, kernel_x, grad)[0],
        torch.autograd.grad(upstream, upstream_x, grad)[0],
    )
