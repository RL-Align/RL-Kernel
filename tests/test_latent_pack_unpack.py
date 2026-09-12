# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Exact-value and raw-bit acceptance for Qwen-Image layout permutations."""

from importlib import import_module
from pathlib import Path

import pytest
import torch

from rl_engine.kernels.ops import base
from rl_engine.kernels.ops.pytorch.packing.latent_pack_unpack import (
    NativeLatentPackOp,
    NativeLatentUnpackOp,
)
from rl_engine.kernels.registry import KernelRegistry, OpBackend

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SIZES = ((1024, 1024), (1328, 1328), (1664, 928))


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().cpu().view(torch.uint8),
        expected.detach().contiguous().cpu().view(torch.uint8),
    )


@pytest.fixture(params=("PYTORCH_CPU", "PYTORCH", "CUDA", "TRITON"))
def ops(request):
    backend = request.param.split("_")[0]
    device = "cpu" if request.param == "PYTORCH_CPU" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    if backend == "CUDA" and (torch.version.hip or not hasattr(base._C, "latent_pack_unpack")):
        pytest.skip("CUDA latent extension unavailable")
    if backend == "TRITON":
        pytest.importorskip("triton")
    return tuple(
        getattr(import_module(module), cls)()
        for direction in ("PACK", "UNPACK")
        for module, cls in [OpBackend[f"{backend}_LATENT_{direction}"].value.rsplit(".", 1)]
    ) + (device,)


# Check exact forward results, gradients, round trips, and batch invariance across backends.
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("image_size", SIZES)
@pytest.mark.parametrize("batch", (1, 2, 4, 8))
def test_forward_backward_exact(ops, dtype, image_size, batch):
    pack, unpack, device = ops
    ih, iw = image_size
    h, w, c = ih // 8, iw // 8, 16
    generator = torch.Generator().manual_seed(386)
    cpu = torch.randn(batch, c, 1, h, w, generator=generator).to(dtype).requires_grad_()
    x = cpu.detach().to(device).requires_grad_()
    expected = NativeLatentPackOp()(cpu, batch, c, h, w)
    packed = pack(x, batch, c, h, w)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    assert_bits(packed, expected)
    restored = unpack(packed, ih, iw, 8)
    assert_bits(restored, cpu)
    # Independent unpack input prevents a paired pack/unpack mistake cancelling out.
    tokens = torch.randn(expected.shape, generator=generator).to(dtype).requires_grad_()
    gpu_tokens = tokens.detach().to(device).requires_grad_()
    spatial = unpack(gpu_tokens, ih, iw, 8)
    ref_spatial = NativeLatentUnpackOp()(tokens, ih, iw, 8)
    assert_bits(spatial, ref_spatial)
    assert_bits(pack(spatial, batch, c, h, w), tokens)
    # Non-contiguous gradients exercise the explicit backward materialization path.
    dp = torch.randn(batch, 4 * c, expected.shape[1], generator=generator).to(dtype)
    dp = dp.transpose(1, 2)
    ds = torch.randn(batch, c, 1, w, h, generator=generator).to(dtype).transpose(-1, -2)
    dx = torch.autograd.grad(packed, x, dp.to(device))[0]
    ref_dx = torch.autograd.grad(expected, cpu, dp)[0]
    dt = torch.autograd.grad(spatial, gpu_tokens, ds.to(device))[0]
    ref_dt = torch.autograd.grad(ref_spatial, tokens, ds)[0]
    assert_bits(dx, ref_dx)
    assert_bits(dt, ref_dt)
    assert_bits(pack(x[:1], 1, c, h, w), packed[:1])
    assert_bits(unpack(gpu_tokens[:1], ih, iw, 8), spatial[:1])


# Check empty and edge shapes, coordinate order, singleton dimensions, and size rounding.
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", ((0, 16, 2, 2), (1, 1, 2, 2), (2, 3, 6, 10), (2, 17, 10, 14), (1, 16, 4, 66))
)
def test_edges_and_coordinate_order(ops, dtype, shape):
    pack, unpack, device = ops
    b, c, h, w = shape
    x = torch.arange(b * c * h * w, dtype=torch.float32).reshape(shape).to(dtype).to(device)
    actual = pack(x, b, c, h, w)
    expected = NativeLatentPackOp()(x.cpu(), b, c, h, w)
    assert_bits(actual, expected)
    assert_bits(unpack(actual, h * 8 + 7, w * 8 + 15, 8), x.unsqueeze(2))
    for view in (x.unsqueeze(1), x.unsqueeze(2)):
        assert_bits(pack(view, b, c, h, w), expected)
    if b and c:
        assert_bits(actual[0, 0, :4], x[0, 0, :2, :2].flatten())


# Check bit preservation for special values and gradients with odd storage offsets.
@pytest.mark.parametrize("dtype", DTYPES)
def test_special_bits_and_unaligned_storage(ops, dtype):
    pack, unpack, device = ops
    patterns = {
        torch.float32: (
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0x7F801234,
            1,
            0x80000001,
            0x7F7FFFFF,
        ),
        torch.float16: (0, 0x8000, 0x7C00, 0xFC00, 0x7E55, 0x7C55, 1, 0x8001, 0x7BFF),
        torch.bfloat16: (0, 0x8000, 0x7F80, 0xFF80, 0x7FC5, 0x7F85, 1, 0x8001, 0x7F7F),
    }
    bits = torch.uint32 if dtype == torch.float32 else torch.uint16
    storage = torch.tensor(patterns[dtype], dtype=bits).view(dtype).repeat(100).to(device)
    x = storage[1 : 1 + 2 * 3 * 6 * 10].view(2, 3, 6, 10).requires_grad_()
    y = pack(x, 2, 3, 6, 10)
    assert_bits(y, NativeLatentPackOp()(x.cpu(), 2, 3, 6, 10))
    assert_bits(unpack(y, 48, 80, 8), x.unsqueeze(2))
    tokens = storage[1:361].view(2, 15, 12).detach().requires_grad_()
    z = unpack(tokens, 48, 80, 8)
    assert_bits(z, NativeLatentUnpackOp()(tokens.cpu(), 48, 80, 8))
    assert_bits(torch.autograd.grad(y, x, tokens)[0], z.squeeze(2))
    assert_bits(torch.autograd.grad(z, tokens, x.unsqueeze(2))[0], y)


# Check forward/backward consistency, gradients through backward, and CUDA stream execution.
def test_forward_backward_and_stream(ops):
    pack, unpack, device = ops
    x = torch.randn(2, 3, 6, 10, device=device, requires_grad=True)
    y = pack(x, 2, 3, 6, 10)
    grad = torch.randn_like(y, requires_grad=True)
    dx = torch.autograd.grad(y, x, grad, create_graph=True)[0]
    probe = torch.randn_like(x)
    assert_bits(torch.autograd.grad(dx, grad, probe)[0], pack(probe, 2, 3, 6, 10))
    if device == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = unpack(pack(x, 2, 3, 6, 10), 48, 80, 8)
        torch.cuda.current_stream().wait_stream(stream)
        assert_bits(result, x.unsqueeze(2))


# Check invalid input rejection and equivalent positional and keyword arguments.
def test_validation(ops):
    pack, unpack, device = ops
    x = torch.empty(1, 3, 6, 10, device=device)
    for args in ((1, 3, 5, 10), (1, 4, 6, 10), (1, 3, 0, 10)):
        with pytest.raises(ValueError):
            pack(x, *args)
    with pytest.raises(ValueError, match="contiguous"):
        pack(x.transpose(-1, -2), 1, 3, 10, 6)
    with pytest.raises(TypeError):
        pack(x.to(torch.float64), 1, 3, 6, 10)
    y = pack(x, 1, 3, 6, 10)
    assert_bits(pack(x, batch_size=1, num_channels_latents=3, height=6, width=10), y)
    assert_bits(unpack(y, height=48, width=80, vae_scale_factor=8), x.unsqueeze(2))
    for args in ((48, 80, 0), (64, 80, 8), (1, 1, 8)):
        with pytest.raises(ValueError):
            unpack(y, *args)


# Check backend fallback order, cached instance reuse, and failure when no backend works.
def test_registry_fallback(monkeypatch):
    registry = KernelRegistry()
    original = registry._get_or_create_backend
    attempted = []

    def native_only(backend):
        attempted.append(backend)
        return original(backend) if backend.name.startswith("PYTORCH") else None

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(registry, "_get_or_create_backend", native_only)
    for direction, op_class in (("pack", NativeLatentPackOp), ("unpack", NativeLatentUnpackOp)):
        attempted.clear()
        op = registry.get_op(f"latent_{direction}", "cuda")
        assert isinstance(op, op_class)
        assert attempted == [
            OpBackend[f"{backend}_LATENT_{direction.upper()}"]
            for backend in ("CUDA", "TRITON", "PYTORCH")
        ]
        assert registry.get_op(f"latent_{direction}", "cpu") is op

    monkeypatch.setattr(registry, "_get_or_create_backend", lambda backend: None)
    for direction in ("pack", "unpack"):
        with pytest.raises(RuntimeError, match="No functional backend"):
            registry.get_op(f"latent_{direction}", "cuda")


# Check that CUDA initialization fails when the extension lacks the latent kernel.
def test_cuda_missing_symbol_gate(monkeypatch):
    from rl_engine.kernels.ops.cuda.packing.latent_pack_unpack import CudaLatentPackOp

    monkeypatch.setattr(base, "_C", object())
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="Rebuild"):
        CudaLatentPackOp()


# Check that the CUDA binding rejects invalid devices, dtypes, shapes, and dimensions.
@pytest.mark.cuda_only
def test_native_binding_validation():
    if not torch.cuda.is_available() or not hasattr(base._C, "latent_pack_unpack"):
        pytest.skip("CUDA latent extension unavailable")
    x = torch.empty(1, 3, 6, 10, device="cuda")
    for invalid in (x.cpu(), x.double(), x.flatten(), x.transpose(-1, -2)):
        with pytest.raises(RuntimeError):
            base._C.latent_pack_unpack(invalid, 1, 3, 6, 10, False)
    for dims in ((-1, 3, 6, 10), (1, 4, 6, 10), (1, 3, 5, 10), (1, 2**62, 6, 10)):
        with pytest.raises(RuntimeError):
            base._C.latent_pack_unpack(x, *dims, False)

