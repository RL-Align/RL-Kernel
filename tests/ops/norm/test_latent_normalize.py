# SPDX-License-Identifier: Apache-2.0
"""CPU semantics plus real CUDA accuracy, raw-byte invariance and rejection tests."""

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.ops.norm.latent_normalize import validate_latents
from rl_engine.reference.norm.latent_normalize import (
    NativeLatentDenormalizeOp,
    NativeLatentNormalizeOp,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA GPU"
)
SHAPES = [(2, 16, 3, 7), (1, 16, 1, 128, 128), (1, 16, 1, 166, 166), (1, 16, 1, 208, 116)]


def inputs(shape, dtype):
    generator = torch.Generator().manual_seed(386)
    x = torch.randn(shape, generator=generator).to(dtype)
    mean = torch.linspace(-1.1, 1.7, 16)
    std_inv = torch.linspace(0.31, 0.89, 16)
    grad = torch.randn(shape, generator=generator).to(dtype)
    return x, mean, std_inv, grad


def reference(inverse):
    return NativeLatentDenormalizeOp() if inverse else NativeLatentNormalizeOp()


def candidate(inverse, block_size=256):
    from rl_engine.backends.cuda.norm.latent_normalize import (
        LatentDenormalizeCudaOp,
        LatentNormalizeCudaOp,
    )

    return (LatentDenormalizeCudaOp if inverse else LatentNormalizeCudaOp)(block_size)


def assert_bytes(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(
        a.detach().cpu().contiguous().view(torch.uint8),
        b.detach().cpu().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_cpu_channel_semantics(inverse):
    x = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1).requires_grad_()
    mean = torch.arange(16, dtype=torch.float32) + 1
    scale = torch.full((16,), 2.0)
    y = reference(inverse)(x, mean, scale)
    expected = (x.detach() / 2 + mean.reshape_as(x)) if inverse else torch.full_like(x, -2)
    assert_bytes(y, expected)
    y.sum().backward()
    assert_bytes(x.grad, torch.full_like(x, 0.5 if inverse else 2.0))


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "frame",
        "empty",
        "layout",
        "dtype",
        "stat_shape",
        "stat_dtype",
        "stat_grad",
        "stat_device",
    ],
)
def test_contract_rejects_unsupported(case):
    x, mean, scale, _ = inputs((1, 16, 3, 7), torch.float32)
    if case == "shape":
        x = x[:, :8]
    elif case == "frame":
        x = torch.empty(1, 16, 2, 3, 7)
    elif case == "empty":
        x = x[:0]
    elif case == "layout":
        x = x.transpose(-1, -2)
    elif case == "dtype":
        x = x.half()
    elif case == "stat_shape":
        mean = mean[:8]
    elif case == "stat_dtype":
        scale = scale.bfloat16()
    elif case == "stat_grad":
        mean.requires_grad_()
    elif case == "stat_device":
        mean = mean.to("meta")
    with pytest.raises((ValueError, TypeError)):
        validate_latents(x, mean, scale)


@CUDA
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_cuda_cpu_reference_forward_backward(shape, dtype, inverse):
    x, mean, scale, grad = inputs(shape, dtype)
    cpu_x = x.float().requires_grad_()
    gold = reference(inverse).forward_fp32(cpu_x, mean, scale)
    gold_grad = torch.autograd.grad(gold, cpu_x, grad.float())[0]
    gpu_x = x.cuda().requires_grad_()
    op = candidate(inverse)
    out = op(gpu_x, mean.cuda(), scale.cuda())
    dx = torch.autograd.grad(out, gpu_x, grad.cuda())[0]
    for actual, expected, judgment in (
        (out, gold, "forward_accuracy"),
        (dx, gold_grad, "gradient_accuracy"),
    ):
        tol = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="elementwise", dtype=dtype
        )
        torch.testing.assert_close(actual.cpu().float(), expected, atol=tol.atol, rtol=tol.rtol)
        # Stronger arithmetic-contract check, including the declared output cast.
        assert_bytes(actual, expected.to(dtype))
    assert op.last_trace["actual_backend"] == "cuda"
    assert op.last_trace["fallback"] is False


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_batch_position_launch_and_replay_invariance(dtype, inverse):
    x, mean, scale, grad = inputs((1, 16, 5, 11), dtype)
    x, mean, scale, grad = (v.cuda() for v in (x, mean, scale, grad))

    def run(value, upstream, block_size):
        value = value.detach().requires_grad_()
        result = candidate(inverse, block_size)(value, mean, scale)
        return result.detach(), torch.autograd.grad(result, value, upstream)[0]

    gold = run(x, grad, 256)
    for block in (128, 256, 512):
        for batch, position in ((1, 0), (3, 0), (3, 2), (7, 4)):
            values = torch.full((batch, 16, 5, 11), 0.37, device="cuda", dtype=dtype)
            upstream = torch.full_like(values, -0.73)
            values[position : position + 1] = x
            upstream[position : position + 1] = grad
            for observed, expected in zip(run(values, upstream, block), gold, strict=True):
                assert_bytes(observed[position : position + 1], expected)
    for observed, expected in zip(run(x, grad, 256), gold, strict=True):
        assert_bytes(observed, expected)
    with torch.inference_mode():
        assert_bytes(candidate(inverse)(x, mean, scale), gold[0])
    # Equivalent 4D and single-frame 5D representations.
    assert_bytes(candidate(inverse)(x.unsqueeze(2), mean, scale).squeeze(2), gold[0])


@CUDA
@pytest.mark.parametrize("inverse", [False, True])
def test_nondefault_stream_and_expanded_gradient(inverse):
    source = inputs((1, 16, 3, 7), torch.bfloat16)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, mean, scale = (v.cuda() for v in source[:3])
        x.requires_grad_()
        out = candidate(inverse)(x, mean, scale)
        out.sum().backward()
    stream.synchronize()
    cpu_x = source[0].float().requires_grad_()
    gold = reference(inverse).forward_fp32(cpu_x, *source[1:3])
    gold.sum().backward()
    assert_bytes(out, gold.bfloat16())
    assert_bytes(x.grad, cpu_x.grad.bfloat16())


@CUDA
@pytest.mark.parametrize("inverse", [False, True])
def test_fp32_rounding_edges(inverse):
    # Tiny values detect FTZ; close values detect reassociation/FMA changes.
    values = torch.tensor([0.0, -0.0, 1e-38, -1e-38, 1.0000001, -1.0000001, 12345.67])
    x = values.repeat(16).reshape(1, 16, 1, 7)
    mean = torch.zeros(16)
    mean[8:] = 1.0000002
    scale = torch.linspace(0.5, 1.5, 16)
    assert_bytes(
        candidate(inverse)(x.cuda(), mean.cuda(), scale.cuda()), reference(inverse)(x, mean, scale)
    )


@CUDA
def test_native_boundary_rejects_bad_metadata():
    from rl_engine.backends.extension import _C

    x, mean, scale, _ = inputs((1, 16, 3, 7), torch.float32)
    x, mean, scale = (v.cuda() for v in (x, mean, scale))
    for bad_x, bad_mean, bad_scale, block in (
        (x.transpose(-1, -2), mean, scale, 256),
        (x.half(), mean, scale, 256),
        (x, mean[:8], scale, 256),
        (x, mean, scale.cpu(), 256),
        (x, mean, scale, 32),
    ):
        with pytest.raises(RuntimeError):
            _C.latent_normalize_forward(bad_x, bad_mean, bad_scale, False, block)


def test_missing_extension_fails_closed(monkeypatch):
    import rl_engine.backends.cuda.norm.latent_normalize as backend

    monkeypatch.setattr(backend, "_C", None)
    with pytest.raises(RuntimeError, match="Rebuild"):
        backend.LatentNormalizeCudaOp()


def test_registry_cpu_and_unsupported_platform():
    from rl_engine.runtime.registry import KernelRegistry

    registry = KernelRegistry()
    assert isinstance(registry.get_op("latent_normalize", "cpu"), NativeLatentNormalizeOp)
    with pytest.raises(RuntimeError, match="unsupported"):
        registry.get_op("latent_normalize", "mps")


@CUDA
def test_registry_cuda_has_no_reference_fallback(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    op = registry.get_op("latent_normalize", "cuda")
    assert type(op).__name__ == "LatentNormalizeCudaOp"
    registry._instance_cache.pop(OpBackend.CUDA_LATENT_NORMALIZE.name, None)
    monkeypatch.setattr(registry, "_get_or_create_backend", lambda backend: None)
    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("latent_normalize", "cuda")
