# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Separate H3 forward parity, the FP32 backward contract, and independent accuracy.

Both supported dtypes (BF16 and FP32) also cover layout/batch/sequence invariance.
FP16 is rejected rather than exempted from the project's accuracy thresholds.
"""

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.minimax_h3.block import golden_apply_rotary
from rl_engine.reference.rope.h3_mm_rope import NativeH3MMRoPEOp
from rl_engine.validation.models.h3_cases import H3_BLOCK_LAYOUTS, h3_fl2va_layout
from rl_engine.validation.models.h3_provider import provider_apply_rotary, provider_rope

DTYPES = (torch.bfloat16, torch.float32)
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm")


@pytest.fixture(params=("native", "triton"))
def implementation(request):
    if request.param == "native":
        return NativeH3MMRoPEOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA or ROCm")
    from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

    return TritonH3MMRoPEOp(), "cuda"


@pytest.fixture(scope="module")
def positions():
    return h3_fl2va_layout(**H3_BLOCK_LAYOUTS["tiny"], device="cpu")["position_ids"]


def _assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().view(torch.uint8),
        expected.detach().contiguous().view(torch.uint8),
    )


def _strided(values):
    """Preserve values while giving every dimension a noncontiguous element stride."""
    storage = torch.empty(
        (*values.shape[:-1], values.shape[-1] * 2), dtype=values.dtype, device=values.device
    )
    storage[..., ::2] = values
    return storage[..., ::2]


def _forward_backward(op, x, cos, sin, upstream):
    x = x.detach().requires_grad_(True)
    y = op(x, cos, sin)
    return y.detach(), torch.autograd.grad(y, x, upstream)[0]


def _gradient_reference(x, cos, sin, upstream):
    """Differentiate the FP32 rotation with forward-quantized coefficients.

    Autograd derives the transpose independently of the operators' handwritten
    backward. The separate accuracy test keeps the original FP32 coefficients.
    """
    _, gradient = _forward_backward(
        golden_apply_rotary,
        x.float(),
        cos.to(x.dtype).float(),
        sin.to(x.dtype).float(),
        upstream.float(),
    )
    return gradient.to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length,heads", [(17, 56), (264, 14)])
@pytest.mark.parametrize("strided", [False, True])
def test_forward_provider_parity_and_backward_contract(
    implementation, positions, dtype, length, heads, strided
):
    """Use full-model and head-sharded geometry with the pinned model's FP32 tables."""
    op, device = implementation
    generator = torch.Generator().manual_seed(420)
    x = torch.randn((2, length, heads, 128), generator=generator).to(device=device, dtype=dtype)
    upstream = torch.randn(x.shape, generator=generator).to(device=device, dtype=dtype)
    cos, sin = provider_rope(positions[:length].to(device))
    if strided:
        x, upstream, cos, sin = map(_strided, (x, upstream, cos, sin))
    actual, grad = _forward_backward(op, x, cos, sin, upstream)
    _assert_bits(actual, provider_apply_rotary(x, cos, sin))
    _assert_bits(grad, _gradient_reference(x, cos, sin, upstream))
    _assert_bits(actual[..., 96:], x[..., 96:])
    _assert_bits(grad[..., 96:], upstream[..., 96:])


def test_backward_rounds_once_using_forward_coefficients(implementation):
    """Pin both decisions: keep coefficient quantization, remove product quantization.

    Distinct table halves also catch using the wrong output's sine in backward.
    This is an arithmetic regression; model tables are covered separately above.
    """
    op, device = implementation
    phases = torch.linspace(-3, 3, 96).reshape(1, 96)
    cos, sin = phases.cos().to(device), phases.sin().to(device)
    x = torch.ones((1, 1, 1, 128), device=device, dtype=torch.bfloat16)
    upstream = torch.full_like(x, 0.5)
    upstream[..., 48:96] = -0.7
    _, grad = _forward_backward(op, x, cos, sin, upstream)
    expected_grad = _gradient_reference(x, cos, sin, upstream)
    _assert_bits(grad, expected_grad)

    _, eager_grad = _forward_backward(provider_apply_rotary, x, cos, sin, upstream)
    _, unquantized_grad = _forward_backward(
        golden_apply_rotary, x.float(), cos, sin, upstream.float()
    )
    assert not torch.equal(expected_grad, eager_grad)
    assert not torch.equal(expected_grad, unquantized_grad.to(x.dtype))


def test_rejects_fp16(implementation):
    op, device = implementation
    x = torch.ones((1, 1, 1, 128), device=device, dtype=torch.float16)
    cos = torch.ones((1, 96), device=device, dtype=torch.float32)
    with pytest.raises(TypeError, match="x must have dtype in"):
        op(x, cos, torch.zeros_like(cos))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [420, 0, 1])
def test_accuracy_against_independent_fp32(implementation, positions, dtype, seed):
    """Accuracy is checked independently of the eager table/product rounding."""
    op, device = implementation
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((2, len(positions), 3, 128), generator=generator).to(device=device, dtype=dtype)
    upstream = torch.randn(x.shape, generator=generator).to(device=device, dtype=dtype)
    cos, sin = provider_rope(positions.to(device))
    actual, grad = _forward_backward(op, x, cos, sin, upstream)
    expected, expected_grad = _forward_backward(
        golden_apply_rotary, x.float(), cos, sin, upstream.float()
    )
    contract = load_contract()
    for judgment, result, reference in (
        ("forward_accuracy", actual, expected),
        ("gradient_accuracy", grad, expected_grad),
    ):
        tolerance = resolve_tolerance(
            contract, judgment=judgment, op_class="elementwise", dtype=dtype
        )
        torch.testing.assert_close(
            result.float(), reference, atol=tolerance.atol, rtol=tolerance.rtol
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("broadcast_gradient", [False, True])
def test_invariance_across_runtime_layouts(implementation, positions, dtype, broadcast_gradient):
    """The same logical rows retain their bits across sequence lengths and batch views."""
    op, device = implementation
    generator = torch.Generator().manual_seed(656)
    x = torch.randn((2, len(positions), 14, 128), generator=generator).to(
        device=device, dtype=dtype
    )
    cos, sin = provider_rope(positions.to(device))
    if broadcast_gradient:
        # A summed loss can supply a broadcast gradient with zero strides.
        upstream = torch.ones((), device=device, dtype=dtype).expand_as(x)
    else:
        upstream = torch.randn(x.shape, generator=generator).to(device=device, dtype=dtype)
    expected, expected_grad = _forward_backward(op, x, cos, sin, upstream)
    repeated, repeated_grad = _forward_backward(op, x, cos, sin, upstream)
    _assert_bits(repeated, expected)
    _assert_bits(repeated_grad, expected_grad)
    with torch.no_grad():
        _assert_bits(op(x, cos, sin), expected)
    with torch.inference_mode():
        _assert_bits(op(x, cos, sin), expected)
    for length in (17, len(positions)):
        for batch in range(x.shape[0]):
            view = _strided(x[batch : batch + 1, :length])
            actual, grad = _forward_backward(
                op,
                view,
                _strided(cos[:length]),
                _strided(sin[:length]),
                upstream[batch : batch + 1, :length],
            )
            _assert_bits(actual, expected[batch : batch + 1, :length])
            _assert_bits(grad, expected_grad[batch : batch + 1, :length])


@needs_gpu
@pytest.mark.parametrize("length", [65535, 65536, 73426])
def test_triton_long_h3_sequences(length):
    """Cover CUDA's former grid-y boundary and a 243-frame, 768x1344 H3 layout.

    The full layout has 40 text + 810 audio + 72,576 video rows. A single head
    keeps this launch-regression test small; full-head geometry is covered above.
    """
    from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

    packed = h3_fl2va_layout(
        num_text_tokens=40,
        num_latent_frames=72,
        latent_height=48,
        latent_width=84,
        num_audio_latents=405,
        device="cpu",
    )
    assert packed["position_ids"].shape == (73426, 3)
    cos, sin = provider_rope(packed["position_ids"][:length].cuda())
    x = torch.randn((1, length, 1, 128), generator=torch.Generator().manual_seed(420))
    x = x.to(device="cuda", dtype=torch.bfloat16)
    upstream = torch.ones((), device="cuda", dtype=x.dtype).expand_as(x)
    actual, grad = _forward_backward(TritonH3MMRoPEOp(), x, cos, sin, upstream)
    _assert_bits(actual, provider_apply_rotary(x, cos, sin))
    _assert_bits(grad, _gradient_reference(x, cos, sin, upstream))
