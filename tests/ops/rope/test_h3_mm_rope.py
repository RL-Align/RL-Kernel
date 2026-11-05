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
from rl_engine.reference.rope.h3_mm_rope import NativeH3MMRoPEOp, model_boundary_reference
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


@pytest.mark.parametrize(
    "invalid,exception,message",
    [
        ("rank", ValueError, "x must have shape"),
        ("channels", ValueError, "x must have shape"),
        ("empty", ValueError, "nonempty"),
        ("cos_shape", ValueError, "cos must have shape"),
        ("sin_dtype", TypeError, "sin must have dtype"),
        ("cos_device", ValueError, "cos must be on x.device"),
        ("sin_trainable", ValueError, "sin must not require gradients"),
    ],
)
def test_input_contract_rejects_invalid_metadata(implementation, invalid, exception, message):
    op, device = implementation
    x = torch.ones(1, 2, 1, 128, device=device, dtype=torch.bfloat16)
    cos = torch.ones(2, 96, device=device)
    sin = torch.zeros_like(cos)
    if invalid == "rank":
        x = x.squeeze(0)
    elif invalid == "channels":
        x = x[..., :64]
    elif invalid == "empty":
        x = x[:, :0]
    elif invalid == "cos_shape":
        cos = cos[:, :48]
    elif invalid == "sin_dtype":
        sin = sin.bfloat16()
    elif invalid == "cos_device":
        cos = cos.to("meta")
    elif invalid == "sin_trainable":
        sin.requires_grad_(True)
    with pytest.raises(exception, match=message):
        op(x, cos, sin)


@needs_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_nondefault_stream_and_graph_replay(dtype, positions):
    """Keep forward and backward on the caller's stream, including graph replay."""
    from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cos, sin = provider_rope(positions[:17].cuda())
        x = torch.randn(1, 17, 14, 128, device="cuda", dtype=dtype, requires_grad=True)
        upstream = torch.randn_like(x)
        op = TritonH3MMRoPEOp()
        for _ in range(3):
            output = op(x, cos, sin)
            torch.autograd.grad(output, x, upstream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = op(x, cos, sin)
            gradient = torch.autograd.grad(output, x, upstream)[0]
        for scale in (1.0, -0.5):
            with torch.no_grad():
                x.mul_(scale)
                upstream.mul_(scale)
            graph.replay()
            expected, expected_grad = _forward_backward(NativeH3MMRoPEOp(), x, cos, sin, upstream)
            _assert_bits(output, expected)
            _assert_bits(gradient, expected_grad)
    stream.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_uses_input_device_and_restores_current_device(positions, dtype):
    from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

    with torch.cuda.device(0):
        cos, sin = provider_rope(positions[:17].to("cuda:1"))
        x = torch.randn(1, 17, 14, 128, device="cuda:1", dtype=dtype)
        upstream = torch.randn_like(x)
        actual, grad = _forward_backward(TritonH3MMRoPEOp(), x, cos, sin, upstream)
        assert torch.cuda.current_device() == 0
        expected, expected_grad = _forward_backward(NativeH3MMRoPEOp(), x, cos, sin, upstream)
        _assert_bits(actual, expected)
        _assert_bits(grad, expected_grad)


def test_registry_exposes_h3_rope_with_reference_fallback():
    from rl_engine.runtime.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    name = "h3_mm_rope_3axis_partial"
    assert registry._priority_map["cuda"][name] == [
        OpBackend.TRITON_H3_MM_ROPE,
        OpBackend.PYTORCH_H3_MM_ROPE,
    ]
    for device in ("cpu", "rocm", "musa", "npu"):
        assert registry._priority_map[device][name] == [OpBackend.PYTORCH_H3_MM_ROPE]
    assert isinstance(registry.get_op(name, device="cpu"), NativeH3MMRoPEOp)
    if torch.cuda.is_available() and torch.version.hip is None:
        from rl_engine.backends.shared.triton.rope.h3_mm_rope import TritonH3MMRoPEOp

        assert isinstance(registry.get_op(name, device="cuda"), TritonH3MMRoPEOp)


def test_semantic_entry_reports_arithmetic_and_rejects_unsupported_profiles():
    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.runtime.semantic_registry import OperatorRequirements, OperatorResolutionError

    session = KernelRegistry().semantic.session()
    request = dict(
        semantic_op="h3_mm_rope_3axis_partial",
        requested_backend="rlkernel.h3_mm_rope.triton",
        target="training",
    )
    resolution = session.resolve(
        **request,
        requirements=OperatorRequirements(
            device="cuda",
            dtype="bfloat16",
            topology={"world_size": 1, "tensor_parallel_size": 1, "context_parallel_size": 1},
        ),
    )
    assert resolution.trace.concrete_backend == "rlkernel.h3_mm_rope.triton"
    assert resolution.descriptor.determinism_or_alignment_properties["fp_fusion"] is False
    for requirement in (
        OperatorRequirements(device="cuda", dtype="float16"),
        OperatorRequirements(device="rocm", dtype="bfloat16"),
        OperatorRequirements(device="cuda", dtype="bfloat16", topology={"world_size": 2}),
    ):
        with pytest.raises(OperatorResolutionError):
            session.resolve(**request, requirements=requirement)
    if torch.cuda.is_available() and torch.version.hip is None:
        op = session.instantiate(resolution)
        provenance = session.instance_provenance(resolution, op)
        assert provenance.concrete_implementation.endswith("TritonH3MMRoPEOp")


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
def test_accuracy_against_model_boundary_reference(implementation, positions, dtype, seed):
    """The hard accuracy gate retains the model's declared storage boundaries."""
    op, device = implementation
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((2, len(positions), 3, 128), generator=generator).to(device=device, dtype=dtype)
    upstream = torch.randn(x.shape, generator=generator).to(device=device, dtype=dtype)
    cos, sin = provider_rope(positions.to(device))
    actual, grad = _forward_backward(op, x, cos, sin, upstream)
    expected, expected_grad = _forward_backward(
        lambda x, c, s: model_boundary_reference(x, c, s, storage_dtype=dtype),
        x.float(),
        cos,
        sin,
        upstream.float(),
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


@pytest.fixture
def captured_rounding_case(implementation):
    """Two BF16 cancellation pairs saved from the long-layout A6000 benchmark.

    Original [token, head, channel] coordinates are [36201, 13, 38] and
    [48538, 3, 11] in [1, 73426, 14, 128], seed 656. Each pair is independent,
    so this exact-value fixture needs neither the full allocation nor a GPU RNG.
    """
    _, device = implementation
    x = torch.zeros((1, 2, 1, 128), dtype=torch.bfloat16, device=device)
    x[0, :, 0, 0] = torch.tensor([3.296875, 3.046875], device=device)
    x[0, :, 0, 48] = torch.tensor([3.609375, 4.5], device=device)
    cos = torch.ones((2, 96), device=device)
    sin = torch.zeros_like(cos)
    for channel in (0, 48):
        cos[:, channel] = torch.tensor([0.7241835594177246, 0.8580111265182495], device=device)
        sin[:, channel] = torch.tensor([0.6896072626113892, 0.5136311054229736], device=device)
    upstream = torch.zeros_like(x)
    upstream[0, :, 0, 0] = torch.tensor([0.045166015625, -1.3203125], device=device)
    upstream[0, :, 0, 48] = torch.tensor([0.298828125, -1.65625], device=device)
    return x, cos, sin, upstream


def test_captured_bf16_pairs_preserve_model_and_backward(implementation, captured_rounding_case):
    """Captured cancellation pairs pass model parity and model-boundary acceptance."""
    op, _ = implementation
    x, cos, sin, upstream = captured_rounding_case
    actual, grad = _forward_backward(op, x, cos, sin, upstream)
    _assert_bits(actual, provider_apply_rotary(x, cos, sin))
    _assert_bits(grad, _gradient_reference(x, cos, sin, upstream))
    assert actual[0, :, 0, 0].tolist() == [-0.125, 0.328125]
    expected, expected_grad = _forward_backward(
        lambda x, c, s: model_boundary_reference(x, c, s, storage_dtype=torch.bfloat16),
        x.float(),
        cos,
        sin,
        upstream.float(),
    )
    for judgment, result, reference in (
        ("forward_accuracy", actual, expected),
        ("gradient_accuracy", grad, expected_grad),
    ):
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="elementwise", dtype=x.dtype
        )
        torch.testing.assert_close(
            result.float(), reference, atol=tolerance.atol, rtol=tolerance.rtol
        )
    _assert_bits(expected.to(x.dtype), actual)
    _assert_bits(expected_grad.to(x.dtype), grad)


def test_captured_bf16_pairs_unrounded_math_diagnostic(implementation, captured_rounding_case):
    """Pin the unresolved mathematical gap; this diagnostic is not acceptance."""
    op, _ = implementation
    x, cos, sin, _ = captured_rounding_case
    actual = op(x, cos, sin)
    reference = NativeH3MMRoPEOp().forward_fp32(x, cos, sin)
    tolerance = resolve_tolerance(
        load_contract(), judgment="forward_accuracy", op_class="elementwise", dtype=x.dtype
    )
    torch.testing.assert_close(reference, golden_apply_rotary(x.float(), cos, sin), atol=0, rtol=0)
    assert reference[0, :, 0, 0].tolist() == [-0.10150837898254395, 0.30291271209716797]
    error = (actual.float() - reference).abs()
    limit = tolerance.atol + tolerance.rtol * reference.abs()
    assert (error > limit).sum().item() == 2
    assert (error > limit)[0, :, 0, 0].all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_references_have_independent_forward_and_vjp(dtype):
    """A nonduplicated table catches transposed sine mistakes in the reference VJP.

    For BF16 operands these FP64 products are exact, providing a separate oracle;
    the production reference still computes in FP32 before each model store.
    """
    generator = torch.Generator().manual_seed(656420)
    x = torch.randn(1, 3, 2, 128, generator=generator).to(dtype).float().requires_grad_()
    c, s = (torch.randn(3, 96, generator=generator) for _ in range(2))
    dy = torch.randn(x.shape, generator=generator)
    y = model_boundary_reference(x, c, s, storage_dtype=dtype)
    dx = torch.autograd.grad(y, x, dy)[0]
    cq, sq = (v.to(dtype).float()[None, :, None, :] for v in (c, s))
    rotated = torch.cat((-x[..., 48:96], x[..., :48]), dim=-1)
    if dtype == torch.bfloat16:
        direct = (x[..., :96].double() * cq.double()).to(dtype).float()
        cross = (rotated.double() * sq.double()).to(dtype).float()
        _assert_bits(y[..., :96], direct + cross)
    else:
        _assert_bits(y, golden_apply_rotary(x, c, s))
    d0 = dy[..., :48] * cq[..., :48] + dy[..., 48:96] * sq[..., 48:96]
    d1 = dy[..., 48:96] * cq[..., 48:96] - dy[..., :48] * sq[..., :48]
    _assert_bits(dx, torch.cat((d0, d1, dy[..., 96:]), dim=-1))
    if dtype == torch.bfloat16:
        assert not torch.equal(dx, dx.bfloat16().float())
        # A plain .to(BF16).float() chain must not replace the identity storage VJP.
        ordinary = (x[..., :96] * cq).bfloat16().float()
        ordinary = ordinary + (rotated * sq).bfloat16().float()
        ordinary_dx = torch.autograd.grad(ordinary, x, dy[..., :96])[0]
        assert not torch.equal(dx[..., :96], ordinary_dx[..., :96])


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_model_boundaries_on_broader_cancellation_samples(scale):
    """Exercise normal and cancellation inputs without restricting the activation scale."""
    generator = torch.Generator().manual_seed(656420)
    x = (torch.randn(1, 4096, 1, 128, generator=generator) * scale).bfloat16()
    phase = torch.rand(4096, 48, generator=generator) * 6 - 3
    c, s = phase.cos().repeat(1, 2), phase.sin().repeat(1, 2)
    # Half normal, half correlated near-cancellation samples. Avoid division near zero.
    x[:, 2048:, :, :48] = (phase[2048:].sin() * scale)[None, :, None, :]
    x[:, 2048:, :, 48:96] = (phase[2048:].cos() * scale)[None, :, None, :]
    reference = model_boundary_reference(x, c, s, storage_dtype=x.dtype)
    _assert_bits(reference.bfloat16(), provider_apply_rotary(x, c, s))
    mathematical = NativeH3MMRoPEOp().forward_fp32(x, c, s)
    _assert_bits(mathematical, golden_apply_rotary(x.float(), c, s))


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


@pytest.mark.parametrize("dtype", DTYPES)
def test_packed_permutation_head_shards_and_checkpoint(implementation, positions, dtype):
    """Logical rows keep their bytes when their positions move with them."""
    from torch.utils.checkpoint import checkpoint

    op, device = implementation
    generator = torch.Generator().manual_seed(656)
    x = torch.randn(2, 17, 14, 128, generator=generator).to(device=device, dtype=dtype)
    upstream = torch.randn(x.shape, generator=generator).to(device=device, dtype=dtype)
    cos, sin = provider_rope(positions[:17].to(device))
    expected, expected_grad = _forward_backward(op, x, cos, sin, upstream)
    permutation = torch.randperm(17, generator=generator).to(device)
    actual, grad = _forward_backward(
        op, x[:, permutation], cos[permutation], sin[permutation], upstream[:, permutation]
    )
    _assert_bits(actual, expected[:, permutation])
    _assert_bits(grad, expected_grad[:, permutation])
    actual, grad = _forward_backward(op, x[:, :, :7], cos, sin, upstream[:, :, :7])
    _assert_bits(actual, expected[:, :, :7])
    _assert_bits(grad, expected_grad[:, :, :7])
    changed = x.clone()
    changed[1].mul_(2)
    actual, grad = _forward_backward(op, changed, cos, sin, upstream)
    _assert_bits(actual[0], expected[0])
    _assert_bits(grad[0], expected_grad[0])
    leaf = x.detach().requires_grad_(True)
    output = checkpoint(op, leaf, cos, sin, use_reentrant=False)
    gradient = torch.autograd.grad(output, leaf, upstream)[0]
    _assert_bits(output, expected)
    _assert_bits(gradient, expected_grad)
