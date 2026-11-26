# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 1x2x2 order, raw bits, inverse, gradients, and batch invariance."""

import argparse

import pytest
import torch

from rl_engine.reference.packing.video_patch_pack_unpack import (
    NativeVideoPatchPackUnpackOp,
    pack_video_patch_reference,
    unpack_video_patch_reference,
)
from rl_engine.runtime.registry import KernelRegistry
from rl_engine.runtime.semantic_registry import OperatorRequirements, OperatorResolutionError
from rl_engine.validation.operators.op_checks import run_operator_suite
from rl_engine.validation.operators.operator_inputs import make_operator_inputs
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SHAPES = ((1, 24, 1, 2, 2), (2, 24, 3, 6, 10), (1, 24, 2, 48, 84))


def _gtest_args(direction, candidate, **overrides):
    values = dict(
        op="video_patch_pack_unpack",
        candidate=candidate,
        arch_key=None,
        batch=2,
        frames=3,
        height=6,
        width=10,
        patch_direction=direction,
        seed=420,
        input_mode="random",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _check_gtest(args, dtype, device):
    case = make_operator_case(args, dtype, torch.device(device))
    candidate = make_candidate(args)
    report = run_operator_suite(
        args.op,
        cases=[case],
        candidates=[candidate],
        check_grad=True,
        grad_mode="random",
        grad_seed=421,
    )
    assert report.passed
    assert case.grad_input_names == ("x",)
    assert args.patch_direction in case.name
    outputs = report.candidates[0].cases[0].outputs
    assert len(outputs) == 2  # One forward output and one input gradient.
    assert all(output.max_abs_error == 0 for output in outputs)
    assert_bits(candidate.fn(**case.inputs), case.gold_fn(**case.inputs).to(dtype))


@pytest.mark.parametrize("direction", ("pack", "unpack"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_public_gtest_reused_forward_and_backward(direction, dtype):
    _check_gtest(_gtest_args(direction, "native"), dtype, "cpu")


@pytest.mark.parametrize("direction", ("pack", "unpack"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_public_gtest_gpu_forward_and_backward(gpu_op, direction, dtype):
    backend = gpu_op.backend_id.split("-", 1)[0]
    _check_gtest(_gtest_args(direction, backend), dtype, "cuda")


@pytest.mark.parametrize(
    "overrides",
    ({"batch": 0}, {"frames": 0}, {"height": 3}, {"width": 5}, {"patch_direction": "other"}),
)
def test_public_gtest_rejects_invalid_input_geometry(overrides):
    args = _gtest_args("pack", "pytorch", **overrides)
    with pytest.raises(ValueError):
        make_operator_inputs(args.op, args, torch.float32, torch.device("cpu"))


def test_public_gtest_cli_parses_video_geometry(monkeypatch):
    from tools.validation.operators.check_operator import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "check_operator.py",
            "--op",
            "video_patch_pack_unpack",
            "--patch-direction",
            "unpack",
            "--batch",
            "2",
            "--frames",
            "3",
            "--height",
            "6",
            "--width",
            "10",
            "--check-grad",
        ],
    )
    args = parse_args()
    inputs = make_operator_inputs(args.op, args, torch.float32, torch.device("cpu"))
    assert inputs["shape"] == (2, 24, 3, 6, 10)
    assert inputs["x"].shape == (2, 45, 96)
    assert inputs["unpack"] and args.check_grad


def assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare all bytes, including signed zero and NaN payload bits."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().cpu().view(torch.uint8),
        expected.detach().contiguous().cpu().view(torch.uint8),
    )


def test_coordinate_order_across_frames():
    shape = (1, 24, 2, 2, 4)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    packed = NativeVideoPatchPackUnpackOp().pack(x)
    assert packed.shape == (1, 4, 96)

    # Check against source coordinates, not another pack/unpack round trip.
    for frame in range(2):
        for patch_col in range(2):
            token_row = frame * 2 + patch_col
            for channel in range(24):
                for dy in range(2):
                    for dx in range(2):
                        feature = channel * 4 + dy * 2 + dx
                        assert (
                            packed[0, token_row, feature]
                            == x[0, channel, frame, dy, patch_col * 2 + dx]
                        )

    # An independently constructed token input checks unpack's direction.
    tokens = torch.arange(4 * 96, dtype=torch.float32).reshape(1, 4, 96)
    unpacked = NativeVideoPatchPackUnpackOp().unpack(tokens, shape)
    for frame in range(2):
        for patch_col in range(2):
            for channel in range(24):
                for dy in range(2):
                    for dx in range(2):
                        assert (
                            unpacked[0, channel, frame, dy, patch_col * 2 + dx]
                            == tokens[0, frame * 2 + patch_col, channel * 4 + dy * 2 + dx]
                        )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_reused_roundtrip_and_both_gradients(shape, dtype):
    generator = torch.Generator().manual_seed(420)
    x = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    packed = NativeVideoPatchPackUnpackOp().pack(x)
    assert_bits(unpack_video_patch_reference(packed, shape), x)

    grad_tokens = torch.randn(packed.shape, generator=generator).to(dtype)
    dx = torch.autograd.grad(packed, x, grad_tokens)[0]
    assert_bits(dx, unpack_video_patch_reference(grad_tokens, shape))

    tokens = torch.randn(packed.shape, generator=generator).to(dtype).requires_grad_()
    unpacked = NativeVideoPatchPackUnpackOp().unpack(tokens, shape)
    grad_latents = torch.randn(shape, generator=generator).to(dtype)
    dtokens = torch.autograd.grad(unpacked, tokens, grad_latents)[0]
    assert_bits(dtokens, pack_video_patch_reference(grad_latents))


def test_reused_batch_invariance():
    op = NativeVideoPatchPackUnpackOp()
    x = torch.arange(24 * 3 * 6 * 10, dtype=torch.float32).reshape(1, 24, 3, 6, 10)
    alone = op.pack(x)
    others = torch.full_like(x, -100)
    assert_bits(op.pack(torch.cat([x, others]))[0], alone[0])
    assert_bits(op.pack(torch.cat([others, x]))[1], alone[0])


def test_invalid_inputs():
    op = NativeVideoPatchPackUnpackOp()
    x = torch.empty((1, 24, 1, 2, 4))
    for bad in (torch.empty((0, 24, 1, 2, 4)), torch.empty((1, 24, 0, 2, 4))):
        with pytest.raises(ValueError, match="positive"):
            op.pack(bad)
    with pytest.raises(ValueError, match="24"):
        op.pack(x[:, :23].contiguous())
    with pytest.raises(ValueError, match="divisible"):
        op.pack(torch.empty((1, 24, 1, 3, 4)))
    with pytest.raises(ValueError, match="contiguous"):
        op.pack(x.transpose(-1, -2))
    with pytest.raises(TypeError, match="dtype"):
        op.pack(x.to(torch.int32))
    with pytest.raises(ValueError, match="shape"):
        op.unpack(torch.empty((1, 3, 96)), tuple(x.shape))
    with pytest.raises(ValueError, match="shape"):
        op.unpack(torch.empty((1, 2, 96)), (1, 24, 1, 2))
    with pytest.raises(TypeError, match="integers"):
        op.unpack(torch.empty((1, 2, 96)), (1, 24, 1.0, 2, 4))
    with pytest.raises(ValueError, match="contiguous"):
        op.unpack(torch.empty((1, 96, 2)).transpose(1, 2), tuple(x.shape))


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_registry_uses_reused_pytorch(device):
    registry = KernelRegistry()
    op = registry.get_op("video_patch_pack_unpack", device=device)
    assert isinstance(op, NativeVideoPatchPackUnpackOp)
    assert op.backend_id == "pytorch-video-patch-pack-unpack-v1"
    assert registry.get_op("video_patch_pack_unpack", device=device) is op


def test_rocm_registry_uses_reused_pytorch(monkeypatch):
    registry = KernelRegistry()
    # ROCm uses torch.device("cuda"); this checks selection, not AMD execution.
    monkeypatch.setattr(torch.version, "hip", "test")
    op = registry.get_op("video_patch_pack_unpack", device="cuda")
    assert isinstance(op, NativeVideoPatchPackUnpackOp)


@pytest.mark.parametrize("direction", ("pack", "unpack"))
def test_public_gold_is_independent_of_reused_candidate(monkeypatch, direction):
    from rl_engine.reference.packing import video_patch_pack_unpack as module

    def broken_upstream(*args, **kwargs):
        raise RuntimeError("Injected upstream failure")

    monkeypatch.setattr(module, "minimax_h3_patchify_video_latent", broken_upstream)
    monkeypatch.setattr(module, "minimax_h3_unpatchify_video_tokens", broken_upstream)
    args = _gtest_args(direction, "pytorch")
    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    expected = (
        unpack_video_patch_reference(case.inputs["x"], case.inputs["shape"])
        if direction == "unpack"
        else pack_video_patch_reference(case.inputs["x"])
    )
    assert_bits(case.gold_fn(**case.inputs), expected)
    with pytest.raises(RuntimeError, match="Injected upstream failure"):
        make_candidate(args).fn(**case.inputs)


def resolve_backend(session, backend, device="cpu", dtype="float32", **overrides):
    requirements = dict(
        device=device,
        dtype=dtype,
        topology={"world_size": 1, "sharding": "unsharded"},
        alignment_properties={
            "channels": 24,
            "patch_size": {"t": 1, "h": 2, "w": 2},
            "bitwise": True,
        },
    )
    requirements.update(overrides)
    return session.resolve(
        semantic_op="video_patch_pack_unpack",
        requested_backend=f"{backend}-video-patch-pack-unpack-v1",
        target="training",
        requirements=OperatorRequirements(**requirements),
    )


def test_strict_reference_executes_registered_implementation():
    session = KernelRegistry().semantic.session()
    resolution = resolve_backend(session, "pytorch")
    op = session.instantiate(resolution)
    x = torch.arange(24 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 24, 2, 4, 6)
    assert_bits(op.pack(x), pack_video_patch_reference(x))
    assert_bits(op.unpack(op.pack(x), tuple(x.shape)), x)
    provenance = session.instance_provenance(resolution, op)
    assert provenance.backend_id == op.backend_id == resolution.trace.concrete_backend
    assert provenance.concrete_implementation.endswith(".NativeVideoPatchPackUnpackOp")
    assert provenance.implementation_fingerprint
    assert resolution.strict and resolution.trace.fallback_attempts == ()


@pytest.mark.parametrize("backend", ("cuda", "triton"))
def test_strict_retired_backend_is_not_registered(backend):
    session = KernelRegistry().semantic.session()
    with pytest.raises(OperatorResolutionError) as error:
        resolve_backend(session, backend, device="cuda")
    assert error.value.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "rocm"},
        {"dtype": "float64"},
        {"topology": {"world_size": 2, "sharding": "unsharded"}},
        {"topology": {"world_size": 1}},
        {"alignment_properties": {"channels": 16}},
        {"alignment_properties": {"patch_size": {"t": 2, "h": 2, "w": 2}}},
    ],
)
def test_strict_dispatch_rejects_unsupported_contract(changes):
    session = KernelRegistry().semantic.session()
    with pytest.raises(OperatorResolutionError) as error:
        resolve_backend(session, "pytorch", **{"device": "cuda", **changes})
    assert error.value.trace.fallback_attempts == ()


def test_strict_unknown_backend_does_not_choose_reference():
    session = KernelRegistry().semantic.session()
    with pytest.raises(OperatorResolutionError, match="no fallback"):
        resolve_backend(session, "missing")


@pytest.fixture(params=("cpu", "cuda"))
def reused_op(request):
    device = request.param
    if device == "cuda" and (not torch.cuda.is_available() or torch.version.hip is not None):
        pytest.skip("NVIDIA CUDA GPU required")
    session = KernelRegistry().semantic.session()
    resolution = resolve_backend(session, "pytorch", device=device)
    op = session.instantiate(resolution)
    assert isinstance(op, NativeVideoPatchPackUnpackOp)
    assert session.instance_provenance(resolution, op).backend_id == op.backend_id
    return op, device


@pytest.fixture
def gpu_op():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA CUDA GPU required")
    session = KernelRegistry().semantic.session()
    resolution = resolve_backend(session, "pytorch", device="cuda")
    op = session.instantiate(resolution)
    assert isinstance(op, NativeVideoPatchPackUnpackOp)
    assert session.instance_provenance(resolution, op).backend_id == op.backend_id
    return op


def test_gpu_registry_executes_reused_pytorch(gpu_op):
    op = KernelRegistry().get_op("video_patch_pack_unpack", device="cuda")
    assert type(op) is type(gpu_op)
    x = torch.randn((1, 24, 3, 6, 10), device="cuda")
    assert op.pack(x).device == x.device
    assert_bits(op.pack(x), pack_video_patch_reference(x))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gpu_reuse_matches_independent_reference(gpu_op, shape, dtype):
    generator = torch.Generator().manual_seed(420)
    cpu_x = torch.randn(shape, generator=generator).to(dtype).requires_grad_()
    gpu_x = cpu_x.detach().to("cuda").requires_grad_()
    expected = pack_video_patch_reference(cpu_x)
    actual = gpu_op.pack(gpu_x)
    assert_bits(actual, expected)
    assert_bits(gpu_op.unpack(actual, shape), cpu_x)

    # Independent unpack input prevents paired errors cancelling each other.
    cpu_tokens = torch.randn(expected.shape, generator=generator).to(dtype).requires_grad_()
    gpu_tokens = cpu_tokens.detach().to("cuda").requires_grad_()
    expected_unpacked = unpack_video_patch_reference(cpu_tokens, shape)
    actual_unpacked = gpu_op.unpack(gpu_tokens, shape)
    assert_bits(actual_unpacked, expected_unpacked)

    grad_tokens = torch.randn(expected.shape, generator=generator).to(dtype)
    grad_latents = torch.randn(shape, generator=generator).to(dtype)
    assert_bits(
        torch.autograd.grad(actual, gpu_x, grad_tokens.to("cuda"))[0],
        torch.autograd.grad(expected, cpu_x, grad_tokens)[0],
    )
    assert_bits(
        torch.autograd.grad(actual_unpacked, gpu_tokens, grad_latents.to("cuda"))[0],
        torch.autograd.grad(expected_unpacked, cpu_tokens, grad_latents)[0],
    )


@pytest.mark.parametrize("width", (30, 32, 34, 62, 64, 66))
@pytest.mark.parametrize("dtype", DTYPES)
def test_gpu_irregular_patch_columns(gpu_op, width, dtype):
    test_gpu_reuse_matches_independent_reference(gpu_op, (2, 24, 3, 6, width), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reused_preserves_special_bits(reused_op, dtype):
    op, device = reused_op
    patterns = {
        torch.float32: (
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC01234,
            0x7F801234,
            1,
        ),
        torch.float16: (0, 0x8000, 0x7C00, 0xFC00, 0x7E55, 0x7C55, 1),
        torch.bfloat16: (0, 0x8000, 0x7F80, 0xFF80, 0x7FC5, 0x7F85, 1),
    }
    shape = (2, 24, 3, 2, 4)
    count = torch.tensor(shape).prod().item()
    bits = torch.uint32 if dtype == torch.float32 else torch.uint16
    repeated = torch.tensor(patterns[dtype], dtype=bits).repeat((count + 6) // 7)[:count]
    cpu_x = repeated.view(dtype).reshape(shape)
    gpu_x = cpu_x.to(device)
    assert_bits(op.pack(gpu_x), pack_video_patch_reference(cpu_x))
    assert_bits(op.unpack(op.pack(gpu_x), shape), cpu_x)

    token_count = 2 * 6 * 96
    token_bits = torch.tensor(patterns[dtype], dtype=bits).repeat((token_count + 6) // 7)
    cpu_tokens = token_bits[:token_count].view(dtype).reshape(2, 6, 96)
    assert_bits(
        op.unpack(cpu_tokens.to(device), shape),
        unpack_video_patch_reference(cpu_tokens, shape),
    )


def test_reused_noncontiguous_grad_and_higher_order(reused_op):
    op, device = reused_op
    shape = (2, 24, 3, 6, 10)
    x = torch.randn(shape, device=device, requires_grad=True)
    packed = op.pack(x)
    grad_tokens = torch.randn(2, 96, packed.shape[1], device=device).transpose(1, 2)
    grad_tokens.requires_grad_()
    dx = torch.autograd.grad(packed, x, grad_tokens, create_graph=True)[0]
    assert_bits(dx, unpack_video_patch_reference(grad_tokens.contiguous(), shape))
    probe = torch.randn_like(x)
    assert_bits(torch.autograd.grad(dx, grad_tokens, probe)[0], op.pack(probe))

    tokens = torch.randn(packed.shape, device=device, requires_grad=True)
    unpacked = op.unpack(tokens, shape)
    grad_latents = torch.randn((2, 24, 3, 10, 6), device=device).transpose(-1, -2)
    grad_latents.requires_grad_()
    dtokens = torch.autograd.grad(unpacked, tokens, grad_latents, create_graph=True)[0]
    assert_bits(dtokens, pack_video_patch_reference(grad_latents.contiguous()))
    token_probe = torch.randn_like(tokens)
    assert_bits(
        torch.autograd.grad(dtokens, grad_latents, token_probe)[0],
        op.unpack(token_probe, shape),
    )


def test_gpu_reuse_batch_invariance_and_stream(gpu_op):
    shape = (1, 24, 3, 6, 10)
    x = torch.randn(shape, device="cuda")
    other = torch.randn_like(x)
    alone = gpu_op.pack(x)
    assert_bits(gpu_op.pack(torch.cat([x, other]))[:1], alone)
    assert_bits(gpu_op.pack(torch.cat([other, x]))[1:], alone)
    assert_bits(gpu_op.pack(x), alone)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        recovered = gpu_op.unpack(gpu_op.pack(x), shape)
    torch.cuda.current_stream().wait_stream(stream)
    assert_bits(recovered, x)


def test_gpu_reuse_768p_layout(gpu_op):
    # 768x1344 pixels at VAE spatial factor 16: latent H=48, W=84.
    shape = (1, 24, 32, 48, 84)
    cpu_x = torch.randn(shape, dtype=torch.bfloat16)
    gpu_x = cpu_x.to("cuda")
    packed = gpu_op.pack(gpu_x)
    assert packed.shape == (1, 32 * 24 * 42, 96)
    assert_bits(packed, pack_video_patch_reference(cpu_x))
    assert_bits(gpu_op.unpack(packed, shape), cpu_x)


@pytest.mark.parametrize("direction", ("pack", "unpack"))
@pytest.mark.parametrize("dtype", DTYPES)
def test_reused_output_and_gradient_invariance(reused_op, direction, dtype):
    """Hold one logical row and its upstream gradient fixed across layouts."""
    op, device = reused_op
    shape = (1, 24, 3, 6, 10)
    token_shape = (1, 45, 96)
    input_shape = shape if direction == "pack" else token_shape
    output_shape = token_shape if direction == "pack" else shape
    generator = torch.Generator(device=device).manual_seed(420)

    def random(size):
        return torch.randn(size, generator=generator, device=device, dtype=dtype)

    def evaluate(values, grad, latent_shape):
        values = values.detach().requires_grad_()
        output = op.pack(values) if direction == "pack" else op.unpack(values, latent_shape)
        dx = torch.autograd.grad(output, values, grad)[0]
        return output.detach(), dx.detach()

    values, grad = random(input_shape), random(output_shape)
    expected, expected_grad = evaluate(values, grad, shape)
    # Repeat both forward and backward, including changing allocation history.
    for _ in range(3):
        output, dx = evaluate(values.clone(), grad.clone(), shape)
        assert_bits(output, expected)
        assert_bits(dx, expected_grad)

    for batch in (2, 4):
        for position in range(batch):
            batch_input = random((batch, *input_shape[1:]))
            batch_grad = random((batch, *output_shape[1:]))
            batch_input[position : position + 1] = values
            batch_grad[position : position + 1] = grad
            for mutation in (False, True):
                if mutation:
                    # Mutate unrelated input AND gradient rows; preserve the probe.
                    batch_input.copy_(random(batch_input.shape))
                    batch_grad.copy_(random(batch_grad.shape))
                    batch_input[position : position + 1] = values
                    batch_grad[position : position + 1] = grad
                output, dx = evaluate(batch_input, batch_grad, (batch, *shape[1:]))
                assert_bits(output[position : position + 1], expected)
                assert_bits(dx[position : position + 1], expected_grad)

    # Add whole temporal patches before/after the probe. Spatial patch geometry
    # stays fixed, so the original three frames still identify the same rows.
    padded_shape = (1, 24, 5, 6, 10)
    padded_tokens = (1, 75, 96)
    latent_slice = (slice(None), slice(None), slice(1, 4))
    token_slice = (slice(None), slice(15, 60))
    in_slice = latent_slice if direction == "pack" else token_slice
    out_slice = token_slice if direction == "pack" else latent_slice
    padded_input = random(padded_shape if direction == "pack" else padded_tokens)
    padded_grad = random(padded_tokens if direction == "pack" else padded_shape)
    padded_input[in_slice] = values
    padded_grad[out_slice] = grad
    output, dx = evaluate(padded_input, padded_grad, padded_shape)
    assert_bits(output[out_slice], expected)
    assert_bits(dx[in_slice], expected_grad)
