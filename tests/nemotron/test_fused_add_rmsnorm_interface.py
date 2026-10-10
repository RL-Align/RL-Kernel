# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Public interfaces: native CPU checks and Triton GPU checks."""

import importlib

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm import NativeFusedAddRMSNormOp

_DTYPES = [
    (torch.float32, torch.float32, torch.float32),
    (torch.float16, torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.float32, torch.bfloat16),
]


@pytest.fixture(scope="module")
def triton_module():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    triton = pytest.importorskip("triton")
    if not hasattr(triton, "jit"):
        pytest.skip("A working Triton runtime is required; type stubs are insufficient")
    return importlib.import_module("rl_engine.kernels.ops.triton.norm.fused_add_rmsnorm")


@pytest.fixture(params=["native", "auto", "sequential", "tiled", "parallel", "fused"])
def implementation(request):
    if request.param == "native":
        return NativeFusedAddRMSNormOp(), "cpu"
    module = request.getfixturevalue("triton_module")
    strategy = None if request.param == "auto" else module.RMSNormWeightGradStrategy(request.param)
    return module.TritonFusedAddRMSNormOp(weight_grad_strategy=strategy), "cuda"


def _rand(shape, seed, *, device="cpu", dtype=torch.float32):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(
        device=device, dtype=dtype
    )


def _inputs(shape, dtypes, device, *, noncontiguous=False):
    inputs = []
    for seed, (tensor_shape, dtype) in enumerate(
        zip((shape, shape, (shape[-1],)), dtypes, strict=True), start=100
    ):
        if noncontiguous:
            padded_shape = (*tensor_shape[:-1], 2 * tensor_shape[-1])
            value = _rand(padded_shape, seed, device=device, dtype=dtype)[..., ::2]
        else:
            value = _rand(tensor_shape, seed, device=device, dtype=dtype)
        inputs.append(value.requires_grad_(True))
    return tuple(inputs)


def _fp64_reference(x, residual, weight, eps):
    # Independent high-precision oracle: no calls into either operator or its backward.
    updated = x + residual
    denominator = (updated.square().mean(dim=-1, keepdim=True) + eps).sqrt()
    return (updated / denominator) * weight, updated


def _backward(outputs, inputs, upstream, branch, **kwargs):
    indices = {"both": (0, 1), "y_only": (0,), "residual_only": (1,)}[branch]
    gradients = torch.autograd.grad(
        tuple(outputs[index] for index in indices),
        inputs,
        grad_outputs=tuple(upstream[index] for index in indices),
        allow_unused=True,
        **kwargs,
    )
    return tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(inputs, gradients, strict=True)
    )


def _assert_close(actual, expected):
    tolerance = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[actual.dtype]
    torch.testing.assert_close(actual, expected.to(actual.dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape", [(7,), (3, 7), (33, 129), (2, 3, 2688), (2, 0, 7)])
@pytest.mark.parametrize("dtypes", _DTYPES)
@pytest.mark.parametrize("branch", ["both", "y_only", "residual_only"])
def test_public_forward_backward_against_fp64(implementation, shape, dtypes, branch):
    op, device = implementation
    inputs = _inputs(shape, dtypes, device)
    reference_inputs = tuple(t.detach().double().requires_grad_(True) for t in inputs)
    upstream = tuple(_rand(shape, seed, device=device) for seed in (110, 111))
    actual = op(*inputs, eps=1e-5)
    expected = _fp64_reference(*reference_inputs, eps=1e-5)
    for output, reference in zip(actual, expected, strict=True):
        assert output.shape == shape
        assert output.dtype == torch.float32
        _assert_close(output, reference)
    actual_gradients = _backward(actual, inputs, upstream, branch)
    expected_gradients = _backward(
        expected, reference_inputs, tuple(t.double() for t in upstream), branch
    )
    for value, gradient, reference in zip(
        inputs, actual_gradients, expected_gradients, strict=True
    ):
        assert gradient.shape == value.shape
        assert gradient.dtype == value.dtype
        _assert_close(gradient, reference)
    if branch == "residual_only":
        assert torch.equal(actual_gradients[0], upstream[1].to(inputs[0].dtype))
        assert torch.equal(actual_gradients[1], upstream[1].to(inputs[1].dtype))
        assert torch.equal(actual_gradients[2], torch.zeros_like(inputs[2]))


@pytest.mark.parametrize("dtypes", _DTYPES)
def test_noncontiguous_inputs_and_upstream_gradients(implementation, dtypes):
    op, device = implementation
    shape = (3, 7)
    inputs = _inputs(shape, dtypes, device, noncontiguous=True)
    assert all(not value.is_contiguous() for value in inputs)
    upstream = tuple(_rand((3, 14), seed, device=device)[:, ::2] for seed in (120, 121))
    reference_inputs = tuple(t.detach().double().requires_grad_(True) for t in inputs)
    actual = op(*inputs)
    expected = _fp64_reference(*reference_inputs, eps=1e-5)
    for output, reference in zip(actual, expected, strict=True):
        _assert_close(output, reference)
    for gradient, reference in zip(
        _backward(actual, inputs, upstream, "both"),
        _backward(expected, reference_inputs, tuple(t.double() for t in upstream), "both"),
        strict=True,
    ):
        _assert_close(gradient, reference)


@pytest.mark.parametrize("trainable", [0, 1, 2])
def test_only_one_input_requires_grad(implementation, trainable):
    op, device = implementation
    inputs = _inputs((3, 7), _DTYPES[0], device)
    for index, value in enumerate(inputs):
        value.requires_grad_(index == trainable)
    outputs = op(*inputs)
    (outputs[0].sum() + outputs[1].sum()).backward()
    assert inputs[trainable].grad is not None
    for index, value in enumerate(inputs):
        if index != trainable:
            assert value.grad is None


@pytest.mark.parametrize("eps", [0.0, -1e-5, float("nan"), float("inf")])
@pytest.mark.parametrize("implementation", ["native", "auto"], indirect=True)
def test_invalid_epsilon(implementation, eps):
    op, device = implementation
    with pytest.raises(ValueError, match="eps"):
        op(*_inputs((3, 7), _DTYPES[0], device), eps=eps)


@pytest.mark.parametrize("invalid_input", [0, 1, 2])
@pytest.mark.parametrize("implementation", ["native", "auto"], indirect=True)
def test_unsupported_input_dtype(implementation, invalid_input):
    op, device = implementation
    inputs = list(_inputs((3, 7), _DTYPES[0], device))
    inputs[invalid_input] = inputs[invalid_input].detach().to(torch.int64)
    with pytest.raises(TypeError, match="dtype"):
        op(*inputs)


@pytest.mark.parametrize(
    "shapes,match",
    [
        (((), (), (1,)), "D > 0"),
        (((2, 0), (2, 0), (0,)), "D > 0"),
        (((2, 7), (1, 7), (7,)), "residual"),
        (((2, 7), (2, 7), (1, 7)), "weight"),
        (((2, 7), (2, 7), (8,)), "weight"),
    ],
)
@pytest.mark.parametrize("implementation", ["native", "auto"], indirect=True)
def test_invalid_shapes(implementation, shapes, match):
    op, device = implementation
    inputs = tuple(torch.zeros(shape, device=device) for shape in shapes)
    with pytest.raises(ValueError, match=match):
        op(*inputs)


def test_native_rejects_mismatched_devices():
    x = torch.zeros(2, 7)
    with pytest.raises(ValueError, match="same device"):
        NativeFusedAddRMSNormOp()(x, torch.empty_like(x, device="meta"), torch.ones(7))


@pytest.mark.parametrize("dtypes", _DTYPES)
def test_forward_reuses_fp32_residual_output_and_saves_row_statistics(triton_module, dtypes):
    inputs = _inputs((2, 3, 2688), dtypes, "cuda", noncontiguous=True)
    outputs = triton_module.TritonFusedAddRMSNormOp()(*inputs)
    saved_residual, inverse_rms, saved_weight = outputs[0].grad_fn.saved_tensors

    # Save the FP32 output itself, without retaining or copying the two inputs.
    assert saved_residual.data_ptr() == outputs[1].data_ptr()
    assert saved_residual.shape == inputs[0].shape
    assert saved_residual.dtype == torch.float32
    assert torch.equal(saved_residual, inputs[0].float() + inputs[1].float())
    assert inverse_rms.shape == (6,)
    assert inverse_rms.dtype == torch.float32
    assert inverse_rms.device == inputs[0].device
    assert torch.equal(saved_weight, inputs[2])

    # Independent FP64 statistics check also covers non-power-of-two row widths.
    updated_fp64 = inputs[0].double() + inputs[1].double()
    expected = (updated_fp64.square().mean(dim=-1) + 1e-5).rsqrt().reshape(-1)
    torch.testing.assert_close(inverse_rms.double(), expected, rtol=2e-6, atol=1e-7)


def test_saved_residual_preserves_low_precision_addition_bits(triton_module):
    # 1 + 1/256 is exactly representable in FP32, but rounds to 1 in BF16.
    x = torch.tensor([[1.0, 2.0]], device="cuda", dtype=torch.bfloat16, requires_grad=True)
    residual = torch.tensor([[1 / 256, 0.0]], device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(2, device="cuda")
    outputs = triton_module.TritonFusedAddRMSNormOp()(x, residual, weight)
    saved_residual = outputs[0].grad_fn.saved_tensors[0]
    expected = torch.tensor([[1 + 1 / 256, 2.0]], device="cuda")
    assert torch.equal(saved_residual, expected)
    assert not torch.equal(saved_residual, expected.bfloat16().float())

    reference_x = x.detach().double().requires_grad_(True)
    reference_y, _ = _fp64_reference(reference_x, residual.double(), weight.double(), 1e-5)
    upstream = torch.tensor([[0.7, -0.4]], device="cuda")
    actual_gradient = torch.autograd.grad(outputs[0], x, upstream)[0]
    expected_gradient = torch.autograd.grad(reference_y, reference_x, upstream.double())[0]
    _assert_close(actual_gradient, expected_gradient)


def test_modifying_saved_residual_output_is_detected(triton_module):
    inputs = _inputs((3, 7), _DTYPES[0], "cuda")
    y, updated_residual = triton_module.TritonFusedAddRMSNormOp()(*inputs)
    with torch.no_grad():
        updated_residual.add_(1.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
@pytest.mark.parametrize("strategy", ["sequential", "tiled", "parallel", "fused"])
def test_repeated_backward_preserves_saved_tensors(triton_module, eps, strategy):
    inputs = _inputs((65, 129), _DTYPES[0], "cuda", noncontiguous=True)
    op = triton_module.TritonFusedAddRMSNormOp(
        weight_grad_strategy=triton_module.RMSNormWeightGradStrategy(strategy)
    )
    outputs = op(*inputs, eps=eps)
    saved = outputs[0].grad_fn.saved_tensors
    before = tuple(value.clone() for value in saved)
    upstream = tuple(_rand((65, 129), seed, device="cuda") for seed in (130, 131))
    first = _backward(outputs, inputs, upstream, "both", retain_graph=True)
    second = _backward(outputs, inputs, upstream, "both")
    for lhs, rhs in zip(first, second, strict=True):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))
    for lhs, rhs in zip(saved, before, strict=True):
        assert torch.equal(lhs.contiguous().view(torch.uint8), rhs.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy", ["sequential", "tiled", "parallel", "fused"])
def test_train_inference_and_row_invariance(triton_module, dtype, strategy):
    op = triton_module.TritonFusedAddRMSNormOp(
        weight_grad_strategy=triton_module.RMSNormWeightGradStrategy(strategy)
    )
    inputs = _inputs((7, 2688), (dtype, dtype, torch.float32), "cuda")
    outputs = op(*inputs)
    inverse_rms = outputs[0].grad_fn.saved_tensors[1]
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            inference_outputs = op(*inputs)
        for training, inference in zip(outputs, inference_outputs, strict=True):
            assert not inference.requires_grad
            assert torch.equal(training.view(torch.uint8), inference.view(torch.uint8))

    upstream = tuple(_rand((7, 2688), seed, device="cuda") for seed in (140, 141))
    full_gradients = _backward(outputs, inputs, upstream, "both")
    order = torch.tensor([5, 2, 0], device="cuda")
    subset_inputs = tuple(value[order].detach().requires_grad_(True) for value in inputs[:2])
    subset_inputs += (inputs[2].detach().requires_grad_(True),)
    subset_outputs = op(*subset_inputs)
    subset_inverse_rms = subset_outputs[0].grad_fn.saved_tensors[1]
    assert torch.equal(inverse_rms[order].view(torch.uint8), subset_inverse_rms.view(torch.uint8))
    subset_gradients = _backward(
        subset_outputs, subset_inputs, tuple(value[order] for value in upstream), "both"
    )
    for full, subset in zip(outputs, subset_outputs, strict=True):
        assert torch.equal(full[order].view(torch.uint8), subset.view(torch.uint8))
    # Weight gradients sum different logical rows; compare only the per-row input gradients.
    for full, subset in zip(full_gradients[:2], subset_gradients[:2], strict=True):
        assert torch.equal(full[order].view(torch.uint8), subset.view(torch.uint8))


@pytest.mark.parametrize("strategy", [None, "sequential", "tiled", "parallel", "fused"])
def test_launches_use_input_device_and_restore_current_device(triton_module, strategy):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two GPUs are required to check a noncurrent input device")
    with torch.cuda.device(0):
        inputs = _inputs((3, 2688), _DTYPES[2], "cuda:1")
        upstream = tuple(_rand((3, 2688), seed, device="cuda:1") for seed in (150, 151))
        reference_inputs = tuple(value.detach().double().requires_grad_(True) for value in inputs)
        expected = _fp64_reference(*reference_inputs, 1e-5)

        op = triton_module.TritonFusedAddRMSNormOp(
            weight_grad_strategy=(
                triton_module.RMSNormWeightGradStrategy(strategy) if strategy is not None else None
            )
        )
        outputs = op(*inputs)
        assert torch.cuda.current_device() == 0
        gradients = _backward(outputs, inputs, upstream, "both")
        assert torch.cuda.current_device() == 0
        expected_gradients = _backward(
            expected, reference_inputs, tuple(value.double() for value in upstream), "both"
        )
        for actual, reference in zip(
            outputs + gradients, expected + expected_gradients, strict=True
        ):
            assert actual.device == torch.device("cuda:1")
            _assert_close(actual, reference)


def test_triton_rejects_cpu_inputs(triton_module):
    with pytest.raises(ValueError, match="device"):
        triton_module.TritonFusedAddRMSNormOp()(*_inputs((3, 7), _DTYPES[0], "cpu"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [8, 9, 32, 33, 16383, 16384, 16385])
def test_auto_boundaries_match_explicit_plan_and_preserve_consistency(triton_module, dtype, rows):
    module = triton_module
    shape = (1, rows, 2688)  # Leading dimensions are flattened for dispatch.
    inputs = _inputs(shape, (dtype, dtype, torch.float32), "cuda")
    upstream = tuple(_rand(shape, seed, device="cuda") for seed in (190, 191))
    if rows <= 8:
        strategy, block_rows, block_cols, warps = "sequential", 1, 128, 4
    elif rows <= 32:
        strategy, block_rows, block_cols, warps = "tiled", 32, 64, 4
    elif rows < 16384:
        strategy, block_rows, block_cols, warps = "tiled", 64, 64, 8
    else:
        strategy, block_rows, block_cols, warps = "fused", 64, 64, 4
    selected_strategy = module.RMSNormWeightGradStrategy(strategy)
    config = module.RMSNormWeightGradConfig(
        block_rows=block_rows, block_cols=block_cols, num_warps=warps
    )
    op = module.TritonFusedAddRMSNormOp()
    actual = op(*inputs)
    assert actual[0].grad_fn.weight_grad_strategy == selected_strategy
    assert actual[0].grad_fn.weight_grad_config == config
    manual = module.TritonFusedAddRMSNormOp(
        weight_grad_strategy=selected_strategy, weight_grad_config=config
    )
    explicit = manual(*inputs)
    gradients = _backward(actual, inputs, upstream, "both", retain_graph=True)
    explicit_gradients = _backward(explicit, inputs, upstream, "both")
    direct_gradients = module._launch_fused_add_rmsnorm_bwd(
        *actual[0].grad_fn.saved_tensors,
        *upstream,
        x_dtype=dtype,
        residual_dtype=dtype,
    )
    for lhs, rhs in zip(actual + gradients, explicit + explicit_gradients, strict=True):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))
    for lhs, rhs in zip(gradients, direct_gradients, strict=True):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))

    expected = NativeFusedAddRMSNormOp()(*inputs)
    expected_gradients = _backward(expected, inputs, upstream, "both")
    for lhs, rhs in zip(actual + gradients[:2], expected + expected_gradients[:2], strict=True):
        _assert_close(lhs, rhs)
    torch.testing.assert_close(
        gradients[2], expected_gradients[2], rtol=2e-5, atol=2e-5 * rows**0.5
    )
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            inferred = op(*inputs)
        for lhs, rhs in zip(actual, inferred, strict=True):
            assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))

    # A smaller microbatch selects another reduction but preserves per-row arithmetic.
    subset_inputs = tuple(t[:, :3].detach().requires_grad_(True) for t in inputs[:2]) + (inputs[2],)
    subset = op(*subset_inputs)
    subset_gradients = _backward(subset, subset_inputs, tuple(g[:, :3] for g in upstream), "both")
    for full, part in zip(actual + gradients[:2], subset + subset_gradients[:2], strict=True):
        assert torch.equal(full[:, :3].contiguous().view(torch.uint8), part.view(torch.uint8))

    # Reusing/mutating the Op or clearing policy caches cannot change a saved graph's plan.
    op.weight_grad_strategy = module.RMSNormWeightGradStrategy.SEQUENTIAL
    op.weight_grad_config = module.RMSNormWeightGradConfig(block_rows=1, block_cols=256)
    module.select_rmsnorm_weight_grad_plan.cache_clear()
    repeated = _backward(actual, inputs, upstream, "both")
    for lhs, rhs in zip(gradients, repeated, strict=True):
        assert torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))


@pytest.mark.parametrize("dtypes", _DTYPES)
def test_weight_strategy_does_not_change_outputs_or_input_gradients(triton_module, dtypes):
    inputs = _inputs((65, 129), dtypes, "cuda")
    upstream = tuple(_rand((65, 129), seed, device="cuda") for seed in (160, 161))
    results = []
    for strategy in triton_module.RMSNormWeightGradStrategy:
        op = triton_module.TritonFusedAddRMSNormOp(weight_grad_strategy=strategy)
        outputs = op(*inputs)
        gradients = _backward(outputs, inputs, upstream, "both")
        results.append(outputs + gradients[:2])
    for candidate in results[1:]:
        for sequential, actual in zip(results[0], candidate, strict=True):
            assert torch.equal(sequential.view(torch.uint8), actual.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy", ["sequential", "tiled", "parallel", "fused"])
def test_config_is_saved_per_forward_and_preserves_row_results(triton_module, dtype, strategy):
    inputs = _inputs((257, 129), (dtype, dtype, torch.float32), "cuda")
    upstream = tuple(_rand((257, 129), seed, device="cuda") for seed in (180, 181))
    config = triton_module.RMSNormWeightGradConfig(block_rows=64, block_cols=64, num_warps=8)
    op = triton_module.TritonFusedAddRMSNormOp(
        weight_grad_strategy=triton_module.RMSNormWeightGradStrategy(strategy),
        weight_grad_config=config,
    )
    outputs = op(*inputs)
    saved_before = tuple(t.clone() for t in outputs[0].grad_fn.saved_tensors)
    first = _backward(outputs, inputs, upstream, "both", retain_graph=True)
    reference_inputs = tuple(t.detach().double().requires_grad_(True) for t in inputs)
    reference = _fp64_reference(*reference_inputs, eps=1e-5)
    gradients = _backward(reference, reference_inputs, tuple(t.double() for t in upstream), "both")
    for actual, expected in zip(outputs + first[:2], reference + gradients[:2], strict=True):
        _assert_close(actual, expected)
    torch.testing.assert_close(first[2], gradients[2].float(), rtol=2e-5, atol=2e-5 * 257**0.5)

    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            inference = op(*inputs)
        for training, inferred in zip(outputs, inference, strict=True):
            assert torch.equal(training.view(torch.uint8), inferred.view(torch.uint8))
    order = torch.tensor([256, 128, 1], device="cuda")
    subset_inputs = tuple(t[order].detach().requires_grad_(True) for t in inputs[:2]) + (inputs[2],)
    subset = op(*subset_inputs)
    for original, selected in zip(outputs, subset, strict=True):
        assert torch.equal(original[order].view(torch.uint8), selected.view(torch.uint8))
    subset_grads = _backward(subset, subset_inputs, tuple(t[order] for t in upstream), "both")
    for original, selected in zip(first[:2], subset_grads[:2], strict=True):
        assert torch.equal(original[order].view(torch.uint8), selected.view(torch.uint8))

    # Changing the reusable Op after forward must not change this graph's backward.
    op.weight_grad_strategy = triton_module.RMSNormWeightGradStrategy.SEQUENTIAL
    op.weight_grad_config = triton_module.RMSNormWeightGradConfig(block_rows=1, block_cols=256)
    second = _backward(outputs, inputs, upstream, "both", retain_graph=True)
    for original, repeated in zip(first, second, strict=True):
        assert torch.equal(original.view(torch.uint8), repeated.view(torch.uint8))
    for before, after in zip(saved_before, outputs[0].grad_fn.saved_tensors, strict=True):
        assert torch.equal(before.view(torch.uint8), after.view(torch.uint8))


@pytest.mark.filterwarnings("error:.*AccumulateGrad.*:UserWarning")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strategy", ["auto", "sequential", "tiled", "parallel", "fused"])
@pytest.mark.parametrize("mode", ["backward", "forward_backward"])
def test_public_graph_replay_matches_eager_without_mutating_live_graph(
    triton_module, dtype, strategy, mode
):
    # Exercise automatic FUSED at its boundary and every explicit strategy at a tail.
    shape = (16384, 2688) if strategy == "auto" else (65, 129)
    selected = None if strategy == "auto" else triton_module.RMSNormWeightGradStrategy(strategy)
    op = triton_module.TritonFusedAddRMSNormOp(weight_grad_strategy=selected)
    inputs = _inputs(shape, (dtype, dtype, torch.float32), "cuda")
    upstream = tuple(_rand(shape, seed, device="cuda") for seed in (190, 191))
    eager_outputs = op(*inputs)
    eager_grads = torch.autograd.grad(eager_outputs, inputs, upstream, retain_graph=True)
    expected = eager_grads if mode == "backward" else eager_outputs + eager_grads
    if strategy == "auto":
        assert eager_outputs[0].grad_fn.weight_grad_strategy == (
            triton_module.RMSNormWeightGradStrategy.FUSED
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Fresh leaves isolate AccumulateGrad stream metadata from the live eager graph.
        leaves = tuple(value.detach().requires_grad_(True) for value in inputs)
        prebuilt = op(*leaves) if mode == "backward" else None

        def call():
            outputs = prebuilt if prebuilt is not None else op(*leaves)
            gradients = torch.autograd.grad(
                outputs, leaves, upstream, retain_graph=prebuilt is not None
            )
            return gradients if mode == "backward" else outputs + gradients

        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(captured, expected, strict=True):
            assert actual.shape == reference.shape and actual.dtype == reference.dtype
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))

    repeated = torch.autograd.grad(eager_outputs, inputs, upstream)
    for actual, reference in zip(repeated, eager_grads, strict=True):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("platform", ["cpu", "musa", "npu"])
def test_registry_native_dispatch(platform, monkeypatch):
    from rl_engine.kernels.registry import KernelRegistry

    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    op = registry.get_op("fused_add_rmsnorm")
    assert isinstance(op, NativeFusedAddRMSNormOp)
    inputs = _inputs((2, 7), _DTYPES[0], "cpu")
    outputs = op(*inputs)
    assert len(outputs) == 2
    for actual, expected in zip(outputs, _fp64_reference(*inputs, 1e-5), strict=True):
        _assert_close(actual, expected)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_registry_fallback_when_triton_unavailable(platform, monkeypatch):
    from rl_engine.kernels.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    load_backend = registry._load_backend
    attempted = []

    def without_triton(backend):
        attempted.append(backend)
        if backend == OpBackend.TRITON_FUSED_ADD_RMSNORM:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", without_triton)
    assert isinstance(registry.get_op("fused_add_rmsnorm"), NativeFusedAddRMSNormOp)
    assert attempted == [
        OpBackend.TRITON_FUSED_ADD_RMSNORM,
        OpBackend.PYTORCH_NATIVE_FUSED_ADD_RMSNORM,
    ]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("backend", ["pytorch", "triton"])
def test_shared_harness_checks_both_outputs_and_all_input_gradients(dtype, backend, request):
    import argparse

    from rl_engine.kernels.gtest.op_checks import run_operator_suite
    from rl_engine.kernels.gtest.operator_specs import make_candidate, make_operator_case

    if backend == "triton":
        request.getfixturevalue("triton_module")
    args = argparse.Namespace(
        op="fused_add_rmsnorm",
        candidate=backend,
        arch_key=None,
        batch=2,
        seq=3,
        normalized_dim=129,
        eps=1e-5,
        seed=434,
        input_mode="random",
    )
    device = torch.device("cuda" if backend == "triton" else "cpu")
    case = make_operator_case(args, dtype, device)
    assert case.inputs["x"].shape == case.inputs["residual"].shape == (2, 3, 129)
    assert case.inputs["weight"].shape == (129,)
    assert case.inputs["weight"].dtype == torch.float32
    assert case.grad_input_names == ("x", "residual", "weight")
    report = run_operator_suite(
        args.op,
        candidates=[make_candidate(args)],
        cases=[case],
        check_grad=True,
    )
    assert report.passed
    assert [len(check.outputs) for check in report.candidates[0].cases] == [5]
