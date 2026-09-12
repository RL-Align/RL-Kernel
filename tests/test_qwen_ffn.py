# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Single-GPU checks for the ROCm-native deterministic Triton Qwen3 FFN."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

import rl_engine.kernels.ops.triton.ffn.ffn as ffn_module
from rl_engine.kernels.ops.triton.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    Qwen3FFNForwardWeights,
    pack_qwen3_ffn_forward_weights,
    qwen3_ffn,
    qwen3_ffn_triton,
    refresh_qwen3_ffn_forward_weights,
)
from rl_engine.platforms.device import device_ctx

_IS_ROCM = getattr(torch.version, "hip", None) is not None
_HAS_GPU = torch.cuda.is_available() and device_ctx.device.type == "cuda"

requires_rocm = pytest.mark.skipif(
    not (_IS_ROCM and _HAS_GPU),
    reason="the Triton FFN acceptance tests require a ROCm GPU",
)


def _randn(shape, *, seed: int, dtype=torch.float32, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.1
    return value.to(device=device, dtype=dtype)


def _assert_same_raw_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.detach()
    expected = expected.detach()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert expected.is_contiguous()
    assert torch.equal(
        actual.reshape(-1).view(torch.uint8),
        expected.reshape(-1).view(torch.uint8),
    )


def _reference(hidden_states, gate_weight, up_weight, down_weight):
    gate = hidden_states @ gate_weight.t()
    up = hidden_states @ up_weight.t()
    return (F.silu(gate) * up) @ down_weight.t()


class _ValidationTensor:
    """Tensor-shaped object for validation-order tests without a GPU."""

    def __init__(self, shape, dtype=torch.bfloat16) -> None:
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.is_cuda = True

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def size(self, dim):
        return self.shape[dim]


def test_qwen3_ffn_public_api_and_model_dimensions_are_pinned():
    assert qwen3_ffn_triton is qwen3_ffn
    assert QWEN3_8B_HIDDEN_SIZE == 4096
    assert QWEN3_8B_INTERMEDIATE_SIZE == 12288

    signature = inspect.signature(qwen3_ffn)
    assert tuple(signature.parameters) == (
        "rmsnorm_output",
        "gate_weight",
        "up_weight",
        "down_weight",
        "forward_weights",
        "tp_group",
        "cp_group",
        "sequence_parallel",
    )
    assert signature.parameters["forward_weights"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["forward_weights"].default is None
    assert signature.parameters["tp_group"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["cp_group"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["sequence_parallel"].default is False


@pytest.mark.parametrize("weight_name", ["gate_weight", "up_weight", "down_weight"])
def test_qwen3_ffn_rejects_non_huggingface_weight_layout(weight_name):
    tensors = {
        "rmsnorm_output": torch.empty((2, 8), dtype=torch.bfloat16),
        "gate_weight": torch.empty((12, 8), dtype=torch.bfloat16),
        "up_weight": torch.empty((12, 8), dtype=torch.bfloat16),
        "down_weight": torch.empty((8, 12), dtype=torch.bfloat16),
    }
    tensors[weight_name] = tensors[weight_name].t().contiguous()

    with pytest.raises(ValueError, match=rf"{weight_name} must have shape"):
        qwen3_ffn(**tensors)


def test_qwen3_ffn_rejects_zero_intermediate_size_before_device_dispatch():
    tensors = {
        "rmsnorm_output": torch.empty((2, 8), dtype=torch.bfloat16),
        "gate_weight": torch.empty((0, 8), dtype=torch.bfloat16),
        "up_weight": torch.empty((0, 8), dtype=torch.bfloat16),
        "down_weight": torch.empty((8, 0), dtype=torch.bfloat16),
    }

    with pytest.raises(ValueError, match="intermediate size must be positive"):
        qwen3_ffn(**tensors)


@pytest.mark.parametrize(
    "input_name",
    ("rmsnorm_output", "gate_weight", "up_weight", "down_weight"),
)
def test_qwen3_ffn_rejects_non_bf16_inputs(input_name, monkeypatch):
    monkeypatch.setattr(ffn_module, "Tensor", _ValidationTensor)
    tensors = {
        "rmsnorm_output": _ValidationTensor((2, 8)),
        "gate_weight": _ValidationTensor((12, 8)),
        "up_weight": _ValidationTensor((12, 8)),
        "down_weight": _ValidationTensor((8, 12)),
    }
    tensors[input_name].dtype = torch.float32

    with pytest.raises(TypeError, match=rf"{input_name} must have dtype bfloat16"):
        ffn_module._validate_ffn_inputs(**tensors)


def test_qwen3_ffn_sequence_parallel_flag_must_be_bool(monkeypatch):
    monkeypatch.setattr(ffn_module, "_validate_ffn_inputs", lambda *args: None)
    tensors = (object(), object(), object(), object())

    with pytest.raises(TypeError, match="sequence_parallel must be a bool"):
        qwen3_ffn(*tensors, sequence_parallel=1)


@requires_rocm
def test_qwen3_ffn_forward_backward_matches_fp32_reference():
    device = device_ctx.device
    hidden = _randn((2, 3, 32), seed=20, dtype=torch.bfloat16, device=device)
    gate_weight = _randn((64, 32), seed=21, dtype=torch.bfloat16, device=device)
    up_weight = _randn((64, 32), seed=22, dtype=torch.bfloat16, device=device)
    down_weight = _randn((32, 64), seed=23, dtype=torch.bfloat16, device=device)
    grad_output = _randn(hidden.shape, seed=24, dtype=torch.bfloat16, device=device)

    reference_inputs = [
        value.detach().cpu().float().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    expected = _reference(*reference_inputs)
    expected.backward(grad_output.cpu().float())

    actual_inputs = [
        value.detach().clone().requires_grad_(True)
        for value in (hidden, gate_weight, up_weight, down_weight)
    ]
    actual = qwen3_ffn(*actual_inputs)
    actual.backward(grad_output)

    assert actual.shape == hidden.shape
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual.cpu().float(), expected.detach(), atol=5e-2, rtol=2e-2)
    for actual_input, reference_input in zip(actual_inputs, reference_inputs, strict=True):
        assert actual_input.grad.dtype is torch.bfloat16
        torch.testing.assert_close(
            actual_input.grad.cpu().float(),
            reference_input.grad,
            atol=5e-2,
            rtol=2e-2,
        )


def _training_step(values, grad_output):
    inputs = [value.detach().clone().requires_grad_(True) for value in values]
    output = qwen3_ffn(*inputs)
    output.backward(grad_output)
    return output.detach(), [value.grad.detach() for value in inputs]


@requires_rocm
def test_qwen3_ffn_repeat_and_train_infer_have_zero_mismatch():
    device = device_ctx.device
    values = (
        _randn((8, 64), seed=30, dtype=torch.bfloat16, device=device),
        _randn((128, 64), seed=31, dtype=torch.bfloat16, device=device),
        _randn((128, 64), seed=32, dtype=torch.bfloat16, device=device),
        _randn((64, 128), seed=33, dtype=torch.bfloat16, device=device),
    )
    grad_output = _randn((8, 64), seed=34, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        inference_first = qwen3_ffn(*values)
        inference_second = qwen3_ffn(*values)
    training_first, gradients_first = _training_step(values, grad_output)
    training_second, gradients_second = _training_step(values, grad_output)

    assert torch.equal(inference_first, inference_second)
    assert torch.equal(inference_first, training_first)
    assert torch.equal(training_first, training_second)
    assert all(
        torch.equal(first, second)
        for first, second in zip(gradients_first, gradients_second, strict=True)
    )


@requires_rocm
def test_qwen3_ffn_output_and_hidden_gradient_are_token_slice_invariant():
    device = device_ctx.device
    hidden = _randn((6, 32), seed=40, dtype=torch.bfloat16, device=device)
    gate_weight = _randn((64, 32), seed=41, dtype=torch.bfloat16, device=device)
    up_weight = _randn((64, 32), seed=42, dtype=torch.bfloat16, device=device)
    down_weight = _randn((32, 64), seed=43, dtype=torch.bfloat16, device=device)
    grad_output = _randn(hidden.shape, seed=44, dtype=torch.bfloat16, device=device)

    full_hidden = hidden.detach().clone().requires_grad_(True)
    full_output = qwen3_ffn(full_hidden, gate_weight, up_weight, down_weight)
    full_output.backward(grad_output)

    slice_hidden = hidden[2:4].detach().clone().requires_grad_(True)
    slice_output = qwen3_ffn(slice_hidden, gate_weight, up_weight, down_weight)
    slice_output.backward(grad_output[2:4])

    assert torch.equal(slice_output, full_output[2:4])
    assert torch.equal(slice_hidden.grad, full_hidden.grad[2:4])


@requires_rocm
def test_qwen3_ffn_forward_weight_pack_is_detached_contiguous_and_transposed():
    device = device_ctx.device
    gate_weight = _randn((67, 35), seed=50, dtype=torch.bfloat16, device=device)
    up_weight = _randn((67, 35), seed=51, dtype=torch.bfloat16, device=device)
    down_weight = _randn((35, 67), seed=52, dtype=torch.bfloat16, device=device)

    packed = pack_qwen3_ffn_forward_weights(gate_weight, up_weight, down_weight)

    assert isinstance(packed, Qwen3FFNForwardWeights)
    expected = (
        gate_weight.t().contiguous(),
        up_weight.t().contiguous(),
        down_weight.t().contiguous(),
    )
    actual = (packed.gate_weight_t, packed.up_weight_t, packed.down_weight_t)
    assert tuple(value.shape for value in actual) == ((35, 67), (35, 67), (67, 35))
    for packed_weight, expected_weight in zip(actual, expected, strict=True):
        assert packed_weight.is_contiguous()
        assert not packed_weight.requires_grad
        assert packed_weight.grad_fn is None
        assert packed_weight.grad is None
        _assert_same_raw_bytes(packed_weight, expected_weight)


@requires_rocm
def test_qwen3_ffn_standard_and_packed_paths_match_all_raw_bytes():
    device = device_ctx.device
    values = (
        _randn((7, 35), seed=60, dtype=torch.bfloat16, device=device),
        _randn((67, 35), seed=61, dtype=torch.bfloat16, device=device),
        _randn((67, 35), seed=62, dtype=torch.bfloat16, device=device),
        _randn((35, 67), seed=63, dtype=torch.bfloat16, device=device),
    )
    grad_output = _randn((7, 35), seed=64, dtype=torch.bfloat16, device=device)

    inference_pack = pack_qwen3_ffn_forward_weights(*values[1:])
    with torch.no_grad():
        standard_inference = qwen3_ffn(*values)
        packed_inference = qwen3_ffn(*values, forward_weights=inference_pack)

    standard_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    packed_inputs = [value.detach().clone().requires_grad_(True) for value in values]
    training_pack = pack_qwen3_ffn_forward_weights(*packed_inputs[1:])
    standard_training = qwen3_ffn(*standard_inputs)
    packed_training = qwen3_ffn(*packed_inputs, forward_weights=training_pack)
    standard_training.backward(grad_output)
    packed_training.backward(grad_output)

    _assert_same_raw_bytes(packed_inference, standard_inference)
    _assert_same_raw_bytes(standard_training, standard_inference)
    _assert_same_raw_bytes(packed_training, standard_training)
    for packed_input, standard_input in zip(packed_inputs, standard_inputs, strict=True):
        assert packed_input.grad is not None
        assert standard_input.grad is not None
        _assert_same_raw_bytes(packed_input.grad, standard_input.grad)
    for packed_weight in (
        training_pack.gate_weight_t,
        training_pack.up_weight_t,
        training_pack.down_weight_t,
    ):
        assert not packed_weight.requires_grad
        assert packed_weight.grad is None


@pytest.mark.parametrize(
    ("weight_index", "weight_name"),
    ((0, "gate_weight"), (1, "up_weight"), (2, "down_weight")),
)
@requires_rocm
def test_qwen3_ffn_rejects_stale_forward_weights_after_inplace_source_update(
    weight_index,
    weight_name,
):
    device = device_ctx.device
    hidden = _randn((2, 8), seed=70, dtype=torch.bfloat16, device=device)
    weights = [
        _randn((12, 8), seed=71, dtype=torch.bfloat16, device=device),
        _randn((12, 8), seed=72, dtype=torch.bfloat16, device=device),
        _randn((8, 12), seed=73, dtype=torch.bfloat16, device=device),
    ]
    packed = pack_qwen3_ffn_forward_weights(*weights)
    with torch.no_grad():
        weights[weight_index].add_(1)

    with pytest.raises(RuntimeError, match=rf"{weight_name} changed after .*refresh"):
        qwen3_ffn(hidden, *weights, forward_weights=packed)


@requires_rocm
def test_qwen3_ffn_forward_weight_refresh_preserves_storage_and_bits():
    device = device_ctx.device
    hidden = _randn((7, 35), seed=80, dtype=torch.bfloat16, device=device)
    weights = [
        _randn((67, 35), seed=81, dtype=torch.bfloat16, device=device),
        _randn((67, 35), seed=82, dtype=torch.bfloat16, device=device),
        _randn((35, 67), seed=83, dtype=torch.bfloat16, device=device),
    ]
    packed = pack_qwen3_ffn_forward_weights(*weights)
    packed_tensors = (
        packed.gate_weight_t,
        packed.up_weight_t,
        packed.down_weight_t,
    )
    data_ptrs = tuple(weight.data_ptr() for weight in packed_tensors)

    with torch.no_grad():
        for weight in weights:
            weight.add_(torch.ones_like(weight))
    refreshed = refresh_qwen3_ffn_forward_weights(packed, *weights)

    assert refreshed is packed
    assert tuple(weight.data_ptr() for weight in packed_tensors) == data_ptrs
    with torch.no_grad():
        standard = qwen3_ffn(hidden, *weights)
        cached = qwen3_ffn(hidden, *weights, forward_weights=packed)
    _assert_same_raw_bytes(cached, standard)


@requires_rocm
def test_qwen3_ffn_forward_weight_pack_works_inside_inference_mode():
    device = device_ctx.device
    with torch.inference_mode():
        hidden = _randn((7, 35), seed=90, dtype=torch.bfloat16, device=device)
        weights = (
            _randn((67, 35), seed=91, dtype=torch.bfloat16, device=device),
            _randn((67, 35), seed=92, dtype=torch.bfloat16, device=device),
            _randn((35, 67), seed=93, dtype=torch.bfloat16, device=device),
        )
        assert all(torch.is_inference(weight) for weight in weights)
        packed = pack_qwen3_ffn_forward_weights(*weights)
        cached = qwen3_ffn(hidden, *weights, forward_weights=packed)
        standard = qwen3_ffn(hidden, *weights)

    assert not torch.is_inference(packed.gate_weight_t)
    assert not torch.is_inference(packed.up_weight_t)
    assert not torch.is_inference(packed.down_weight_t)
    _assert_same_raw_bytes(cached, standard)


@requires_rocm
def test_qwen3_ffn_rejects_mutated_packed_forward_weight():
    device = device_ctx.device
    hidden = _randn((2, 8), seed=100, dtype=torch.bfloat16, device=device)
    weights = (
        _randn((12, 8), seed=101, dtype=torch.bfloat16, device=device),
        _randn((12, 8), seed=102, dtype=torch.bfloat16, device=device),
        _randn((8, 12), seed=103, dtype=torch.bfloat16, device=device),
    )
    packed = pack_qwen3_ffn_forward_weights(*weights)
    with torch.no_grad():
        packed.gate_weight_t.add_(1)

    with pytest.raises(RuntimeError, match="packed gate_weight changed; refresh"):
        qwen3_ffn(hidden, *weights, forward_weights=packed)


@requires_rocm
def test_qwen3_ffn_refresh_updates_already_captured_cuda_graph():
    device = device_ctx.device
    hidden = _randn((7, 35), seed=110, dtype=torch.bfloat16, device=device)
    weights = [
        _randn((67, 35), seed=111, dtype=torch.bfloat16, device=device),
        _randn((67, 35), seed=112, dtype=torch.bfloat16, device=device),
        _randn((35, 67), seed=113, dtype=torch.bfloat16, device=device),
    ]
    packed = pack_qwen3_ffn_forward_weights(*weights)
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream), torch.no_grad():
        for _ in range(3):
            qwen3_ffn(hidden, *weights, forward_weights=packed)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream), torch.no_grad():
        static_output = qwen3_ffn(hidden, *weights, forward_weights=packed)
    graph.replay()
    torch.cuda.synchronize()
    before_refresh = static_output.clone()

    with torch.no_grad():
        weights[0].add_(1)
    packed.refresh_(*weights)
    graph.replay()
    torch.cuda.synchronize()
    after_refresh = static_output.clone()
    with torch.no_grad():
        expected = qwen3_ffn(hidden, *weights)

    assert not torch.equal(
        before_refresh.view(torch.uint8),
        after_refresh.view(torch.uint8),
    )
    _assert_same_raw_bytes(after_refresh, expected)
