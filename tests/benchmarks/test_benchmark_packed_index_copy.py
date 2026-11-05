# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Check timing boundaries and all input-gradient paths without a GPU."""

import pytest
import torch

from benchmarks.operators.packing.benchmark_packed_index_copy import (
    _check_accuracy,
    _make_workload,
    build_arg_parser,
    run_benchmark,
)
from rl_engine.reference.packing.packed_index_copy import NativePackedIndexCopyOp


def _inputs():
    sources = (
        torch.tensor([[[1.0, -2.0], [3.0, -4.0]]], dtype=torch.bfloat16),
        torch.tensor([[[5.0, -6.0]]]),
        torch.tensor([[[7.0, -8.0]]]),
    )
    indices = (torch.tensor([4, 0]), torch.tensor([1]), torch.tensor([3]))
    grad_y = torch.tensor(
        [[[0.5, -2.0], [3.0, 4.0], [-5.0, 6.0], [7.0, -8.0], [9.0, 10.0]]],
        dtype=torch.bfloat16,
    )
    return sources, indices, 5, grad_y


@pytest.mark.parametrize("mode", ["forward", "backward", "forward_backward"])
def test_workload_graph_boundaries_without_accumulating_gradients(mode):
    sources, indices, sequence_length, grad_y = _inputs()
    inputs = (*sources, *indices, grad_y)
    originals = [value.clone() for value in inputs]
    calls, leaves, gradients = [], [], [[], [], []]
    native = NativePackedIndexCopyOp()

    def observed_op(text, video, audio, text_indices, video_indices, audio_indices, length):
        calls.append(torch.is_grad_enabled())
        if torch.is_grad_enabled():
            leaves.append((text, video, audio))
            for leaf, observed in zip(leaves[-1], gradients, strict=True):
                leaf.register_hook(lambda grad, observed=observed: observed.append(grad.clone()))
        return native(text, video, audio, text_indices, video_indices, audio_indices, length)

    fn = _make_workload(observed_op, sources, indices, sequence_length, grad_y, mode)
    assert len(calls) == (1 if mode == "backward" else 0)
    assert fn() is None
    assert fn() is None
    assert len(calls) == (1 if mode == "backward" else 2)
    if mode == "forward":
        assert calls == [False, False]
        assert gradients == [[], [], []]
    else:
        assert len(leaves) == (1 if mode == "backward" else 2)
        assert all(value.grad is None for invocation in leaves for value in invocation)
        if mode == "forward_backward":
            assert all(first is not second for first, second in zip(*leaves, strict=True))
        for observed, source, index in zip(gradients, sources, indices, strict=True):
            assert len(observed) == 2
            expected = grad_y.index_select(1, index).to(source.dtype)
            for grad in observed:
                torch.testing.assert_close(grad, expected, rtol=0, atol=0)
    for actual, original in zip(inputs, originals, strict=True):
        assert actual.grad is None and not actual.requires_grad
        assert torch.equal(actual, original)


def test_benchmark_rejects_cpu_instead_of_timing_fallback():
    args = build_arg_parser().parse_args(["--device", "cpu"])
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA GPU"):
        run_benchmark(args)


@pytest.mark.parametrize("modality,label", [(0, "d_text"), (1, "d_video"), (2, "d_audio")])
def test_accuracy_gate_rejects_wrong_gradient(modality, label):
    native = NativePackedIndexCopyOp()

    def wrong_gradient(*args):
        if args[modality].requires_grad:
            args[modality].register_hook(lambda grad: grad + 1)
        return native(*args)

    with pytest.raises(AssertionError, match=label):
        _check_accuracy(native, wrong_gradient, *_inputs())


@pytest.mark.parametrize("inference_only", [False, True])
def test_accuracy_gate_rejects_wrong_output(inference_only):
    native = NativePackedIndexCopyOp()

    def wrong_output(*args):
        y = native(*args)
        return y if inference_only and torch.is_grad_enabled() else y + 1

    label = "inference_output" if inference_only else "output"
    with pytest.raises(AssertionError, match=label):
        _check_accuracy(native, wrong_output, *_inputs())
