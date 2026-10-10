# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton acceptance tests for joint-attention softmax (issue #386)."""

import pytest
import torch

from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.testing.bitwise import tensor_bytes_equal

_CONTRACT = load_contract()


def _accuracy_tolerance(dtype: torch.dtype, judgment: str) -> tuple[float, float]:
    """Return rtol/atol from the shared reduction contract."""
    spec = resolve_tolerance(
        _CONTRACT,
        judgment=judgment,
        op_class="reduction",
        dtype=dtype,
    )
    return spec.rtol, spec.atol


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)


def test_forward_fp32_matches_cpu_reference_byte_for_byte():
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    scores_cpu = torch.linspace(-10.0, 10.0, 513, dtype=torch.float32).reshape(1, 513)
    expected = NativeJointAttnSoftmaxOp().forward_fp32(scores_cpu)

    probabilities = TritonJointAttnSoftmaxOp().forward_fp32(scores_cpu.cuda()).cpu()

    assert tensor_bytes_equal(probabilities, expected)


def test_forward_bf16_casts_once_at_final_triton_write():
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    scores_cpu = torch.linspace(-10.0, 10.0, 513, dtype=torch.bfloat16).reshape(1, 513)
    expected = NativeJointAttnSoftmaxOp().forward(scores_cpu)

    probabilities = TritonJointAttnSoftmaxOp().forward(scores_cpu.cuda()).cpu()

    assert probabilities.dtype == torch.bfloat16
    assert tensor_bytes_equal(probabilities, expected)


def test_backward_fp32_matches_cpu_reference_byte_for_byte():
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386)
    scores = torch.randn(2, 513, dtype=torch.float32, generator=generator)
    upstream = torch.randn(2, 513, dtype=torch.float32, generator=generator)

    expected_scores = scores.clone().requires_grad_(True)
    NativeJointAttnSoftmaxOp().forward_fp32(expected_scores).backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    TritonJointAttnSoftmaxOp().forward_fp32(actual_scores).backward(upstream.cuda())

    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)


def test_backward_bf16_matches_cpu_reference_byte_for_byte():
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    scores = torch.linspace(-10.0, 10.0, 513, dtype=torch.bfloat16).reshape(1, 513)
    upstream = torch.linspace(1.0, -1.0, 513, dtype=torch.bfloat16).reshape(1, 513)

    expected_scores = scores.clone().requires_grad_(True)
    expected_probabilities = NativeJointAttnSoftmaxOp().forward(expected_scores)
    expected_probabilities.backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    actual_probabilities = TritonJointAttnSoftmaxOp().forward(actual_scores)
    actual_probabilities.backward(upstream.cuda())

    assert tensor_bytes_equal(actual_probabilities.cpu(), expected_probabilities)
    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)


@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_fully_masked_row_returns_zero_probabilities_and_gradients(dtype, output_fp32):
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    finite_row = torch.linspace(-4.0, 4.0, 513, dtype=torch.float32).to(dtype)
    scores = torch.stack([finite_row, finite_row.new_full((513,), float("-inf"))])
    upstream = torch.linspace(-1.0, 1.0, 513, dtype=torch.float32).repeat(2, 1)
    if not output_fp32:
        upstream = upstream.to(dtype)

    expected_scores = scores.clone().requires_grad_(True)
    reference_op = NativeJointAttnSoftmaxOp()
    expected_probabilities = (
        reference_op.forward_fp32(expected_scores)
        if output_fp32
        else reference_op.forward(expected_scores)
    )
    expected_probabilities.backward(upstream)

    actual_scores = scores.cuda().requires_grad_(True)
    triton_op = TritonJointAttnSoftmaxOp()
    actual_probabilities = (
        triton_op.forward_fp32(actual_scores) if output_fp32 else triton_op.forward(actual_scores)
    )
    actual_probabilities.backward(upstream.cuda())

    assert tensor_bytes_equal(actual_probabilities.cpu(), expected_probabilities)
    assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)
    assert tensor_bytes_equal(
        actual_probabilities[1].cpu(), torch.zeros_like(expected_probabilities[1])
    )
    assert tensor_bytes_equal(
        actual_scores.grad[1].cpu(), torch.zeros_like(expected_scores.grad[1])
    )


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf")], ids=["nan", "posinf"])
def test_unsupported_nonfinite_scores_propagate_nan(invalid_score):
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    scores = torch.tensor(
        [[invalid_score, float("-inf")]],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    probabilities = TritonJointAttnSoftmaxOp().forward_fp32(scores)
    probabilities.backward(torch.ones_like(probabilities))

    assert torch.isnan(probabilities).any()
    assert torch.isnan(scores.grad).any()


def test_registry_falls_back_to_triton_when_cuda_symbol_is_unavailable(monkeypatch):
    from rl_engine.kernels.ops.base import _C
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp
    from rl_engine.kernels.registry import KernelRegistry

    monkeypatch.delattr(_C, "joint_attn_softmax_forward")
    operation = KernelRegistry().get_op("joint_attn_softmax", device="cuda")

    assert isinstance(operation, TritonJointAttnSoftmaxOp)


def test_triton_trace_records_the_frozen_arithmetic_contract():
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    operation = TritonJointAttnSoftmaxOp()

    assert operation.provenance == {
        "selected_backend": "triton_cuda",
        "reduction_order": "tile256_tree_128_to_1_then_left_to_right",
        "accumulator_precision": "fp32",
        "split_k": False,
        "stream_k": False,
        "tf32": False,
        "kernel_fingerprint": "joint-attn-softmax-v2-logical-mask-tile256-exp7",
        "fallback": False,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_matches_cuda_forward_and_backward_byte_for_byte(dtype):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386)
    scores = (torch.randn(2, 777, dtype=torch.float32, generator=generator) * 8.0).to(dtype)
    upstream = torch.randn(2, 777, dtype=torch.float32, generator=generator).to(dtype)

    cuda_scores = scores.cuda().requires_grad_(True)
    cuda_probabilities = JointAttnSoftmaxCudaOp().forward(cuda_scores)
    cuda_probabilities.backward(upstream.cuda())

    triton_scores = scores.cuda().requires_grad_(True)
    triton_probabilities = TritonJointAttnSoftmaxOp().forward(triton_scores)
    triton_probabilities.backward(upstream.cuda())

    assert tensor_bytes_equal(triton_probabilities, cuda_probabilities)
    assert tensor_bytes_equal(triton_scores.grad, cuda_scores.grad)


def test_triton_forward_and_backward_are_byte_invariant_to_batch_companions():
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    target = torch.linspace(-5.0, 5.0, 513, device="cuda", dtype=torch.float32)
    upstream = torch.linspace(1.0, -1.0, 513, device="cuda", dtype=torch.float32)
    operation = TritonJointAttnSoftmaxOp()

    alone_scores = target.clone().requires_grad_(True)
    alone_probabilities = operation.forward_fp32(alone_scores)
    alone_probabilities.backward(upstream)

    batched_scores = torch.stack([target.flip(0), target, torch.zeros_like(target)])
    batched_scores.requires_grad_(True)
    batched_upstream = torch.stack([torch.zeros_like(upstream), upstream, upstream.flip(0)])
    batched_probabilities = operation.forward_fp32(batched_scores)
    batched_probabilities.backward(batched_upstream)

    assert tensor_bytes_equal(batched_probabilities[1], alone_probabilities)
    assert tensor_bytes_equal(batched_scores.grad[1], alone_scores.grad)


def test_masked_prompt_padding_is_batch_invariant():
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    target = torch.linspace(-5.0, 5.0, 769, device="cuda", dtype=torch.float32)
    # The first 512 keys are text positions: 73 valid tokens, then padding.
    target[73:512] = float("-inf")
    upstream = torch.linspace(1.0, -1.0, 769, device="cuda", dtype=torch.float32)
    operation = TritonJointAttnSoftmaxOp()

    alone_scores = target.clone().requires_grad_(True)
    alone_probabilities = operation.forward_fp32(alone_scores)
    alone_probabilities.backward(upstream)

    batched_scores = torch.stack([target.flip(0), target, torch.zeros_like(target)])
    batched_scores.requires_grad_(True)
    batched_upstream = torch.stack([torch.zeros_like(upstream), upstream, upstream.flip(0)])
    batched_probabilities = operation.forward_fp32(batched_scores)
    batched_probabilities.backward(batched_upstream)

    assert tensor_bytes_equal(batched_probabilities[1], alone_probabilities)
    assert tensor_bytes_equal(batched_scores.grad[1], alone_scores.grad)


@pytest.mark.parametrize("upstream_kind", ["regular", "negative_zero"])
@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_explicit_key_padding_mask_preserves_prompt_padding_bytes_across_backends(
    upstream_kind, dtype, output_fp32
):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    text_key_count = 73
    image_key_count = 257
    prompt_padding_count = 512 - text_key_count
    image_start = text_key_count + prompt_padding_count
    logical_scores = torch.linspace(-5.0, 5.0, text_key_count + image_key_count).to(dtype)
    upstream_dtype = torch.float32 if output_fp32 else dtype
    logical_upstream = (
        torch.linspace(1.0, -1.0, logical_scores.numel()).to(upstream_dtype)
        if upstream_kind == "regular"
        else torch.full((logical_scores.numel(),), -0.0, dtype=upstream_dtype)
    )

    padded_scores = torch.cat(
        [
            logical_scores[:text_key_count],
            logical_scores.new_zeros(prompt_padding_count),
            logical_scores[text_key_count:],
        ]
    ).unsqueeze(0)
    key_padding_mask = torch.cat(
        [
            torch.ones(text_key_count, dtype=torch.bool),
            torch.zeros(prompt_padding_count, dtype=torch.bool),
            torch.ones(image_key_count, dtype=torch.bool),
        ]
    ).unsqueeze(0)
    padded_upstream = torch.cat(
        [
            logical_upstream[:text_key_count],
            logical_upstream.new_zeros(prompt_padding_count),
            logical_upstream[text_key_count:],
        ]
    ).unsqueeze(0)

    compact_scores = logical_scores.unsqueeze(0).clone().requires_grad_(True)
    reference = NativeJointAttnSoftmaxOp()
    compact_probabilities = (
        reference.forward_fp32(compact_scores) if output_fp32 else reference.forward(compact_scores)
    )
    compact_probabilities.backward(logical_upstream.unsqueeze(0))

    results = []
    for operation in (
        NativeJointAttnSoftmaxOp(),
        JointAttnSoftmaxCudaOp(),
        TritonJointAttnSoftmaxOp(),
    ):
        device = "cpu" if isinstance(operation, NativeJointAttnSoftmaxOp) else "cuda"
        backend_scores = padded_scores.to(device).detach().requires_grad_(True)
        probabilities = (
            operation.forward_fp32(
                backend_scores,
                key_padding_mask=key_padding_mask.to(device),
            )
            if output_fp32
            else operation.forward(
                backend_scores,
                key_padding_mask=key_padding_mask.to(device),
            )
        )
        probabilities.backward(padded_upstream.to(device))
        results.append((probabilities.cpu(), backend_scores.grad.cpu()))

    for probabilities, gradients in results:
        logical_probabilities = torch.cat(
            [probabilities[:, :text_key_count], probabilities[:, image_start:]], dim=-1
        )
        logical_gradients = torch.cat(
            [gradients[:, :text_key_count], gradients[:, image_start:]], dim=-1
        )
        assert tensor_bytes_equal(logical_probabilities, compact_probabilities)
        assert tensor_bytes_equal(logical_gradients, compact_scores.grad)
        assert torch.count_nonzero(probabilities[:, text_key_count:image_start]) == 0
        assert torch.count_nonzero(gradients[:, text_key_count:image_start]) == 0


def test_masked_benchmark_reports_prompt_layout():
    from argparse import Namespace

    from benchmarks.benchmark_joint_attn_softmax import run_benchmark

    report = run_benchmark(
        Namespace(
            rows=2,
            dtype="bf16",
            backward=False,
            warmup=0,
            iterations=1,
            backends=["cuda", "triton"],
            cases={"synthetic": 513},
            valid_text_keys=73,
        )
    )

    assert report["schema_version"] == "rlkernel.joint_attn_softmax_benchmark.v3"
    assert report["configuration"]["valid_text_keys"] == 73
    assert report["configuration"]["warmup"] == 0
    assert report["configuration"]["iterations"] == 1
    assert report["environment"]["git_commit"] != "unavailable"
    assert all(record["masked_prompt_keys"] == 439 for record in report["results"])
    assert all(record["byte_equal_to_cuda"] for record in report["results"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_explicit_key_padding_mask_uses_each_batch_layout_across_backends(dtype, output_fp32):
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386)
    scores = torch.randn(3, 2, 3, 513, generator=generator).to(dtype)
    upstream = torch.randn(3, 2, 3, 513, generator=generator)
    if not output_fp32:
        upstream = upstream.to(dtype)
    key_padding_mask = torch.ones(3, 513, dtype=torch.bool)
    key_padding_mask[0, 73:256] = False
    key_padding_mask[1, :128] = False
    key_padding_mask[1, 400:] = False
    key_padding_mask[2] = False
    expanded_mask = key_padding_mask[:, None, None, :].expand_as(scores)

    expected_scores = scores.clone().requires_grad_(True)
    reference = NativeJointAttnSoftmaxOp()
    expected_probabilities = (
        reference.forward_fp32(expected_scores, key_padding_mask=key_padding_mask)
        if output_fp32
        else reference.forward(expected_scores, key_padding_mask=key_padding_mask)
    )
    expected_probabilities.backward(upstream)

    for operation in (JointAttnSoftmaxCudaOp(), TritonJointAttnSoftmaxOp()):
        actual_scores = scores.cuda().requires_grad_(True)
        actual_probabilities = (
            operation.forward_fp32(
                actual_scores,
                key_padding_mask=key_padding_mask.cuda(),
            )
            if output_fp32
            else operation.forward(
                actual_scores,
                key_padding_mask=key_padding_mask.cuda(),
            )
        )
        actual_probabilities.backward(upstream.cuda())

        assert tensor_bytes_equal(actual_probabilities.cpu(), expected_probabilities)
        assert tensor_bytes_equal(actual_scores.grad.cpu(), expected_scores.grad)
        assert torch.count_nonzero(actual_probabilities.masked_select(~expanded_mask.cuda())) == 0
        assert torch.count_nonzero(actual_scores.grad.masked_select(~expanded_mask.cuda())) == 0


def test_explicit_key_padding_mask_preserves_empty_batch_across_backends():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    for operation in (
        NativeJointAttnSoftmaxOp(),
        JointAttnSoftmaxCudaOp(),
        TritonJointAttnSoftmaxOp(),
    ):
        device = "cpu" if isinstance(operation, NativeJointAttnSoftmaxOp) else "cuda"
        scores = torch.empty(0, 2, 3, 257, device=device, requires_grad=True)
        key_padding_mask = torch.empty(0, 257, device=device, dtype=torch.bool)

        probabilities = operation.forward_fp32(
            scores,
            key_padding_mask=key_padding_mask,
        )
        probabilities.sum().backward()

        assert probabilities.shape == scores.shape
        assert scores.grad is not None
        assert scores.grad.shape == scores.shape


@pytest.mark.parametrize(
    "dtype,output_fp32",
    [(torch.float32, True), (torch.bfloat16, False), (torch.bfloat16, True)],
)
def test_fully_masked_tail_tile_preserves_valid_key_bytes(dtype, output_fp32):
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    scores = torch.linspace(-4.0, 4.0, 256, device="cuda", dtype=torch.float32).to(dtype)
    upstream = torch.linspace(-1.0, 1.0, 256, device="cuda", dtype=torch.float32)
    if not output_fp32:
        upstream = upstream.to(dtype)

    short_scores = scores.clone().requires_grad_(True)
    long_scores = torch.cat([scores, scores.new_full((1,), float("-inf"))]).requires_grad_(True)
    op = TritonJointAttnSoftmaxOp()

    short_probabilities = op.forward_fp32(short_scores) if output_fp32 else op(short_scores)
    long_probabilities = op.forward_fp32(long_scores) if output_fp32 else op(long_scores)
    short_probabilities.backward(upstream)
    long_probabilities.backward(torch.cat([upstream, upstream.new_zeros(1)]))

    assert tensor_bytes_equal(long_probabilities[:256], short_probabilities)
    assert tensor_bytes_equal(long_scores.grad[:256], short_scores.grad)
    assert long_probabilities[-1] == 0
    assert long_scores.grad[-1] == 0


@pytest.mark.parametrize(
    ("image_shape", "joint_key_length"),
    [
        ((1024, 1024), 4608),
        ((1328, 1328), 7401),
        ((1664, 928), 6544),
    ],
)
def test_qwen_image_shapes_match_all_backends_forward_and_backward(image_shape, joint_key_length):
    """Cover 512 text tokens plus H/16 * W/16 packed image tokens."""
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    height, width = image_shape
    assert 512 + (height // 16) * (width // 16) == joint_key_length

    generator = torch.Generator().manual_seed(386 + joint_key_length)
    scores = torch.randn(1, joint_key_length, dtype=torch.float32, generator=generator)
    upstream = torch.randn(1, joint_key_length, dtype=torch.float32, generator=generator)

    cpu_scores = scores.clone().requires_grad_(True)
    cpu_probabilities = NativeJointAttnSoftmaxOp().forward_fp32(cpu_scores)
    cpu_probabilities.backward(upstream)

    cuda_scores = scores.cuda().requires_grad_(True)
    cuda_probabilities = JointAttnSoftmaxCudaOp().forward_fp32(cuda_scores)
    cuda_probabilities.backward(upstream.cuda())

    triton_scores = scores.cuda().requires_grad_(True)
    triton_probabilities = TritonJointAttnSoftmaxOp().forward_fp32(triton_scores)
    triton_probabilities.backward(upstream.cuda())

    assert tensor_bytes_equal(cuda_probabilities.cpu(), cpu_probabilities)
    assert tensor_bytes_equal(triton_probabilities, cuda_probabilities)
    assert tensor_bytes_equal(cuda_scores.grad.cpu(), cpu_scores.grad)
    assert tensor_bytes_equal(triton_scores.grad, cuda_scores.grad)


def test_noncontiguous_input_matches_all_backends_forward_and_backward():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    generator = torch.Generator().manual_seed(386)
    scores = torch.randn(513, 3, dtype=torch.float32, generator=generator).transpose(0, 1)
    upstream = torch.randn(3, 513, dtype=torch.float32, generator=generator)
    assert not scores.is_contiguous()

    cpu_scores = scores.detach().requires_grad_(True)
    cpu_probabilities = NativeJointAttnSoftmaxOp().forward_fp32(cpu_scores)
    cpu_probabilities.backward(upstream)

    cuda_scores = scores.cuda().detach().requires_grad_(True)
    cuda_probabilities = JointAttnSoftmaxCudaOp().forward_fp32(cuda_scores)
    cuda_probabilities.backward(upstream.cuda())

    triton_scores = scores.cuda().detach().requires_grad_(True)
    triton_probabilities = TritonJointAttnSoftmaxOp().forward_fp32(triton_scores)
    triton_probabilities.backward(upstream.cuda())

    assert tensor_bytes_equal(cuda_probabilities.cpu(), cpu_probabilities)
    assert tensor_bytes_equal(triton_probabilities, cuda_probabilities)
    assert tensor_bytes_equal(cuda_scores.grad.cpu(), cpu_scores.grad)
    assert tensor_bytes_equal(triton_scores.grad, cuda_scores.grad)


def test_materialized_negative_infinity_mask_matches_all_backends():
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    scores = torch.linspace(-5.0, 5.0, 513, dtype=torch.float32).reshape(1, 513)
    masked_columns = torch.tensor([0, 255, 256, 512])
    scores[:, masked_columns] = float("-inf")
    upstream = torch.linspace(1.0, -1.0, 513, dtype=torch.float32).reshape(1, 513)

    cpu_scores = scores.clone().requires_grad_(True)
    cpu_probabilities = NativeJointAttnSoftmaxOp().forward_fp32(cpu_scores)
    cpu_probabilities.backward(upstream)

    cuda_scores = scores.cuda().requires_grad_(True)
    cuda_probabilities = JointAttnSoftmaxCudaOp().forward_fp32(cuda_scores)
    cuda_probabilities.backward(upstream.cuda())

    triton_scores = scores.cuda().requires_grad_(True)
    triton_probabilities = TritonJointAttnSoftmaxOp().forward_fp32(triton_scores)
    triton_probabilities.backward(upstream.cuda())

    assert torch.count_nonzero(cpu_probabilities[:, masked_columns]) == 0
    assert torch.count_nonzero(cpu_scores.grad[:, masked_columns]) == 0
    assert tensor_bytes_equal(cuda_probabilities.cpu(), cpu_probabilities)
    assert tensor_bytes_equal(triton_probabilities, cuda_probabilities)
    assert tensor_bytes_equal(cuda_scores.grad.cpu(), cpu_scores.grad)
    assert tensor_bytes_equal(triton_scores.grad, cuda_scores.grad)


@pytest.mark.parametrize("leading_masked_tiles", [1, 2])
@pytest.mark.parametrize("finite_tail_keys", [1, 257])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("output_fp32", [False, True])
def test_leading_fully_masked_tiles_still_produce_finite_forward_and_backward(
    leading_masked_tiles, finite_tail_keys, dtype, output_fp32
):
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    key_length = leading_masked_tiles * 256 + finite_tail_keys
    masked_keys = leading_masked_tiles * 256
    scores = torch.linspace(-5.0, 5.0, key_length, dtype=torch.float32).to(dtype)
    scores[:masked_keys] = float("-inf")
    upstream = torch.linspace(1.0, -1.0, key_length, dtype=torch.float32)

    reference_scores = scores.detach().clone().float().requires_grad_(True)
    expected_fp32 = torch.softmax(reference_scores, dim=-1)
    expected = expected_fp32 if output_fp32 else expected_fp32.to(dtype)
    expected.backward(upstream.to(expected.dtype))

    actual_scores = scores.detach().cuda().requires_grad_(True)
    operation = TritonJointAttnSoftmaxOp()
    actual = (
        operation.forward_fp32(actual_scores) if output_fp32 else operation.forward(actual_scores)
    )
    actual.backward(upstream.to(actual.dtype).cuda())

    cpu_scores = scores.detach().clone().requires_grad_(True)
    reference_op = NativeJointAttnSoftmaxOp()
    cpu_probabilities = (
        reference_op.forward_fp32(cpu_scores) if output_fp32 else reference_op.forward(cpu_scores)
    )
    cpu_probabilities.backward(upstream.to(cpu_probabilities.dtype))

    assert torch.isfinite(actual).all()
    assert torch.isfinite(actual_scores.grad).all()
    assert torch.count_nonzero(actual[:masked_keys]) == 0
    assert torch.count_nonzero(actual_scores.grad[:masked_keys]) == 0
    forward_rtol, forward_atol = _accuracy_tolerance(dtype, "forward_accuracy")
    gradient_rtol, gradient_atol = _accuracy_tolerance(dtype, "gradient_accuracy")
    torch.testing.assert_close(
        actual.cpu(), expected.detach(), rtol=forward_rtol, atol=forward_atol
    )
    torch.testing.assert_close(
        actual_scores.grad.cpu(),
        reference_scores.grad.to(dtype),
        rtol=gradient_rtol,
        atol=gradient_atol,
    )
    assert tensor_bytes_equal(actual.cpu(), cpu_probabilities.detach())
    assert tensor_bytes_equal(actual_scores.grad.cpu(), cpu_scores.grad)
