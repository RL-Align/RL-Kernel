# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Full Qwen-Image score-matrix acceptance smoke for joint-attention softmax."""

from time import perf_counter

import pytest
import torch

from rl_engine.testing.bitwise import tensor_bytes_equal

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)


@pytest.mark.parametrize(
    ("image_shape", "key_length"),
    [
        ((1024, 1024), 4608),
        ((1328, 1328), 7401),
        ((1664, 928), 6544),
    ],
)
def test_full_qwen_image_scores_match_cuda_triton_and_sampled_cpu_reference(
    image_shape: tuple[int, int], key_length: int
) -> None:
    """Validate complete [B, H, Q, K] scores with Q=K, not just a sample row."""
    from rl_engine.kernels.ops.cuda.attention.joint_attn_softmax import JointAttnSoftmaxCudaOp
    from rl_engine.kernels.ops.pytorch.attention.joint_attn_softmax import NativeJointAttnSoftmaxOp
    from rl_engine.kernels.ops.triton.attention.joint_attn_softmax import TritonJointAttnSoftmaxOp

    height, width = image_shape
    assert 512 + (height // 16) * (width // 16) == key_length
    shape = (1, 1, key_length, key_length)

    # Reserve headroom for two backends' outputs/gradients and the FP32 backward state.
    matrix_bytes = key_length * key_length * torch.bfloat16.itemsize
    minimum_free_bytes = 1_000_000_000 + 16 * matrix_bytes
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < minimum_free_bytes:
        pytest.skip(
            f"full {shape} BF16 acceptance needs at least "
            f"{minimum_free_bytes / 2**30:.2f} GiB free GPU memory; "
            f"only {free_bytes / 2**30:.2f} GiB is free"
        )

    torch.cuda.reset_peak_memory_stats()
    started_at = perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(386 + key_length)
    scores = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    upstream = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    selected_rows = (0, key_length // 2, key_length - 1)
    scores[0, 0, selected_rows[0], :256] = float("-inf")
    scores[0, 0, selected_rows[1], :512] = float("-inf")
    scores.requires_grad_(True)

    cuda_probabilities = JointAttnSoftmaxCudaOp().forward(scores)
    (cuda_grad_scores,) = torch.autograd.grad(cuda_probabilities, scores, upstream)
    triton_probabilities = TritonJointAttnSoftmaxOp().forward(scores)
    (triton_grad_scores,) = torch.autograd.grad(triton_probabilities, scores, upstream)

    for probabilities, grad_scores in (
        (cuda_probabilities, cuda_grad_scores),
        (triton_probabilities, triton_grad_scores),
    ):
        assert probabilities.shape == shape
        assert probabilities.dtype == torch.bfloat16
        assert grad_scores.shape == shape
        assert grad_scores.dtype == torch.bfloat16
    assert tensor_bytes_equal(triton_probabilities, cuda_probabilities)
    assert tensor_bytes_equal(triton_grad_scores, cuda_grad_scores)

    cpu_scores = scores[0, 0, list(selected_rows)].detach().cpu().requires_grad_(True)
    cpu_upstream = upstream[0, 0, list(selected_rows)].cpu()
    cpu_probabilities = NativeJointAttnSoftmaxOp().forward(cpu_scores)
    (cpu_grad_scores,) = torch.autograd.grad(cpu_probabilities, cpu_scores, cpu_upstream)

    for probabilities, grad_scores in (
        (cuda_probabilities, cuda_grad_scores),
        (triton_probabilities, triton_grad_scores),
    ):
        assert tensor_bytes_equal(probabilities[0, 0, list(selected_rows)].cpu(), cpu_probabilities)
        assert tensor_bytes_equal(grad_scores[0, 0, list(selected_rows)].cpu(), cpu_grad_scores)

    torch.cuda.synchronize()
    print(
        f"full shape {shape}: {perf_counter() - started_at:.2f}s, "
        f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )
