# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

pytest.importorskip("triton")

from rl_engine.backends.shared.triton.embedding import embedding as embedding_module  # noqa: E402


@dataclass
class _FakeForwardKernel:
    grids: list[tuple[int, ...]] = field(default_factory=list)

    def __getitem__(self, grid: tuple[int, ...]):
        self.grids.append(grid)

        def launch(
            ids: torch.Tensor,
            weight: torch.Tensor,
            out: torch.Tensor,
            n_tokens: int,
            vocab_size: int,
            scale: float,
            *,
            hidden: int,
            block_h: int,
            apply_scale: bool,
        ) -> None:
            del vocab_size, block_h, apply_scale
            assert n_tokens == ids.numel()
            assert hidden == weight.size(1)
            out.copy_((weight.index_select(0, ids).float() * scale).to(out.dtype))

        return launch


@dataclass
class _FakeBackwardKernel:
    grids: list[tuple[int, ...]] = field(default_factory=list)

    def __getitem__(self, grid: tuple[int, ...]):
        self.grids.append(grid)

        def launch(
            sorted_ids: torch.Tensor,
            sorted_grad_rows: torch.Tensor,
            grad_weight: torch.Tensor,
            n_tokens: int,
            vocab_size: int,
            scale: float,
            *,
            hidden: int,
            block_h: int,
            apply_scale: bool,
            num_warps: int,
            enable_fp_fusion: bool,
        ) -> None:
            del block_h, apply_scale, num_warps, enable_fp_fusion
            assert n_tokens == sorted_ids.numel()
            assert hidden == sorted_grad_rows.size(1)
            for position in range(n_tokens):
                token = int(sorted_ids[position])
                if position and token == int(sorted_ids[position - 1]):
                    continue
                if token < 0 or token >= vocab_size:
                    continue
                accumulator = torch.zeros(hidden, dtype=torch.float32)
                row = position
                while row < n_tokens and int(sorted_ids[row]) == token:
                    accumulator = accumulator + sorted_grad_rows[row].float() * scale
                    row += 1
                grad_weight[token].copy_(accumulator.to(grad_weight.dtype))

        return launch


def _install_fake_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeForwardKernel, _FakeBackwardKernel]:
    forward = _FakeForwardKernel()
    backward = _FakeBackwardKernel()
    monkeypatch.setattr(embedding_module, "_embedding_fwd", forward)
    monkeypatch.setattr(embedding_module, "_embedding_bwd", backward)
    monkeypatch.setattr(embedding_module, "record_backward", lambda *args, **kwargs: None)
    return forward, backward


def _left_fold_grad_weight(
    token_ids: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    vocab_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    ids = token_ids.reshape(-1).long()
    rows = grad_output.reshape(ids.numel(), grad_output.shape[-1])
    result = torch.zeros((vocab_size, rows.size(1)), dtype=output_dtype)
    for token in range(vocab_size):
        accumulator = torch.zeros(rows.size(1), dtype=torch.float32)
        for position in range(ids.numel()):
            if int(ids[position]) == token:
                accumulator = accumulator + rows[position].float()
        result[token].copy_(accumulator.to(output_dtype))
    return result


def _run_fake_autograd(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    differentiable_weight = weight.detach().clone().requires_grad_(True)
    output = embedding_module._TritonEmbeddingFunction.apply(token_ids, differentiable_weight)
    output.backward(grad_output)
    assert differentiable_weight.grad is not None
    return output.detach(), differentiable_weight.grad.detach()


def test_stable_sort_preserves_original_positions_within_each_token() -> None:
    ids = torch.tensor([4, 1, 4, 2, 1, 4], dtype=torch.long)
    positions = torch.arange(ids.numel(), dtype=torch.float32).unsqueeze(1)

    sorted_ids, sorted_positions = embedding_module._stable_sort_token_rows(ids, positions)

    assert sorted_ids.tolist() == [1, 1, 2, 4, 4, 4]
    assert sorted_positions.squeeze(1).tolist() == [1.0, 4.0, 3.0, 0.0, 2.0, 5.0]


def test_backward_grid_scales_with_tokens_instead_of_vocab() -> None:
    grid = embedding_module._embedding_backward_grid(n_tokens=32, hidden=4096)

    assert grid == (32, 32)
    assert grid[0] * grid[1] == 1024
    assert grid[0] * grid[1] < 151936 * 4096


def test_embedding_backward_contract_rejects_misaligned_rows() -> None:
    ids = torch.tensor([1, 2, 1], dtype=torch.long)

    with pytest.raises(ValueError, match=r"shape \[3, 4\]"):
        embedding_module._embedding_grad_weight(
            ids,
            torch.ones(2, 4),
            weight_shape=(8, 4),
            weight_dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="flattened"):
        embedding_module._embedding_grad_weight(
            ids.reshape(1, 3),
            torch.ones(3, 4),
            weight_shape=(8, 4),
            weight_dtype=torch.float32,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,flat_ids",
    [
        ((2, 3), [5, 1, 5, 2, 1, 5]),
        ((2, 3), [5, 1, 4, 2, 0, 3]),
        ((0, 3), []),
    ],
)
def test_fake_autograd_matches_stable_fp32_left_fold(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    flat_ids: list[int],
) -> None:
    forward_kernel, backward_kernel = _install_fake_kernels(monkeypatch)
    vocab_size, hidden = 8, 7
    token_ids = torch.tensor(flat_ids, dtype=torch.int32).reshape(shape)
    weight = torch.linspace(-1.0, 1.0, vocab_size * hidden, dtype=torch.float32).reshape(
        vocab_size, hidden
    )
    weight = weight.to(dtype)
    grad_output = torch.arange(
        token_ids.numel() * hidden,
        dtype=torch.float32,
    ).reshape(*shape, hidden)
    grad_output = (grad_output / 13.0).to(dtype)

    output, grad_weight = _run_fake_autograd(token_ids, weight, grad_output)
    expected_grad = _left_fold_grad_weight(
        token_ids,
        grad_output,
        vocab_size=vocab_size,
        output_dtype=dtype,
    )

    assert output.dtype == dtype
    assert torch.equal(output, weight[token_ids.long()])
    assert torch.equal(grad_weight, expected_grad)
    if token_ids.numel():
        assert forward_kernel.grids == [(token_ids.numel(),)]
        assert backward_kernel.grids == [
            embedding_module._embedding_backward_grid(token_ids.numel(), hidden)
        ]
    else:
        assert not forward_kernel.grids
        assert not backward_kernel.grids


def test_backward_uses_each_permutations_original_position_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_kernels(monkeypatch)
    vocab_size, hidden = 6, 3
    base_ids = torch.tensor([2, 1, 2, 2, 1, 2], dtype=torch.long)
    base_rows = torch.tensor(
        [
            [8192.0, 1.0, 0.5],
            [3.0, 2.0, 1.0],
            [1.0, -1.0, 0.25],
            [-8192.0, 4.0, -0.5],
            [5.0, 6.0, 7.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=torch.float16,
    )
    weight = torch.zeros((vocab_size, hidden), dtype=torch.float16)

    for order in ([0, 1, 2, 3, 4, 5], [3, 4, 2, 0, 1, 5], [5, 2, 0, 1, 4, 3]):
        permutation = torch.tensor(order, dtype=torch.long)
        ids = base_ids.index_select(0, permutation).reshape(2, 3)
        rows = base_rows.index_select(0, permutation).reshape(2, 3, hidden)
        _, grad_weight = _run_fake_autograd(ids, weight, rows)
        expected = _left_fold_grad_weight(
            ids,
            rows,
            vocab_size=vocab_size,
            output_dtype=weight.dtype,
        )
        assert torch.equal(grad_weight, expected)


def test_flatten_equivalent_batch_shapes_share_forward_and_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_kernels(monkeypatch)
    flat_ids = torch.tensor([3, 1, 3, 2, 3, 1], dtype=torch.long)
    flat_rows = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 7.0
    weight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 11.0

    baseline_output, baseline_grad = _run_fake_autograd(flat_ids, weight, flat_rows)
    for leading_shape in ((2, 3), (3, 2), (1, 6)):
        output, grad = _run_fake_autograd(
            flat_ids.reshape(leading_shape),
            weight,
            flat_rows.reshape(*leading_shape, 4),
        )
        assert torch.equal(output.reshape(-1, 4), baseline_output)
        assert torch.equal(grad, baseline_grad)


def test_forward_path_does_not_depend_on_weight_requires_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward_kernel, _ = _install_fake_kernels(monkeypatch)
    ids = torch.tensor([[2, 0, 2], [1, 4, 3]], dtype=torch.int16)
    weight = torch.randn(5, 4, dtype=torch.bfloat16)

    inference = embedding_module._TritonEmbeddingFunction.apply(ids, weight)
    training = embedding_module._TritonEmbeddingFunction.apply(ids, weight.requires_grad_(True))

    assert torch.equal(inference, training)
    assert len(forward_kernel.grids) == 2


def test_explicit_token_validation_accepts_boundaries_and_rejects_invalid_ids() -> None:
    embedding_module.TritonEmbeddingOp.validate_token_ids(torch.tensor([0, 4]), 5)

    with pytest.raises(ValueError, match=r"\[0, 5\)"):
        embedding_module.TritonEmbeddingOp.validate_token_ids(torch.tensor([-1, 2]), 5)
    with pytest.raises(ValueError, match=r"\[0, 5\)"):
        embedding_module.TritonEmbeddingOp.validate_token_ids(torch.tensor([1, 5]), 5)
    with pytest.raises(TypeError, match="integer dtype"):
        embedding_module.TritonEmbeddingOp.validate_token_ids(torch.tensor([1.5]), 5)


def test_hot_path_range_assertion_keeps_condition_as_a_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[torch.Tensor, str]] = []

    def capture(condition: torch.Tensor, message: str) -> None:
        captured.append((condition, message))

    monkeypatch.setattr(torch, "_assert_async", capture)
    embedding_module._assert_valid_token_ids_async(torch.tensor([0, 3, 1]), 4)

    assert len(captured) == 1
    assert captured[0][0].dtype == torch.bool
    assert captured[0][0].ndim == 0
    assert captured[0][1] == "embedding token_ids must be in [0, 4)"


@pytest.mark.parametrize(
    "token_ids,weight,error_type,message",
    [
        (torch.tensor([0.0]), torch.ones(3, 2), TypeError, "integer dtype"),
        (torch.tensor([True]), torch.ones(3, 2), TypeError, "integer dtype"),
        (torch.tensor([0]), torch.ones(3, 2, dtype=torch.int32), TypeError, "fp16"),
        (torch.tensor([0]), torch.ones(3), ValueError, "weight must be"),
    ],
)
def test_embedding_input_contract_rejects_invalid_dtypes_and_shapes(
    token_ids: torch.Tensor,
    weight: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        embedding_module._validate_embedding_inputs(token_ids, weight)


def test_backward_kernel_has_no_atomic_updates() -> None:
    source = embedding_module._embedding_bwd.src

    assert "atomic_" not in source
    assert "while (row < n_tokens) & continuing" in source


requires_nvidia_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA is required",
)


@requires_nvidia_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,flat_ids",
    [
        ((2, 4), [7, 1, 7, 2, 1, 7, 3, 7]),
        ((2, 4), [7, 1, 6, 2, 0, 5, 3, 4]),
        ((0, 4), []),
    ],
)
def test_triton_embedding_cuda_matches_stable_fp32_left_fold(
    dtype: torch.dtype,
    shape: tuple[int, ...],
    flat_ids: list[int],
) -> None:
    vocab_size, hidden = 11, 64
    token_ids = torch.tensor(flat_ids, device="cuda", dtype=torch.int32).reshape(shape)
    weight = torch.linspace(
        -1.0,
        1.0,
        vocab_size * hidden,
        device="cuda",
        dtype=torch.float32,
    ).reshape(vocab_size, hidden)
    weight = weight.to(dtype).requires_grad_(True)
    grad_output = torch.arange(
        token_ids.numel() * hidden,
        device="cuda",
        dtype=torch.float32,
    ).reshape(*shape, hidden)
    grad_output = (grad_output / 17.0).to(dtype)

    output = embedding_module.TritonEmbeddingOp().forward(token_ids, weight)
    output.backward(grad_output)
    expected_grad = _left_fold_grad_weight(
        token_ids.cpu(),
        grad_output.cpu(),
        vocab_size=vocab_size,
        output_dtype=dtype,
    )

    assert torch.equal(output, weight.detach()[token_ids.long()])
    assert weight.grad is not None
    assert torch.equal(weight.grad.cpu(), expected_grad)


@requires_nvidia_cuda
def test_triton_embedding_cuda_backward_is_repeatable() -> None:
    token_ids = torch.tensor(
        [[5, 2, 5, 1], [2, 5, 5, 2]],
        device="cuda",
        dtype=torch.long,
    )
    weight = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)

    gradients = []
    for _ in range(3):
        differentiable_weight = weight.detach().clone().requires_grad_(True)
        embedding_module.TritonEmbeddingOp().forward(token_ids, differentiable_weight).backward(
            grad_output
        )
        assert differentiable_weight.grad is not None
        gradients.append(differentiable_weight.grad.detach())

    assert torch.equal(gradients[0], gradients[1])
    assert torch.equal(gradients[0], gradients[2])


# The oracles below do not touch the kernels, so a kernel change cannot move both sides.
_BITS_DTYPE = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}
# +-0, +-smallest and largest subnormal, smallest normal, NaNs (quiet, signed, payload,
# signaling), +-inf, largest finite and one.
_EDGE_BITS = {
    torch.float32: [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000,
        0x7FC00000, 0xFFC00000, 0x7FC12345, 0x7F812345, 0xFF812345,
        0x7F800000, 0xFF800000, 0x7F7FFFFF, 0x3F800000,
    ],
    torch.bfloat16: [
        0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080,
        0x7FC0, 0xFFC0, 0x7FC5, 0x7F85, 0xFF85,
        0x7F80, 0xFF80, 0x7F7F, 0x3F80,
    ],
    torch.float16: [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400,
        0x7E00, 0xFE00, 0x7E55, 0x7C55, 0xFC55,
        0x7C00, 0xFC00, 0x7BFF, 0x3C00,
    ],
}  # fmt: skip


def _accelerator_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    return "cuda"


@pytest.mark.parametrize("dtype", _EDGE_BITS)
def test_plain_forward_stores_the_weight_bits_unchanged(dtype: torch.dtype) -> None:
    device = _accelerator_device()
    edge_row = torch.tensor(_EDGE_BITS[dtype]).to(_BITS_DTYPE[dtype]).view(dtype)
    weight = torch.stack((edge_row, torch.ones_like(edge_row))).to(device)
    token_ids = torch.tensor([[0, 1], [0, 0]], device=device)

    output = embedding_module.TritonEmbeddingOp()(token_ids, weight)

    # Raw bits: a float comparison would reject every NaN and accept -0 for +0.
    expected = weight.index_select(0, token_ids.reshape(-1)).view(_BITS_DTYPE[dtype])
    assert torch.equal(output.reshape(4, -1).view(_BITS_DTYPE[dtype]), expected)


@pytest.mark.parametrize("dtype", _EDGE_BITS)
def test_plain_backward_matches_left_fold_oracle(dtype: torch.dtype) -> None:
    device = _accelerator_device()
    vocab_size, hidden = 37, 64
    generator = torch.Generator().manual_seed(415)
    token_ids = torch.randint(0, vocab_size, (4, 16), generator=generator)
    token_ids[1, :8] = 7
    grad_output = torch.randn(4, 16, hidden, generator=generator).to(dtype)
    weight = torch.randn(vocab_size, hidden, generator=generator).to(device=device, dtype=dtype)
    weight.requires_grad_(True)

    output = embedding_module.TritonEmbeddingOp()(token_ids.to(device), weight)
    (grad_weight,) = torch.autograd.grad(output, weight, grad_outputs=grad_output.to(device))

    expected = _left_fold_grad_weight(
        token_ids, grad_output, vocab_size=vocab_size, output_dtype=dtype
    )
    assert torch.equal(grad_weight.cpu(), expected)
