# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import pytest
import torch

from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.ops.embedding.canonical_embedding import canonical_embedding


def _run_permutation(order: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    base_ids = torch.tensor([2, 2, 2, 1], dtype=torch.long)
    base_grads = torch.tensor([[1.0e20], [-1.0e20], [3.0], [7.0]], dtype=torch.float32)
    base_keys = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.long)
    permutation = torch.tensor(order, dtype=torch.long)
    ids = base_ids.index_select(0, permutation)
    grads = base_grads.index_select(0, permutation)
    keys = base_keys.index_select(0, permutation)
    weight = torch.zeros((4, 1), dtype=torch.float32, requires_grad=True)

    with canonical_backward_session() as session:
        output = canonical_embedding(
            ids,
            weight,
            keys,
            forward_op=lambda token_ids, table: table[token_ids],
            family="cuda-sm90",
        )
        output.backward(grads)
        session.validate_complete()

    assert weight.grad is not None
    return output.detach(), weight.grad.detach()


def test_canonical_embedding_gradient_is_invariant_to_physical_permutation() -> None:
    direct_output, direct_grad = _run_permutation([0, 1, 2, 3])
    permuted_output, permuted_grad = _run_permutation([0, 2, 1, 3])

    assert torch.equal(direct_output[[0, 2, 1, 3]], permuted_output)
    assert torch.equal(direct_grad, permuted_grad)
    assert direct_grad[2, 0].item() == 3.0


def test_canonical_embedding_excludes_inactive_logical_rows() -> None:
    ids = torch.tensor([2, 2], dtype=torch.long)
    keys = torch.tensor([[0, 0], [-1, -1]], dtype=torch.long)
    weight = torch.zeros((4, 1), dtype=torch.float32, requires_grad=True)

    with canonical_backward_session() as session:
        output = canonical_embedding(
            ids,
            weight,
            keys,
            forward_op=lambda token_ids, table: table[token_ids],
            family="cuda",
        )
        output.backward(torch.tensor([[5.0], [100.0]]))
        session.validate_complete()

    assert weight.grad is not None
    assert weight.grad[2, 0].item() == 5.0


def _run_chunked_uses(chunks: list[list[int]]) -> torch.Tensor:
    base_ids = torch.tensor([2, 2, 2, 1], dtype=torch.long)
    base_grads = torch.tensor([[1.0e20], [-1.0e20], [3.0], [7.0]], dtype=torch.float32)
    base_keys = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.long)
    weight = torch.zeros((4, 1), dtype=torch.float32, requires_grad=True)

    outputs: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    with canonical_backward_session() as session:
        for indices in chunks:
            selection = torch.tensor(indices, dtype=torch.long)
            outputs.append(
                canonical_embedding(
                    base_ids.index_select(0, selection),
                    weight,
                    base_keys.index_select(0, selection),
                    forward_op=lambda token_ids, table: table[token_ids],
                    family="cuda",
                )
            )
            gradients.append(base_grads.index_select(0, selection))
        torch.autograd.backward(outputs, gradients)
        session.validate_complete()

    assert weight.grad is not None
    return weight.grad.detach()


def test_canonical_embedding_combines_multiple_uses_in_logical_order() -> None:
    single_use = _run_chunked_uses([[0, 1, 2, 3]])
    reversed_chunks = _run_chunked_uses([[2, 3], [0, 1]])

    assert torch.equal(single_use, reversed_chunks)
    assert single_use[2, 0].item() == 3.0


def test_canonical_embedding_rejects_unknown_backward_family() -> None:
    ids = torch.tensor([1], dtype=torch.long)
    keys = torch.tensor([[0, 0]], dtype=torch.long)
    weight = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)

    with canonical_backward_session():
        with pytest.raises(RuntimeError, match="unsupported canonical embedding backend"):
            canonical_embedding(
                ids,
                weight,
                keys,
                forward_op=lambda token_ids, table: table[token_ids],
                family="unknown",
            )


def test_canonical_embedding_rejects_misaligned_logical_keys() -> None:
    ids = torch.tensor([0, 1], dtype=torch.long)
    weight = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)

    with canonical_backward_session():
        with pytest.raises(ValueError, match="logical keys"):
            canonical_embedding(
                ids,
                weight,
                torch.tensor([0, 1], dtype=torch.long),
                forward_op=lambda token_ids, table: table[token_ids],
                family="pytorch",
            )
