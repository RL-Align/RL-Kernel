import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape,seq,batch",
    [
        ((2, 4, 513, 16), 2, 0),
        ((2, 513, 4, 16), 1, 0),
        ((513, 2, 4, 16), 0, 1),
        ((513, 2, 4), 0, 1),
    ],
)
def test_permutation_forward_backward_matches_deterministic_gather(dtype, shape, seq, batch):
    from rl_engine.backends.cuda.attention.permutation import permute_sequence

    torch.manual_seed(7)
    x = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    order = torch.stack([torch.randperm(513, device="cuda") for _ in range(2)])
    index_shape = [1] * len(shape)
    index_shape[seq], index_shape[batch] = 513, 2
    index = (order if batch < seq else order.t()).reshape(index_shape).expand(shape)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = torch.gather(reference, seq, index)
        actual = permute_sequence(x, order, sequence_dim=seq, batch_dim=batch)
        grad = torch.randn_like(expected)
        grad.reshape(-1)[:4] = torch.tensor([-0.0, 0.0, 1e-30, -1e-30], device="cuda", dtype=dtype)
        expected.backward(grad)
        actual.backward(grad)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(x.grad.view(torch.uint8), reference.grad.view(torch.uint8))
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("rows,width", [(1, 7), (3, 513), (32, 151936)])
def test_unique_boolean_scatters_preserve_complete_support(rows, width):
    from rl_engine.backends.cuda.sampling.unique_scatter import (
        copy_unique_rows_,
        scatter_permuted_columns,
    )

    torch.manual_seed(rows)
    scores = torch.randn(rows, width, device="cuda")
    ids = scores.argsort(dim=1)
    values = scores > 0
    expected = torch.zeros_like(values).scatter_(1, ids, values)
    assert torch.equal(scatter_permuted_columns(values, ids), expected)
    indices = torch.arange(0, rows * 2, 2, device="cuda")
    output = torch.ones((rows * 2, width), device="cuda", dtype=torch.bool)
    reference = output.clone()
    reference.index_copy_(0, indices, expected)
    copy_unique_rows_(output, indices, expected)
    assert torch.equal(output, reference)
