# SPDX-License-Identifier: Apache-2.0
"""INK-21 shared-weight gradient contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.shared_weight_grad import accumulate_shared_weight_gradients


def test_three_loop_visits_accumulate_in_declared_order():
    gradients = {0: torch.full((2, 3), 1.0), 1: torch.full((2, 3), 2.0), 2: torch.full((2, 3), 3.0)}
    output = accumulate_shared_weight_gradients(gradients)
    assert torch.equal(output, torch.full((2, 3), 6.0))


def test_reduce_scatter_keeps_the_declared_local_shard():
    gradients = {
        0: torch.arange(12, dtype=torch.float32).reshape(4, 3),
        1: torch.ones(4, 3),
        2: torch.ones(4, 3),
    }
    output = accumulate_shared_weight_gradients(gradients, shard=(1, 3))
    expected = gradients[0][1:3] + 2
    assert torch.equal(output, expected)


def test_gradient_accumulation_remains_differentiable():
    gradients = {visit: torch.randn(3, 4, requires_grad=True) for visit in range(3)}
    accumulate_shared_weight_gradients(gradients).square().mean().backward()
    for gradient in gradients.values():
        assert gradient.grad is not None and torch.isfinite(gradient.grad).all()


def test_missing_visit_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        accumulate_shared_weight_gradients({0: torch.ones(2, 2)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_shared_weight_gradient_matches_reference_and_backward():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    from rl_engine.models.inkling2.shared_weight_grad import accumulate_shared_weight_gradients_cuda

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_shared_weight_grad_forward"):
        pytest.skip("Inkling 2 shared-weight CUDA extension is unavailable")
    torch.manual_seed(23)
    gradients = {
        visit: torch.randn(4, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for visit in range(3)
    }
    native = accumulate_shared_weight_gradients_cuda(gradients, shard=(1, 3))
    reference = accumulate_shared_weight_gradients(gradients, shard=(1, 3))
    torch.testing.assert_close(native, reference, atol=2e-2, rtol=0)
    native.float().square().mean().backward()
    assert all(gradient.grad is not None for gradient in gradients.values())
