from types import SimpleNamespace

import pytest
import torch

from rl_engine.distributed.algorithms.canonical_cp import (
    CPRMSNorm,
    packed_gather_indices,
    weight_gradient,
)
from rl_engine.ops.gemm import det_gemm


def test_checkpoint_recompute_retains_forward_layout():
    from torch.utils.checkpoint import checkpoint

    from rl_engine.distributed.algorithms.canonical_cp import bind_layout, current_layout

    layout = object()
    seen = []

    def function(x):
        seen.append((torch.is_grad_enabled(), current_layout()))
        return x.square()

    x = torch.ones(3, requires_grad=True)
    output = checkpoint(bind_layout(function, layout), x, use_reentrant=True)
    assert current_layout() is None
    output.sum().backward()
    assert seen == [(False, layout), (True, layout)]
    assert current_layout() is None
    assert torch.equal(x.grad, torch.full_like(x, 2))


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
def test_packed_rows_restore_token_order_without_padding(cp):
    sequences = [torch.arange(13), torch.arange(101, 128), torch.arange(200, 218)]
    ranks = []
    for rank in range(cp):
        local = []
        for seq in sequences:
            if cp == 1:
                local.append(seq)
            else:
                width = (seq.numel() + 2 * cp - 1) // (2 * cp)
                padded = torch.nn.functional.pad(seq, (0, 2 * cp * width - seq.numel()), value=-1)
                pieces = padded.chunk(2 * cp)
                local.extend([pieces[rank], pieces[2 * cp - rank - 1]])
        ranks.append(torch.nn.functional.pad(torch.cat(local), (0, 7), value=-2))
    order = packed_gather_indices([len(s) for s in sequences], len(ranks[0]), cp)
    assert torch.equal(torch.cat(ranks)[order], torch.cat(sequences))


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_canonical_parameter_gradient_cancels_cp_loss_scale(monkeypatch, cp, chunks):
    monkeypatch.setattr(det_gemm, "det_gemm_linear_weight_gradient", lambda x, dy: dy.t() @ x)
    gen = torch.Generator().manual_seed(18)
    x = torch.randn(21, 16, generator=gen).bfloat16()
    dy = torch.randn(21, 32, generator=gen).bfloat16()
    layout = SimpleNamespace(cp_world=cp, cp_rank=0, gather_many=lambda *args: (x, dy))
    actual = weight_gradient(x[:1], dy[:1], layout, chunks)
    assert torch.equal(actual / cp, dy.t() @ x)


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
def test_norm_parameter_gradient_uses_all_logical_rows(cp):
    gen = torch.Generator().manual_seed(2)
    full_x = torch.randn(16, 1, 32, generator=gen).bfloat16()
    full_dy = torch.randn(16, 1, 32, generator=gen).bfloat16()
    w = torch.randn(32, generator=gen).bfloat16().requires_grad_()
    reference_w = w.detach().clone().requires_grad_()
    torch.nn.functional.rms_norm(full_x, (32,), reference_w, 1e-6).backward(full_dy)
    layout = SimpleNamespace(cp_world=cp, cp_rank=0, gather_many=lambda *args: (full_x, full_dy))
    x = full_x[: 16 // cp].detach().requires_grad_()
    CPRMSNorm.apply(x, w, 1e-6, layout, False).backward(full_dy[: 16 // cp])
    assert torch.equal(w.grad / cp, reference_w.grad)


@pytest.mark.parametrize("replicas", [1, 2, 4, 8])
def test_single_contributor_eliminates_ring_average_rounding(replicas):
    from rl_engine.distributed.algorithms.canonical_cp import replica_parameter_gradient

    gradient = torch.randn(10003, generator=torch.Generator().manual_seed(29))
    average = torch.zeros_like(gradient)
    replicated_average = torch.zeros_like(gradient)
    for rank in range(replicas):
        contribution = replica_parameter_gradient(gradient.clone(), replicas, rank)
        average.add_(contribution / replicas)
        replicated_average.add_(gradient / replicas)
    assert torch.equal(average.view(torch.int32), gradient.view(torch.int32))
    if replicas == 8:
        assert not torch.equal(replicated_average, gradient)


def test_non_owner_norm_keeps_activation_gradients():
    x = torch.randn(4, 1, 16, requires_grad=True)
    weight = torch.ones(16, requires_grad=True)
    dy = torch.randn_like(x)
    local = x.detach().clone().requires_grad_()
    torch.nn.functional.rms_norm(local, (16,), weight.detach(), 1e-6).backward(dy)
    layout = SimpleNamespace(cp_world=2, cp_rank=1, gather_many=lambda *args: (x.detach(), dy))
    CPRMSNorm.apply(x, weight, 1e-6, layout, False).backward(dy)
    assert torch.equal(x.grad, local.grad)
    assert torch.count_nonzero(weight.grad) == 0


def test_ffn_replica_wrapper_preserves_exact_ring_average():
    from rl_engine.distributed.algorithms.canonical_cp import bind_layout
    from rl_engine.integrations.engines.train.megatron.operators import _MegatronCPWeightGradient

    gradient = torch.randn(10003, generator=torch.Generator().manual_seed(29))
    result = torch.zeros_like(gradient)
    for rank in range(8):
        weight = torch.zeros_like(gradient, requires_grad=True)
        layout = SimpleNamespace(cp_rank=rank)
        output = bind_layout(lambda w: _MegatronCPWeightGradient.apply(w, 8), layout)(weight)
        # Canonical loss removes the CP multiplier before model backward.
        output.backward(gradient)
        result.add_(weight.grad / 8)
    assert torch.equal(result.view(torch.int32), gradient.view(torch.int32))


@pytest.mark.parametrize("cp", [1, 2, 4, 8])
def test_loss_preserves_forward_value_and_unscales_model_derivatives(cp):
    from rl_engine.distributed.algorithms.canonical_cp import CanonicalLossGradient

    x = torch.tensor(0.375, requires_grad=True)
    scaled = x.square() * cp
    actual = CanonicalLossGradient.apply(scaled, cp)
    assert torch.equal(actual, scaled)
    actual.mul_(2)
    actual.backward()
    assert x.grad.item() == 1.5


def test_shared_head_norm_does_not_downscale_subnormal_gradients(monkeypatch):
    x = torch.ones(4, 1, 4, 8)
    dy = torch.full_like(x, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).item() * 3)
    reference = torch.ones(8, requires_grad=True)
    torch.nn.functional.rms_norm(x, (8,), reference, 1e-6).backward(dy)
    total = torch.zeros_like(reference)
    for rank in range(2):
        call = [0]

        def gather(outputs, value, group):
            full = x if call[0] == 0 else dy
            call[0] += 1
            for output, part in zip(outputs, full.chunk(2, dim=-2)):
                output.copy_(part)

        monkeypatch.setattr(torch.distributed, "all_gather", gather)
        local_x = x.chunk(2, dim=-2)[rank].contiguous().requires_grad_()
        local_dy = dy.chunk(2, dim=-2)[rank].contiguous()
        weight = torch.ones(8, requires_grad=True)
        layout = SimpleNamespace(
            cp_world=1,
            cp_rank=0,
            tp_world=2,
            tp_rank=rank,
            tp_group=None,
            gather_many=lambda *values: values,
        )
        CPRMSNorm.apply(local_x, weight, 1e-6, layout, True).backward(local_dy)
        total.add_(weight.grad)
    assert torch.count_nonzero(reference.grad) == 8
    assert torch.equal(total.view(torch.int32), reference.grad.view(torch.int32))
