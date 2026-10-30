# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Mandatory shared CUDA GEMM and MoE checks; these do not claim model L2."""

import pytest
import torch
import torch.nn.functional as F

from rl_engine.models.qwen3_next.qwen3_next_forward import (
    combine_routes,
    shared_linear,
    shared_moe,
    stable_top10_routes,
)
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def random(shape, seed=801, *, grad=False, dtype=torch.bfloat16):
    assert torch.cuda.is_available(), "Mandatory shared-forward CUDA gates cannot skip"
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(shape, generator=gen, device="cuda", dtype=dtype)
        .mul_(0.02)
        .requires_grad_(grad)
    )


@pytest.mark.parametrize(
    "k,n", [(2048, 3072), (2048, 16), (1024, 2048), (2048, 256), (128, 2048), (2048, 37984)]
)
@torch.no_grad()
def test_qwen_linear_shapes_batch_chunk_and_reorder(k, n):
    x, weight = random((17, k)), random((n, k), seed=802)
    output = shared_linear(x, weight)
    exact(
        torch.cat(
            [
                shared_linear(x[:1], weight),
                shared_linear(x[1:4], weight),
                shared_linear(x[4:], weight),
            ]
        ),
        output,
    )
    order = torch.arange(16, -1, -1, device="cuda")
    exact(shared_linear(x[order], weight), output[order])


def test_shared_linear_vjp_and_empty_tokens():
    x, weight, bias = (
        random((4, 2048), grad=True),
        random((16, 2048), 802, grad=True),
        random((16,), 803, grad=True),
    )
    output = shared_linear(x, weight, bias)
    grad = random(output.shape, 804)
    output.backward(grad)
    refs = [value.detach().double().requires_grad_() for value in (x, weight, bias)]
    F.linear(*refs).backward(grad.double())
    for value, ref in zip((x, weight, bias), refs):
        torch.testing.assert_close(value.grad.float(), ref.grad.float(), atol=0.002, rtol=0.016)
        assert (value.grad.double() - ref.grad).norm() / ref.grad.norm() < 0.016
    with torch.no_grad():
        exact(output, shared_linear(x, weight, bias))
    empty = random((0, 2048), grad=True)
    weight.grad = bias.grad = None
    shared_linear(empty, weight, bias).sum().backward()
    assert empty.grad.shape == empty.shape
    assert torch.count_nonzero(weight.grad) == 0 and torch.count_nonzero(bias.grad) == 0


def test_stable_route_tie_boundary_and_router_gradient():
    logits = random((4, 512), dtype=torch.float32)
    logits.zero_().requires_grad_()
    routes = stable_top10_routes(logits)
    assert torch.equal(routes.indices, torch.arange(10, device="cuda").expand(4, 10))
    exact(routes.weights, torch.full((4, 10), 0.1, device="cuda"))
    (routes.weights * torch.arange(10, device="cuda")).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().max() > 0
    bad = logits.detach().clone()
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        stable_top10_routes(bad)


def test_combine_has_fixed_route_order_and_differentiable_weights():
    outputs = random((4, 10, 2048), grad=True)
    weights = random((4, 10), 802, dtype=torch.float32, grad=True)
    actual = combine_routes(outputs, weights)
    exact(
        torch.cat([combine_routes(outputs[i : i + 1], weights[i : i + 1]) for i in range(4)]),
        actual,
    )
    actual.float().square().sum().backward()
    assert weights.grad.abs().max() > 0 and outputs.grad.abs().max() > 0
    refs = [value.detach().double().requires_grad_() for value in (outputs, weights)]
    expected = (refs[0] * refs[1].unsqueeze(-1)).sum(dim=1)
    expected.backward((2 * actual.float()).double())
    for value, reference in zip((outputs, weights), refs):
        torch.testing.assert_close(
            value.grad.float(), reference.grad.float(), atol=0.002, rtol=0.016
        )


def test_qwen_moe_actual_tp4_shapes_and_independent_training_forward():
    x = random((4, 2048), grad=True)
    router = random((512, 2048), 802, grad=True)
    gate_up = random((512, 256, 2048), 803, grad=True)
    down = random((512, 2048, 128), 804, grad=True)
    output, routes = shared_moe(x, router, gate_up, down)
    with torch.no_grad():
        row_output, row_routes = shared_moe(x[:1], router, gate_up, down)
        exact(row_output, output[:1])
        exact(row_routes.weights, routes.weights[:1])
        assert torch.equal(row_routes.indices, routes.indices[:1])
    output.float().square().sum().backward()
    for value in (x, router, gate_up, down):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0
    active = routes.indices.unique()
    inactive = torch.ones(512, dtype=torch.bool, device="cuda")
    inactive[active] = False
    assert torch.count_nonzero(gate_up.grad[inactive]) == 0
    assert torch.count_nonzero(down.grad[inactive]) == 0


def test_moe_vjp_matches_independent_double_formula():
    # A small hidden width isolates the derivative; the previous gate covers
    # the unmodified official TP4 widths and expert count.
    x = random((2, 32)).mul_(10).requires_grad_()
    router = random((512, 32), 802).mul_(5).requires_grad_()
    gate_up = random((512, 16, 32), 803).mul_(10).requires_grad_()
    down = random((512, 32, 8), 804).mul_(10).requires_grad_()
    actual, routes = shared_moe(x, router, gate_up, down)
    grad = random(actual.shape, 805)
    actual.backward(grad)
    xr, rr, gu, dr = [v.detach().double().requires_grad_() for v in (x, router, gate_up, down)]
    probability = (xr @ rr.t()).softmax(-1)
    selected = probability.gather(1, routes.indices)
    selected = selected / selected.sum(-1, keepdim=True)
    rows = []
    for token in range(2):
        slots = []
        for slot, expert in enumerate(routes.indices[token].tolist()):
            gate, up = F.linear(xr[token], gu[expert]).chunk(2)
            slots.append(F.linear(F.silu(gate) * up, dr[expert]) * selected[token, slot])
        rows.append(torch.stack(slots).sum(0))
    torch.stack(rows).backward(grad.double())
    for value, reference in zip((x, router, gate_up, down), (xr, rr, gu, dr)):
        assert torch.isfinite(value.grad).all()
        # BF16 intermediates are rounded by the actual provider; this compares
        # derivative correctness, never the raw-bit L1/L2 acceptance criterion.
        torch.testing.assert_close(
            value.grad.float(), reference.grad.float(), atol=0.00002, rtol=0.016
        )
        assert (value.grad.double() - reference.grad).norm() / reference.grad.norm() < 0.016


@torch.no_grad()
def test_route_weights_are_batch_invariant():
    logits = random((32, 512), 805, dtype=torch.float32).mul_(100)
    whole = stable_top10_routes(logits)
    for count in (1, 8, 17):
        part = stable_top10_routes(logits[:count])
        assert torch.equal(part.indices, whole.indices[:count])
        exact(part.weights, whole.weights[:count])
    moe_x, router = random((32, 2048), 806), random((512, 2048), 807)
    gate_up, down = random((512, 256, 2048), 808), random((512, 2048, 128), 809)
    whole_out, _ = shared_moe(moe_x, router, gate_up, down)
    part_out, _ = shared_moe(moe_x[:8], router, gate_up, down)
    exact(part_out, whole_out[:8])


class _PerExpertReference(torch.autograd.Function):
    """The previous provider: one pinned vLLM GEMM pair per expert, in expert order."""

    @staticmethod
    def forward(ctx, x, gate_up, down, route_indices):
        from rl_engine.models.qwen3_next.qwen3_next_forward import _vllm_linear

        output = x.new_empty((x.shape[0] * 10, x.shape[1]))
        flat = route_indices.flatten()
        for expert in flat.unique().tolist():
            slots = (flat == expert).nonzero().squeeze(1)
            rows = x.index_select(0, slots // 10)
            gate, up = _vllm_linear()(rows, gate_up[expert]).chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * up.float()).to(x.dtype)
            output.index_copy_(0, slots, _vllm_linear()(activated, down[expert]))
        return output.reshape(x.shape[0], 10, x.shape[1])


@torch.no_grad()
@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("tokens", [1, 7, 13, 300])
def test_grouped_experts_are_bitwise_the_per_expert_gemms(width, tokens):
    from rl_engine.models.qwen3_next.qwen3_next_forward import _RoutedExperts, shared_router

    x, router = random((tokens, 2048), 811), random((512, 2048), 812)
    gate_up, down = random((512, 2 * width, 2048), 813), random((512, 2048, width), 814)
    indices = stable_top10_routes(shared_router(x, router)).indices
    exact(
        _RoutedExperts.apply(x, gate_up, down, indices),
        _PerExpertReference.apply(x, gate_up, down, indices),
    )


@torch.no_grad()
@pytest.mark.parametrize("width", [128, 512])
def test_grouped_projection_is_bitwise_the_per_expert_gemm(width):
    # The backward recomputes [gate, up] with the grouped kernel; equality here
    # leaves every gradient bitwise what the per-expert backward produced.
    from rl_engine.models.qwen3_next.qwen3_next_forward import (
        _vllm_linear,
        grouped_route_linear,
        shared_router,
    )

    x, router = random((300, 2048), 817), random((512, 2048), 818)
    gate_up = random((512, 2 * width, 2048), 819)
    indices = stable_top10_routes(shared_router(x, router)).indices
    grouped = grouped_route_linear(x, gate_up, indices).reshape(-1, 2 * width)
    flat = indices.flatten()
    for expert in flat.unique().tolist():
        slots = (flat == expert).nonzero().squeeze(1)
        exact(grouped[slots], _vllm_linear()(x[slots // 10], gate_up[expert]), name=f"e{expert}")


def test_grouped_experts_fail_closed_on_the_tensor_descriptor_path(monkeypatch):
    from rl_engine.models.qwen3_next import qwen3_next_forward as provider

    prepare, dispatch, _ = provider._vllm_grouped()
    monkeypatch.setattr(provider, "_vllm_grouped", lambda: (prepare, dispatch, lambda: True))
    x = random((4, 2048), 815)
    with pytest.raises(RuntimeError, match="VLLM_TRITON_USE_TD"):
        provider.grouped_route_linear(
            x, random((512, 256, 2048), 816), torch.zeros(4, 10, device="cuda", dtype=torch.long)
        )
