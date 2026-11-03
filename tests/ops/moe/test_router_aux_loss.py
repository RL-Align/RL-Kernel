# SPDX-License-Identifier: Apache-2.0
"""Normal routing aux loss: independent formula/VJP, overflow and invariance."""

from dataclasses import replace

import pytest
import torch

from rl_engine.ops.moe.learned_route import prepare_capacity_domain, prepare_learned_route_plan
from rl_engine.ops.moe.router_aux_loss import (
    AuxLossConfig,
    prepare_router_aux_context,
    router_aux_loss,
)
from rl_engine.reference.moe.router_aux_loss import router_aux_loss_fp64

gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
)
VERSIONS = dict(
    absolute_layer=3, model_version="m1", bias_version="b1", domain_version="d1", plan_version="p1"
)


def bits(value):
    return value.detach().contiguous().view(torch.int32)


def setup(x, ids, bias, capacity=2, scale=1.5, alpha=0.01, global_ids=None):
    domain = prepare_capacity_domain(
        ids, capacity, absolute_layer=3, model_version="m1", domain_version="d1"
    )
    plan = prepare_learned_route_plan(
        x,
        ids,
        bias,
        domain,
        bias_version="b1",
        plan_version="p1",
        routed_scaling_factor=scale,
        global_expert_ids=global_ids,
    )
    config = AuxLossConfig(True, alpha, "aux.v1")
    context = prepare_router_aux_context(plan, config=config, version="ctx.v1")
    return plan, config, context


def loss(x, ids, bias, plan, config, context, **kwargs):
    return router_aux_loss(x, ids, bias, plan, config=config, context=context, **VERSIONS, **kwargs)


def test_disabled_default_has_no_kernel_collective_or_backward_branch(monkeypatch):
    import rl_engine.ops.moe.router_aux_loss as op

    def forbidden(*args, **kwargs):
        pytest.fail("disabled auxiliary loss must not call a numerical dependency")

    monkeypatch.setattr(op, "implementation", forbidden)
    monkeypatch.setattr(op, "learned_route", forbidden)
    result = router_aux_loss(None, None, None, None)
    assert result.loss is None and result.routes == ()
    assert result.execution["enabled"] is False and result.execution["collective_called"] is False
    assert result.execution["config_version"] == "router_aux.disabled.v1"


@pytest.mark.parametrize(
    "config",
    [
        AuxLossConfig(1, 0.1, "a"),
        AuxLossConfig(True, None, "a"),
        AuxLossConfig(True, -1, "a"),
        AuxLossConfig(True, float("nan"), "a"),
        AuxLossConfig(True, 1e100, "a"),
        AuxLossConfig(True, 1e-100, "a"),
        AuxLossConfig(True, 0.1, ""),
    ],
)
def test_invalid_config_rejected_without_gpu(config):
    with pytest.raises((TypeError, ValueError)):
        config.validate()


@gpu
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("capacity", [None, 0, 2])
def test_normal_discrete_formula_and_gradient_independent_fp64(seed, capacity):
    gen = torch.Generator().manual_seed(340 + seed)
    x = (torch.randn(17, 256, generator=gen) * 5).cuda().requires_grad_()
    ids = torch.randperm(17, generator=gen).long() + 100
    bias = (torch.randn(256, generator=gen) * 0.4).cuda().requires_grad_()
    plan, config, context = setup(x, ids, bias, capacity)
    result = loss(x, ids, bias, plan, config, context)
    oracle_x = x.detach().cpu().double().requires_grad_()
    ref, experts, columns, keep, positions, counts, masses = router_aux_loss_fp64(
        oracle_x, ids, bias.cpu(), plan.domain.capacities, config.validate()
    )
    assert torch.equal(result.expert_ids.cpu(), experts)
    assert torch.equal(result.counts.cpu(), counts) and int(result.counts.sum()) == 6 * 17
    assert torch.equal(result.routes[0].surviving_mask.cpu(), keep)
    assert torch.equal(result.routes[0].dispatch_positions.cpu(), positions)
    torch.testing.assert_close(
        result.loss.detach().cpu().double(), ref.detach(), rtol=3e-6, atol=2e-8
    )
    torch.testing.assert_close(result.masses.cpu().double(), masses.detach(), rtol=2e-6, atol=1e-7)
    dx, db = torch.autograd.grad(result.loss, (x, bias), allow_unused=True)
    mathdx = torch.autograd.grad(ref, oracle_x)[0]
    torch.testing.assert_close(dx.cpu().double(), mathdx, rtol=5e-4, atol=2e-8)
    assert db is None
    selected = torch.zeros_like(x, dtype=torch.bool).scatter_(1, columns.cuda(), keep.cuda())
    assert dx[~selected].count_nonzero() == 0
    if capacity == 0:
        assert result.loss == 0 and dx.count_nonzero() == 0


@gpu
def test_overflow_denominator_gradient_and_pre_capacity_frequency_negative_controls():
    x = torch.full((3, 256), -10.0, device="cuda")
    x[0, :6] = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], device="cuda")
    x[1, :5] = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0], device="cuda")
    x[1, 6] = 1
    x[2, :4] = torch.tensor([6.0, 5.0, 4.0, 3.0], device="cuda")
    x[2, 7:9] = 2
    x.requires_grad_()
    ids = torch.tensor([10, 20, 30])
    bias = torch.zeros(256, device="cuda")
    capacities = torch.full((256,), 3, dtype=torch.int64)
    capacities[0] = 1
    plan, config, context = setup(x, ids, bias, capacities, alpha=1.0)
    result = loss(x, ids, bias, plan, config, context)
    dx = torch.autograd.grad(result.loss, x)[0]
    assert result.expert_ids[1].tolist() == [0, 1, 2, 3, 4, 6]
    assert not result.routes[0].surviving_mask[1, 0] and dx[1, 0] == 0
    # A positive-frequency expert with no other effect still participates in
    # the denominator, so its derivative can be negative.
    assert dx[1, 6] < 0 and dx[1, 1] > 0
    accepted = torch.zeros(256, device="cuda")
    for row in range(3):
        for slot in range(6):
            if result.routes[0].surviving_mask[row, slot]:
                accepted[result.expert_ids[row, slot]] += 1
    wrong_frequency = accepted / (6 * 3)
    wrong_loss = 256 * ((wrong_frequency) * (result.masses / 3)).sum()
    assert abs(float((result.loss - wrong_loss).detach())) > 0.01
    # An independent, deliberately detached denominator misses cross-slot
    # terms. A tolerance check on the scalar alone would not catch this bug.
    from rl_engine.reference.moe.learned_route import score_fp64

    oracle = x.detach().cpu().double().requires_grad_()
    chosen = result.expert_ids.cpu()
    mask = result.routes[0].surviving_mask.cpu()
    selected = score_fp64(oracle).gather(1, chosen)
    survived = torch.where(mask, selected, 0.0)
    wrong_probabilities = survived / survived.sum(1, keepdim=True).clamp_min(1e-20).detach()
    frequency = result.counts.cpu().double() / (6 * 3)
    wrong_objective = 256 * (wrong_probabilities * frequency[chosen]).sum() / 3
    detached_dx = torch.autograd.grad(wrong_objective, oracle)[0]
    assert detached_dx[1, 6] > 0 and dx[1, 6] < 0
    for row, col in [(1, 0), (1, 1), (1, 6), (1, 200)]:
        observed = []
        for sign in (1, -1):
            changed = x.detach().clone()
            changed[row, col] += sign * 0.005
            p, c, ctx = setup(changed, ids, bias, capacities, alpha=1.0)
            current = loss(changed, ids, bias, p, c, ctx)
            assert torch.equal(current.expert_ids, result.expert_ids)
            observed.append(float(current.loss))
        assert float(dx[row, col]) == pytest.approx(
            (observed[0] - observed[1]) / 0.01, rel=0.01, abs=0.001
        )


@gpu
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5, 2.0])
def test_loss_and_gradient_exclude_route_scale_even_zero(scale):
    gen = torch.Generator().manual_seed(341)
    x = torch.randn(17, 256, generator=gen).cuda().requires_grad_()
    ids = torch.arange(17)
    bias = torch.zeros(256, device="cuda")
    p, c, ctx = setup(x, ids, bias, 2, scale=scale)
    result = loss(x, ids, bias, p, c, ctx)
    dx = torch.autograd.grad(result.loss, x)[0]
    p2, c2, ctx2 = setup(x, ids, bias, 2, scale=1.0)
    reference = loss(x, ids, bias, p2, c2, ctx2)
    reference_dx = torch.autograd.grad(reference.loss, x)[0]
    assert torch.equal(bits(result.loss), bits(reference.loss)) and torch.equal(
        bits(dx), bits(reference_dx)
    )
    if scale == 0:
        assert result.routes[0].weights.count_nonzero() == 0 and dx.count_nonzero() > 0


@gpu
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "padded"])
def test_microbatch_domain_reduction_order_permutation_and_padding(layout):
    gen = torch.Generator().manual_seed(342)
    x = torch.randn(65, 256, generator=gen).cuda().requires_grad_()
    ids = torch.randperm(65, generator=gen).long() + 100
    bias = torch.zeros(256, device="cuda")
    p, c, ctx = setup(x, ids, bias, 2)
    baseline = loss(x, ids, bias, p, c, ctx)
    baseline_dx = torch.autograd.grad(baseline.loss, x)[0]
    for m in (1, 16, 17, 32, 65):
        order = torch.randperm(65, generator=gen)
        local = x.detach()[order.cuda()].clone()
        if layout == "transpose":
            local = local.T.contiguous().T
        if layout == "padded":
            storage = torch.empty(65, 515, device="cuda")
            view = storage[:, 1:513:2]
            view.copy_(local)
            local = view
        local.requires_grad_()
        current = loss(local, ids[order], bias, p, c, ctx, microbatch_size=m)
        dx = torch.autograd.grad(current.loss, local)[0]
        assert torch.equal(bits(current.loss), bits(baseline.loss))
        assert torch.equal(bits(dx), bits(baseline_dx[order.cuda()]))
    # Invalid padding rows are excluded from N, routing, and gradient flow.
    padded = torch.cat(
        (x.detach(), torch.full((3, 256), float("nan"), device="cuda"))
    ).requires_grad_()
    mask = torch.tensor([True] * 65 + [False] * 3)
    current = loss(
        padded,
        torch.cat((ids, torch.tensor([-1, -1, -1]))),
        bias,
        p,
        c,
        ctx,
        valid_mask=mask,
        microbatch_size=17,
    )
    dx = torch.autograd.grad(current.loss, padded)[0]
    assert torch.equal(bits(current.loss), bits(baseline.loss))
    assert torch.equal(bits(dx[:65]), bits(baseline_dx)) and dx[65:].count_nonzero() == 0
    assert (
        current.execution["count_denominator"] == 6 * 65
        and current.execution["probability_denominator"] == 65
    )


@gpu
def test_empty_domain_zero_alpha_and_nonfinite_upstream():
    ids = torch.empty(0, dtype=torch.int64)
    bias = torch.zeros(256, device="cuda")
    x = torch.empty(0, 256, device="cuda", requires_grad=True)
    p, c, ctx = setup(x, ids, bias, None)
    result = loss(x, ids, bias, p, c, ctx)
    dx = torch.autograd.grad(result.loss, x)[0]
    assert result.loss == 0 and dx.shape == (0, 256) and not result.execution["acceptance_sample"]
    x = torch.randn(2, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    p, c, ctx = setup(x, ids, bias, None, alpha=0.0)
    result = loss(x, ids, bias, p, c, ctx)
    assert result.loss == 0 and torch.autograd.grad(result.loss, x)[0].count_nonzero() == 0
    p, c, ctx = setup(x, ids, bias, None)
    result = loss(x, ids, bias, p, c, ctx)
    with pytest.raises(ValueError, match="finite auxiliary"):
        result.loss.backward(torch.tensor(float("nan"), device="cuda"))


@gpu
def test_context_versions_ownership_mutation_and_graph_fail_closed():
    x = torch.randn(3, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20, 30])
    bias = torch.zeros(256, device="cuda")
    p, c, ctx = setup(x, ids, bias, None)
    for owned in [((10, 20),), ((10, 20, 20),), ((10, 20, 31),)]:
        with pytest.raises(ValueError, match="cover the complete"):
            prepare_router_aux_context(p, config=c, ownership=owned, version="ctx.v1")
    with pytest.raises(ValueError, match="immutable"):
        loss(x, ids, bias, p, c, replace(ctx, version="other"))
    with pytest.raises(ValueError, match="configuration/plan"):
        loss(x, ids, bias, p, replace(c, version="other"), ctx)
    with pytest.raises(ValueError, match="all locally owned"):
        loss(x[:2], ids[:2], bias, p, c, ctx)
    with pytest.raises(ValueError, match="microbatch size"):
        loss(x, ids, bias, p, c, ctx, microbatch_size=True)
    run = loss(x, ids, bias, p, c, ctx)
    run.probabilities.data[0, 0] += 1
    with pytest.raises(RuntimeError, match="state changed"):
        run.loss.backward()
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            loss(x, ids, bias, p, c, ctx)


@gpu
@pytest.mark.parametrize("value", [-300.0, -120.0, -94.0, -16.0, 1000.0])
def test_extreme_tails_repeated_vjp_and_higher_order_rejection(value):
    x = torch.full((2, 256), value, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    bias = torch.zeros(256, device="cuda")
    p, c, ctx = setup(x, ids, bias, None, alpha=1.0)
    first = loss(x, ids, bias, p, c, ctx)
    dx = torch.autograd.grad(first.loss, x)[0]
    second = loss(x, ids, bias, p, c, ctx)
    dx2 = torch.autograd.grad(second.loss, x, create_graph=True)[0]
    assert torch.equal(bits(first.loss), bits(second.loss)) and torch.equal(bits(dx), bits(dx2))
    assert torch.isfinite(dx).all()
    if value == -300:
        assert dx.count_nonzero() > 0
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dx2.sum(), x)


@gpu
def test_training_total_loss_opt_in_two_updates_and_old_plan_rejected():
    gen = torch.Generator().manual_seed(343)
    x = torch.nn.Parameter(torch.randn(17, 256, generator=gen).cuda())
    ids = torch.randperm(17, generator=gen).long()
    bias = torch.zeros(256, device="cuda")
    upstream = torch.randn(17, 6, generator=gen).cuda()
    optimizer = torch.optim.SGD([x], lr=0.1)
    initial = x.detach().clone()
    for _ in range(2):
        p, c, ctx = setup(x, ids, bias, 2)
        current = loss(x, ids, bias, p, c, ctx, microbatch_size=7)
        main = sum(
            (route.weights * upstream[i * 7 : i * 7 + route.weights.shape[0]]).sum()
            for i, route in enumerate(current.routes)
        )
        separate = (
            torch.autograd.grad(main, x, retain_graph=True)[0]
            + torch.autograd.grad(current.loss, x, retain_graph=True)[0]
        )
        (main + current.loss).backward()
        torch.testing.assert_close(x.grad, separate, rtol=1e-6, atol=1e-8)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with pytest.raises(ValueError, match="logits changed"):
            loss(x, ids, bias, p, c, ctx)
    assert not torch.equal(x, initial)


@gpu
def test_alpha_overflow_and_unsupported_context_rejected():
    x = torch.zeros(2, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    bias = torch.zeros(256, device="cuda")
    p, c, ctx = setup(x, ids, bias, None, alpha=1e38)
    with pytest.raises(ValueError, match="overflowed"):
        loss(x, ids, bias, p, c, ctx)
    with pytest.raises(TypeError, match="P4 RCCL"):
        prepare_router_aux_context(p, config=c, collective=object(), version="ctx.v1")


@gpu
def test_global_expert_column_permutation_and_stream_workspace_isolation():
    gen = torch.Generator().manual_seed(344)
    x = torch.randn(17, 256, generator=gen).cuda().requires_grad_()
    ids = torch.randperm(17, generator=gen).long()
    bias = torch.randn(256, generator=gen).cuda() * 0.1
    p, c, ctx = setup(x, ids, bias, 2)
    original = loss(x, ids, bias, p, c, ctx)
    dx = torch.autograd.grad(original.loss, x)[0]
    permutation = torch.randperm(256, generator=gen)
    changed = x.detach()[:, permutation.cuda()].requires_grad_()
    p2, c2, ctx2 = setup(changed, ids, bias, 2, global_ids=permutation)
    current = loss(changed, ids, bias, p2, c2, ctx2)
    perm_dx = torch.autograd.grad(current.loss, changed)[0]
    assert torch.equal(original.expert_ids, current.expert_ids)
    assert torch.equal(bits(original.loss), bits(current.loss))
    assert torch.equal(bits(dx[:, permutation.cuda()]), bits(perm_dx))
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            value = x.detach().clone().requires_grad_()
            run = loss(value, ids, bias, p, c, ctx)
            outputs.append((run.loss, torch.autograd.grad(run.loss, value)[0]))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert all(
        torch.equal(bits(y), bits(original.loss)) and torch.equal(bits(g), bits(dx))
        for y, g in outputs
    )
    with torch.inference_mode():
        for _ in range(3):
            temporary = loss(x, ids, bias, p, c, ctx)
            del temporary
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for _ in range(50):
            temporary = loss(x, ids, bias, p, c, ctx)
            del temporary
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() <= baseline


@gpu
@pytest.mark.parametrize("field", ["logits", "bias"])
def test_original_input_data_mutation_rejected_before_backward(field):
    x = torch.randn(3, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20, 30])
    bias = torch.zeros(256, device="cuda")
    p, c, ctx = setup(x, ids, bias, None)
    result = loss(x, ids, bias, p, c, ctx)
    tensor = x if field == "logits" else bias
    tensor.data.flatten()[0] += 1
    with pytest.raises(RuntimeError, match="input state changed"):
        result.loss.backward()
