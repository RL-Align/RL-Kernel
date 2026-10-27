# SPDX-License-Identifier: Apache-2.0
"""Task-32 bias, complete-domain admission, post-mask weights and denominator VJP."""

from dataclasses import replace

import pytest
import torch

from rl_engine.ops.moe.learned_route import (
    _digest,
    _plan_record,
    learned_route,
    prepare_capacity_domain,
    prepare_learned_route_plan,
)
from rl_engine.reference.moe.learned_route import learned_route_reference, vjp_fp64, weights_fp64

gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def bits(x):
    return x.detach().contiguous().view(torch.int32)


def make_plan(x, ids, bias, capacity=None, global_ids=None, scale=1.5, domain_ids=None):
    domain = prepare_capacity_domain(
        ids if domain_ids is None else domain_ids,
        capacity,
        absolute_layer=3,
        model_version="m1",
        domain_version="d1",
    )
    return prepare_learned_route_plan(
        x,
        ids,
        bias,
        domain,
        bias_version="b1",
        plan_version="p1",
        global_expert_ids=global_ids,
        routed_scaling_factor=scale,
    )


def route(x, ids, bias, plan, **kwargs):
    options = dict(
        absolute_layer=3,
        model_version="m1",
        bias_version="b1",
        domain_version="d1",
        plan_version="p1",
    )
    options.update(kwargs)
    return learned_route(x, ids, bias, plan, **options)


@gpu
def test_bias_only_selects_direct_unbiased_gather_not_subtracting_rounded_bias():
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    with torch.no_grad():
        x[0, :6] = torch.arange(6, device="cuda")
    bias = torch.zeros(256, device="cuda", requires_grad=True)
    with torch.no_grad():
        bias[:6] = 1e8
    ids = torch.tensor([100], dtype=torch.int64)
    plan = make_plan(x, ids, bias)
    result = route(x, ids, bias, plan)
    assert result.expert_ids.tolist() == [[0, 1, 2, 3, 4, 5]]
    ref = weights_fp64(x.detach().cpu(), result.column_indices.cpu(), result.surviving_mask.cpu())
    torch.testing.assert_close(result.weights.detach().cpu().double(), ref, rtol=2e-6, atol=1e-8)
    assert not torch.allclose(result.weights, torch.full_like(result.weights, 0.25), atol=0.01)
    # Subtracting 1e8 from FP32(score+1e8) loses every original small score.
    raw = torch.sqrt(torch.nn.functional.softplus(x.detach()))
    wrong = (raw[0, :6] + bias[:6]).detach() - bias[:6].detach()
    assert wrong.count_nonzero() == 0 and result.weights.count_nonzero() == 6
    dx, db = torch.autograd.grad(
        result.weights, (x, bias), torch.arange(6.0, device="cuda").reshape(1, 6), allow_unused=True
    )
    assert db is None and torch.isfinite(dx).all()


@gpu
def test_capacity_priority_is_global_token_order_not_batch_or_score_order():
    ids = torch.tensor([900, 2, 17, 1, 30])
    x = torch.zeros(5, 256, device="cuda")
    bias = torch.zeros(256, device="cuda")
    capacity = torch.zeros(256, dtype=torch.int64)
    capacity[:6] = torch.tensor([1, 2, 0, 3, 5, 1])
    plan = make_plan(x, ids, bias, capacity)
    result = route(x, ids, bias, plan)
    ref, experts, columns, keep, positions = learned_route_reference(
        x, ids, bias, plan.domain.capacities
    )
    assert torch.equal(result.expert_ids.cpu(), experts)
    assert torch.equal(result.surviving_mask.cpu(), keep)
    assert torch.equal(result.dispatch_positions.cpu(), positions)
    assert result.surviving_mask[3].tolist() == [True, True, False, True, True, True]
    assert result.surviving_mask[0].tolist() == [False, False, False, False, True, False]
    torch.testing.assert_close(result.weights.detach().cpu().double(), ref, rtol=2e-6, atol=1e-8)
    # Overflow slots stay in their original positions; no compaction/new experts.
    assert result.expert_ids.tolist() == [list(range(6))] * 5
    assert torch.count_nonzero(result.weights[~result.surviving_mask]) == 0


@gpu
@pytest.mark.parametrize("capacity", [None, 0, 1, 2, 100])
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5, 2.0])
def test_capacity_and_scale_modes_normal_forward_and_gradient(capacity, scale):
    gen = torch.Generator().manual_seed(320)
    x = torch.randn(17, 256, generator=gen).cuda().requires_grad_()
    bias = (torch.randn(256, generator=gen) * 0.2).cuda()
    ids = torch.randperm(17, generator=gen).long() * 100 + 7
    plan = make_plan(x, ids, bias, capacity, scale=scale)
    result = route(x, ids, bias, plan)
    reference, experts, columns, keep, positions = learned_route_reference(
        x, ids, bias, plan.domain.capacities, scale=scale
    )
    assert torch.equal(result.expert_ids.cpu(), experts)
    assert torch.equal(result.surviving_mask.cpu(), keep)
    assert torch.equal(result.dispatch_positions.cpu(), positions)
    torch.testing.assert_close(
        result.weights.detach().cpu().double(), reference, rtol=2e-6, atol=1e-8
    )
    grad = torch.randn(17, 6, generator=gen).cuda()
    dx = torch.autograd.grad(result.weights, x, grad)[0]
    refdx = vjp_fp64(x.detach().cpu(), columns, keep, grad.cpu(), scale)
    torch.testing.assert_close(dx.cpu().double(), refdx, rtol=3e-4, atol=2e-7)
    selected_keep = torch.zeros_like(x, dtype=torch.bool).scatter_(
        1, result.column_indices.long(), result.surviving_mask
    )
    assert torch.count_nonzero(dx[~selected_keep]) == 0
    if capacity == 0 or scale == 0:
        assert result.weights.count_nonzero() == 0 and dx.count_nonzero() == 0


@gpu
def test_denominator_gradient_is_present_after_overflow_masking():
    x = torch.zeros(2, 256, device="cuda", requires_grad=True)
    bias = torch.zeros(256, device="cuda")
    ids = torch.tensor([1, 2])
    capacity = torch.zeros(256, dtype=torch.int64)
    capacity[:6] = torch.tensor([1, 2, 2, 2, 2, 2])
    plan = make_plan(x, ids, bias, capacity)
    result = route(x, ids, bias, plan)
    g = torch.zeros_like(result.weights)
    g[1, 1] = 1.0
    dx = torch.autograd.grad(result.weights, x, g)[0]
    # Five experts survive in row 1. A loss on slot 1 must also differentiate
    # the other four surviving denominator terms; the dropped slot gets zero.
    assert dx[1, 0].item() == 0
    assert dx[1, 1].item() > 0 and (dx[1, 2:6] < 0).all()
    ref = vjp_fp64(
        x.detach().cpu(), result.column_indices.cpu(), result.surviving_mask.cpu(), g.cpu()
    )
    torch.testing.assert_close(dx.cpu().double(), ref, rtol=2e-6, atol=1e-8)
    # A detached denominator would miss these negative terms entirely.
    assert dx[1, 2:6].abs().min().item() > 0.01


@gpu
@pytest.mark.parametrize("seed", range(10))
def test_random_normal_routing_oracle_and_repeated_vjp(seed):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(17, 256, generator=gen) * 10).cuda().requires_grad_()
    bias = (torch.randn(256, generator=gen) * 0.1).cuda()
    ids = torch.randperm(17, generator=gen).long() + 1000
    capacity = torch.randint(0, 4, (256,), generator=gen)
    plan = make_plan(x, ids, bias, capacity)
    a = route(x, ids, bias, plan)
    ref, e, c, k, p = learned_route_reference(x, ids, bias, plan.domain.capacities)
    assert torch.equal(a.expert_ids.cpu(), e) and torch.equal(a.surviving_mask.cpu(), k)
    torch.testing.assert_close(a.weights.detach().cpu().double(), ref, rtol=2e-6, atol=1e-8)
    grad = torch.randn(17, 6, generator=gen).cuda()
    dx = torch.autograd.grad(a.weights, x, grad)[0]
    b = route(x, ids, bias, plan)
    dx2 = torch.autograd.grad(b.weights, x, grad)[0]
    assert torch.equal(bits(a.weights), bits(b.weights)) and torch.equal(bits(dx), bits(dx2))
    torch.testing.assert_close(
        dx.cpu().double(), vjp_fp64(x.detach().cpu(), c, k, grad.cpu()), rtol=3e-4, atol=2e-7
    )


@gpu
@pytest.mark.parametrize("kind", ["contiguous", "padded", "transpose"])
def test_fixed_domain_microbatch_bi_and_plan_input_permutation(kind):
    gen = torch.Generator().manual_seed(321)
    full = torch.randn(257, 256, generator=gen).cuda().requires_grad_()
    bias = torch.zeros(256, device="cuda")
    ids = torch.randperm(257, generator=gen).long() + 100
    plan = make_plan(full, ids, bias, 3)
    whole = route(full, ids, bias, plan)
    all_grad = torch.randn(257, 6, generator=gen).cuda()
    whole_dx = torch.autograd.grad(whole.weights, full, all_grad)[0]
    for m in (1, 16, 17, 32, 65, 257):
        rows = torch.cat(
            (torch.tensor([128]), torch.tensor([i for i in range(257) if i != 128])[: m - 1])
        )
        rows = rows.roll(m // 2)
        x = full.detach()[rows.cuda()].clone()
        if kind == "transpose":
            x = x.T.contiguous().T
        if kind == "padded":
            storage = torch.empty(m, 515, device="cuda")
            view = storage[:, 1:513:2]
            view.copy_(x)
            x = view
        x.requires_grad_()
        result = route(x, ids[rows], bias, plan)
        assert torch.equal(bits(result.weights), bits(whole.weights[rows.cuda()]))
        assert torch.equal(result.expert_ids, whole.expert_ids[rows.cuda()])
        assert torch.equal(result.surviving_mask, whole.surviving_mask[rows.cuda()])
        dx = torch.autograd.grad(result.weights, x, all_grad[rows.cuda()])[0]
        assert torch.equal(bits(dx), bits(whole_dx[rows.cuda()]))
        with torch.inference_mode():
            replay = route(x, ids[rows], bias, plan)
        assert torch.equal(bits(result.weights), bits(replay.weights))
    permutation = torch.randperm(257, generator=gen)
    rebuilt = make_plan(full[permutation.cuda()], ids[permutation], bias, 3)
    assert rebuilt.sha256 == plan.sha256


@gpu
def test_shared_immutable_plan_across_streams_and_repeated_live_allocation():
    gen = torch.Generator().manual_seed(322)
    x = torch.randn(17, 256, generator=gen).cuda().requires_grad_()
    bias = torch.zeros(256, device="cuda")
    ids = torch.arange(17) + 50
    grad = torch.randn(17, 6, generator=gen).cuda()
    plan = make_plan(x, ids, bias, 2)
    original_plan_hash = plan.sha256
    reference = route(x, ids, bias, plan)
    reference_dx = torch.autograd.grad(reference.weights, x, grad)[0]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            local = x.detach().clone().requires_grad_()
            result = route(local, ids, bias, plan)
            dx = torch.autograd.grad(result.weights, local, grad)[0]
            results.append((result, dx))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for result, dx in results:
        assert torch.equal(bits(result.weights), bits(reference.weights))
        assert torch.equal(bits(dx), bits(reference_dx))
        assert torch.equal(result.dispatch_positions, reference.dispatch_positions)
    # Validation synchronizes with the host. This checks stream ownership and
    # workspace/state isolation, without claiming overlapping GPU execution.
    with torch.inference_mode():
        for _ in range(3):
            temporary = route(x, ids, bias, plan)
            del temporary
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        for _ in range(50):
            temporary = route(x, ids, bias, plan)
            assert torch.equal(bits(temporary.weights), bits(reference.weights))
            del temporary
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == baseline
    assert plan.sha256 == original_plan_hash == _digest(_plan_record(plan))


@gpu
@pytest.mark.parametrize(
    "invalid",
    [
        "bf16",
        "wrong_shape",
        "nan_logits",
        "inf_bias",
        "duplicate_ids",
        "float_ids",
        "bad_global_ids",
    ],
)
def test_invalid_normal_inputs_fail_before_route(invalid):
    x = torch.zeros(2, 256, device="cuda")
    ids = torch.tensor([1, 2])
    bias = torch.zeros(256, device="cuda")
    gids = None
    if invalid == "bf16":
        x = x.bfloat16()
    if invalid == "wrong_shape":
        x = x[:, :255]
    if invalid == "nan_logits":
        x[0, 0] = torch.nan
    if invalid == "inf_bias":
        bias[0] = torch.inf
    if invalid == "duplicate_ids":
        ids = torch.tensor([1, 1])
    if invalid == "float_ids":
        ids = ids.float()
    if invalid == "bad_global_ids":
        gids = torch.zeros(256, dtype=torch.int64)
    with pytest.raises((TypeError, ValueError)):
        make_plan(x, ids, bias, global_ids=gids)


@gpu
@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_upstream_rejected(invalid):
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([1])
    bias = torch.zeros(256, device="cuda")
    plan = make_plan(x, ids, bias)
    r = route(x, ids, bias, plan)
    with pytest.raises(ValueError, match="finite learned weight gradients"):
        torch.autograd.grad(r.weights, x, torch.full_like(r.weights, invalid))


@gpu
def test_higher_derivatives_and_changed_capacity_domain_rejected():
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([1])
    bias = torch.zeros(256, device="cuda")
    plan = make_plan(x, ids, bias)
    r = route(x, ids, bias, plan)
    dx = torch.autograd.grad(r.weights, x, torch.ones_like(r.weights), create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dx.sum(), x)
    bad_domain = replace(plan.domain, capacities=(0,) * 256)
    with pytest.raises(ValueError, match="domain identity"):
        route(x, ids, bias, replace(plan, domain=bad_domain))


@gpu
def test_expert_column_permutation_preserves_logical_plan_and_slots():
    gen = torch.Generator().manual_seed(322)
    x = torch.randn(17, 256, generator=gen).cuda()
    ids = torch.arange(17)
    bias = torch.randn(256, generator=gen).cuda()
    plan = make_plan(x, ids, bias, 2)
    original = route(x, ids, bias, plan)
    permutation = torch.randperm(256, generator=gen).cuda()
    permuted = x[:, permutation]
    plan2 = make_plan(permuted, ids, bias, 2, global_ids=permutation)
    result = route(permuted, ids, bias, plan2)
    assert plan.sha256 == plan2.sha256
    assert torch.equal(result.expert_ids, original.expert_ids)
    assert torch.equal(bits(result.weights), bits(original.weights))


@gpu
@pytest.mark.parametrize("value", [-300.0, -200.0, -120.0, -94.0, 80.0, 1000.0])
def test_extreme_negative_scores_clamped_denominator_and_local_gradient(value):
    x = torch.full((1, 256), value, device="cuda", requires_grad=True)
    ids = torch.tensor([0])
    bias = torch.zeros(256, device="cuda")
    plan = make_plan(x, ids, bias)
    r = route(x, ids, bias, plan)
    g = torch.tensor([[1.0, -2.0, 3.0, -0.5, 0.25, 0.0]], device="cuda")
    dx = torch.autograd.grad(r.weights, x, g)[0]
    ref = weights_fp64(x.detach().cpu(), r.column_indices.cpu(), r.surviving_mask.cpu())
    torch.testing.assert_close(
        r.weights.detach().cpu().double(), ref, rtol=2e-6, atol=(2**-146) * 1.5 / 1e-20
    )
    refdx = vjp_fp64(x.detach().cpu(), r.column_indices.cpu(), r.surviving_mask.cpu(), g.cpu())
    torch.testing.assert_close(dx.cpu().double(), refdx, rtol=3e-5, atol=2**-146)
    if value == -300.0:
        assert r.weights.count_nonzero() == 0 and dx.count_nonzero() > 0


@gpu
@pytest.mark.parametrize(
    "field,value",
    [
        ("absolute_layer", 4),
        ("model_version", "m2"),
        ("bias_version", "b2"),
        ("domain_version", "d2"),
        ("plan_version", "p2"),
    ],
)
def test_version_and_layer_barriers(field, value):
    x = torch.zeros(1, 256, device="cuda")
    ids = torch.tensor([1])
    b = torch.zeros(256, device="cuda")
    plan = make_plan(x, ids, b)
    with pytest.raises(ValueError, match="barrier mismatch"):
        route(x, ids, b, plan, **{field: value})


@gpu
@pytest.mark.parametrize("target", ["bias", "logits"])
def test_data_mutation_cannot_reuse_a_stale_plan(target):
    x = torch.zeros(1, 256, device="cuda")
    ids = torch.tensor([1])
    b = torch.zeros(256, device="cuda")
    plan = make_plan(x, ids, b)
    (b if target == "bias" else x).data.add_(1)
    with pytest.raises(ValueError, match="changed"):
        route(x, ids, b, plan)


@gpu
def test_capacity_plan_identity_and_normal_reselection_not_fixed_replay():
    x = torch.zeros(1, 256, device="cuda")
    ids = torch.tensor([1])
    b = torch.zeros(256, device="cuda")
    plan = make_plan(x, ids, b)
    row = replace(plan.tokens[0], keep=(False,) * 6)
    corrupt = replace(plan, tokens=(row,))
    with pytest.raises(ValueError, match="plan identity"):
        route(x, ids, b, corrupt)
    row = replace(plan.tokens[0], expert_ids=(10, 11, 12, 13, 14, 15))
    consistent = replace(plan, tokens=(row,))
    consistent = replace(consistent, sha256=_digest(_plan_record(consistent)))
    with pytest.raises(ValueError, match="normal learned selection differs"):
        route(x, ids, b, consistent)


@gpu
def test_complete_domain_is_required_for_planning_and_duplicate_consumer_ids_reject():
    x = torch.zeros(2, 256, device="cuda")
    ids = torch.tensor([1, 2])
    b = torch.zeros(256, device="cuda")
    with pytest.raises(ValueError, match="complete declared"):
        make_plan(x[:1], ids[:1], b, domain_ids=ids)
    plan = make_plan(x, ids, b)
    with pytest.raises(ValueError, match="distinct nonnegative"):
        route(x, torch.tensor([1, 1]), b, plan)
    with pytest.raises(ValueError, match="not in declared"):
        route(x[:1], torch.tensor([999]), b, plan)


@gpu
def test_empty_domain_graph_and_saved_tensor_mutation():
    b = torch.zeros(256, device="cuda")
    x = torch.zeros(0, 256, device="cuda", requires_grad=True)
    ids = torch.empty(0, dtype=torch.int64)
    plan = make_plan(x, ids, b)
    r = route(x, ids, b, plan)
    r.weights.sum().backward()
    assert x.grad.shape == (0, 256)
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([0])
    plan = make_plan(x, ids, b)
    r = route(x, ids, b, plan)
    r.surviving_mask[0, 0] = False
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        r.weights.sum().backward()
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            route(x, ids, b, plan)


@pytest.mark.parametrize("capacity", [-1, True, 1.5, [-1] * 256, [0] * 255, torch.zeros(256)])
def test_invalid_capacity_cpu(capacity):
    with pytest.raises((TypeError, ValueError)):
        prepare_capacity_domain(
            torch.tensor([1, 2]),
            capacity,
            absolute_layer=3,
            model_version="m1",
            domain_version="d1",
        )


@pytest.mark.parametrize("layer", [0, 1, 2, 43, -1, True])
def test_invalid_learned_absolute_layers_cpu(layer):
    with pytest.raises(ValueError, match="absolute layers"):
        prepare_capacity_domain(
            torch.tensor([1]), 1, absolute_layer=layer, model_version="m1", domain_version="d1"
        )


def test_capacity_domain_detaches_input_metadata():
    ids = torch.tensor([5, 1, 9])
    caps = torch.ones(256, dtype=torch.int64)
    domain = prepare_capacity_domain(
        ids, caps, absolute_layer=3, model_version="m1", domain_version="d1"
    )
    ids.fill_(0)
    caps.fill_(0)
    assert domain.token_ids == (1, 5, 9) and domain.capacities == (1,) * 256


def test_independent_vjp_by_finite_difference_after_masking():
    logits = torch.linspace(-3, 3, 256, dtype=torch.float64).reshape(1, 256)
    columns = torch.tensor([[250, 1, 129, 7, 100, 3]])
    keep = torch.tensor([[True, False, True, True, False, True]])
    grad = torch.tensor([[1.0, -2.0, 3.0, -0.5, 0.25, 0.0]], dtype=torch.float64)
    derivative = vjp_fp64(logits, columns, keep, grad)
    for column in columns[0]:
        delta = torch.zeros_like(logits)
        delta[0, column] = 1e-5
        fd = (
            (
                weights_fp64(logits + delta, columns, keep)
                - weights_fp64(logits - delta, columns, keep)
            )
            * grad
        ).sum() / 2e-5
        torch.testing.assert_close(derivative[0, column], fd, rtol=1e-7, atol=1e-10)
