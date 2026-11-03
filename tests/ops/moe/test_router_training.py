# SPDX-License-Identifier: Apache-2.0
"""Task 33: normal selection first, conditional gradients and diagnostic replay."""

import json

import pytest
import torch

from rl_engine.models.deepseek_v4.router_training import DeepSeekV4RouterTraining
from rl_engine.ops.moe.hash_route import prepare_hash_route_table
from rl_engine.ops.moe.learned_route import (
    _digest,
    prepare_capacity_domain,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.ops.moe.router_training import (
    RouteRecord,
    compare_route_records,
    compare_vjp_records,
    diagnostic_fixed_route,
    load_route_record,
    normal_hash_route,
    normal_learned_route,
    record_route_vjp,
    save_route_record,
)
from rl_engine.reference.moe.learned_route import learned_route_reference, score_fp64, vjp_fp64

gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm GPU"
)


def bits(tensor):
    return tensor.detach().contiguous().view(torch.int32)


def make_learned(x, ids, bias, capacity=2, gids=None):
    domain = prepare_capacity_domain(
        ids, capacity, absolute_layer=3, model_version="m1", domain_version="d1"
    )
    plan = prepare_learned_route_plan(
        x, ids, bias, domain, bias_version="b1", plan_version="p1", global_expert_ids=gids
    )
    return normal_learned_route(
        x,
        ids,
        bias,
        plan,
        record_version="r1",
        absolute_layer=3,
        model_version="m1",
        bias_version="b1",
        domain_version="d1",
        plan_version="p1",
    ), plan


def replay(x, ids, record, **kwargs):
    data = record.payload()
    return diagnostic_fixed_route(
        x, ids, record, absolute_layer=data["absolute_layer"], versions=data["versions"], **kwargs
    )


def replace_payload(record, change):
    payload = record.payload()
    change(payload)
    return RouteRecord(json.dumps(payload, sort_keys=True, separators=(",", ":")), _digest(payload))


@gpu
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("capacity", [None, 0, 2])
def test_normal_learned_discrete_math_gradient_then_diagnostic_replay(seed, capacity):
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(17, 256, generator=gen) * 10).cuda().requires_grad_()
    bias = (torch.randn(256, generator=gen) * 0.3).cuda().requires_grad_()
    ids = torch.randperm(17, generator=gen).long() + 100
    result, plan = make_learned(x, ids, bias, capacity)
    reference, experts, columns, keep, positions = learned_route_reference(
        x, ids, bias, plan.domain.capacities
    )
    # Mandatory order: discrete choices before any tolerance/value comparison.
    assert torch.equal(result.expert_ids.cpu(), experts)
    rows = {r["logical_id"]: r for r in result.record.payload()["tokens"]}
    assert [rows[int(t)]["keep"] for t in ids] == keep.tolist()
    assert [rows[int(t)]["positions"] for t in ids] == positions.tolist()
    torch.testing.assert_close(
        result.weights.detach().cpu().double(), reference, rtol=2e-6, atol=1e-8
    )
    grad = torch.randn(17, 6, generator=gen).cuda()
    dx, vjp = record_route_vjp(result, grad, loss_version="loss.mean17.v1", retain_graph=True)
    torch.testing.assert_close(
        dx.cpu().double(),
        vjp_fp64(x.detach().cpu(), columns, keep, grad.cpu()),
        rtol=3e-4,
        atol=2e-7,
    )
    assert torch.autograd.grad(result.weights, bias, grad, allow_unused=True)[0] is None
    assert not result.expert_ids.requires_grad
    diagnostic = replay(x.detach().clone().requires_grad_(), ids, result.record)
    replay_dx, replay_vjp = record_route_vjp(diagnostic, grad, loss_version="loss.mean17.v1")
    assert torch.equal(bits(result.weights), bits(diagnostic.weights))
    assert torch.equal(bits(dx), bits(replay_dx))
    with pytest.raises(ValueError, match="diagnostic only"):
        compare_route_records(diagnostic.record, result.record)
    compared = compare_vjp_records(replay_vjp, vjp, require_normal=False)
    assert compared["status"] == "match" and not compared["acceptance"]


@gpu
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("layer", [0, 1, 2])
def test_normal_hash_table_selection_and_math_vjp_then_replay(renormalize, scale, layer):
    gen = torch.Generator().manual_seed(330 + layer)
    x = torch.randn(17, 256, generator=gen).cuda().requires_grad_()
    ids = torch.randperm(17, generator=gen).long() + 100
    source = torch.stack([torch.randperm(256, generator=gen)[:6] for _ in range(7)]).cuda()
    table = prepare_hash_route_table(
        source,
        absolute_layer=layer,
        model_version="m1",
        table_version="t1",
        expected_sha256=tensor_sha256(source),
    )
    vocab = (torch.arange(17, device="cuda") % 7).long()
    result = normal_hash_route(
        x,
        ids,
        vocab,
        table,
        record_version="r1",
        absolute_layer=layer,
        model_version="m1",
        table_version="t1",
        renormalize=renormalize,
        routed_scaling_factor=scale,
    )
    columns = source[vocab].cpu()
    assert torch.equal(result.expert_ids.cpu(), columns)
    scores = score_fp64(x.detach().cpu()).gather(1, columns)
    denominator = scores.sum(1, keepdim=True).clamp_min(1e-20) if renormalize else 1.0
    torch.testing.assert_close(
        result.weights.detach().cpu().double(), scores / denominator * scale, rtol=2e-6, atol=1e-8
    )
    grad = torch.randn(17, 6, generator=gen).cuda()
    dx, vjp = record_route_vjp(result, grad, loss_version="loss.v1")
    if renormalize:
        mathdx = vjp_fp64(
            x.detach().cpu(), columns, torch.ones_like(columns, dtype=torch.bool), grad.cpu(), scale
        )
    else:
        z = x.detach().cpu().double().requires_grad_()
        mathdx = torch.autograd.grad(
            (score_fp64(z).gather(1, columns) * scale), z, grad.cpu().double()
        )[0]
    torch.testing.assert_close(dx.cpu().double(), mathdx, rtol=3e-4, atol=2e-7)
    diagnostic = replay(x.detach().clone().requires_grad_(), ids, result.record)
    replay_dx, replay_vjp = record_route_vjp(diagnostic, grad, loss_version="loss.v1")
    assert torch.equal(bits(result.weights), bits(diagnostic.weights))
    assert torch.equal(bits(dx), bits(replay_dx))
    assert compare_vjp_records(replay_vjp, vjp, require_normal=False)["status"] == "match"


@gpu
def test_saved_json_roundtrip_owned_state_and_reordered_tokens(tmp_path):
    x = torch.zeros(17, 256, device="cuda", requires_grad=True)
    ids = torch.arange(17).roll(3) + 100
    result, _ = make_learned(x, ids, torch.zeros(256, device="cuda"), 2)
    path = tmp_path / "route.json"
    save_route_record(result.record, path)
    saved = load_route_record(path)
    assert saved.sha256 == result.record.sha256
    owned = saved.payload()
    owned["tokens"][0]["experts"][0] = 255
    assert saved.payload()["tokens"][0]["experts"][0] == 0
    order = torch.arange(16, -1, -1)
    reordered = replay(x.detach()[order.cuda()].requires_grad_(), ids[order], saved)
    assert torch.equal(bits(reordered.weights), bits(result.weights[order.cuda()]))
    assert compare_route_records(reordered.record, saved, require_normal=False)["status"] == "match"
    payload = json.loads(path.read_text())
    payload["payload"]["tokens"][0]["weight_bits"][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        load_route_record(path)


@gpu
def test_changed_logits_fixed_replay_cannot_mask_normal_selection_bug():
    x = torch.zeros(2, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    bias = torch.zeros(256, device="cuda")
    old, _ = make_learned(x, ids, bias, None)
    changed = x.detach().clone()
    changed[:, 200:206] = 1000
    changed.requires_grad_()
    normal, _ = make_learned(changed, ids, bias, None)
    with pytest.raises(ValueError, match="logits changed"):
        replay(changed, ids, old.record)
    fixed = replay(changed, ids, old.record, allow_changed_logits=True)
    assert torch.equal(fixed.expert_ids, old.expert_ids)
    assert not torch.equal(normal.expert_ids, old.expert_ids)
    result = compare_route_records(normal.record, old.record)
    assert result["status"] == "discrete_mismatch" and not result["numeric_compared"]
    # Even perfect agreement under an injected route cannot earn acceptance.
    assert not compare_route_records(fixed.record, fixed.record, require_normal=False)["acceptance"]


@gpu
@pytest.mark.parametrize("state", ["logits", "bias", "weights", "expert_ids"])
def test_data_mutation_between_forward_and_backward_rejected(state):
    x = torch.randn(2, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    bias = torch.zeros(256, device="cuda")
    result, _ = make_learned(x, ids, bias, None)
    tensor = {
        "logits": x,
        "bias": bias,
        "weights": result.weights,
        "expert_ids": result.expert_ids,
    }[state]
    tensor.data.flatten()[0] += 1
    with pytest.raises(RuntimeError, match="state changed"):
        result.weights.sum().backward()


@gpu
def test_normal_microbatches_column_permutation_and_grad_records():
    gen = torch.Generator().manual_seed(333)
    x = torch.randn(65, 256, generator=gen).cuda().requires_grad_()
    ids = torch.randperm(65, generator=gen).long() + 100
    bias = torch.zeros(256, device="cuda")
    full, plan = make_learned(x, ids, bias, 2)
    grad = torch.randn(65, 6, generator=gen).cuda()
    dx, full_vjp = record_route_vjp(full, grad, loss_version="full65.v1")
    for m in (1, 16, 17, 32, 65):
        order = torch.arange(m).roll(3)
        local = x.detach()[order.cuda()].T.contiguous().T.requires_grad_()
        run = normal_learned_route(
            local,
            ids[order],
            bias,
            plan,
            record_version="r1",
            absolute_layer=3,
            model_version="m1",
            bias_version="b1",
            domain_version="d1",
            plan_version="p1",
        )
        local_dx, _ = record_route_vjp(run, grad[order.cuda()], loss_version="full65.v1")
        assert torch.equal(bits(run.weights), bits(full.weights[order.cuda()]))
        assert torch.equal(bits(local_dx), bits(dx[order.cuda()]))
    order = torch.randperm(65, generator=gen)
    perm = torch.randperm(256, generator=gen)
    transposed = x.detach()[order.cuda()][:, perm.cuda()].requires_grad_()
    rebuilt, _ = make_learned(transposed, ids[order], bias, 2, perm)
    _, perm_vjp = record_route_vjp(rebuilt, grad[order.cuda()], loss_version="full65.v1")
    assert compare_route_records(rebuilt.record, full.record)["acceptance"]
    assert compare_vjp_records(perm_vjp, full_vjp)["acceptance"]


@gpu
@pytest.mark.parametrize("width", [32, 4096])
def test_upstream_bf16_lora_two_updates_through_frozen_gate(width):
    gen = torch.Generator().manual_seed(334)
    hidden = (torch.randn(17, width, generator=gen) * 0.3).cuda().bfloat16()
    gate = torch.nn.Parameter(
        (torch.randn(256, width, generator=gen) * 0.1).cuda().bfloat16(), requires_grad=False
    )
    a = torch.nn.Parameter((torch.randn(4, width, generator=gen) * 0.1).cuda().bfloat16())
    b = torch.nn.Parameter((torch.randn(width, 4, generator=gen) * 0.1).cuda().bfloat16())
    bias = torch.zeros(256, device="cuda")
    ids = torch.randperm(17, generator=gen).long() + 100
    upstream = torch.randn(17, 6, generator=gen).cuda()
    initial = a.detach().clone(), b.detach().clone()
    optimizer = torch.optim.SGD([a, b], lr=0.2)
    old_plan = None
    for step in range(3):
        version = f"m{step}"
        adapter = DeepSeekV4RouterTraining(
            gate, absolute_layer=3, model_version=version, bias=bias, bias_version="b1"
        )
        current = (
            hidden + torch.nn.functional.linear(torch.nn.functional.linear(hidden, a), b) * 0.5
        )
        domain = prepare_capacity_domain(
            ids, 2, absolute_layer=3, model_version=version, domain_version=f"d{step}"
        )
        plan = adapter.prepare_plan(current, ids, domain, plan_version=f"p{step}")
        if old_plan is not None:
            with pytest.raises(ValueError, match="version barrier"):
                adapter(current, ids, record_version="r1", plan=old_plan)
        run = adapter(current, ids, record_version="r1", plan=plan)
        assert (
            run.record.payload()["execution"]["gate_projection"]["backward_backend"]
            == "rlk.triton.fixed_fp32_dot"
        )
        ref, experts, columns, keep, _ = learned_route_reference(
            run.logits, ids, bias, domain.capacities
        )
        assert torch.equal(run.expert_ids.cpu(), experts)
        torch.testing.assert_close(run.weights.detach().cpu().double(), ref, rtol=2e-6, atol=1e-8)
        math_dx = vjp_fp64(run.logits.detach().cpu(), columns, keep, upstream.cpu())
        math_hidden_grad = (math_dx @ gate.detach().cpu().double()).bfloat16().cuda()
        expected = torch.autograd.grad(current, (a, b), math_hidden_grad, retain_graph=True)
        actual = torch.autograd.grad(run.weights, (a, b), upstream, retain_graph=True)
        run.weights.backward(upstream)
        assert torch.equal(bits(a.grad), bits(actual[0])) and torch.equal(
            bits(b.grad), bits(actual[1])
        )
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got.float(), want.float(), rtol=0.03, atol=2e-5)
            assert torch.isfinite(got).all() and got.count_nonzero() > 0
        assert gate.grad is None and adapter.selection_bias.grad is None
        if step < 2:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        old_plan = plan
    assert not torch.equal(a, initial[0]) and not torch.equal(b, initial[1])


@gpu
def test_finite_differences_recompute_normal_selection_and_capacity():
    x = torch.linspace(3, -3, 256, device="cuda").repeat(2, 1).requires_grad_()
    ids = torch.tensor([10, 20])
    bias = torch.zeros(256, device="cuda")
    capacity = torch.full((256,), 2, dtype=torch.int64)
    capacity[0] = 1
    result, _ = make_learned(x, ids, bias, capacity)
    upstream = torch.zeros(2, 6, device="cuda")
    upstream[1, 1] = 1
    dx, _ = record_route_vjp(result, upstream, loss_version="finite-difference.v1")
    assert dx[1, 0] == 0 and dx[1, 2] < 0
    for row, column in [(1, 0), (1, 1), (1, 2), (1, 200), (0, 2)]:
        objectives = []
        for sign in (1, -1):
            perturbed = x.detach().clone()
            perturbed[row, column] += sign * 0.005
            normal, _ = make_learned(perturbed, ids, bias, capacity)
            assert torch.equal(normal.expert_ids, result.expert_ids)
            objectives.append(float((normal.weights.double() * upstream.double()).sum()))
        difference = (objectives[0] - objectives[1]) / 0.01
        assert float(dx[row, column]) == pytest.approx(difference, rel=0.003, abs=1e-5)


@gpu
def test_shared_plan_records_across_streams_and_live_allocation_stability():
    gen = torch.Generator().manual_seed(335)
    x = torch.randn(17, 256, generator=gen).cuda().requires_grad_()
    ids = torch.arange(17) + 100
    bias = torch.zeros(256, device="cuda")
    baseline, plan = make_learned(x, ids, bias, 2)
    grad = torch.randn(17, 6, generator=gen).cuda()
    _, reference_vjp = record_route_vjp(baseline, grad, loss_version="loss.v1")

    def normal(local):
        return normal_learned_route(
            local,
            ids,
            bias,
            plan,
            record_version="r1",
            absolute_layer=3,
            model_version="m1",
            bias_version="b1",
            domain_version="d1",
            plan_version="p1",
        )

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    records = []
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            actual = normal(x.detach().clone().requires_grad_())
            _, vjp = record_route_vjp(actual, grad, loss_version="loss.v1")
            records.append(vjp)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert all(compare_vjp_records(record, reference_vjp)["acceptance"] for record in records)
    with torch.inference_mode():
        for _ in range(3):
            temporary = normal(x)
            del temporary
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for _ in range(50):
            temporary = normal(x)
            assert compare_route_records(temporary.record, baseline.record)["acceptance"]
            del temporary
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated


@gpu
@pytest.mark.parametrize("value", [-300.0, -120.0, -94.0, -16.0, 1000.0])
def test_tails_repeated_vjp_and_replay(value):
    x = torch.full((2, 256), value, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    bias = torch.zeros(256, device="cuda")
    result, _ = make_learned(x, ids, bias, None)
    grad = torch.arange(12.0, device="cuda").reshape(2, 6) - 4
    dx, first = record_route_vjp(result, grad, loss_version="loss.v1", retain_graph=True)
    dx2, second = record_route_vjp(result, grad, loss_version="loss.v1")
    assert torch.equal(bits(dx), bits(dx2)) and compare_vjp_records(first, second)["acceptance"]
    diagnostic = replay(x.detach().clone().requires_grad_(), ids, result.record)
    replay_dx, _ = record_route_vjp(diagnostic, grad, loss_version="loss.v1")
    assert torch.isfinite(dx).all() and torch.equal(bits(dx), bits(replay_dx))
    if value == -300:
        assert dx.count_nonzero() > 0


@gpu
def test_hash_model_adapter_and_frozen_parameter_barriers():
    gen = torch.Generator().manual_seed(339)
    source = torch.stack([torch.randperm(256, generator=gen)[:6] for _ in range(7)]).cuda()
    table = prepare_hash_route_table(
        source,
        absolute_layer=0,
        model_version="m1",
        table_version="t1",
        expected_sha256=tensor_sha256(source),
    )
    gate = torch.nn.Parameter(
        torch.randn(256, 32, generator=gen).cuda().bfloat16(), requires_grad=False
    )
    hidden = torch.randn(17, 32, generator=gen).cuda().bfloat16().requires_grad_()
    adapter = DeepSeekV4RouterTraining(gate, absolute_layer=0, model_version="m1", hash_table=table)
    run = adapter(
        hidden,
        torch.arange(17),
        record_version="r1",
        vocabulary_ids=torch.arange(17, device="cuda") % 7,
    )
    run.weights.backward(torch.randn_like(run.weights))
    assert hidden.grad.count_nonzero() > 0 and gate.grad is None
    hidden.grad = None
    run = adapter(
        hidden,
        torch.arange(17),
        record_version="r1",
        vocabulary_ids=torch.arange(17, device="cuda") % 7,
    )
    gate.data[0, 0] += 1
    with pytest.raises(RuntimeError, match="state changed"):
        run.weights.sum().backward()
    with pytest.raises(ValueError, match="frozen Gate changed"):
        adapter(
            hidden,
            torch.arange(17),
            record_version="r1",
            vocabulary_ids=torch.arange(17, device="cuda") % 7,
        )
    with pytest.raises(ValueError, match="frozen loaded"):
        DeepSeekV4RouterTraining(
            torch.nn.Parameter(gate.clone()), absolute_layer=0, model_version="m1", hash_table=table
        )


@gpu
def test_replay_versions_invalid_gradients_empty_and_graph_fail_closed():
    from rl_engine.ops.moe.router_training import load_vjp_record, save_vjp_record
    import tempfile
    from pathlib import Path

    x = torch.zeros(2, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([10, 20])
    result, _ = make_learned(x, ids, torch.zeros(256, device="cuda"), None)
    data = result.record.payload()
    with pytest.raises(ValueError, match="version barrier"):
        diagnostic_fixed_route(x, ids, result.record, absolute_layer=4, versions=data["versions"])
    with pytest.raises(ValueError, match="version barrier"):
        diagnostic_fixed_route(
            x,
            ids,
            result.record,
            absolute_layer=3,
            versions={**data["versions"], "model_version": "m2"},
        )
    with pytest.raises(ValueError, match="outside"):
        replay(x, torch.tensor([10, 30]), result.record)
    with pytest.raises(TypeError, match="FP32 upstream"):
        record_route_vjp(
            result, torch.zeros_like(result.weights, dtype=torch.bfloat16), loss_version="l1"
        )
    with pytest.raises(ValueError, match="shape/device"):
        record_route_vjp(result, torch.zeros(1, 6, device="cuda"), loss_version="l1")
    diagnostic = replay(x.detach().clone().requires_grad_(), ids, result.record)
    with pytest.raises(ValueError, match="finite"):
        diagnostic.weights.backward(torch.full_like(diagnostic.weights, float("nan")))
    _, vjp = record_route_vjp(result, torch.ones_like(result.weights), loss_version="l1")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "vjp.json"
        save_vjp_record(vjp, path)
        assert load_vjp_record(path) == vjp
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            replay(x, ids, result.record)
    empty = torch.empty(0, 256, device="cuda", requires_grad=True)
    empty_ids = torch.empty(0, dtype=torch.int64)
    empty_run, _ = make_learned(empty, empty_ids, torch.zeros(256, device="cuda"), None)
    empty_replay = replay(empty, empty_ids, empty_run.record)
    empty_dx, _ = record_route_vjp(
        empty_replay, torch.empty(0, 6, device="cuda"), loss_version="empty"
    )
    assert empty_dx.shape == (0, 256)
    assert not compare_route_records(empty_run.record, empty_run.record)["acceptance"]
