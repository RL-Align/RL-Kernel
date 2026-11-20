# SPDX-License-Identifier: Apache-2.0
"""Task-30 absolute-layer lookup, table identity, slot order and fixed-route VJP."""

from pathlib import Path

import pytest
import torch

from rl_engine.contracts.operators.hash_route import CHECKPOINT_TABLE_SHA256
from rl_engine.models.deepseek_v4.hash_router import DeepSeekV4HashRouter
from rl_engine.ops.moe.hash_route import hash_route, prepare_hash_route_table, tensor_sha256
from rl_engine.reference.moe.hash_route import hash_route_fp64, hash_route_vjp_fp64

gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def bits(x):
    return x.detach().contiguous().view(torch.int32)


def prepare(t, layer=0, **kwargs):
    return prepare_hash_route_table(
        t,
        absolute_layer=layer,
        model_version="model-v1",
        table_version="tables-v1",
        expected_sha256=tensor_sha256(t),
        **kwargs,
    )


def route(x, ids, table, **kwargs):
    return hash_route(
        x,
        ids,
        table,
        absolute_layer=table.absolute_layer,
        model_version="model-v1",
        table_version="tables-v1",
        **kwargs,
    )


def synthetic_table(dtype=torch.int64):
    # Deliberately not sorted. Ties in logits must not reorder these positions.
    base = torch.tensor([250, 1, 60, 7, 3, 199], dtype=dtype, device="cuda")
    return (base[None, :] + torch.arange(23, device="cuda", dtype=dtype)[:, None]) % 256


def layout(x, kind):
    if kind == "transpose":
        return x.T.contiguous().T
    if kind == "padded":
        data = torch.zeros((x.shape[0], x.shape[1] * 2 + 3), device=x.device)
        view = data[:, 1 : 2 * x.shape[1] + 1 : 2]
        view.copy_(x)
        return view
    return x


@gpu
@pytest.mark.parametrize("layer", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_absolute_layers_slots_tied_logits_and_integer_table_profiles(layer, dtype):
    table = synthetic_table(dtype)
    prepared = prepare(table, layer)
    ids = torch.tensor([0, 22, 7, 0], device="cuda", dtype=dtype)
    x = torch.zeros(4, 256, device="cuda", requires_grad=True)
    result = route(x, ids, prepared)
    assert torch.equal(result.expert_ids, table[ids.long()].long())
    assert result.expert_ids.dtype == torch.int64 and not result.expert_ids.requires_grad
    torch.testing.assert_close(result.weights, torch.full((4, 6), 0.25, device="cuda"))
    assert result.execution["table_source_sha256"] == tensor_sha256(table)
    assert result.execution["absolute_layer"] == layer


@gpu
@pytest.mark.parametrize("kind", ["contiguous", "padded", "transpose"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_score_weight_and_vjp_bytes_across_mixed_batches(kind, renormalize):
    gen = torch.Generator().manual_seed(30)
    table = synthetic_table()
    prepared = prepare(table)
    row = (torch.randn(1, 256, generator=gen) * 20).cuda()
    row_grad = torch.randn(1, 6, generator=gen).cuda()
    baseline_x = row.clone().requires_grad_()
    baseline_ids = torch.tensor([7], device="cuda")
    baseline = route(baseline_x, baseline_ids, prepared, renormalize=renormalize)
    baseline_dx = torch.autograd.grad(baseline.weights, baseline_x, row_grad)[0]
    for m in (1, 16, 17, 32, 65, 257):
        x = (torch.randn(m, 256, generator=gen) * 20).cuda()
        ids = (torch.arange(m, device="cuda") * 11) % 23
        grad = torch.randn(m, 6, generator=gen).cuda()
        positions = sorted({0, m // 2, m - 1})
        x[positions] = row
        ids[positions] = 7
        grad[positions] = row_grad
        padded_ids = torch.zeros(m * 2 + 1, device="cuda", dtype=torch.int64)
        padded_ids[1::2] = ids
        x = layout(x, kind).requires_grad_()
        result = route(x, padded_ids[1::2], prepared, renormalize=renormalize)
        dx = torch.autograd.grad(result.weights, x, layout(grad, kind))[0]
        with torch.inference_mode():
            replay = route(x, ids, prepared, renormalize=renormalize)
        assert torch.equal(result.expert_ids, table[ids].long())
        assert torch.equal(bits(result.weights), bits(replay.weights))
        for index in positions:
            assert torch.equal(bits(result.weights[index]), bits(baseline.weights[0]))
            assert torch.equal(bits(dx[index]), bits(baseline_dx[0]))
        expected, _ = hash_route_fp64(
            x.detach().cpu(), ids.cpu(), table.cpu(), renormalize=renormalize
        )
        expected_dx = hash_route_vjp_fp64(
            x.detach().cpu(), ids.cpu(), table.cpu(), grad.cpu(), renormalize=renormalize
        )
        torch.testing.assert_close(
            result.weights.detach().cpu().double(), expected, rtol=2e-6, atol=2**-146
        )
        torch.testing.assert_close(dx.cpu().double(), expected_dx, rtol=3e-4, atol=2e-7)


@gpu
@pytest.mark.parametrize("seed", range(10))
def test_random_accuracy_and_gradient_repeatability(seed):
    gen = torch.Generator().manual_seed(seed)
    table = synthetic_table()
    prepared = prepare(table)
    x = (torch.rand(17, 256, generator=gen) * 320 - 160).cuda().requires_grad_()
    ids = (torch.arange(17, device="cuda") * 7) % 23
    grad = torch.randn(17, 6, generator=gen).cuda()
    first = route(x, ids, prepared)
    dx = torch.autograd.grad(first.weights, x, grad)[0]
    again = route(x, ids, prepared)
    dx_again = torch.autograd.grad(again.weights, x, grad)[0]
    assert torch.equal(bits(first.weights), bits(again.weights))
    assert torch.equal(bits(dx), bits(dx_again))
    ref, _ = hash_route_fp64(x.detach().cpu(), ids.cpu(), table.cpu())
    ref_dx = hash_route_vjp_fp64(x.detach().cpu(), ids.cpu(), table.cpu(), grad.cpu())
    torch.testing.assert_close(first.weights.detach().cpu().double(), ref, rtol=2e-6, atol=2**-146)
    torch.testing.assert_close(dx.cpu().double(), ref_dx, rtol=3e-4, atol=2e-7)
    selected = torch.zeros_like(x, dtype=torch.bool).scatter_(1, first.expert_ids, True)
    assert torch.count_nonzero(dx[~selected]) == 0


@gpu
@pytest.mark.parametrize("value", [-300.0, -200.0, -120.0, -94.0, -90.0, 80.0, 1000.0])
def test_clamped_denominator_extremes_and_finite_gradients(value):
    table = synthetic_table()
    prepared = prepare(table)
    ids = torch.tensor([0], device="cuda")
    x = torch.full((1, 256), value, device="cuda", requires_grad=True)
    grad = torch.tensor([[1.0, -2.0, 3.0, -0.5, 0.25, 0.0]], device="cuda")
    result = route(x, ids, prepared)
    dx = torch.autograd.grad(result.weights, x, grad)[0]
    ref, _ = hash_route_fp64(x.detach().cpu(), ids.cpu(), table.cpu())
    ref_dx = hash_route_vjp_fp64(x.detach().cpu(), ids.cpu(), table.cpu(), grad.cpu())
    assert torch.isfinite(result.weights).all() and torch.isfinite(dx).all()
    # FP32 subnormal score rounding is amplified by division by the clamped
    # 1e-20 denominator. Account for that independently of normal-value rtol.
    torch.testing.assert_close(
        result.weights.detach().cpu().double(), ref, rtol=2e-6, atol=(2**-146) * 1.5 / 1e-20
    )
    torch.testing.assert_close(dx.cpu().double(), ref_dx, rtol=3e-5, atol=2**-146)
    if value == -300.0:
        assert result.weights.count_nonzero() == 0 and dx.count_nonzero() > 0
    if value == -200.0:
        rounded_score = torch.exp(torch.tensor(value / 2, dtype=torch.float64)).float()
        expected = ((rounded_score / torch.tensor(1e-20)) * 1.5).expand(1, 6)
        assert (bits(result.weights).cpu().long() - bits(expected).long()).abs().max() <= 2
    if value == -94.0:
        # Detect AITER's old sum+epsilon denominator: clamp and addition differ here.
        score = torch.exp(torch.tensor(value / 2, dtype=torch.float64))
        additive = (score / (6 * score + 1e-20)) * 1.5
        assert abs(result.weights[0, 0].item() - additive.item()) > 0.01


@gpu
@pytest.mark.parametrize("scale", [0.0, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("renormalize", [False, True])
def test_scaling_position_and_single_gradient_path(scale, renormalize):
    table = synthetic_table()
    prepared = prepare(table)
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([0], device="cuda")
    result = route(x, ids, prepared, renormalize=renormalize, routed_scaling_factor=scale)
    result.weights.sum().backward()
    ref, _ = hash_route_fp64(
        x.detach().cpu(), ids.cpu(), table.cpu(), renormalize=renormalize, scale=scale
    )
    torch.testing.assert_close(result.weights.detach().cpu().double(), ref)
    if scale == 0:
        assert torch.count_nonzero(x.grad) == 0


@gpu
@pytest.mark.parametrize("layer", [0, 1, 2])
def test_real_0731_tables_all_layers_vocabulary_edges_and_hashes(layer):
    path = Path(f"artifacts/task30/0731-layer{layer}-tid2eid.pt")
    assert path.exists(), "prepare the pinned task-30 checkpoint tables before GPU acceptance"
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    source = fixture["table"]
    assert tensor_sha256(source) == CHECKPOINT_TABLE_SHA256[layer]
    table = prepare(source.cuda(), layer)
    ids = torch.tensor([0, 129279, 17, 10000, 0], device="cuda")
    x = torch.linspace(-80, 80, 5 * 256, device="cuda").reshape(5, 256).requires_grad_()
    result = route(x, ids, table)
    assert torch.equal(result.expert_ids.cpu(), source[ids.cpu()])
    reference, _ = hash_route_fp64(x.detach().cpu(), ids.cpu(), source)
    torch.testing.assert_close(
        result.weights.detach().cpu().double(), reference, rtol=2e-6, atol=2**-146
    )
    dx = torch.autograd.grad(result.weights, x, torch.ones_like(result.weights))[0]
    assert torch.isfinite(dx).all()


@gpu
def test_absolute_layer_adapter_for_pp_stage_and_configured_layers():
    source = synthetic_table()
    tables = {layer: prepare((source + layer) % 256, layer) for layer in (0, 1, 2)}
    adapter = DeepSeekV4HashRouter(tables)
    ids = torch.tensor([0], device="cuda")
    x = torch.zeros(1, 256, device="cuda")
    for layer in (0, 1, 2):
        r = adapter(
            x, ids, absolute_layer=layer, model_version="model-v1", table_version="tables-v1"
        )
        assert torch.equal(r.expert_ids, ((source + layer) % 256)[ids])
    stage = DeepSeekV4HashRouter({2: tables[2]})
    with pytest.raises(ValueError, match="absolute layer"):
        stage(x, ids, absolute_layer=0, model_version="model-v1", table_version="tables-v1")
    custom = prepare(source, 8, hash_layers=(4, 8, 12))
    assert route(x, ids, custom).execution["absolute_layer"] == 8


@gpu
def test_owned_snapshot_and_mutation_version_guards():
    source = synthetic_table()
    prepared = prepare(source)
    ids = torch.tensor([0], device="cuda")
    x = torch.zeros(1, 256, device="cuda")
    expected = route(x, ids, prepared).expert_ids.clone()
    source.data.add_(1)
    assert torch.equal(route(x, ids, prepared).expert_ids, expected)
    prepared._prepared.data[0, 0] = 9
    with pytest.raises(ValueError, match="content changed"):
        route(x, ids, prepared)


@gpu
def test_prepared_table_reshape_preserving_bytes_is_rejected():
    prepared = prepare(synthetic_table())
    prepared._prepared.data = prepared._prepared.data.reshape(6, 23)
    with pytest.raises(ValueError, match="layout changed"):
        route(torch.zeros(1, 256, device="cuda"), torch.tensor([0], device="cuda"), prepared)


@gpu
@pytest.mark.parametrize(
    "field,value",
    [
        ("model_version", "model-v2"),
        ("table_version", "tables-v2"),
        ("absolute_layer", 1),
        ("absolute_layer", True),
    ],
)
def test_stale_or_wrong_layer_rejected(field, value):
    table = prepare(synthetic_table())
    x = torch.zeros(1, 256, device="cuda")
    ids = torch.tensor([0], device="cuda")
    kwargs = dict(absolute_layer=0, model_version="model-v1", table_version="tables-v1")
    kwargs[field] = value
    with pytest.raises(ValueError, match="version|absolute layer"):
        hash_route(x, ids, table, **kwargs)


@gpu
@pytest.mark.parametrize("token", [-1, 23, 2**32])
def test_out_of_range_token_ids_never_clamped(token):
    with pytest.raises(ValueError, match="outside table vocabulary"):
        route(
            torch.zeros(1, 256, device="cuda"),
            torch.tensor([token], device="cuda"),
            prepare(synthetic_table()),
        )


@gpu
@pytest.mark.parametrize("invalid", ["duplicate", "negative", "too_large", "checksum"])
def test_bad_table_fails_before_i64_narrowing(invalid):
    table = synthetic_table()
    if invalid == "duplicate":
        table[0, 0] = table[0, 1]
    if invalid == "negative":
        table[0, 0] = -1
    if invalid == "too_large":
        table[0, 0] = 2**32
    kwargs = dict(
        absolute_layer=0,
        model_version="v1",
        table_version="t1",
        expected_sha256=tensor_sha256(table),
    )
    if invalid == "checksum":
        kwargs["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        prepare_hash_route_table(table, **kwargs)


@gpu
def test_empty_input_saved_indices_and_graph_rejection():
    table = prepare(synthetic_table())
    x = torch.zeros(0, 256, device="cuda", requires_grad=True)
    ids = torch.empty(0, device="cuda", dtype=torch.int64)
    result = route(x, ids, table)
    result.weights.sum().backward()
    assert x.grad.shape == x.shape and result.expert_ids.shape == (0, 6)
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    ids = torch.tensor([0], device="cuda")
    result = route(x, ids, table)
    result.expert_ids[0, 0] = 8
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.weights.sum().backward()
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            route(x, ids, table)


@gpu
@pytest.mark.parametrize(
    "kind",
    [
        "bf16",
        "wrong_width",
        "wrong_ids_dtype",
        "wrong_ids_shape",
        "expanded",
        "nan",
        "inf",
        "bad_scale",
        "bad_renormalize",
    ],
)
def test_unsupported_inputs_rejected(kind):
    table = prepare(synthetic_table())
    x = torch.zeros(1, 256, device="cuda")
    ids = torch.tensor([0], device="cuda")
    kwargs = {}
    if kind == "bf16":
        x = x.bfloat16()
    if kind == "wrong_width":
        x = x[:, :255]
    if kind == "wrong_ids_dtype":
        ids = ids.float()
    if kind == "wrong_ids_shape":
        ids = ids.reshape(1, 1)
    if kind == "expanded":
        x = x.expand(2, -1)
        ids = ids.repeat(2)
    if kind == "nan":
        x[0, 0] = torch.nan
    if kind == "inf":
        x[0, 0] = torch.inf
    if kind == "bad_scale":
        kwargs["routed_scaling_factor"] = torch.inf
    if kind == "bad_renormalize":
        kwargs["renormalize"] = 1
    with pytest.raises((ValueError, TypeError)):
        route(x, ids, table, **kwargs)


@gpu
@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_upstream_rejected(value):
    table = prepare(synthetic_table())
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    result = route(x, torch.tensor([0], device="cuda"), table)
    with pytest.raises(ValueError, match="finite upstream"):
        torch.autograd.grad(result.weights, x, torch.full_like(result.weights, value))


@gpu
def test_saved_logit_mutation_and_higher_derivatives_rejected():
    table = prepare(synthetic_table())
    ids = torch.tensor([0], device="cuda")
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    result = route(x, ids, table)
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.weights.sum().backward()
    x = torch.zeros(1, 256, device="cuda", requires_grad=True)
    result = route(x, ids, table)
    dx = torch.autograd.grad(result.weights, x, torch.ones_like(result.weights), create_graph=True)[
        0
    ]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dx.sum(), x)


@pytest.mark.parametrize(
    "layer,hash_layers", [(True, (0, 1, 2)), (3, (0, 1, 2)), (-1, (0, 1, 2)), (0, (0, 0))]
)
def test_invalid_absolute_layer_configuration_cpu(layer, hash_layers):
    with pytest.raises(ValueError):
        prepare_hash_route_table(
            torch.zeros(2, 6, dtype=torch.int64),
            absolute_layer=layer,
            model_version="v",
            table_version="t",
            expected_sha256="0" * 64,
            hash_layers=hash_layers,
        )


def test_independent_fp64_vjp_by_finite_difference():
    x = torch.linspace(-10, 10, 256, dtype=torch.float64).reshape(1, 256)
    ids = torch.tensor([0])
    table = torch.tensor([[250, 1, 60, 7, 3, 199]])
    grad = torch.tensor([[1.0, -2.0, 3.0, -0.5, 0.25, 0.0]], dtype=torch.float64)
    vjp = hash_route_vjp_fp64(x, ids, table, grad)
    for expert in table[0]:
        delta = torch.zeros_like(x)
        delta[0, expert] = 1e-5
        before, _ = hash_route_fp64(x - delta, ids, table)
        after, _ = hash_route_fp64(x + delta, ids, table)
        expected = ((after - before) * grad).sum() / (2e-5)
        torch.testing.assert_close(vjp[0, expert], expected, rtol=1e-7, atol=1e-10)
