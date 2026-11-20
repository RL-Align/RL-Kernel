# SPDX-License-Identifier: Apache-2.0
"""Task-31 exact score/ID ordering, payload bytes, BI and local gather VJP."""

import pytest
import torch

from rl_engine.ops.moe.stable_topk6 import stable_topk6
from rl_engine.reference.moe.stable_topk6 import stable_topk6_reference

gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


def bits(x):
    return x.detach().contiguous().view(torch.int32)


def select(x, ids=None):
    return stable_topk6(
        x,
        global_expert_ids=ids,
        global_expert_count=x.shape[1],
        model_version="task31-synthetic-v1",
    )


def layout(x, kind):
    if kind == "transpose":
        return x.T.contiguous().T
    if kind == "padded":
        storage = torch.zeros(x.shape[0], 2 * x.shape[1] + 3, device=x.device)
        view = storage[:, 1 : 2 * x.shape[1] + 1 : 2]
        view.copy_(x)
        return view
    return x


def assert_reference(x, ids, result):
    values, expected_ids, columns = stable_topk6_reference(x, ids)
    assert torch.equal(result.expert_ids.cpu(), expected_ids)
    assert torch.equal(result.column_indices.cpu().long(), columns)
    assert torch.equal(bits(result.selected_scores).cpu(), bits(values))


@gpu
@pytest.mark.parametrize("experts", [6, 7, 31, 127, 128, 129, 255, 256, 257, 513, 1025, 32767])
def test_complete_global_tail_shapes_match_independent_oracle(experts):
    gen = torch.Generator().manual_seed(31)
    x = (torch.randn(2, experts, generator=gen) * 100).cuda()
    ids = torch.randperm(experts, generator=gen).cuda()
    result = select(x, ids)
    assert_reference(x, ids, result)


@gpu
@pytest.mark.parametrize(
    "value", [0.0, -0.0, 1.0, -1.0, torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]
)
def test_all_ties_prefer_small_global_ids_and_keep_payload_bits(value):
    x = torch.full((17, 256), value, device="cuda")
    ids = torch.arange(255, -1, -1, device="cuda")
    result = select(x, ids)
    assert torch.equal(result.expert_ids, torch.arange(6, device="cuda").expand(17, -1))
    assert_reference(x, ids, result)


@gpu
def test_slots_are_score_order_not_expert_number_order():
    x = torch.full((1, 256), -100.0, device="cuda")
    chosen = torch.tensor([250, 1, 129, 7, 100, 3], device="cuda")
    x[0, chosen] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0], device="cuda")
    result = select(x)
    assert torch.equal(result.expert_ids[0], chosen)
    assert not torch.equal(result.expert_ids[0], chosen.sort().values)


@gpu
@pytest.mark.parametrize("pattern", ["ties", "score_order"])
def test_original_aiter_entry_is_a_negative_control(pattern):
    from aiter.ops.triton.moe.moe_routing.topk import topk

    x = torch.ones(1, 256, device="cuda")
    if pattern == "score_order":
        x.fill_(-100.0)
        x[0, torch.tensor([250, 1, 129, 7, 100, 3], device="cuda")] = torch.arange(
            10.0, 4.0, -1.0, device="cuda"
        )
    _, old_ids, _ = topk(x, 6, apply_softmax=False, score_mode="softmax")
    _, reference_ids, _ = stable_topk6_reference(x)
    assert not torch.equal(old_ids.cpu().long(), reference_ids)
    assert_reference(x, None, select(x))


@gpu
def test_sixth_place_ties_across_stream_blocks_are_stable():
    x = torch.full((1, 256), 2.0, device="cuda")
    leaders = torch.tensor([255, 128, 127, 126, 125], device="cuda")
    x[0, leaders] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0], device="cuda")
    result = select(x)
    assert torch.equal(
        result.expert_ids[0], torch.tensor([255, 128, 127, 126, 125, 0], device="cuda")
    )
    assert_reference(x, None, result)


@gpu
@pytest.mark.parametrize("base", [-100.0, -1.0, 0.0, 1.0, 100.0])
def test_adjacent_fp32_scores_never_use_an_epsilon_tie(base):
    x = torch.full((1, 256), -torch.finfo(torch.float32).max, device="cuda")
    v = torch.tensor(base, dtype=torch.float32)
    values = []
    for _ in range(10):
        values.append(v.item())
        v = torch.nextafter(v, torch.tensor(torch.inf))
    # Larger score deliberately has larger ID. Treating near values as ties is wrong.
    x[0, 10:20] = torch.tensor(values, device="cuda")
    result = select(x)
    assert torch.equal(result.expert_ids[0], torch.arange(19, 13, -1, device="cuda"))
    assert_reference(x, None, result)


@gpu
def test_signed_zero_and_positive_negative_subnormal_ordering():
    raw = torch.tensor(
        [0, -2147483648, 1, -2147483647, 2, -2147483646, 3, -2147483645, 4, -2147483644],
        dtype=torch.int32,
    )
    x = torch.full((1, 256), -1.0, device="cuda")
    x[0, :10] = raw.view(torch.float32).cuda()
    result = select(x)
    assert torch.equal(result.expert_ids[0], torch.tensor([8, 6, 4, 2, 0, 1], device="cuda"))
    assert_reference(x, None, result)


@gpu
@pytest.mark.parametrize("kind", ["contiguous", "padded", "transpose"])
@pytest.mark.parametrize("expert_profile", ["canonical", "permuted"])
def test_batch_layout_neighbors_payload_and_gradient_bi(kind, expert_profile):
    gen = torch.Generator().manual_seed(311)
    experts = 256
    ids = None if expert_profile == "canonical" else torch.randperm(experts, generator=gen).cuda()
    row = (torch.randint(-10, 10, (1, experts), generator=gen).float()).cuda()
    row_grad = torch.randn(1, 6, generator=gen).cuda()
    bx = row.clone().requires_grad_()
    base = select(bx, ids)
    base_dx = torch.autograd.grad(base.selected_scores, bx, row_grad)[0]
    for m in (1, 16, 17, 32, 65, 257):
        x = torch.randn(m, experts, generator=gen).cuda()
        grad = torch.randn(m, 6, generator=gen).cuda()
        positions = sorted({0, m // 2, m - 1})
        x[positions] = row
        grad[positions] = row_grad
        x = layout(x, kind).requires_grad_()
        result = select(x, ids)
        dx = torch.autograd.grad(result.selected_scores, x, layout(grad, kind))[0]
        with torch.inference_mode():
            replay = select(x, ids)
        assert torch.equal(result.expert_ids, replay.expert_ids)
        assert torch.equal(bits(result.selected_scores), bits(replay.selected_scores))
        for pos in positions:
            assert torch.equal(result.expert_ids[pos], base.expert_ids[0])
            assert torch.equal(bits(result.selected_scores[pos]), bits(base.selected_scores[0]))
            assert torch.equal(bits(dx[pos]), bits(base_dx[0]))
        assert_reference(x, ids, result)
        expected = torch.zeros_like(x).scatter_(1, result.column_indices.long(), grad)
        assert torch.equal(bits(dx), bits(expected))


@gpu
@pytest.mark.parametrize("seed", range(10))
def test_random_ties_streaming_merges_and_global_number_permutations(seed):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randint(-10, 11, (17, 1025), generator=gen).float().cuda().requires_grad_()
    ids = torch.randperm(1025, generator=gen).cuda()
    result = select(x, ids)
    assert_reference(x, ids, result)
    permutation = torch.randperm(1025, generator=gen).cuda()
    permuted = select(x[:, permutation], ids[permutation])
    assert torch.equal(result.expert_ids, permuted.expert_ids)
    assert torch.equal(bits(result.selected_scores), bits(permuted.selected_scores))


@gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_global_map_noncontiguous_integer_profiles_and_owned_snapshot(dtype):
    x = torch.ones(1, 256, device="cuda", requires_grad=True)
    storage = torch.zeros(513, device="cuda", dtype=dtype)
    storage[1::2] = torch.arange(255, -1, -1, device="cuda", dtype=dtype)
    ids = storage[1::2]
    result = select(x, ids)
    ids.data.fill_(0)
    grad = torch.tensor([[-0.0, 1.0, -1.0, 2.0, -2.0, 3.0]], device="cuda")
    dx = torch.autograd.grad(result.selected_scores, x, grad)[0]
    assert torch.equal(result.expert_ids[0], torch.arange(6, device="cuda"))
    expected = torch.zeros_like(x).scatter_(1, result.column_indices.long(), grad)
    assert torch.equal(bits(dx), bits(expected))


@gpu
@pytest.mark.parametrize("m", [0, 1, 17])
def test_empty_rows_and_expanded_sum_backward(m):
    x = torch.zeros(m, 256, device="cuda", requires_grad=True)
    result = select(x)
    result.selected_scores.sum().backward()
    assert result.expert_ids.shape == (m, 6) and x.grad.shape == x.shape
    assert result.selected_scores.dtype == torch.float32
    assert not result.expert_ids.requires_grad and not result.column_indices.requires_grad


@gpu
@pytest.mark.parametrize(
    "bad",
    ["duplicate", "negative", "too_large", "overflow_i64", "float", "wrong_length", "expanded"],
)
def test_invalid_global_numbering_never_falls_back_to_local_columns(bad):
    ids = torch.arange(256, device="cuda")
    if bad == "duplicate":
        ids[0] = ids[1]
    if bad == "negative":
        ids[0] = -1
    if bad == "too_large":
        ids[0] = 256
    if bad == "overflow_i64":
        ids[0] = 2**32
    if bad == "float":
        ids = ids.float()
    if bad == "wrong_length":
        ids = ids[:-1]
    if bad == "expanded":
        ids = ids[:1].expand(256)
    with pytest.raises((ValueError, TypeError)):
        select(torch.ones(1, 256, device="cuda"), ids)


@gpu
def test_incomplete_shard_rejected_instead_of_claiming_global_top6():
    with pytest.raises(ValueError, match="complete global scores"):
        stable_topk6(
            torch.ones(17, 128, device="cuda"),
            global_expert_ids=torch.arange(128, 256, device="cuda"),
            global_expert_count=256,
            model_version="v1",
        )


@gpu
@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_scores_and_gradients_rejected(invalid):
    x = torch.ones(1, 256, device="cuda")
    x[0, 0] = invalid
    with pytest.raises(ValueError, match="must be finite"):
        select(x)
    x = torch.ones(1, 256, device="cuda", requires_grad=True)
    r = select(x)
    with pytest.raises(ValueError, match="finite upstream"):
        torch.autograd.grad(r.selected_scores, x, torch.full_like(r.selected_scores, invalid))


@gpu
@pytest.mark.parametrize("target", ["scores", "selected_scores", "expert_ids", "column_indices"])
def test_saved_tensor_mutation_detected(target):
    x = torch.ones(1, 256, device="cuda", requires_grad=True)
    r = select(x)
    with torch.no_grad():
        (x if target == "scores" else getattr(r, target)).add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        r.selected_scores.sum().backward()


@gpu
def test_graph_and_higher_order_derivative_rejected():
    x = torch.ones(1, 256, device="cuda", requires_grad=True)
    r = select(x)
    dx = torch.autograd.grad(
        r.selected_scores, x, torch.ones_like(r.selected_scores), create_graph=True
    )[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(dx.sum(), x)
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            select(x)


@pytest.mark.parametrize(
    "shape,count,error",
    [
        ((1, 256), True, ValueError),
        ((1, 256), 5, ValueError),
        ((1, 256), 32768, ValueError),
        ((1, 128), 256, ValueError),
        ((256,), 256, ValueError),
        ((1, 256), 256, RuntimeError),
    ],
)
def test_cpu_request_rejection_before_backend_import(shape, count, error):
    with pytest.raises(error):
        stable_topk6(torch.ones(shape), global_expert_count=count, model_version="v1")


def test_version_and_dtype_required():
    with pytest.raises(ValueError, match="model_version"):
        stable_topk6(torch.ones(1, 256), model_version="")
    with pytest.raises(TypeError, match="FP32"):
        stable_topk6(torch.ones(1, 256, dtype=torch.bfloat16), model_version="v1")


def test_cpu_reference_orders_zero_ties_by_global_id():
    scores = torch.tensor([[0.0, -0.0, 1.0, -1.0, 2.0, 2.0]])
    ids = torch.tensor([5, 0, 4, 1, 3, 2])
    values, chosen, _ = stable_topk6_reference(scores, ids)
    assert chosen.tolist() == [[2, 3, 4, 0, 5, 1]]
    assert bits(values).tolist() == bits(torch.tensor([[2.0, 2.0, 1.0, -0.0, 0.0, -1.0]])).tolist()
