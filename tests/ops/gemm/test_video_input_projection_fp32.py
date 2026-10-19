# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 FP32 projection accuracy, actual autograd, and logical-row invariance."""

import json
import os

import pytest
import torch

from rl_engine.contracts.numerical import load_contract
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.ops.gemm.canonical_linear import canonical_linear_fp32
from rl_engine.ops.gemm.video_input_projection_fp32 import VideoInputProjectionFP32Op
from rl_engine.reference.gemm.video_input_projection_fp32 import NativeVideoInputProjectionFP32Op

HAS_SM90 = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() == (9, 0)
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")),
    ],
)
@pytest.mark.parametrize(
    "key_rows",
    [
        [],
        [[0, 0]],
        [[0, 0], [0, 1], [1, 0]],
        [[0, 1], [0, 0], [1, 0]],
        [[0, 0], [-1, 0], [1, 0]],
        [[0, 0], [0, 0]],
        [[0, 2**63 - 1], [1, -(2**63)]],
        [[0, 0, 1], [0, 1, -1], [1, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 0, 1]],
    ],
)
def test_canonical_order_proof_preserves_stable_normalization(key_rows, device, monkeypatch):
    from rl_engine.ops.autograd.canonical_backward import (
        CanonicalBackwardSession,
        _keys_are_canonical,
    )

    monkeypatch.setattr(
        "rl_engine.ops.autograd.canonical_backward._CANONICAL_ORDER_REUSE_MIN_ROWS", 0
    )
    columns = len(key_rows[0]) if key_rows else 2
    keys = torch.tensor(key_rows, dtype=torch.int64, device=device).reshape(-1, columns)
    values = torch.arange(4 * len(key_rows), device=device).reshape(4, len(key_rows)).t()
    expected_order = sorted(
        (i for i, key in enumerate(key_rows) if key[0] >= 0), key=lambda i: key_rows[i]
    )
    ordered = expected_order == list(range(len(key_rows)))
    # Equal keys still require the stable fallback, even in physical order.
    strict = ordered and len({tuple(key) for key in key_rows}) == len(key_rows)
    assert _keys_are_canonical(keys) == strict
    if not strict:
        assert not _keys_are_canonical(keys)
    elif device == "cuda":

        def unexpected_sort(*args, **kwargs):
            raise AssertionError("canonical keys must reuse their physical order")

        monkeypatch.setattr(torch, "argsort", unexpected_sort)
    session = CanonicalBackwardSession()
    session.register("weight", keys)
    order, actual = session._normalize("weight", [values])
    expected = values.index_select(
        0, torch.tensor(expected_order, dtype=torch.int64, device=device)
    )
    assert order.tolist() == expected_order
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()


def test_canonical_normalization_reads_changed_keys(monkeypatch):
    from rl_engine.ops.autograd.canonical_backward import CanonicalBackwardSession

    monkeypatch.setattr(
        "rl_engine.ops.autograd.canonical_backward._CANONICAL_ORDER_REUSE_MIN_ROWS", 0
    )
    keys = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
    values = torch.tensor([[1.0], [2.0]])
    session = CanonicalBackwardSession()
    session.register("weight", keys)
    _, first = session._normalize("weight", [values])
    keys.copy_(keys.flip(0))
    _, second = session._normalize("weight", [values])
    assert torch.equal(first, values)
    assert torch.equal(second, values.flip(0))


def test_cpu_callers_retain_stable_normalization(monkeypatch):
    from rl_engine.ops.autograd import canonical_backward as canonical

    def unexpected_proof(keys):
        raise AssertionError("existing callers must retain stable normalization")

    monkeypatch.setattr(canonical, "_keys_are_canonical", unexpected_proof)
    monkeypatch.setattr(canonical, "_CANONICAL_ORDER_REUSE_MIN_ROWS", 0)
    session = canonical.CanonicalBackwardSession()
    keys = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
    values = torch.tensor([[1.0], [2.0]])
    session.register("weight", keys)
    order, actual = session._normalize("weight", [values])
    assert order.tolist() == [0, 1]
    assert torch.equal(actual, values)


def test_small_canonical_uses_retain_stable_normalization(monkeypatch):
    from rl_engine.ops.autograd import canonical_backward as canonical

    def unexpected_proof(keys):
        raise AssertionError("small inputs must avoid the order proof overhead")

    monkeypatch.setattr(canonical, "_keys_are_canonical", unexpected_proof)
    session = canonical.CanonicalBackwardSession()
    keys = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
    values = torch.tensor([[1.0], [2.0]])
    session.register("weight", keys)
    order, actual = session._normalize("weight", [values])
    assert order.tolist() == [0, 1]
    assert torch.equal(actual, values)


def _op():
    from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
        TritonVideoInputProjectionFP32Op,
    )

    return TritonVideoInputProjectionFP32Op()


def _keys(rows, device):
    return torch.stack(
        (torch.zeros(rows, dtype=torch.int64, device=device), torch.arange(rows, device=device)),
        dim=1,
    )


def _data(rows, device="cuda", *, seed=420):
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(rows, 96, generator=generator, device=device) * 0.125,
        torch.randn(5376, 96, generator=generator, device=device) * 0.0625,
        torch.randn(5376, generator=generator, device=device) * 0.125,
        torch.randn(rows, 5376, generator=generator, device=device) * 0.0625,
    )


def _run(op, data, keys, chunks=None, *, train_weight=True, train_bias=True):
    x, weight, bias, grad = [None if t is None else t.detach().clone() for t in data]
    x.requires_grad_(True)
    weight.requires_grad_(train_weight)
    if bias is not None:
        bias.requires_grad_(train_bias)
    cuts = chunks if chunks is not None else [torch.arange(x.shape[0], device=x.device)]
    with canonical_backward_session() as session:
        outputs = [
            op(
                x.index_select(0, ix),
                weight,
                bias,
                logical_keys=keys.index_select(0, ix),
                parameter_id="proj_in",
            )
            for ix in cuts
        ]
        torch.autograd.backward(outputs, [grad.index_select(0, ix) for ix in cuts])
        session.validate_complete()
    # Scatter the per-call outputs back to physical row order.
    y = torch.empty(x.shape[0], 5376, device=x.device)
    for ix, out in zip(cuts, outputs, strict=True):
        y.index_copy_(0, ix, out.detach())
    return y, x.grad, weight.grad, None if bias is None else bias.grad


def _assert_accuracy(actual, expected, kind):
    threshold = load_contract()["judgments"][kind]["by_op_class"]["reduction"]["float32"]
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), atol=threshold["atol"], rtol=threshold["rtol"]
    )


def _assert_bytes(a, b):
    assert torch.equal(
        a.detach().contiguous().view(torch.int32), b.detach().contiguous().view(torch.int32)
    )


@pytest.mark.parametrize("output", ["y", "dx", "dw", "dbias"])
@pytest.mark.parametrize(
    "candidate,reference,violations,non_finite",
    [
        (1.0, 1.0, 0, 0),
        (2.0, 1.0, 1, 0),
        (float("nan"), 1.0, 1, 1),
        (1.0, float("nan"), 1, 1),
        (float("inf"), 1.0, 1, 1),
        (1.0, float("inf"), 1, 1),
        (float("inf"), float("inf"), 1, 1),
        (-float("inf"), 1.0, 1, 1),
    ],
)
def test_benchmark_accuracy_rejects_non_finite(
    output, candidate, reference, violations, non_finite
):
    from benchmarks.operators.gemm.benchmark_video_input_projection_fp32 import _accuracy

    names = ("y", "dx", "dw", "dbias")
    actual = [torch.ones(2) for _ in names]
    gold = [torch.ones(2) for _ in names]
    actual[names.index(output)][0] = candidate
    gold[names.index(output)][0] = reference
    report = _accuracy(actual, gold)
    for name in names:
        assert report[name]["out_of_tolerance"] == (violations if name == output else 0)
        assert report[name]["non_finite_elements"] == (non_finite if name == output else 0)
    if non_finite:
        assert report[output]["max_abs_error"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("with_bias", [False, True])
def test_canonical_linear_bias_and_legacy_call(with_bias):
    torch.manual_seed(420)
    x = torch.randn(7, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    bias = torch.randn(5, requires_grad=True) if with_bias else None
    grad = torch.randn(7, 5)
    with canonical_backward_session() as session:
        y = canonical_linear_fp32(
            x, weight, _keys(7, "cpu"), parameter_id="weight", family="pytorch", bias=bias
        )
        y.backward(grad)
        session.validate_complete()
    torch.testing.assert_close(y, torch.nn.functional.linear(x, weight, bias))
    torch.testing.assert_close(x.grad, grad @ weight)
    torch.testing.assert_close(weight.grad, grad.t() @ x)
    if with_bias:
        torch.testing.assert_close(bias.grad, grad.sum(0))


@pytest.mark.parametrize("with_bias", [False, True])
def test_canonical_linear_custom_matmul_covers_all_vjps(with_bias, monkeypatch):
    from rl_engine.ops.autograd.backward_runtime import (
        reset_backward_runtime,
        snapshot_backward_runtime,
    )
    from rl_engine.ops.autograd.canonical_backward import CanonicalBackwardSession

    torch.manual_seed(424)
    x = torch.randn(7, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    bias = torch.randn(5, requires_grad=True) if with_bias else None
    grad = torch.randn(7, 5)
    calls, registrations = [], []
    original_register = CanonicalBackwardSession.register

    def register(self, parameter_id, keys, **kwargs):
        registrations.append(parameter_id)
        return original_register(self, parameter_id, keys, **kwargs)

    def gemm(a, b, *, bias=None):
        calls.append((a.shape, b.shape, a.stride(0), bias is not None))
        value = a @ b
        return value if bias is None else value + bias

    def two_argument_gemm(a, b):
        return gemm(a, b)

    matmul_op = gemm if with_bias else two_argument_gemm
    monkeypatch.setattr(CanonicalBackwardSession, "register", register)
    reset_backward_runtime()
    try:
        with canonical_backward_session() as session:
            y = canonical_linear_fp32(
                x,
                weight,
                _keys(7, "cpu"),
                parameter_id="weight",
                family="pytorch",
                bias=bias,
                matmul_op=matmul_op,
            )
            y.backward(grad)
            session.validate_complete()
        torch.testing.assert_close(y, torch.nn.functional.linear(x, weight, bias))
        torch.testing.assert_close(x.grad, grad @ weight.detach())
        torch.testing.assert_close(weight.grad, grad.t() @ x.detach())
        expected_shapes = [((7, 3), (3, 5)), ((7, 5), (5, 3)), ((5, 7), (7, 3))]
        if with_bias:
            torch.testing.assert_close(bias.grad, grad.sum(0))
            expected_shapes.append(((1, 7), (7, 5)))
        assert [(a, b) for a, b, _, _ in calls] == expected_shapes
        assert calls[0][3] == with_bias
        assert all(not call[3] for call in calls[1:])
        assert calls[2][2] == 1
        assert registrations == (["weight", "weight.bias"] if with_bias else ["weight"])
        provenance = snapshot_backward_runtime()["det_gemm"]
        assert provenance["kernel_id"] == f"{matmul_op.__module__}.{matmul_op.__qualname__}"
        assert provenance["family"] == "pytorch"
    finally:
        reset_backward_runtime()


def test_canonical_linear_default_triton_retains_existing_dispatch(monkeypatch):
    from rl_engine.ops.autograd.backward_runtime import (
        reset_backward_runtime,
        snapshot_backward_runtime,
    )
    from rl_engine.ops.gemm import canonical_linear

    calls = []

    def existing_gemm(a, b, *, output_dtype):
        assert output_dtype == torch.float32
        calls.append((a.shape, b.shape))
        return a @ b

    monkeypatch.setattr(canonical_linear, "_triton_gemm", existing_gemm)
    x = torch.randn(7, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    grad = torch.randn(7, 5)
    reset_backward_runtime()
    try:
        with canonical_backward_session() as session:
            output = canonical_linear_fp32(
                x, weight, _keys(7, "cpu"), parameter_id="weight", family="triton"
            )
            output.backward(grad)
            session.validate_complete()
        torch.testing.assert_close(output, x @ weight.t())
        torch.testing.assert_close(x.grad, grad @ weight.detach())
        torch.testing.assert_close(weight.grad, grad.t() @ x.detach())
        assert calls == [((7, 3), (3, 5)), ((7, 5), (5, 3)), ((5, 7), (7, 3))]
        assert snapshot_backward_runtime()["det_gemm"]["kernel_id"] == (
            "rl_engine.backends.shared.triton.gemm.det_gemm._triton_gemm"
        )
    finally:
        reset_backward_runtime()


def test_video_projection_declares_backend_and_matmul_separately():
    from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
        TritonVideoInputProjectionFP32Op,
        projection_gemm_fp32,
    )

    assert TritonVideoInputProjectionFP32Op.family == "triton"
    assert TritonVideoInputProjectionFP32Op.matmul_op is projection_gemm_fp32


def test_input_gradient_without_parameter_session():
    x = torch.ones(3, 2, requires_grad=True)
    weight = torch.ones(4, 2)
    canonical_linear_fp32(x, weight, None, parameter_id="w", family="pytorch").sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4))


class _CPUAdapter(VideoInputProjectionFP32Op):
    family = "pytorch"

    def _check_backend(self, x):
        pass


@pytest.mark.parametrize("frozen", ["weight", "bias", "both"])
def test_frozen_parameter_registration(frozen):
    result = _run(
        _CPUAdapter(),
        _data(2, "cpu"),
        _keys(2, "cpu"),
        train_weight=frozen not in ("weight", "both"),
        train_bias=frozen not in ("bias", "both"),
    )
    assert (result[2] is None) == (frozen in ("weight", "both"))
    assert (result[3] is None) == (frozen in ("bias", "both"))


@pytest.mark.parametrize(
    "invalid", ["dtype", "shape", "empty", "bias", "noncontiguous", "keys", "duplicates", "session"]
)
def test_invalid_inputs(invalid):
    x, weight, bias, _ = _data(2, "cpu")
    keys = _keys(2, "cpu")
    if invalid == "dtype":
        x = x.bfloat16()
    elif invalid == "shape":
        x = x[:, :95].contiguous()
    elif invalid == "empty":
        x = x[:0]
    elif invalid == "bias":
        bias = bias[:10]
    elif invalid == "noncontiguous":
        x = torch.empty(2, 192)[:, ::2]
    elif invalid == "keys":
        keys = keys.int()
    elif invalid == "duplicates":
        keys.zero_()
    weight.requires_grad_(True)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        if invalid == "session":
            _CPUAdapter()(x, weight, bias, logical_keys=keys, parameter_id="p")
        else:
            with canonical_backward_session():
                _CPUAdapter()(x, weight, bias, logical_keys=keys, parameter_id="p")


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("rows", [1, 9, 17, 65, 129, 257])
@pytest.mark.parametrize("with_bias", [False, True])
def test_forward_and_actual_gradients(rows, with_bias):
    data = list(_data(rows))
    if not with_bias:
        data[2] = None
    # _run clones optional bias separately to keep the production call identical.
    if not with_bias:
        x, weight, _, grad = data
        x, weight = x.requires_grad_(), weight.requires_grad_()
        with canonical_backward_session() as session:
            y = _op()(x, weight, logical_keys=_keys(rows, "cuda"), parameter_id="p")
            y.backward(grad)
            session.validate_complete()
        actual = (y, x.grad, weight.grad)
    else:
        actual = _run(_op(), data, _keys(rows, "cuda"))
    cx, cw = data[0].detach().cpu().requires_grad_(), data[1].detach().cpu().requires_grad_()
    cb = None if not with_bias else data[2].detach().cpu().requires_grad_()
    gold = NativeVideoInputProjectionFP32Op()(cx, cw, cb)
    gold.backward(data[3].cpu())
    expected = (gold, cx.grad, cw.grad) if cb is None else (gold, cx.grad, cw.grad, cb.grad)
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        _assert_accuracy(a, b, "forward_accuracy" if i == 0 else "gradient_accuracy")


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("rows", [9, 65, 127, 128, 129, 257])
def test_parameter_bi_permutation_chunk_padding(rows):
    op, data, keys = _op(), _data(rows), _keys(rows, "cuda")
    _check_parameter_bi(op, data, keys)


def _check_parameter_bi(op, data, keys):
    rows = data[0].shape[0]
    baseline = _run(op, data, keys)
    order = torch.randperm(rows, device="cuda")
    permuted = _run(
        op,
        [t.index_select(0, order) if i in (0, 3) else t for i, t in enumerate(data)],
        keys.index_select(0, order),
    )
    inverse = order.argsort()
    for i in range(4):
        _assert_bytes(baseline[i], permuted[i].index_select(0, inverse) if i < 2 else permuted[i])
    chunks = list(torch.tensor_split(order, 3))
    chunked = _run(op, data, keys, chunks)
    for a, b in zip(baseline, chunked, strict=True):
        _assert_bytes(a, b)
    padded = [
        torch.cat((data[0], torch.randn(7, 96, device="cuda"))),
        data[1],
        data[2],
        torch.cat((data[3], torch.randn(7, 5376, device="cuda"))),
    ]
    padded_keys = torch.cat((keys, torch.full((7, 2), -1, device="cuda", dtype=torch.int64)))
    result = _run(op, padded, padded_keys)
    for i, (a, b) in enumerate(zip(baseline, result, strict=True)):
        _assert_bytes(a, b[:rows] if i < 2 else b)
    repeated = _run(op, data, keys)
    for a, b in zip(baseline, repeated, strict=True):
        _assert_bytes(a, b)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
def test_row_placement_and_unrelated_rows():
    op, data = _op(), _data(65)
    single = _run(op, [data[0][:1], data[1], data[2], data[3][:1]], _keys(1, "cuda"))
    data[0][37] = data[0][0]
    data[3][37] = data[3][0]
    batch = _run(op, data, _keys(65, "cuda"))
    _assert_bytes(single[0][0], batch[0][37])
    _assert_bytes(single[1][0], batch[1][37])
    changed = [t.clone() for t in data]
    changed[0][0:30] += 3
    changed[3][0:30] -= 2
    result = _run(op, changed, _keys(65, "cuda"))
    _assert_bytes(batch[0][37], result[0][37])
    _assert_bytes(batch[1][37], result[1][37])


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
def test_noncontiguous_upstream_and_forward_graph():
    op = _op()
    x, weight, bias, grad = _data(17)
    with canonical_backward_session() as session:
        y = op(
            x.requires_grad_(),
            weight.requires_grad_(),
            bias.requires_grad_(),
            logical_keys=_keys(17, "cuda"),
            parameter_id="p",
        )
        y.sum().backward()  # Expanded scalar upstream has zero strides.
        session.validate_complete()
    _assert_accuracy(x.grad, torch.ones_like(grad).cpu() @ weight.cpu(), "gradient_accuracy")
    with torch.no_grad():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                expected = op(x, weight, bias)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(x, weight, bias)
        graph.replay()
        _assert_bytes(output, expected)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
def test_real_weights_patch_path():
    path = os.environ.get("H3_PROJECTION_WEIGHTS")
    if not path:
        pytest.skip("set H3_PROJECTION_WEIGHTS to pinned H3 safetensors fixture")
    from safetensors import safe_open
    from safetensors.torch import load_file

    from rl_engine.reference.packing.video_patch_pack_unpack import NativeVideoPatchPackUnpackOp

    with safe_open(path, framework="pt") as fixture:
        metadata = fixture.metadata()
        assert metadata["repository"] == "MiniMaxAI/MiniMax-H3"
        assert metadata["checkpoint_revision"] == "42ed227ee7df40d41602854ae760620d6eb651fe"
    tensors = load_file(path)
    weight = tensors["proj_in.weight"].cuda().float().contiguous()
    bias = tensors["proj_in.bias"].cuda().float().contiguous()
    torch.manual_seed(420)
    latent = torch.randn(1, 24, 13, 48, 84, device="cuda", dtype=torch.bfloat16)
    x = NativeVideoPatchPackUnpackOp().pack(latent).reshape(-1, 96).float()
    grad = torch.randn(x.shape[0], 5376, device="cuda") * 0.0625
    actual = _run(_op(), (x, weight, bias, grad), _keys(x.shape[0], "cuda"))
    cx, cw, cb = [t.detach().cpu().requires_grad_() for t in (x, weight, bias)]
    gold = NativeVideoInputProjectionFP32Op()(cx, cw, cb)
    gold.backward(grad.cpu())
    for i, (a, b) in enumerate(zip(actual, (gold, cx.grad, cw.grad, cb.grad), strict=True)):
        _assert_accuracy(a, b, "forward_accuracy" if i == 0 else "gradient_accuracy")
    order = torch.arange(x.shape[0] - 1, -1, -1, device="cuda")
    chunked = _run(
        _op(),
        (x, weight, bias, grad),
        _keys(x.shape[0], "cuda"),
        list(torch.tensor_split(order, 3)),
    )
    for a, b in zip(actual, chunked, strict=True):
        _assert_bytes(a, b)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
def test_default_cuda_dispatch():
    from rl_engine.runtime.registry import KernelRegistry

    op = KernelRegistry().get_op("video_input_projection_fp32", device="cuda")
    assert op.backend_info()["backend_id"] == "triton-video-input-projection-fp32-v11"
    x, weight, bias, _ = _data(3)
    with torch.no_grad():
        _assert_bytes(op(x, weight, bias), _op()(x, weight, bias))


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
def test_reused_fp32_gemm():
    from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

    x, weight, _, _ = _data(17)
    output = _triton_gemm_fp32(x, weight.t().contiguous())
    _assert_accuracy(output, x.cpu() @ weight.cpu().t(), "forward_accuracy")


def _fp32_tree_reference(a, b):
    # Evaluate the existing plan with independent CPU FP32 scalar arithmetic.
    from rl_engine.backends.shared.triton.gemm.det_gemm import _build_tree_plan

    a, b = a.cpu(), b.cpu()
    plan = _build_tree_plan(a.shape[1])
    nodes = {}
    for start, length, node in zip(
        plan.leaf_starts, plan.leaf_lengths, plan.leaf_nodes, strict=True
    ):
        value = torch.zeros(a.shape[0], b.shape[1])
        for k in range(start, start + length):
            value = value + a[:, k, None] * b[k, None, :]
        nodes[node] = value
    for level in plan.reduction_levels:
        for lower, upper, output in level:
            nodes[output] = nodes[lower] + nodes[upper]
    return nodes[plan.root]


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("k", [31, 33, 129, 5376])
def test_fp32_tree_preserves_fp32_nodes(k):
    from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_tree_gemm

    torch.manual_seed(420)
    a, b = torch.randn(3, k, device="cuda"), torch.randn(k, 5, device="cuda")
    output = _triton_tree_gemm(a, b, output_dtype=torch.float32)
    assert output.dtype == torch.float32
    _assert_bytes(output.cpu(), _fp32_tree_reference(a, b))
    with torch.inference_mode():
        inference = _triton_tree_gemm(a, b, output_dtype=torch.float32)
    _assert_bytes(inference, output)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
def test_fp32_tree_workspace_chunking(monkeypatch):
    from rl_engine.backends.shared.triton.gemm import det_gemm
    from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
        projection_gemm_fp32,
    )

    a, b = torch.randn(7, 129, device="cuda"), torch.randn(129, 5, device="cuda")
    expected = projection_gemm_fp32(a, b)
    plan = det_gemm._build_tree_plan(129)
    monkeypatch.setattr(det_gemm, "_MAX_TREE_WORKSPACE_ELEMENTS", plan.node_count * 5 * 2)
    actual = projection_gemm_fp32(a, b)
    _assert_bytes(actual, expected)
    monkeypatch.setattr(det_gemm, "_MAX_TREE_WORKSPACE_ELEMENTS", plan.node_count * 2)
    # Exercise output-column chunking when even one whole row exceeds budget.
    _assert_bytes(projection_gemm_fp32(a, b), expected)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("rows", [129, 13104])
@pytest.mark.parametrize("seed", [420, 421, 422, 423])
def test_standard_normal_long_reduction(rows, seed):
    # Unscaled inputs expose cancellation error in the 5376-term input VJP.
    data = _data(rows, seed=seed)
    data = tuple(t / scale for t, scale in zip(data, (0.125, 0.0625, 0.125, 0.0625), strict=True))
    actual = _run(_op(), data, _keys(rows, "cuda"))
    cx, cw, cb = [t.detach().cpu().requires_grad_() for t in data[:3]]
    gold = NativeVideoInputProjectionFP32Op()(cx, cw, cb)
    gold.backward(data[3].cpu())
    for i, (a, b) in enumerate(zip(actual, (gold, cx.grad, cw.grad, cb.grad), strict=True)):
        _assert_accuracy(a, b, "forward_accuracy" if i == 0 else "gradient_accuracy")
    _check_parameter_bi(_op(), data, _keys(rows, "cuda"))


def test_cpu_fp32_reference_long_dot_accuracy():
    from rl_engine.reference.gemm.video_input_projection_fp32 import _reference_matmul_fp32

    generator = torch.Generator().manual_seed(420)
    a = torch.randn(3, 13104, generator=generator)
    b = torch.randn(13104, 5, generator=generator)
    expected = (a.double() @ b.double()).float()
    actual = _reference_matmul_fp32(a, b)
    assert actual.dtype == torch.float32
    _assert_accuracy(actual, expected, "gradient_accuracy")


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("k", [127, 128, 129, 5376, 13104])
def test_fp32_chunked_gemm_strides_and_workspace(k, monkeypatch):
    from rl_engine.backends.shared.triton.gemm import det_gemm
    from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
        projection_gemm_fp32,
    )
    from rl_engine.reference.gemm.video_input_projection_fp32 import _reference_matmul_fp32

    torch.manual_seed(420)
    a = torch.randn(k, 7, device="cuda").t()
    b = torch.randn(k, 96, device="cuda")
    expected = projection_gemm_fp32(a.contiguous(), b)
    actual = projection_gemm_fp32(a, b)
    _assert_bytes(actual, expected)
    _assert_accuracy(actual, _reference_matmul_fp32(a, b), "gradient_accuracy")
    chunks = (k + 127) // 128
    monkeypatch.setattr(det_gemm, "_MAX_TREE_WORKSPACE_ELEMENTS", chunks * 96 * 2)
    _assert_bytes(projection_gemm_fp32(a, b), expected)


@pytest.mark.parametrize("bias_uses", ["all", "partial", "distinct"])
def test_canonical_bias_sharing_requires_matching_contributions(bias_uses, monkeypatch):
    from rl_engine.ops.autograd.canonical_backward import CanonicalBackwardSession

    torch.manual_seed(421)
    x = torch.randn(7, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    biases = [torch.randn(5, requires_grad=True) for _ in range(2)]
    grad = torch.randn(7, 5)
    indices = [torch.tensor([4, 1, 6]), torch.tensor([0, 5, 2, 3])]
    keys = _keys(7, "cpu")
    keys[2, 0] = -1
    normalized = []
    original = CanonicalBackwardSession._normalize

    def observe(self, parameter_id, values):
        result = original(self, parameter_id, values)
        normalized.append((parameter_id, result[1]))
        return result

    monkeypatch.setattr(CanonicalBackwardSession, "_normalize", observe)
    with canonical_backward_session() as session:
        outputs = []
        for use, ix in enumerate(indices):
            bias = (
                None
                if bias_uses == "partial" and use == 1
                else biases[use if bias_uses == "distinct" else 0]
            )
            outputs.append(
                canonical_linear_fp32(
                    x[ix],
                    weight,
                    keys[ix],
                    parameter_id="weight",
                    family="pytorch",
                    bias=bias,
                    bias_parameter_id=f"bias.{use if bias_uses == 'distinct' else 0}",
                )
            )
        torch.autograd.backward(outputs, [grad[ix] for ix in indices])
        session.validate_complete()
        assert not session._normalized
    active = keys[:, 0] >= 0
    torch.testing.assert_close(weight.grad, grad[active].t() @ x.detach()[active])
    torch.testing.assert_close(x.grad, grad @ weight.detach())
    for use, bias in enumerate(biases):
        if use == 1 and bias_uses != "distinct":
            assert bias.grad is None
            continue
        ix = torch.arange(7) if bias_uses == "all" else indices[use]
        ix = ix[active[ix]]
        torch.testing.assert_close(bias.grad, grad[ix].sum(0))
    if bias_uses == "all":
        assert len(normalized) == 2
        assert normalized[0][1] is normalized[1][1]
    else:
        assert all(a[1] is not b[1] for i, a in enumerate(normalized) for b in normalized[i + 1 :])


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("k", [64, 65, 128, 1024, 13104])
def test_fp32_tree_default_schedule_and_bias_match_scalar_reference(k):
    from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_tree_gemm

    torch.manual_seed(422)
    # Multirow FP32 calls use the original schedule; bias retains one-row fusion.
    a = torch.randn(k, 3, device="cuda").t()
    b = torch.randn(k, 5, device="cuda")

    def gemm(a, b):
        return _triton_tree_gemm(
            a,
            b,
            output_dtype=torch.float32,
            preserve_a_strides=True,
        )

    result = gemm(a, b)
    _assert_bytes(result.cpu(), _fp32_tree_reference(a, b))
    _assert_bytes(gemm(a[:1], b), result[:1])


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("k", [128, 1024, 13104])
def test_compact_bias_tree_workspace_preserves_chunked_outputs(k, monkeypatch):
    from rl_engine.backends.shared.triton.gemm import det_gemm
    from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
        projection_gemm_fp32,
    )

    a, b = torch.ones(1, k, device="cuda"), torch.randn(k, 5, device="cuda")
    original_plan = det_gemm._device_tree_plan(k, a.device)
    assert not original_plan.fp32_fused
    assert original_plan.workspace_node_count == original_plan.host.node_count
    plan = det_gemm._device_tree_plan(k, a.device, fp32_fuse_leaves=True)
    assert plan.fp32_fused
    assert plan.workspace_node_count < plan.host.node_count
    node_arrays = [plan.rocm_leaf_reduction[-1]]
    node_arrays.extend(
        nodes for pair in plan.rocm_fused_reduction_pairs_after_leaf for nodes in pair
    )
    for nodes in node_arrays:
        assert bool(torch.all((nodes >= 0) & (nodes < plan.workspace_node_count)))
    expected = projection_gemm_fp32(a, b)
    monkeypatch.setattr(det_gemm, "_MAX_TREE_WORKSPACE_ELEMENTS", plan.workspace_node_count * 2)
    _assert_bytes(projection_gemm_fp32(a, b), expected)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("k", [129, 5376, 13104])
def test_shortened_fp32_leaves_preserve_padded_arithmetic(rows, k, monkeypatch):
    from rl_engine.backends.shared.triton.gemm import det_gemm

    class PaddedLeaves:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kwargs["LEAF_LOOP_BOUND"] = det_gemm._BLOCK_K
                return self.kernel[grid](*args, **kwargs)

            return launch

    torch.manual_seed(423)
    a = torch.randn(k, rows, device="cuda").t()
    b = torch.randn(k, 5, device="cuda")
    inputs = [(a, b)]
    # Include signed zero, gradual underflow, and non-finite propagation.
    for value in (-0.0, torch.finfo(torch.float32).tiny / 8, float("inf"), float("nan")):
        inputs.append((torch.full_like(a, value), b))
    actual = [
        det_gemm._triton_tree_gemm(a, b, output_dtype=torch.float32, preserve_a_strides=True)
        for a, b in inputs
    ]
    for name in ("_det_gemm_kernel", "_det_gemm_tree_leaf_reduce_rocm_kernel"):
        monkeypatch.setattr(det_gemm, name, PaddedLeaves(getattr(det_gemm, name)))
    for (a, b), output in zip(inputs, actual, strict=True):
        padded = det_gemm._triton_tree_gemm(
            a, b, output_dtype=torch.float32, preserve_a_strides=True
        )
        _assert_bytes(output, padded)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("rows", [1, 17, 129])
def test_forward_bias_fusion_preserves_fp32_rounding(rows):
    from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

    x, weight, bias, _ = _data(rows)
    gemm = _triton_gemm_fp32
    rhs = weight.t().contiguous()
    expected = gemm(x, rhs) + bias
    actual = gemm(x, rhs, bias=bias)
    _assert_bytes(actual, expected)
    with torch.no_grad():
        _assert_bytes(_op()(x, weight, bias), expected)


@pytest.mark.skipif(not HAS_SM90, reason="requires CUDA SM90")
@pytest.mark.parametrize("frozen", ["weight", "bias", "both"])
def test_backend_frozen_parameters_preserve_trainable_gradients(frozen):
    op = _op()
    data, keys = _data(129), _keys(129, "cuda")
    expected = _run(op, data, keys)
    actual = _run(
        op,
        data,
        keys,
        train_weight=frozen not in ("weight", "both"),
        train_bias=frozen not in ("bias", "both"),
    )
    for i, (value, gold) in enumerate(zip(actual, expected, strict=True)):
        if i == 2 and frozen in ("weight", "both") or i == 3 and frozen in ("bias", "both"):
            assert value is None
        else:
            _assert_bytes(value, gold)
