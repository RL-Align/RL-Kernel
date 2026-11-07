# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Attention output precision, row invariance and real TP1/2/4/8 equivalence."""

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.backends.cuda.model_specific.qwenimage import tp_attn_out_bias_gemm as module
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.ops.autograd.vjp_fp32 import row_local_bias_fp32
from rl_engine.reference.model_specific.qwenimage.tp_attn_out_bias_gemm import (
    HIDDEN,
    TP_SIZES,
    QwenImageTPAttnOutBiasGemmReference,
    shard_attn_out_parameters,
    validate_attn_out,
)
from rl_engine.validation.fixtures.qwenimage_attn_out import (
    ATTN_OUT_LAYOUTS,
    ATTN_OUT_WORKLOADS,
    attn_out_workload_rows,
    canonical_attn_out_indices,
    make_attn_out_workload,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def equal_bytes(a, b):
    if a is None or b is None:
        return a is b
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8),
            b.contiguous().reshape(-1).view(torch.uint8),
        )
    )


def inputs(rows, dtype, seed=386):
    generator = torch.Generator().manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(device="cuda", dtype=dtype)

    x = rand((rows, HIDDEN))
    weight = rand((HIDDEN, HIDDEN), HIDDEN**-0.5)
    bias = rand((HIDDEN,), 0.1)
    grad = rand(x.shape, HIDDEN**-0.5)
    keys = torch.stack(
        (torch.zeros(rows, device="cuda", dtype=torch.int64), torch.arange(rows, device="cuda")), 1
    )
    return x, weight, bias, grad, keys


def native(x, weight, bias, grad, keys):
    order = canonical_attn_out_indices(keys)
    y = _C.qwenimage_attn_out_tree_gemm(x.float().contiguous(), weight.float().contiguous())
    if bias is not None:
        y = y + bias.float()
    y = y.to(x.dtype)
    dx = _C.qwenimage_attn_out_tree_gemm(
        grad.float().contiguous(), weight.t().float().contiguous()
    ).to(x.dtype)
    dw = _C.qwenimage_attn_out_weight_grad(
        grad[order].float().contiguous(), x[order].float().contiguous()
    ).to(weight.dtype)
    db = None if bias is None else row_local_bias_fp32(grad[order]).to(bias.dtype)
    return y[order], dx[order], dw, db


def adapted(op, x, weight, bias, grad, keys, cuts=None):
    x = x.detach().requires_grad_()
    weight = weight.detach().requires_grad_()
    bias = None if bias is None else bias.detach().requires_grad_()
    cuts = [(0, x.shape[0])] if cuts is None else cuts
    with canonical_backward_session() as session:
        ys = [
            op(x[s:e], weight, bias, logical_keys=keys[s:e], parameter_id="attn_out")
            for s, e in cuts
        ]
        parameters = (x, weight) if bias is None else (x, weight, bias)
        grads = torch.autograd.grad(ys, parameters, [grad[s:e] for s, e in cuts])
    session.validate_complete()
    order = canonical_attn_out_indices(keys)
    return torch.cat(ys)[order], grads[0][order], grads[1], None if bias is None else grads[2]


@pytest.mark.parametrize("world", TP_SIZES)
def test_whole_head_weight_columns_and_replicated_bias(world):
    weight = torch.arange(HIDDEN).float()[None].expand(HIDDEN, -1)
    bias = torch.arange(HIDDEN).float()
    shards = []
    for rank in range(world):
        local, local_bias = shard_attn_out_parameters(weight, bias, tp_world=world, tp_rank=rank)
        heads = torch.arange(rank * (24 // world), (rank + 1) * (24 // world))
        expected = (heads[:, None] * 128 + torch.arange(128)).flatten().float()
        assert torch.equal(local[0], expected)
        assert local_bias is bias
        shards.append(local[0])
    assert torch.equal(torch.cat(shards), torch.arange(HIDDEN).float())


def test_geometry_rejections():
    x = torch.empty(1, HIDDEN)
    w = torch.empty(HIDDEN, HIDDEN)
    with pytest.raises(ValueError, match="TP size"):
        validate_attn_out(x, w, None, 3)
    with pytest.raises(ValueError, match="x must"):
        validate_attn_out(x[:0], w, None)
    with pytest.raises(ValueError, match="weight"):
        validate_attn_out(x, w[:, :-1], None)
    with pytest.raises(ValueError, match="bias"):
        validate_attn_out(x, w, torch.empty(HIDDEN // 2))
    with pytest.raises(TypeError, match="dtype"):
        validate_attn_out(x, w.double(), None)
    with pytest.raises(ValueError, match="rank"):
        shard_attn_out_parameters(w, None, tp_world=2, tp_rank=2)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 129])
@pytest.mark.parametrize("bias_present", [False, True])
def test_tp1_native_all_gradients(dtype, rows, bias_present):
    x, w, b, grad, keys = inputs(rows, dtype)
    b = b if bias_present else None
    expected = native(x, w, b, grad, keys)
    actual = adapted(module.QwenImageTPAttnOutBiasGemmCudaOp(), x, w, b, grad, keys)
    assert all(equal_bytes(a, z) for a, z in zip(actual, expected))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_independent_cpu_reference_and_accuracy(dtype):
    x, w, b, grad, keys = inputs(2, dtype)
    cpu = [value.cpu().detach().requires_grad_() for value in (x, w, b)]
    y = QwenImageTPAttnOutBiasGemmReference()(*cpu)
    grads = torch.autograd.grad(y, cpu, grad.cpu())
    actual = adapted(module.QwenImageTPAttnOutBiasGemmCudaOp(), x, w, b, grad, keys)
    assert all(equal_bytes(a.cpu(), z) for a, z in zip(actual, (y, *grads)))
    independent = torch.nn.functional.linear(cpu[0].float(), cpu[1].float(), cpu[2].float())
    torch.testing.assert_close(
        actual[0].float().cpu(),
        independent,
        atol=0.05 if dtype == torch.bfloat16 else 1e-4,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
    )


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_layouts_and_capacity_chunking(dtype):
    x, w, b, grad, keys = inputs(17, dtype)
    op = module.QwenImageTPAttnOutBiasGemmCudaOp()
    expected = adapted(op, x, w, b, grad, keys)
    permutation = torch.randperm(17, device=x.device)
    padded = (
        torch.cat((x[permutation], torch.ones_like(x[:3]))),
        torch.cat((grad[permutation], torch.zeros_like(grad[:3]))),
        torch.cat((keys[permutation], torch.full_like(keys[:3], -1))),
    )
    for values, cuts in [((x, grad, keys), [(0, 1), (1, 8), (8, 17)]), (padded, None)]:
        a, g, k = values
        actual = adapted(op, a, w, b, g, k, cuts)
        assert all(equal_bytes(v, z) for v, z in zip(actual, expected))

    # TP1 transport mock exercises row-capacity splitting without changing sums.
    class IdentityCollective:
        world_size, rank = 1, 0
        backend_id = "cuda_ipc_fixed_tree"
        max_size_bytes = HIDDEN * 4 * 3
        device = x.device

        def all_reduce(self, partial):
            assert partial.shape[0] <= 3
            return partial.clone()

    chunked = adapted(
        module.QwenImageTPAttnOutBiasGemmCudaOp(IdentityCollective()), x, w, b, grad, keys
    )
    assert all(equal_bytes(v, z) for v, z in zip(chunked, expected))


@cuda
@pytest.mark.parametrize("frozen", ["weight", "bias", "both"])
def test_noncontiguous_upstream_and_frozen_parameters(frozen):
    x, w, b, grad, keys = inputs(3, torch.float32)
    expected = native(x, w, b, grad, keys)
    x.requires_grad_()
    w.requires_grad_(frozen not in ("weight", "both"))
    b.requires_grad_(frozen not in ("bias", "both"))
    storage = torch.empty(3, HIDDEN * 2, device="cuda")
    storage[:, ::2] = grad
    with canonical_backward_session() as session:
        y = module.QwenImageTPAttnOutBiasGemmCudaOp()(
            x, w, b, logical_keys=keys, parameter_id="freeze"
        )
        y.backward(storage[:, ::2])
    session.validate_complete()
    assert equal_bytes(x.grad, expected[1])
    assert equal_bytes(w.grad, expected[2] if w.requires_grad else None)
    assert equal_bytes(b.grad, expected[3] if b.requires_grad else None)


@cuda
def test_session_and_transport_rejections():
    from types import SimpleNamespace

    x, w, b, grad, keys = inputs(1, torch.float32)
    op = module.QwenImageTPAttnOutBiasGemmCudaOp()
    with pytest.raises(RuntimeError, match="canonical backward session"):
        op(x, w.requires_grad_(), b)
    with canonical_backward_session():
        with pytest.raises(ValueError, match="logical_keys and parameter_id"):
            op(x, w, b)
    collective = SimpleNamespace(world_size=2, rank=0, backend_id="ordinary_sum")
    with pytest.raises(ValueError, match="deterministic collective"):
        module.QwenImageTPAttnOutBiasGemmCudaOp(collective)
    collective.backend_id = "cuda_ipc_fixed_tree"
    collective.max_size_bytes = HIDDEN * 4 - 1
    with pytest.raises(ValueError, match="one FP32 output row"):
        module.QwenImageTPAttnOutBiasGemmCudaOp(collective)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tp1_noncurrent_device(dtype):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs for the current-device regression")
    values = inputs(3, dtype)
    x, w, b, grad, keys = [value.to("cuda:1") for value in values]
    expected = native(x, w, b, grad, keys)
    original = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        actual = adapted(module.QwenImageTPAttnOutBiasGemmCudaOp(), x, w, b, grad, keys)
        assert torch.cuda.current_device() == 0
        assert all(equal_bytes(a, z) for a, z in zip(actual, expected))
    finally:
        torch.cuda.set_device(original)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_both_streams_share_one_session(dtype):
    op = module.QwenImageTPAttnOutBiasGemmCudaOp()
    fixtures = [inputs(3, dtype, 386), inputs(2, dtype, 387)]
    expected = [native(*values) for values in fixtures]
    parameters, outputs, upstream = [], [], []
    with canonical_backward_session() as session:
        for name, (x, w, b, grad, keys) in zip(("image", "text"), fixtures):
            x.requires_grad_()
            w.requires_grad_()
            b.requires_grad_()
            parameters.extend((x, w, b))
            outputs.append(op(x, w, b, logical_keys=keys, parameter_id=name + ".attn_out"))
            upstream.append(grad)
        grads = torch.autograd.grad(outputs, parameters, upstream)
    session.validate_complete()
    for stream, gold in enumerate(expected):
        actual = outputs[stream], *grads[stream * 3 : (stream + 1) * 3]
        assert all(equal_bytes(a, z) for a, z in zip(actual, gold))


@cuda
def test_registry_inference_and_missing_extension_fail_closed(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    x, w, b, grad, keys = inputs(1, torch.bfloat16)
    op = KernelRegistry().get_op("qwenimage_tp_attn_out_bias_gemm", device=x.device)
    with torch.no_grad():
        actual = op(x.requires_grad_(), w.requires_grad_(), b)
    expected = (_C.qwenimage_attn_out_tree_gemm(x.float(), w.float()) + b.float()).to(x.dtype)
    assert equal_bytes(actual, expected)
    assert op.implementation_info(x, w, b)["fallback"] is False
    with pytest.raises(ValueError, match="NVIDIA"):
        op(x.cpu(), w.cpu(), b.cpu())
    with pytest.raises(ValueError, match="CPU"):
        QwenImageTPAttnOutBiasGemmReference()(x, w, b)
    monkeypatch.delattr(_C, "qwenimage_attn_out_tree_gemm")
    with pytest.raises(RuntimeError, match="rebuild"):
        module.QwenImageTPAttnOutBiasGemmCudaOp()


def check_workload(op, workload, dtype, world=1, rank=0):
    fixture = make_attn_out_workload(workload, dtype, "cuda")
    checks = 0
    width = HIDDEN // world
    owned = slice(rank * width, (rank + 1) * width)
    for stream in fixture["streams"].values():
        x, grad, keys, _ = attn_out_workload_rows(stream, "batched")
        expected = native(x, stream["weight"], stream["bias"], grad, keys)
        if rank == 0:
            # Independent CPU matmuls check numerical accuracy at real lengths.
            # Complete tensors are compared byte for byte with TP1 below.
            order = canonical_attn_out_indices(keys)
            selected = order[:3]
            w_cpu = stream["weight"].float().cpu()
            g_cpu = grad[selected].float().cpu()
            y_cpu = torch.nn.functional.linear(
                x[selected].float().cpu(), w_cpu, stream["bias"].float().cpu()
            )
            dx_cpu = g_cpu @ w_cpu
            gx = grad[order, :64].float().cpu()
            rows = x[order, :32].float().cpu()
            dw_cpu = gx.t() @ rows
            db_cpu = gx.sum(0)
            tolerance = (0.05, 0.02) if dtype == torch.bfloat16 else (1e-4, 1e-4)
            torch.testing.assert_close(
                expected[0][:3].float().cpu(), y_cpu, atol=tolerance[0], rtol=tolerance[1]
            )
            grad_tolerance = (0.1, 0.02) if dtype == torch.bfloat16 else (1e-4, 1e-4)
            for actual, target in (
                (expected[1][:3], dx_cpu),
                (expected[2][:64, :32], dw_cpu),
                (expected[3][:64], db_cpu),
            ):
                torch.testing.assert_close(
                    actual.float().cpu(),
                    target,
                    atol=grad_tolerance[0],
                    rtol=grad_tolerance[1],
                )
        expected = expected[0], expected[1][:, owned], expected[2][:, owned], expected[3]
        weight, bias = shard_attn_out_parameters(
            stream["weight"], stream["bias"], tp_world=world, tp_rank=rank
        )
        stream["x"] = stream["x"][..., owned]
        for layout in ATTN_OUT_LAYOUTS:
            local_x, g, k, cuts = attn_out_workload_rows(stream, layout)
            actual = adapted(op, local_x, weight, bias, g, k, cuts)
            assert all(equal_bytes(a, z) for a, z in zip(actual, expected)), (
                workload,
                layout,
                world,
                rank,
            )
            checks += 1
    return checks


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("workload", ATTN_OUT_WORKLOADS)
def test_model_layouts(dtype, workload):
    assert check_workload(module.QwenImageTPAttnOutBiasGemmCudaOp(), workload, dtype) == 14


def real_tp_worker(rank, world, directory):
    from pathlib import Path

    from rl_engine.distributed import create_deterministic_collective

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=5),
    )
    checks = 0
    try:
        with create_deterministic_collective(device=torch.device("cuda", rank)) as collective:
            op = module.QwenImageTPAttnOutBiasGemmCudaOp(collective)
            for dtype in (torch.float32, torch.bfloat16):
                for rows in (1, 7, 129):
                    x, w, b, grad, keys = inputs(rows, dtype)
                    expected = native(x, w, b, grad, keys)
                    width = HIDDEN // world
                    owned = slice(rank * width, (rank + 1) * width)
                    local_w, local_b = shard_attn_out_parameters(w, b, tp_world=world, tp_rank=rank)
                    actual = adapted(op, x[:, owned], local_w, local_b, grad, keys)
                    gold = expected[0], expected[1][:, owned], expected[2][:, owned], expected[3]
                    assert all(equal_bytes(a, z) for a, z in zip(actual, gold))
                    checks += 1
                for workload in ATTN_OUT_WORKLOADS:
                    checks += check_workload(op, workload, dtype, world, rank)
        (Path(directory) / f"rank-{rank}.json").write_text(
            json.dumps({"tp": world, "rank": rank, "checks": checks, "byte_equal": True})
        )
    finally:
        dist.destroy_process_group()


@cuda
@pytest.mark.parametrize("world", [2, 4, 8])
def test_real_tp_all_results(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} NVIDIA GPUs")
    mp.spawn(real_tp_worker, args=(world, str(tmp_path)), nprocs=world, join=True)
    for rank in range(world):
        report = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert report["checks"] == 174 and report["byte_equal"]


def scalar_fmaf():
    """Use the host C library as an oracle independent of torch and CUDA."""
    import ctypes
    import ctypes.util

    library = ctypes.CDLL(ctypes.util.find_library("m") or None)
    oracle = library.fmaf
    oracle.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    oracle.restype = ctypes.c_float
    return oracle


def scalar_tree(a, b):
    import ctypes

    fma = scalar_fmaf()
    roots = []
    for start in range(0, len(a), 32):
        value = 0.0
        for index in range(start, min(start + 32, len(a))):
            value = fma(a[index], b[index], value)
        roots.append(value)

    def merge(begin, end):
        if end - begin == 1:
            return roots[begin]
        middle = (begin + end) // 2
        return ctypes.c_float(merge(begin, middle) + merge(middle, end)).value

    return merge(0, len(roots))


def test_cpu_fma_primitive_matches_scalar_oracle():
    generator = torch.Generator().manual_seed(386)
    a, b, c = torch.randn((3, 256), generator=generator)
    # Include operands which distinguish a fused FMA from two rounded operations.
    a[:2] = torch.tensor([14872.8408203125, -0.016603613272309303])
    b[:2] = torch.tensor([1.8524175882339478, -2.194535970687866])
    c[:2] = torch.tensor([1026.0933837890625, -0.03500966727733612])
    oracle = scalar_fmaf()
    expected = torch.tensor(
        [oracle(x, y, z) for x, y, z in zip(a.tolist(), b.tolist(), c.tolist())]
    )
    assert equal_bytes(torch.addcmul(c, a, b), expected)


@cuda
@pytest.mark.parametrize("world", TP_SIZES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiled_tree_against_scalar_fmaf(world, dtype):
    x, w, _, _, _ = inputs(17, dtype)
    width = HIDDEN // world
    x, w = x[:, :width].float().contiguous(), w[:, :width].float().contiguous()
    actual = _C.qwenimage_attn_out_tree_gemm(x, w)
    for row, col in ((0, 0), (1, 31), (15, 63), (16, 64), (16, 3071)):
        expected = scalar_tree(x[row].cpu().tolist(), w[col].cpu().tolist())
        assert equal_bytes(actual[row, col].cpu(), torch.tensor(expected))


@cuda
@pytest.mark.parametrize("rows", [1, 31, 32, 33, 129])
def test_tiled_weight_fold_tail_against_scalar_fmaf(rows):
    x, _, _, grad, _ = inputs(rows, torch.float32)
    if rows == 1:
        x[0, 0], grad[0, 0] = 1e-30, -1e-30
    actual = _C.qwenimage_attn_out_weight_grad(grad, x)
    oracle = scalar_fmaf()
    for n, k in ((0, 0), (15, 63), (16, 64), (3071, 3071)):
        value = 0.0
        for g, a in zip(grad[:, n].cpu().tolist(), x[:, k].cpu().tolist()):
            value = oracle(g, a, value)
        assert equal_bytes(actual[n, k].cpu(), torch.tensor(value))


@cuda
def test_tiled_native_rejects_unsupported_geometry():
    a = torch.zeros((1, 32), device="cuda")
    with pytest.raises(RuntimeError, match="reduction must"):
        _C.qwenimage_attn_out_tree_gemm(a, a)
    with pytest.raises(RuntimeError, match="float32"):
        _C.qwenimage_attn_out_tree_gemm(a.double(), a.double())
    with pytest.raises(RuntimeError, match="TP1/2/4/8"):
        _C.qwenimage_attn_out_weight_grad(a, a)
