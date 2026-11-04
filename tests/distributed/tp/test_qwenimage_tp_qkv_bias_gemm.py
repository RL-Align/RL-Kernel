# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""QKV accuracy, row invariance and simulated/real CUDA TP topologies."""

import json
import queue
import traceback
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.backends.cuda.model_specific.qwenimage import tp_qkv_bias_gemm as module
from rl_engine.backends.extension import _C
from rl_engine.contracts.numerical import load_contract
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.reference.model_specific.qwenimage.tp_qkv_bias_gemm import (
    HIDDEN,
    TP_SIZES,
    QwenImageTPQKVBiasGemmReference,
    shard_qkv_parameters,
    validate_qkv,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def equal_bytes(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def indices(world, rank, device="cpu"):
    # Independent head-based oracle, rather than splitting the packed matrix once.
    heads = torch.arange(rank * (24 // world), (rank + 1) * (24 // world), device=device)
    channels = (heads[:, None] * 128 + torch.arange(128, device=device)).flatten()
    return torch.cat([channels + q * HIDDEN for q in range(3)])


@pytest.mark.parametrize("world", TP_SIZES)
def test_head_shards_cover_all_columns_and_parameters(world):
    weight = torch.arange(3 * HIDDEN).float()[:, None].expand(-1, HIDDEN)
    bias = torch.arange(3 * HIDDEN).float()
    owned = []
    for rank in range(world):
        local_weight, local_bias = shard_qkv_parameters(weight, bias, tp_world=world, tp_rank=rank)
        expected = indices(world, rank)
        assert torch.equal(local_weight[:, 0], expected.float())
        assert torch.equal(local_weight[:, -1], expected.float())
        assert torch.equal(local_bias, expected.float())
        owned.append(expected)
    assert torch.equal(torch.cat(owned).sort().values, torch.arange(3 * HIDDEN))


@pytest.mark.parametrize("world", TP_SIZES)
def test_gathered_chunks_restore_global_64_channel_order(world):
    full = torch.arange(144).float()[:, None, None].expand(144, 1, HIDDEN)
    shards = [
        full.view(3, 48, 1, HIDDEN)[:, r * (48 // world) : (r + 1) * (48 // world)].reshape(
            -1, 1, HIDDEN
        )
        for r in range(world)
    ]
    actual = module._ordered_input_partials(torch.cat(shards), world)
    assert equal_bytes(actual, full)


def test_geometry_and_backend_rejections():
    x, w = torch.empty(1, HIDDEN), torch.empty(3 * HIDDEN, HIDDEN)
    with pytest.raises(ValueError, match="TP size"):
        validate_qkv(x, w, None, 3)
    with pytest.raises(ValueError, match="rank"):
        shard_qkv_parameters(w, None, tp_world=2, tp_rank=2)
    with pytest.raises(ValueError, match="x must"):
        validate_qkv(x[:0], w, None)
    with pytest.raises(TypeError, match="only"):
        validate_qkv(x.half(), w.half(), None)
    with pytest.raises(ValueError, match="deterministic collective"):
        module._tp_world(SimpleNamespace(world_size=2, rank=0, backend_id="nccl_all_reduce"))


def make_inputs(rows, dtype, seed=386):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=gen) * scale).to(device="cuda", dtype=dtype)

    return (
        rand((rows, HIDDEN)),
        rand((3 * HIDDEN, HIDDEN), HIDDEN**-0.5),
        rand((3 * HIDDEN,), 0.1),
        rand((rows, 3 * HIDDEN), (3 * HIDDEN) ** -0.5),
    )


def keys_for(x):
    return torch.stack(
        (
            torch.zeros(x.shape[0], device=x.device, dtype=torch.int64),
            torch.arange(x.shape[0], device=x.device),
        ),
        dim=1,
    )


def evaluate(
    x, w, b, grad, *, keys=None, cuts=None, collective=None, frozen=(), parameter_id="image.qkv"
):
    values = [
        v.detach().requires_grad_(name not in frozen) if v is not None else None
        for name, v in zip(("x", "weight", "bias"), (x, w, b))
    ]
    xx, ww, bb = values
    keys = keys_for(x) if keys is None else keys
    cuts = [(0, x.shape[0])] if cuts is None else cuts
    op = module.QwenImageTPQKVBiasGemmCudaOp(collective)
    with canonical_backward_session() as session:
        ys = [
            op(xx[s:e], ww, bb, logical_keys=keys[s:e], parameter_id=parameter_id) for s, e in cuts
        ]
        torch.autograd.backward(list(reversed(ys)), [grad[s:e] for s, e in reversed(cuts)])
    session.validate_complete()
    return torch.cat(ys), tuple(v.grad if v is not None else None for v in values)


def assert_accuracy(actual, expected, dtype, judgment):
    spec = load_contract()["judgments"][judgment]["by_op_class"]["reduction"][
        str(dtype).split(".")[-1]
    ]
    torch.testing.assert_close(
        actual.float().cpu(), expected.float().cpu(), atol=spec["atol"], rtol=spec["rtol"]
    )


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("seed", [386, 387])
@pytest.mark.parametrize("rows", [1, 17, 129, 4096, 6032, 6889])
def test_cpu_reference_and_original_native_bytes(dtype, seed, rows):
    x, w, b, grad = make_inputs(rows, dtype, seed)
    y, (dx, dw, db) = evaluate(x, w, b, grad)
    cpu = [v.detach().cpu().float().requires_grad_() for v in (x, w, b)]
    ref = QwenImageTPQKVBiasGemmReference().forward_fp32(*cpu)
    rg = torch.autograd.grad(ref, cpu, grad.cpu().float())
    assert_accuracy(y, ref, dtype, "forward_accuracy")
    for actual, expected in zip((dx, dw, db), rg):
        assert_accuracy(actual, expected, dtype, "gradient_accuracy")
    assert equal_bytes(y, _C.qwenimage_det_linear_forward(x, w, b, 0, False)[0])
    assert equal_bytes(dx, _C.qwenimage_det_linear_backward_input(grad.float(), w, dtype))
    native_dw, native_db = _C.qwenimage_det_linear_backward_weight(grad.float(), x, dtype, True)
    assert equal_bytes(dw, native_dw)
    assert equal_bytes(db, native_db)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [17, 129, 6889])
def test_permutation_microbatch_padding_and_repeat(dtype, rows):
    x, w, b, grad = make_inputs(rows, dtype)
    keys = keys_for(x)
    y, gs = evaluate(x, w, b, grad, keys=keys)
    for position in (0, rows // 2, rows - 1):
        sy, sg = evaluate(x[position : position + 1], w, b, grad[position : position + 1])
        assert equal_bytes(sy, y[position : position + 1])
        assert equal_bytes(sg[0], gs[0][position : position + 1])
    order = torch.arange(rows - 1, -1, -1, device=x.device)
    py, pg = evaluate(x[order], w, b, grad[order], keys=keys[order])
    assert equal_bytes(py[order], y)
    assert equal_bytes(pg[0][order], gs[0])
    assert all(equal_bytes(a, z) for a, z in zip(pg[1:], gs[1:]))
    cy, cg = evaluate(x, w, b, grad, keys=keys, cuts=[(0, rows // 2), (rows // 2, rows)])
    assert equal_bytes(cy, y)
    assert all(equal_bytes(a, z) for a, z in zip(cg, gs))
    pad_x = torch.cat((x, torch.ones_like(x[:3])))
    pad_g = torch.cat((grad, torch.ones_like(grad[:3])))
    pad_keys = torch.cat((keys, torch.full_like(keys[:3], -1)))
    zy, zg = evaluate(pad_x, w, b, pad_g, keys=pad_keys)
    assert equal_bytes(zy[:rows], y)
    assert equal_bytes(zg[0][:rows], gs[0])
    assert all(equal_bytes(a, z) for a, z in zip(zg[1:], gs[1:]))
    ry, rg = evaluate(x, w, b, grad)
    assert equal_bytes(ry, y)
    assert all(equal_bytes(a, z) for a, z in zip(rg, gs))


class _SimulatedGather:
    """Single-GPU test transport: run the peer kernels, then concatenate bytes.

    This is not evidence of a real process group or inter-GPU communication.
    """

    backend_id = "cuda_ipc_fixed_tree"
    max_size_bytes = 64 * 1024**2

    def __init__(self, grad, weight, world, rank):
        self.grad, self.weight = grad, weight
        self.world_size, self.rank = world, rank
        self.device = grad.device
        self.start = 0

    def all_gather(self, partials):
        rows = partials.shape[1]
        peers = []
        for rank in range(self.world_size):
            if rank == self.rank:
                peers.append(partials)
            else:
                idx = indices(self.world_size, rank, self.device)
                peers.append(
                    _C.qwenimage_det_linear_input_partials(
                        self.grad[self.start : self.start + rows].index_select(1, idx).float(),
                        self.weight.index_select(0, idx),
                    )
                )
        self.start += rows
        return torch.cat(peers)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [17, 129, 6889])
def test_simulated_tp_matches_original_native_all_gradients(dtype, world, rows):
    x, w, b, grad = make_inputs(rows, dtype)
    full_y = _C.qwenimage_det_linear_forward(x, w, b, 0, False)[0]
    full_dx = _C.qwenimage_det_linear_backward_input(grad.float(), w, dtype)
    full_dw, full_db = _C.qwenimage_det_linear_backward_weight(grad.float(), x, dtype, True)
    # Every parameter shard is checked; dX on one rank sees all peer partials.
    for rank in range(world):
        idx = indices(world, rank, x.device)
        local_w, local_b = shard_qkv_parameters(w, b, tp_world=world, tp_rank=rank)
        transport = _SimulatedGather(grad, w, world, rank)
        y, (dx, dw, db) = evaluate(x, local_w, local_b, grad[:, idx], collective=transport)
        assert equal_bytes(y, full_y[:, idx])
        assert equal_bytes(dx, full_dx)
        assert equal_bytes(dw, full_dw[idx])
        assert equal_bytes(db, full_db[idx])


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [255, 256, 257, 2048])
def test_cancellation_parameter_gradients(dtype, rows):
    x = torch.ones((rows, HIDDEN), device="cuda", dtype=dtype)
    x[:, 1::2] = 1.5
    w = torch.zeros((3 * HIDDEN, HIDDEN), device="cuda", dtype=dtype)
    b = torch.zeros(3 * HIDDEN, device="cuda", dtype=dtype)
    grad = torch.ones((rows, 3 * HIDDEN), device="cuda", dtype=dtype)
    grad[0], grad[-1] = 2**26, -(2**26)
    _, (_, dw, db) = evaluate(x, w, b, grad, frozen=("x",))
    gold_db = grad.double().cpu().sum(0)
    gold_dw = gold_db[:, None] * x[0].double().cpu()[None, :]
    assert equal_bytes(dw.cpu(), gold_dw.to(dtype))
    assert equal_bytes(db.cpu(), gold_db.to(dtype))


@cuda
@pytest.mark.parametrize("frozen", [("weight",), ("bias",), ("weight", "bias")])
def test_frozen_parameters_and_noncontiguous_upstream(frozen):
    x, w, b, grad = make_inputs(17, torch.bfloat16)
    grad = torch.stack((grad, grad), -1)[..., 0]
    assert not grad.is_contiguous()
    _, gs = evaluate(x, w, b, grad, frozen=frozen)
    for name, value in zip(("x", "weight", "bias"), gs):
        assert (value is None) == (name in frozen)


@cuda
def test_two_stream_parameters_stay_separate_in_one_session():
    op = module.QwenImageTPQKVBiasGemmCudaOp()
    streams = [make_inputs(3, torch.bfloat16, seed) for seed in (386, 387)]
    values = [[v.detach().requires_grad_() for v in args[:3]] for args in streams]
    with canonical_backward_session() as session:
        ys = [
            op(*args, logical_keys=keys_for(args[0]), parameter_id=name)
            for name, args in zip(("image.qkv", "text.qkv"), values)
        ]
        torch.autograd.backward(ys, [args[3] for args in streams])
    session.validate_complete()
    for (x, w, b), (_, _, _, grad) in zip(values, streams):
        dw, db = _C.qwenimage_det_linear_backward_weight(grad.float(), x, w.dtype, True)
        assert equal_bytes(w.grad, dw)
        assert equal_bytes(b.grad, db)


@cuda
def test_bias_is_added_before_the_bf16_output_conversion():
    x = torch.zeros((1, HIDDEN), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((3 * HIDDEN, HIDDEN), device="cuda", dtype=x.dtype)
    b = torch.full((3 * HIDDEN,), 2**-8, device="cuda", dtype=x.dtype)
    x[0, 0], x[0, 1] = 1, 2**-8
    w[:, :2] = 1
    with torch.no_grad():
        y = module.QwenImageTPQKVBiasGemmCudaOp()(x, w, b)
    assert equal_bytes(y, torch.full_like(y, 1.0078125))


@cuda
def test_existing_cuda_collective_tp1_integration(tmp_path):
    import torch.distributed as dist

    from rl_engine.distributed import create_deterministic_collective

    if dist.is_initialized():
        pytest.skip("this test owns its single-rank process group")
    x, w, b, grad = make_inputs(2, torch.bfloat16)
    expected, expected_grads = evaluate(x, w, b, grad)
    dist.init_process_group("gloo", init_method=(tmp_path / "group").as_uri(), rank=0, world_size=1)
    try:
        with create_deterministic_collective(device=x.device) as collective:
            payload = torch.arange(96, device=x.device, dtype=torch.float32).reshape(3, 32)
            assert equal_bytes(collective.all_gather(payload), payload)
            actual, actual_grads = evaluate(x, w, b, grad, collective=collective)
            assert equal_bytes(actual, expected)
            assert all(equal_bytes(a, z) for a, z in zip(actual_grads, expected_grads))
    finally:
        dist.destroy_process_group()


@cuda
def test_inference_no_bias_registry_and_fail_closed(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    x, w, b, _ = make_inputs(1, torch.bfloat16)
    op = KernelRegistry().get_op("qwenimage_tp_qkv_bias_gemm", device=x.device)
    with torch.no_grad():
        y = op(x.requires_grad_(), w.requires_grad_())
    assert equal_bytes(y, _C.qwenimage_det_linear_forward(x, w, None, 0, False)[0])
    assert op.implementation_info(x, w, b)["fallback"] is False
    with pytest.raises(ValueError, match="NVIDIA"):
        op(x.cpu(), w.cpu(), b.cpu())
    with pytest.raises(RuntimeError, match="session"):
        op(x, w, b)
    monkeypatch.delattr(_C, "qwenimage_det_linear_input_partials")
    with pytest.raises(RuntimeError, match="rebuild"):
        module.QwenImageTPQKVBiasGemmCudaOp()


def _real_tp_worker(rank, world, init_method, result_queue):
    import torch.distributed as dist

    from rl_engine.distributed import create_deterministic_collective

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    cases = []
    try:
        dist.init_process_group(
            "nccl",
            init_method=init_method,
            rank=rank,
            world_size=world,
            timeout=timedelta(minutes=5),
        )
        with create_deterministic_collective(device=torch.device("cuda", rank)) as collective:
            for dtype in (torch.bfloat16, torch.float32):
                for seed in (386, 387):
                    for rows in (1, 17, 129, 4096, 6032, 6889):
                        x, w, b, grad = make_inputs(rows, dtype, seed)
                        idx = indices(world, rank, x.device)
                        local_w, local_b = shard_qkv_parameters(w, b, tp_world=world, tp_rank=rank)
                        local_grad = grad.index_select(1, idx)
                        stream = "image.qkv" if seed == 386 else "text.qkv"
                        gold_y = _C.qwenimage_det_linear_forward(x, w, b, 0, False)[0][:, idx]
                        gold_dx = _C.qwenimage_det_linear_backward_input(grad.float(), w, dtype)
                        gold_dw, gold_db = _C.qwenimage_det_linear_backward_weight(
                            grad.float(), x, dtype, True
                        )
                        gold = (gold_y, gold_dx, gold_dw[idx], gold_db[idx])
                        del gold_dw, gold_db

                        def check(xx, gg, kk, cuts=None, inverse=None, active_rows=None):
                            y, grads = evaluate(
                                xx,
                                local_w,
                                local_b,
                                gg,
                                keys=kk,
                                cuts=cuts,
                                collective=collective,
                                parameter_id=stream,
                            )
                            dx, dw, db = grads
                            if inverse is not None:
                                y, dx = y[inverse], dx[inverse]
                            if active_rows is not None:
                                y, dx = y[:active_rows], dx[:active_rows]
                            for name, value, expected in zip(
                                ("Y", "dX", "dW", "dbias"), (y, dx, dw, db), gold
                            ):
                                assert equal_bytes(value, expected), (
                                    world,
                                    rank,
                                    dtype,
                                    seed,
                                    rows,
                                    name,
                                )

                        keys = keys_for(x)
                        check(x, local_grad, keys)
                        # All ranks execute the same logical layout and collective order.
                        order = torch.arange(rows - 1, -1, -1, device=x.device)
                        check(x[order], local_grad[order], keys[order], inverse=order)
                        if rows > 1:
                            check(x, local_grad, keys, cuts=[(0, rows // 2), (rows // 2, rows)])
                        pad_x = torch.cat((x, torch.ones_like(x[:1])))
                        pad_g = torch.cat((local_grad, torch.ones_like(local_grad[:1])))
                        pad_keys = torch.cat((keys, torch.full_like(keys[:1], -1)))
                        check(pad_x, pad_g, pad_keys, active_rows=rows)
                        check(x, local_grad, keys)
                        with torch.no_grad():
                            inference = module.QwenImageTPQKVBiasGemmCudaOp(collective)(
                                x, local_w, local_b
                            )
                        assert equal_bytes(inference, gold_y)
                        cases.append(
                            {
                                "dtype": str(dtype),
                                "seed": seed,
                                "rows": rows,
                                "layouts": ["base", "reverse"]
                                + (["microbatch"] if rows > 1 else [])
                                + ["padding", "repeat"],
                                "outputs": ["Y", "dX", "dW", "dbias"],
                                "byte_equal": True,
                            }
                        )
                        if rank == 0:
                            print(
                                f"TP{world} {dtype} seed={seed} M={rows}: bytes passed", flush=True
                            )
                        del x, w, b, grad, local_grad
                        dist.barrier()
        dist.destroy_process_group()
        result_queue.put({"ok": True, "rank": rank, "cases": cases})
    except Exception:
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise


@cuda
@pytest.mark.parametrize("world", [2, 4, 8])
def test_real_tp_matches_native_all_gradients(world, tmp_path):
    if not torch.distributed.is_nccl_available() or torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} NVIDIA GPUs and NCCL")
    if torch.distributed.is_initialized():
        pytest.skip("this test owns its multi-rank process group")
    torch.cuda.empty_cache()
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()
    init_method = (tmp_path / "nccl-group").as_uri()
    processes = [
        ctx.Process(target=_real_tp_worker, args=(rank, world, init_method, results_queue))
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    results = []
    try:
        for _ in processes:
            result = results_queue.get(timeout=900)
            results.append(result)
            if not result["ok"]:
                break
    except queue.Empty:
        pytest.fail(f"timed out waiting for TP{world} QKV workers")
    finally:
        for process in processes:
            if len(results) != world or any(not r["ok"] for r in results):
                if process.is_alive():
                    process.terminate()
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=30)
        results_queue.close()
        results_queue.join_thread()
        (tmp_path / f"real-tp{world}.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        assert result["ok"], result.get("traceback")
    assert len(results) == world
    assert all(process.exitcode == 0 for process in processes)
