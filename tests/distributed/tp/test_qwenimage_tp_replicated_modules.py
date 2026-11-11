# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Replicated-module precision, canonical gradients and real TP ownership."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.operators.gemm.benchmark_tp_attn_out_bias_gemm import equal_bytes
from rl_engine.backends.cuda.model_specific.qwenimage.replicated_modules import broadcast_condition
from rl_engine.distributed import create_deterministic_collective
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_WORKLOADS, qkv_workload_rows
from rl_engine.validation.fixtures.qwenimage_replicated import (
    MODULES,
    evaluate_replicated,
    make_replicated_case,
    make_replicated_workload,
    replicated_layout,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")


@cuda
@pytest.mark.parametrize("kind", MODULES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_complete_small_precision_and_layouts(kind, dtype):
    x, p, g, k = make_replicated_case(kind, 9, dtype, "cuda", requests=2)
    expected = evaluate_replicated(kind, x, p, g, k, [(0, 9)])
    cpu = evaluate_replicated(
        kind,
        x.float().cpu(),
        tuple(v.float().cpu() for v in p),
        g.float().cpu(),
        k.cpu(),
        [(0, 9)],
        backend="pytorch",
    )
    for index, (value, gold) in enumerate(zip(expected, cpu)):
        atol = (0.05 if index == 0 else 0.1) if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(
            value.float().cpu(),
            gold.float(),
            atol=atol,
            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
        )
    for layout in ("reverse", "microbatch", "padding", "repeat"):
        a, dy, keys, cuts = replicated_layout(x, g, k, layout)
        actual = evaluate_replicated(kind, a, p, dy, keys, cuts)
        assert all(equal_bytes(left, right) for left, right in zip(expected, actual)), layout


def test_condition_shared_consumers_and_padding():
    condition = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    grad = torch.tensor([[1.0, 1.0], [2**-24, -(2**-24)], [-1.0, -1.0], [0.0, 0.0]])
    keys = torch.tensor([[0, 0], [0, 1], [0, 2], [-1, -1]], dtype=torch.int64)
    values = []
    for order in ((0, 1), (1, 0)):
        c = condition.clone().requires_grad_()
        with canonical_backward_session() as session:
            ys, gs = [], []
            for use in order:
                kk = torch.stack((keys[:, 0], torch.full_like(keys[:, 0], use), keys[:, 1]), 1)
                ys.append(broadcast_condition(c, kk, parameter_id="condition"))
                gs.append(grad * (use + 1))
            values.append(torch.autograd.grad(ys, c, gs)[0])
        session.validate_complete()
    assert equal_bytes(values[0], values[1])
    assert torch.equal(values[0][1], torch.zeros(2))


def _worker(rank, world, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=10),
    )
    checked = 0
    records = []
    with create_deterministic_collective(device=torch.device("cuda", rank)) as collective:
        for dtype in (torch.bfloat16, torch.float32):
            for kind in MODULES:
                x, p, g, k = make_replicated_case(kind, 17, dtype, "cuda", requests=2)
                gold = evaluate_replicated(kind, x, p, g, k, [(0, 17)])
                for layout in ("batched", "reverse", "microbatch", "padding", "repeat"):
                    a, dy, keys, cuts = replicated_layout(x, g, k, layout)
                    c = None
                    expected = gold
                    if kind == "qk_norm":
                        owned = slice(rank * 24 // world, (rank + 1) * 24 // world)
                        a, dy = a[:, owned].contiguous(), dy[:, owned].contiguous()
                        expected = (gold[0][:, owned], gold[1][:, owned], gold[2])
                        c = collective
                    actual = evaluate_replicated(kind, a, p, dy, keys, cuts, collective=c)
                    assert all(equal_bytes(v, w) for v, w in zip(actual, expected)), (
                        rank,
                        kind,
                        layout,
                        dtype,
                    )
                    checked += len(actual)
                if kind != "block_norm":
                    # These complete gradients must not be summed over replicas.
                    assert not equal_bytes(gold[-1], gold[-1] * world)
            # All six model workloads, including sequential CFG as a call layout.
            for name in QKV_WORKLOADS:
                for kind in MODULES:
                    stream = "text" if kind in ("text_input", "qk_norm") else "image"
                    fixture = make_replicated_workload(kind, name, dtype, "cuda", stream=stream)
                    p = fixture["parameters"]
                    a, dy, keys, cuts = qkv_workload_rows(fixture, "batched")
                    if kind == "qk_norm":
                        a, dy = a.reshape(-1, 24, 128), dy.reshape(-1, 24, 128)
                    gold = evaluate_replicated(kind, a, p, dy, keys, cuts)
                    layouts = (
                        "batched",
                        "sequential_cfg",
                        "requests",
                        "microbatch",
                        "reverse",
                        "padding",
                        "repeat",
                    )
                    for layout in layouts:
                        if layout == "sequential_cfg" and fixture["metadata"]["cfg_branches"] == 1:
                            continue
                        aa, gg, kk, cc = qkv_workload_rows(fixture, layout)
                        expected = gold
                        c = None
                        if kind == "qk_norm":
                            owned = slice(rank * 24 // world, (rank + 1) * 24 // world)
                            aa = aa.reshape(-1, 24, 128)[:, owned].contiguous()
                            gg = gg.reshape(-1, 24, 128)[:, owned].contiguous()
                            expected = (gold[0][:, owned], gold[1][:, owned], gold[2])
                            c = collective
                        actual = evaluate_replicated(kind, aa, p, gg, kk, cc, collective=c)
                        assert all(equal_bytes(v, w) for v, w in zip(actual, expected)), (
                            rank,
                            kind,
                            name,
                            layout,
                            dtype,
                        )
                        checked += len(actual)
                        records.append(
                            {
                                "dtype": str(dtype),
                                "module": kind,
                                "workload": name,
                                "layout": layout,
                                "tensors": len(actual),
                                "byte_equal": True,
                            }
                        )
                    Path(directory, f"progress-{rank}.json").write_text(json.dumps(records))
        checked += check_tp_boundaries(rank, world, collective)
    Path(directory, f"rank-{rank}.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "tp": world,
                "tensor_comparisons": checked,
                "byte_equal": True,
                "cases": records,
            }
        )
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
def test_real_tp(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} NVIDIA GPUs")
    directory = tmp_path / f"tp-{world}"
    directory.mkdir()
    mp.spawn(_worker, args=(world, str(directory)), nprocs=world, join=True)
    assert len(list(directory.glob("rank-*.json"))) == world


@cuda
@pytest.mark.parametrize("kind", ["image_input", "qk_norm", "modulation_apply"])
def test_noncontiguous_inputs_and_upstream(kind):
    x, parameters, grad, keys = make_replicated_case(kind, 17, torch.bfloat16, "cuda")
    expected = evaluate_replicated(kind, x, parameters, grad, keys, [(0, 17)])

    def strided(value):
        storage = torch.empty((*value.shape, 2), device=value.device, dtype=value.dtype)
        storage[..., 0] = value
        return storage[..., 0]

    actual = evaluate_replicated(
        kind, strided(x), tuple(strided(p) for p in parameters), strided(grad), keys, [(0, 17)]
    )
    assert all(equal_bytes(a, b) for a, b in zip(expected, actual))


@cuda
@pytest.mark.parametrize("kind", ["image_input", "qk_norm"])
@pytest.mark.parametrize("frozen", ["parameters", "input"])
def test_frozen_leaves_and_inference(kind, frozen):
    from rl_engine.validation.fixtures.qwenimage_replicated import module_forward

    x, parameters, grad, keys = make_replicated_case(kind, 7, torch.bfloat16, "cuda")
    gold = evaluate_replicated(kind, x, parameters, grad, keys, [(0, 7)])
    x.requires_grad_(frozen != "input")
    parameters = tuple(p.requires_grad_(frozen != "parameters") for p in parameters)
    with canonical_backward_session() as session:
        y = module_forward(kind, x, parameters, keys)
        y.backward(grad)
    session.validate_complete()
    assert x.grad is None if frozen == "input" else equal_bytes(x.grad, gold[1])
    for p, expected in zip(parameters, gold[2:]):
        assert p.grad is None if frozen == "parameters" else equal_bytes(p.grad, expected)
    with torch.no_grad(), canonical_backward_session() as session:
        y = module_forward(kind, x, parameters, keys)
    session.validate_complete()
    assert equal_bytes(y, gold[0])


@cuda
def test_qk_noncurrent_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    from rl_engine.validation.fixtures.qwenimage_replicated import module_forward

    original = torch.cuda.current_device()
    try:
        x, p, g, k = make_replicated_case("qk_norm", 7, torch.float32, "cuda:1")
        torch.cuda.set_device(1)
        expected = evaluate_replicated("qk_norm", x, p, g, k, [(0, 7)])
        torch.cuda.set_device(0)
        actual = evaluate_replicated("qk_norm", x, p, g, k, [(0, 7)])
        assert all(equal_bytes(a, b) for a, b in zip(expected, actual))
        with torch.no_grad():
            y = module_forward("qk_norm", x, p, k)
        assert equal_bytes(y, expected[0])
        assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(original)


def check_tp_boundaries(rank, world, collective):
    """Compose the three TP projections with norm/modulation VJPs, not attention."""
    import torch.nn.functional as F

    from rl_engine.backends.cuda.model_specific.qwenimage.replicated_modules import (
        replicated_linear,
        replicated_qk_rmsnorm,
    )
    from rl_engine.backends.cuda.model_specific.qwenimage.tp_attn_out_bias_gemm import (
        QwenImageTPAttnOutBiasGemmCudaOp,
        shard_attn_out_parameters,
    )
    from rl_engine.backends.cuda.model_specific.qwenimage.tp_mlp_gemm_gelu import (
        QwenImageTPMlpGemmGeluCudaOp,
        shard_mlp_parameters,
    )
    from rl_engine.backends.cuda.model_specific.qwenimage.tp_qkv_bias_gemm import (
        QwenImageTPQKVBiasGemmCudaOp,
        shard_qkv_parameters,
    )

    generator = torch.Generator().manual_seed(391)

    def rand(shape, scale=1):
        return (torch.randn(shape, generator=generator) * scale).to(
            device="cuda", dtype=torch.bfloat16
        )

    parameters = (
        rand((3072, 64), 64**-0.5),
        rand((3072,), 0.02),
        rand((9216, 3072), 3072**-0.5),
        rand((9216,), 0.02),
        rand((128,), 0.1) + 1,
        rand((128,), 0.1) + 1,
        rand((3072, 3072), 3072**-0.5),
        rand((3072,), 0.02),
        rand((12288, 3072), 3072**-0.5),
        rand((12288,), 0.02),
        rand((3072, 12288), 12288**-0.5),
        rand((3072,), 0.02),
        rand((64, 3072), 3072**-0.5),
        rand((64,), 0.02),
        rand((1, 9216), 0.1),
    )
    x, grad = rand((3, 64), 0.3), rand((3, 64), 0.125)
    keys = torch.stack(
        (torch.zeros(3, device="cuda", dtype=torch.int64), torch.arange(3, device="cuda")), 1
    )

    def shard(p):
        qw, qb = shard_qkv_parameters(p[2], p[3], tp_world=world, tp_rank=rank)
        ow, ob = shard_attn_out_parameters(p[6], p[7], tp_world=world, tp_rank=rank)
        u, ub, d, db = shard_mlp_parameters(*p[8:12], tp_world=world, tp_rank=rank)
        return (*p[:2], qw, qb, *p[4:6], ow, ob, u, ub, d, db, *p[12:])

    def run(parameters, collective, layout):
        a, g, k, cuts = replicated_layout(x, grad, keys, layout)
        a = a.detach().requires_grad_()
        p = tuple(v.detach().requires_grad_() for v in parameters)
        qkv = QwenImageTPQKVBiasGemmCudaOp(collective)
        out = QwenImageTPAttnOutBiasGemmCudaOp(collective)
        mlp = QwenImageTPMlpGemmGeluCudaOp(collective)

        def linear(v, w, b, kk, name):
            return replicated_linear(v, w, b, logical_keys=kk, parameter_id="boundary." + name)

        ys = []
        with canonical_backward_session() as session:
            for start, stop in cuts:
                kk = k[start:stop]
                h = linear(a[start:stop], p[0], p[1], kk, "image_in")
                shift, scale, gate = broadcast_condition(
                    p[14], kk, parameter_id="boundary.condition"
                ).chunk(3, -1)
                normalized = F.layer_norm(h.float(), (3072,), eps=1e-6).to(h.dtype)
                z = normalized * (1 + scale) + shift
                packed = qkv(z, p[2], p[3], logical_keys=kk, parameter_id="boundary.qkv")
                q, key, v = packed.chunk(3, -1)
                heads = 24 if collective is None else 24 // world
                q = replicated_qk_rmsnorm(
                    q.reshape(-1, heads, 128),
                    p[4],
                    logical_keys=kk,
                    parameter_id="boundary.q_gain",
                    collective=collective,
                )
                key = replicated_qk_rmsnorm(
                    key.reshape(-1, heads, 128),
                    p[5],
                    logical_keys=kk,
                    parameter_id="boundary.k_gain",
                    collective=collective,
                )
                # A row-local linear probe sends gradients through Q, K and V.
                # It supplies no softmax, RoPE or full-attention qualification.
                probe = q.reshape_as(v) * 0.25 + key.reshape_as(v) * 0.125 + v * 0.5
                h = h + gate * out(probe, p[6], p[7], logical_keys=kk, parameter_id="boundary.out")
                h = h + mlp(h, *p[8:12], logical_keys=kk, parameter_id="boundary.mlp")
                ys.append(linear(h, p[12], p[13], kk, "image_out"))
            derivatives = torch.autograd.grad(ys, (a, *p), [g[s:e] for s, e in cuts])
        session.validate_complete()
        order = torch.argsort(k[:, 1])
        return (
            torch.cat(ys)[order].detach(),
            derivatives[0][order].detach(),
            *(t.detach() for t in derivatives[1:]),
        )

    gold = run(parameters, None, "batched")
    expected = (*gold[:2], *shard(gold[2:]))
    local = shard(parameters)
    for layout in ("batched", "reverse", "microbatch"):
        actual = run(local, collective, layout)
        assert all(equal_bytes(a, b) for a, b in zip(actual, expected)), (
            "boundary",
            rank,
            world,
            layout,
        )
    return len(expected) * 3


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_real_reduction_cpu_slices(dtype):
    """Check long parameter sums independently, not only against native TP1."""
    import torch.nn.functional as F

    for kind in ("image_input", "image_output", "qk_norm", "modulation_apply"):
        stream = "text" if kind == "qk_norm" else "image"
        fixture = make_replicated_workload(
            kind, "batch2-cfg-text1024", dtype, "cuda", stream=stream
        )
        x, grad, keys, cuts = qkv_workload_rows(fixture, "batched")
        parameters = fixture["parameters"]
        if kind == "qk_norm":
            x, grad = x.reshape(-1, 24, 128), grad.reshape(-1, 24, 128)
        actual = evaluate_replicated(kind, x, parameters, grad, keys, cuts)
        active = keys[:, 0] >= 0
        a, g = x[active].float().cpu(), grad[active].float().cpu()
        atol, rtol = (0.1, 0.02) if dtype == torch.bfloat16 else (1e-4, 1e-4)
        if kind in ("image_input", "image_output"):
            dw = g[:, :8].t() @ a[:, :8]
            db = g[:, :8].sum(0)
            torch.testing.assert_close(actual[2][:8, :8].float().cpu(), dw, atol=atol, rtol=rtol)
            torch.testing.assert_close(actual[3][:8].float().cpu(), db, atol=atol, rtol=rtol)
        elif kind == "qk_norm":
            rstd = torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-6)
            dg = (g * a * rstd).sum((0, 1))
            torch.testing.assert_close(actual[2].float().cpu(), dg, atol=atol, rtol=rtol)
        else:
            c = parameters[0].float().cpu()
            sample = keys[active, 0].cpu()
            shift, scale, gate = c[sample].chunk(3, -1)
            normalized = F.layer_norm(a, (3072,), eps=1e-6)
            rows = torch.cat(
                (g * gate, g * gate * normalized, g * (normalized * (1 + scale) + shift)), -1
            )
            expected = torch.stack([rows[sample == i].sum(0) for i in range(c.shape[0])])
            torch.testing.assert_close(actual[2].float().cpu(), expected, atol=atol, rtol=rtol)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunks", [2, 6])
def test_original_silu_projection_arithmetic(dtype, chunks):
    from rl_engine.backends.cuda.model_specific.qwenimage.adaln_mod_projection import (
        projection_backward,
        projection_forward,
    )
    from rl_engine.backends.cuda.model_specific.qwenimage.replicated_modules import (
        replicated_silu_projection,
    )

    x, p, grad, keys = make_replicated_case("image_modulation", 3, dtype, "cuda")
    weight, bias = p[0][: chunks * 3072].contiguous(), p[1][: chunks * 3072].contiguous()
    grad = grad[:, : chunks * 3072].contiguous()
    y, act = projection_forward(x, weight, bias)
    expected = (y, *projection_backward(grad, x, act, weight))
    leaves = tuple(v.detach().requires_grad_() for v in (x, weight, bias))
    with canonical_backward_session() as session:
        actual = replicated_silu_projection(*leaves, logical_keys=keys, parameter_id="original")
        derivatives = torch.autograd.grad(actual, leaves, grad)
    session.validate_complete()
    assert all(equal_bytes(a, b) for a, b in zip((actual, *derivatives), expected))


@cuda
def test_distinct_stream_parameters_in_shared_session():
    from rl_engine.validation.fixtures.qwenimage_replicated import module_forward

    cases = [
        (kind, make_replicated_case(kind, 3, torch.bfloat16, "cuda", seed=seed))
        for kind, seed in (("image_modulation", 386), ("text_modulation", 387))
    ]
    expected = [evaluate_replicated(kind, x, p, g, k, [(0, 3)]) for kind, (x, p, g, k) in cases]
    leaves = [
        (x.detach().requires_grad_(), *(v.detach().requires_grad_() for v in p))
        for _, (x, p, _, _) in cases
    ]
    with canonical_backward_session() as session:
        ys = [
            module_forward(kind, values[0], values[1:], k)
            for (kind, (_, _, _, k)), values in zip(cases, leaves)
        ]
        derivatives = torch.autograd.grad(
            ys, tuple(v for group in leaves for v in group), [case[1][2] for case in cases]
        )
    session.validate_complete()
    for index, gold in enumerate(expected):
        actual = (ys[index], *derivatives[index * 3 : index * 3 + 3])
        assert all(equal_bytes(a, b) for a, b in zip(actual, gold))
