# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fixed-tree MLP precision, canonical gradients and real TP topology checks."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmarks.operators.gemm.benchmark_tp_mlp_gemm_gelu import (
    equal_bytes,
    make_inputs,
    mlp_calls,
    native_mlp,
)
from rl_engine.backends.cuda.model_specific.qwenimage import tp_mlp_gemm_gelu as module
from rl_engine.backends.extension import _C
from rl_engine.reference.gemm.mlp_up_gemm_gelu import (
    gelu_tanh_fp32,
    left_fold_bias_gradient,
    left_fold_weight_gradient,
    mlp_up_gemm_gelu_reference_gate,
    tree_gemm,
)
from rl_engine.reference.model_specific.qwenimage.tp_mlp_gemm_gelu import (
    QwenImageTPMlpGemmGeluReference,
    shard_mlp_parameters,
    validate_mlp,
)
from rl_engine.validation.fixtures.qwenimage_mlp import (
    MLP_LAYOUTS,
    MLP_WORKLOADS,
    canonical_mlp_indices,
    make_mlp_workload,
    mlp_workload_rows,
)

cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def adapted(op, x, parameters, grad, keys, cuts=None):
    x = x.detach().requires_grad_()
    parameters = tuple(p.detach().requires_grad_() for p in parameters)
    _, fb = mlp_calls(op, x, parameters, grad, keys, cuts or [(0, x.shape[0])])
    result = fb()
    order = canonical_mlp_indices(keys)
    return torch.cat(result[0])[order], result[1][order], *result[2:]


@cuda
@pytest.mark.parametrize("rows", [1, 17, 129])
@pytest.mark.parametrize("seed", [386, 387])
def test_forward_and_gradients(rows, seed):
    values = make_inputs(rows, seed)
    x, parameters, grad, keys = values
    actual = adapted(module.QwenImageTPMlpGemmGeluCudaOp(), x, parameters, grad, keys)
    expected = native_mlp(x, *parameters, grad, keys)
    assert all(equal_bytes(a, b) for a, b in zip(actual, expected))


@cuda
@pytest.mark.parametrize("reduction", [1536, 3072, 6144, 12288])
def test_fp32_partial_entry(reduction):
    generator = torch.Generator().manual_seed(386)
    a, b = [
        (torch.randn(shape, generator=generator) * 0.1).to(device="cuda", dtype=torch.bfloat16)
        for shape in [(17, reduction), (32, reduction)]
    ]
    _, pre = _C.mlp_up_gemm_gelu_cuda_forward(a, b, None, True)
    actual = _C.mlp_up_gemm_gelu_cuda_tree_fp32(a, b)
    assert actual.dtype == torch.float32 and equal_bytes(actual, pre)
    independent = tree_gemm(a[:2].cpu(), b[:4].t().cpu())
    assert equal_bytes(actual[:2, :4].cpu(), independent)


@cuda
def test_independent_cpu_precision():
    x, parameters, grad, keys = make_inputs(3)
    up_weight, up_bias, down_weight, down_bias = parameters
    hidden, pre = _C.mlp_up_gemm_gelu_cuda_forward(x, up_weight, up_bias, True)
    cpu_pre = tree_gemm(x[:2].cpu(), up_weight[:8].t().cpu()) + up_bias[:8].float().cpu()
    assert equal_bytes(pre[:2, :8].cpu(), cpu_pre)
    torch.testing.assert_close(
        hidden[:2, :8].float().cpu(), gelu_tanh_fp32(cpu_pre), atol=0.05, rtol=0.02
    )
    cpu_y = tree_gemm(hidden[:2].cpu(), down_weight[:8].t().cpu()) + down_bias[:8].float().cpu()
    result = adapted(module.QwenImageTPMlpGemmGeluCudaOp(), x, parameters, grad, keys)
    torch.testing.assert_close(result[0][:2, :8].float().cpu(), cpu_y, atol=0.05, rtol=0.02)
    dh = _C.mlp_up_gemm_gelu_cuda_dx(grad, down_weight)
    gate = _C.mlp_up_gemm_gelu_cuda_gate(dh, pre)
    cpu_dh = tree_gemm(grad[:2].cpu(), down_weight[:, :8].cpu()).to(torch.bfloat16)
    assert equal_bytes(dh[:2, :8].cpu(), cpu_dh)
    cpu_gate = mlp_up_gemm_gelu_reference_gate(cpu_pre, cpu_dh)
    torch.testing.assert_close(gate[:2, :8].float().cpu(), cpu_gate.float(), atol=0.1, rtol=0.02)
    cpu_dx = tree_gemm(gate[:2].cpu(), up_weight[:, :8].cpu()).to(torch.bfloat16)
    assert equal_bytes(result[1][:2, :8].cpu(), cpu_dx)
    for index, rows, grads in [(2, x, gate), (4, hidden, grad)]:
        cpu_dw = left_fold_weight_gradient(grads[:, :8].cpu(), rows[:, :8].cpu()).to(torch.bfloat16)
        cpu_db = left_fold_bias_gradient(grads[:, :8].cpu()).to(torch.bfloat16)
        assert equal_bytes(result[index][:8, :8].cpu(), cpu_dw)
        assert equal_bytes(result[index + 1][:8].cpu(), cpu_db)


@cuda
def test_complete_cpu_reference_and_cuda_gradients():
    from rl_engine.runtime.registry import KernelRegistry

    x, parameters, grad, keys = make_inputs(1, 389)
    cpu = tuple(value.detach().cpu().requires_grad_() for value in (x, *parameters))
    reference = KernelRegistry().get_op("qwenimage_tp_mlp_gemm_gelu", device=cpu[0].device)
    output = reference(*cpu)
    gradients = torch.autograd.grad(output, cpu, grad.cpu())
    actual = adapted(module.QwenImageTPMlpGemmGeluCudaOp(), x, parameters, grad, keys)
    for index, (value, expected) in enumerate(zip(actual, (output, *gradients))):
        torch.testing.assert_close(
            value.float().cpu(), expected.float(), atol=0.05 if index == 0 else 0.1, rtol=0.02
        )


@cuda
@pytest.mark.parametrize("layout", ["reverse", "microbatch", "padding"])
def test_parameter_bi(layout):
    x, parameters, grad, keys = make_inputs(17)
    expected = native_mlp(x, *parameters, grad, keys)
    stream = {"x": x[None, None], "grad": grad[None, None], "keys": keys[None, None]}
    local_x, g, k, cuts = mlp_workload_rows(stream, layout)
    result = adapted(module.QwenImageTPMlpGemmGeluCudaOp(), local_x, parameters, g, k, cuts)
    assert all(equal_bytes(a, b) for a, b in zip(result, expected))


@cuda
@pytest.mark.parametrize("frozen", [0, 1, 2, 3, "all"])
def test_frozen_parameters(frozen):
    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    x, parameters, grad, keys = make_inputs(7)
    x.requires_grad_()
    parameters = tuple(
        p.requires_grad_(frozen != "all" and i != frozen) for i, p in enumerate(parameters)
    )
    op = module.QwenImageTPMlpGemmGeluCudaOp()
    expected = native_mlp(x.detach(), *parameters, grad, keys)
    with canonical_backward_session() as session:
        y = op(x, *parameters, logical_keys=keys, parameter_id="frozen")
        y.backward(grad)
    session.validate_complete()
    assert equal_bytes(x.grad, expected[1])
    for i, p in enumerate(parameters):
        assert equal_bytes(p.grad, expected[i + 2]) if p.requires_grad else p.grad is None


@cuda
def test_registry_inference_and_validation(monkeypatch):
    from rl_engine.runtime.registry import KernelRegistry

    x, parameters, grad, keys = make_inputs(1)
    op = KernelRegistry().get_op("qwenimage_tp_mlp_gemm_gelu", device=x.device)
    with torch.no_grad():
        y = op(x, *parameters)
    assert equal_bytes(y, native_mlp(x, *parameters, grad, keys)[0])
    assert op.implementation_info(x, *parameters)["fallback"] is False
    with pytest.raises(TypeError, match="bfloat16"):
        validate_mlp(x.float(), *(p.float() for p in parameters))
    with pytest.raises(ValueError, match="CPU"):
        QwenImageTPMlpGemmGeluReference()(x, *parameters)
    with pytest.raises(RuntimeError, match="session"):
        op(x, *[p.requires_grad_() for p in parameters])
    monkeypatch.delattr(_C, "mlp_up_gemm_gelu_cuda_tree_fp32")
    with pytest.raises(RuntimeError, match="rebuild"):
        module.QwenImageTPMlpGemmGeluCudaOp()


@cuda
def test_independent_streams_in_shared_session():
    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    inputs = [make_inputs(3, 386), make_inputs(5, 387)]
    expected = [native_mlp(x, *parameters, grad, keys) for x, parameters, grad, keys in inputs]
    leaves = [
        (x.requires_grad_(), *(p.requires_grad_() for p in parameters))
        for x, parameters, grad, keys in inputs
    ]
    op = module.QwenImageTPMlpGemmGeluCudaOp()
    with canonical_backward_session() as session:
        ys = [
            op(*values, logical_keys=inputs[i][3], parameter_id=name)
            for i, (values, name) in enumerate(zip(leaves, ("image.mlp", "text.mlp")))
        ]
        gradients = torch.autograd.grad(
            ys, tuple(value for group in leaves for value in group), [group[2] for group in inputs]
        )
    session.validate_complete()
    for i in range(2):
        actual = (ys[i], *gradients[5 * i : 5 * (i + 1)])
        assert all(equal_bytes(a, b) for a, b in zip(actual, expected[i]))


@cuda
def test_noncontiguous_inputs_and_upstream():
    x, parameters, grad, keys = make_inputs(7)
    expected = native_mlp(x, *parameters, grad, keys)
    x = x.t().contiguous().t()
    grad = grad.t().contiguous().t()
    parameters = tuple(p.t().contiguous().t() if p.ndim == 2 else p for p in parameters)
    assert not x.is_contiguous() and not grad.is_contiguous()
    actual = adapted(module.QwenImageTPMlpGemmGeluCudaOp(), x, parameters, grad, keys)
    assert all(equal_bytes(a, b) for a, b in zip(actual, expected))


def test_shape_and_topology_validation():
    x = torch.empty(1, 3072, dtype=torch.bfloat16)
    parameters = (
        torch.empty(12288, 3072, dtype=x.dtype),
        torch.empty(12288, dtype=x.dtype),
        torch.empty(3072, 12288, dtype=x.dtype),
        torch.empty(3072, dtype=x.dtype),
    )
    validate_mlp(x, *parameters)
    local = shard_mlp_parameters(*parameters, tp_world=2, tp_rank=0)
    assert all(
        p.untyped_storage().nbytes() == p.numel() * p.element_size() for p in local
    ), "local shards must not retain the full checkpoint storage"
    for world in (0, 3, True):
        with pytest.raises(ValueError, match="TP size"):
            validate_mlp(x, *parameters, world)
    with pytest.raises(ValueError, match="4194240"):
        validate_mlp(x.expand(4194241, -1), *parameters)
    with pytest.raises(ValueError, match="rank"):
        shard_mlp_parameters(*parameters, tp_world=2, tp_rank=2)


def check_real_precision(x, parameters, grad, keys, expected):
    """Check selected contractions against CPU FP32 at each real task length."""
    up_weight, up_bias, down_weight, down_bias = parameters
    hidden, pre = _C.mlp_up_gemm_gelu_cuda_forward(x, up_weight, up_bias, True)
    dh = _C.mlp_up_gemm_gelu_cuda_dx(grad, down_weight)
    gate = _C.mlp_up_gemm_gelu_cuda_gate(dh, pre)
    order = canonical_mlp_indices(keys)
    selected = order[:3]
    cpu_pre = x[selected].float().cpu() @ up_weight.float().cpu().t() + up_bias.float().cpu()
    torch.testing.assert_close(pre[selected].cpu(), cpu_pre, atol=0.05, rtol=0.02)
    cpu_y = (
        hidden[selected].float().cpu() @ down_weight[:32].float().cpu().t()
        + down_bias[:32].float().cpu()
    )
    cpu_dx = gate[selected].float().cpu() @ up_weight[:, :32].float().cpu()
    for actual, reference in [(expected[0][:3, :32], cpu_y), (expected[1][:3, :32], cpu_dx)]:
        torch.testing.assert_close(actual.float().cpu(), reference, atol=0.1, rtol=0.02)
    for index, rows, grads in [(2, x, gate), (4, hidden, grad)]:
        cpu_rows = rows[order, :32].float().cpu()
        cpu_grads = grads[order, :32].float().cpu()
        cpu_dw = cpu_grads.t() @ cpu_rows
        cpu_db = cpu_grads.sum(0)
        torch.testing.assert_close(
            expected[index][:32, :32].float().cpu(), cpu_dw, atol=0.1, rtol=0.02
        )
        torch.testing.assert_close(
            expected[index + 1][:32].float().cpu(), cpu_db, atol=0.1, rtol=0.02
        )


def check_workload(op, workload, world=1, rank=0):
    fixture = make_mlp_workload(workload, "cuda")
    checks = 0
    width = 12288 // world
    own = slice(rank * width, (rank + 1) * width)
    for stream_name, stream in fixture["streams"].items():
        parameters = tuple(stream[k] for k in ("up_weight", "up_bias", "down_weight", "down_bias"))
        x, grad, keys, _ = mlp_workload_rows(stream, "batched")
        full = native_mlp(x, *parameters, grad, keys)
        if rank == 0:
            check_real_precision(x, parameters, grad, keys, full)
        expected = (full[0], full[1], full[2][own], full[3][own], full[4][:, own], full[5])
        local = shard_mlp_parameters(*parameters, tp_world=world, tp_rank=rank)
        del full, parameters
        for layout in MLP_LAYOUTS:
            xx, g, k, cuts = mlp_workload_rows(stream, layout)
            actual = adapted(op, xx, local, g, k, cuts)
            for name, value, gold in zip(
                ("Y", "dX", "dW_up", "db_up", "dW_down", "db_down"), actual, expected
            ):
                assert equal_bytes(value, gold), (
                    workload,
                    stream_name,
                    layout,
                    world,
                    rank,
                    name,
                    int((value != gold).sum()),
                    (value.float() - gold.float()).abs().max().item(),
                )
            checks += 1
    return checks


@cuda
@pytest.mark.parametrize("workload", MLP_WORKLOADS)
def test_model_layouts(workload):
    assert check_workload(module.QwenImageTPMlpGemmGeluCudaOp(), workload) == 14


def check_adversarial_backward_order(op, world, rank):
    """Opposite rank-local priorities must not mix two independent requests."""
    from rl_engine.ops.autograd.canonical_backward import canonical_backward_session

    x, parameters, grad, keys = make_inputs(18, 388)
    full = native_mlp(x, *parameters, grad, keys)
    width = 12288 // world
    own = slice(rank * width, (rank + 1) * width)
    expected = (full[0], full[1], full[2][own], full[3][own], full[4][:, own], full[5])
    local = tuple(
        p.detach().requires_grad_()
        for p in shard_mlp_parameters(*parameters, tp_world=world, tp_rank=rank)
    )
    x.requires_grad_()
    with canonical_backward_session() as session:
        ys = [
            op(
                x[start : start + 9],
                *local,
                logical_keys=keys[start : start + 9],
                parameter_id="adversarial",
            )
            for start in (0, 9)
        ]
        for index, y in enumerate(ys):
            # Test-only scheduler perturbation; the graph edge must take precedence.
            y.grad_fn._set_sequence_nr(index + 1 if rank % 2 == 0 else 2 - index)
        grads = torch.autograd.grad(ys, (x, *local), [grad[:9], grad[9:]])
    session.validate_complete()
    assert not session._qwenimage_mlp_tails
    actual = (torch.cat(ys), *grads)
    assert all(equal_bytes(a, b) for a, b in zip(actual, expected))


def real_tp_worker(rank, world, directory):
    from rl_engine.distributed import create_deterministic_collective

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=10),
    )
    checks = 0
    try:
        with create_deterministic_collective(device=torch.device("cuda", rank)) as collective:
            op = module.QwenImageTPMlpGemmGeluCudaOp(collective)
            for rows in [1, 17, 129]:
                x, parameters, grad, keys = make_inputs(rows)
                full = native_mlp(x, *parameters, grad, keys)
                width = 12288 // world
                own = slice(rank * width, (rank + 1) * width)
                expected = (full[0], full[1], full[2][own], full[3][own], full[4][:, own], full[5])
                local = shard_mlp_parameters(*parameters, tp_world=world, tp_rank=rank)
                actual = adapted(op, x, local, grad, keys)
                assert all(equal_bytes(a, b) for a, b in zip(actual, expected))
                checks += 1
            del x, parameters, grad, keys, full, expected, local, actual
            check_adversarial_backward_order(op, world, rank)
            checks += 1
            for workload in MLP_WORKLOADS:
                checks += check_workload(op, workload, world, rank)
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps({"tp": world, "rank": rank, "checks": checks, "byte_equal": True})
        )
    finally:
        dist.destroy_process_group()


@cuda
@pytest.mark.parametrize("world", [2, 4, 8])
def test_real_tp(world, tmp_path):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} NVIDIA GPUs")
    mp.spawn(real_tp_worker, args=(world, str(tmp_path)), nprocs=world, join=True)
    assert len(list(tmp_path.glob("rank-*.json"))) == world
