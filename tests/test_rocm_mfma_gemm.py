# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Bitwise contract tests for the ROCm MFMA deterministic GEMM."""

from __future__ import annotations

import importlib
import os

import pytest
import torch

IS_ROCM = getattr(torch.version, "hip", None) is not None
IS_GFX942 = (
    IS_ROCM
    and torch.cuda.is_available()
    and str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")).startswith("gfx942")
)

pytestmark = pytest.mark.skipif(not IS_GFX942, reason="MFMA det GEMM targets ROCm gfx942")

if IS_GFX942:
    from rl_engine.kernels.ops.triton.matmul import mfma_gemm as M

DEV = "cuda"
# Qwen3-8B TP4 local shapes as (K, N).
QWEN_TP4_SHAPES = [(4096, 1536), (1024, 4096), (4096, 6144), (3072, 4096)]


def _rand(*shape, scale=1.0):
    return (torch.randn(*shape, device=DEV) * scale).to(torch.bfloat16)


def _configs():
    return [
        M.MfmaGemmConfig(16, 16, 1, waves_per_eu=0, num_stages=1, group_m=1),
        M.MfmaGemmConfig(16, 32, 2, waves_per_eu=0, num_stages=2, group_m=1),
        M.MfmaGemmConfig(32, 128, 4, waves_per_eu=2, num_stages=2, group_m=8),
        M.MfmaGemmConfig(64, 64, 4, waves_per_eu=2, num_stages=2, group_m=4),
        M.MfmaGemmConfig(64, 128, 4, waves_per_eu=2, num_stages=3, group_m=8),
        M.MfmaGemmConfig(128, 128, 4, waves_per_eu=2, num_stages=2, group_m=8),
        M.MfmaGemmConfig(128, 128, 8, waves_per_eu=1, num_stages=2, group_m=16),
        M.MfmaGemmConfig(128, 256, 8, waves_per_eu=2, num_stages=2, group_m=8),
        M.MfmaGemmConfig(256, 128, 8, waves_per_eu=1, num_stages=2, group_m=8),
    ]


def test_qwen_tp4_decode_config_selection():
    default = M.MfmaGemmConfig(16, 32, 2, waves_per_eu=0, num_stages=2, group_m=1)
    wide = M.MfmaGemmConfig(32, 64, 4, waves_per_eu=0, num_stages=2, group_m=1)
    vocab = M.MfmaGemmConfig(16, 128, 4, waves_per_eu=2, num_stages=2, group_m=1)

    assert M.select_config(1, 1536, 4096) == default
    assert M.select_config(4, 1536, 4096) == wide
    assert M.select_config(4, 6144, 4096) == wide
    assert M.select_config(4, 37984, 4096) == vocab
    assert M.select_config(4, 4096, 1024) == default
    assert M.select_config(4, 4096, 3072) == default


@pytest.mark.parametrize("k_size,n_size", QWEN_TP4_SHAPES)
def test_forward_matches_fp32_reference(k_size, n_size):
    a = _rand(300, k_size)
    w = _rand(n_size, k_size, scale=0.02)
    out = M.mfma_linear(a, w)
    ref = a.float() @ w.float().t()
    assert out.shape == (300, n_size)
    tol = 8e-3 * ref.abs().max().item()
    assert (out.float() - ref).abs().max().item() <= tol


@pytest.mark.parametrize("k_size,n_size", QWEN_TP4_SHAPES)
def test_every_schedule_and_layout_is_bitwise_identical(k_size, n_size):
    a = _rand(4096 + 37, k_size)
    w = _rand(n_size, k_size, scale=0.02)
    reference = M.mfma_gemm(a, w.t())
    layouts = {"nk_view": w.t(), "kn_contiguous": w.t().contiguous()}
    for config in _configs():
        for name, b in layouts.items():
            for force_split in (False, True):
                out = M.mfma_gemm(a, b, config=config, force_split=force_split)
                assert torch.equal(out, reference), (config, name, force_split)


@pytest.mark.parametrize("k_size,n_size", QWEN_TP4_SHAPES)
def test_rows_are_batch_invariant(k_size, n_size):
    a = _rand(4096 + 37, k_size)
    w = _rand(n_size, k_size, scale=0.02)
    reference = M.mfma_linear(a, w)
    configs = _configs()
    for rows, start in ((1, 5), (7, 100), (8, 0), (32, 1000), (33, 4000), (129, 77), (1024, 3000)):
        sub = a[start : start + rows]
        for config in (configs[0], configs[2], configs[8]):
            for force_split in (False, True):
                out = M.mfma_gemm(sub, w.t(), config=config, force_split=force_split)
                assert torch.equal(out, reference[start : start + rows]), (rows, start, config)
    strided = a[::3][:64]
    assert torch.equal(M.mfma_linear(strided, w), reference[::3][:64])


def test_column_major_a_operand_is_bitwise_identical():
    grad = _rand(4096, 6144)
    a = _rand(4096, 4096)
    reference = M.mfma_gemm(grad.t().contiguous(), a)
    assert torch.equal(M.mfma_gemm(grad.t(), a), reference)
    assert torch.equal(M.mfma_linear_weight_gradient(a, grad), reference)
    ref = grad.float().t() @ a.float()
    assert (reference.float() - ref).abs().max().item() <= 8e-3 * ref.abs().max().item()


def test_output_buffer_and_narrowed_staging_views():
    a = _rand(24, 4096)
    w = _rand(1536, 4096, scale=0.02)
    reference = M.mfma_linear(a, w)
    staging = torch.zeros((32, 1536), dtype=torch.bfloat16, device=DEV)
    out = M.mfma_linear(a, w, out=staging.narrow(0, 0, 24))
    assert out.data_ptr() == staging.data_ptr()
    assert torch.equal(staging[:24], reference)
    assert torch.equal(staging[24:], torch.zeros_like(staging[24:]))


def test_ragged_k_and_ragged_mn_match_padded_computation():
    a = _rand(45, 1000)
    b = _rand(1000, 77)
    out = M.mfma_gemm(a, b)
    a_pad = torch.zeros((45, 1024), dtype=torch.bfloat16, device=DEV)
    a_pad[:, :1000] = a
    b_pad = torch.zeros((1024, 77), dtype=torch.bfloat16, device=DEV)
    b_pad[:1000] = b
    assert torch.equal(out, M.mfma_gemm(a_pad, b_pad))
    ref = a.float() @ b.float()
    assert (out.float() - ref).abs().max().item() <= 8e-3 * ref.abs().max().item()


def test_backward_is_deterministic_and_close_to_reference():
    a = _rand(512, 4096).requires_grad_(True)
    w = _rand(1536, 4096, scale=0.02).requires_grad_(True)
    grad = _rand(512, 1536)
    outputs = []
    for _ in range(2):
        out = M.MfmaLinearFn.apply(a, w)
        out.backward(grad)
        outputs.append((out.detach().clone(), a.grad.clone(), w.grad.clone()))
        a.grad = None
        w.grad = None
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert torch.equal(outputs[0][2], outputs[1][2])
    ref_da = grad.float() @ w.detach().float()
    ref_dw = grad.float().t() @ a.detach().float()
    assert (outputs[0][1].float() - ref_da).abs().max().item() <= 8e-3 * ref_da.abs().max().item()
    assert (outputs[0][2].float() - ref_dw).abs().max().item() <= 8e-3 * ref_dw.abs().max().item()


def test_generic_gemm_backward_matches_linear_backward_bitwise():
    a = _rand(256, 3072).requires_grad_(True)
    w = _rand(4096, 3072, scale=0.02).requires_grad_(True)
    grad = _rand(256, 4096)
    linear_out = M.MfmaLinearFn.apply(a, w)
    linear_out.backward(grad)
    linear_grads = (a.grad.clone(), w.grad.clone())
    a.grad = None
    w.grad = None
    generic_out = M.MfmaGemmFn.apply(a, w.t())
    generic_out.backward(grad)
    assert torch.equal(linear_out, generic_out)
    assert torch.equal(linear_grads[0], a.grad)
    assert torch.equal(linear_grads[1], w.grad)


def test_facade_routes_to_mfma_by_default(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_DET_GEMM_BACKEND", raising=False)
    from rl_engine.kernels.ops.rocm.matmul import det_gemm as facade

    facade = importlib.reload(facade)
    assert facade.det_gemm_backend() == "triton_mfma"
    assert facade.det_gemm_backend_id() == M.MFMA_GEMM_CONTRACT_ID
    a = _rand(40, 4096)
    w = _rand(1536, 4096, scale=0.02)
    assert torch.equal(facade.det_gemm_linear(a, w), M.mfma_linear(a, w))
    assert torch.equal(facade.det_gemm_linear_prepared(a, w.t().contiguous()), M.mfma_linear(a, w))
    grad = _rand(40, 1536)
    assert torch.equal(
        facade.det_gemm_linear_input_gradient(grad, w), M.mfma_linear_input_gradient(grad, w)
    )
    assert torch.equal(
        facade.det_gemm_linear_weight_gradient(a, grad), M.mfma_linear_weight_gradient(a, grad)
    )
    op = facade.RocmDetGemmOp()
    assert torch.equal(op(a, w.t().contiguous()), M.mfma_linear(a, w))
    assert torch.equal(op.linear(a, w), M.mfma_linear(a, w))


def test_facade_tree_backend_remains_selectable(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "triton_tree")
    from rl_engine.kernels.ops.rocm.matmul import det_gemm as facade

    facade = importlib.reload(facade)
    try:
        assert facade.det_gemm_backend() == "triton_tree"
        assert facade.det_gemm_backend_id() == "rlkernel.det_gemm.triton_tree_rocm.v1"
        from rl_engine.kernels.ops.triton.matmul.det_gemm import _triton_tree_gemm

        a = _rand(8, 1024)
        w = _rand(256, 1024, scale=0.02)
        assert torch.equal(facade.det_gemm_linear(a, w), _triton_tree_gemm(a, w.t().contiguous()))
    finally:
        monkeypatch.delenv("RL_KERNEL_DET_GEMM_BACKEND", raising=False)
        importlib.reload(facade)


def test_facade_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "cublas")
    from rl_engine.kernels.ops.rocm.matmul import det_gemm as facade

    with pytest.raises(RuntimeError, match="RL_KERNEL_DET_GEMM_BACKEND"):
        importlib.reload(facade)
    monkeypatch.delenv("RL_KERNEL_DET_GEMM_BACKEND", raising=False)
    importlib.reload(facade)
    assert os.getenv("RL_KERNEL_DET_GEMM_BACKEND") is None
