import os

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.kernels.registry import OpBackend, kernel_registry


def test_adaln_modulation_forward_and_shared_backward():
    x = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]], requires_grad=True)
    # shift, scale, gate; the gate has its own downstream gradient.
    modulation = torch.tensor([[0.5, -0.5, 1.0, 0.0, 2.0, 3.0]], requires_grad=True)
    y, gate = NativeAdaLNModulationOp()(x, modulation, eps=0.0)
    torch.testing.assert_close(y, torch.tensor([[[-1.5, 0.5], [-1.5, 0.5]]]))
    torch.testing.assert_close(gate, torch.tensor([[[2.0, 3.0]]]))

    (y * torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])).sum().backward(retain_graph=True)
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(
        modulation.grad, torch.tensor([[4.0, 6.0, -4.0, 6.0, 0.0, 0.0]])
    )
    x.grad.zero_()
    modulation.grad.zero_()
    gate.sum().backward()
    torch.testing.assert_close(modulation.grad, torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]))


def test_adaln_modulation_rejects_wrong_modulation_shape():
    with pytest.raises(ValueError, match="modulation must have shape"):
        NativeAdaLNModulationOp()(torch.zeros(1, 2, 4), torch.zeros(1, 4))


def test_adaln_registry_trace_reports_selected_backend():
    op, trace = kernel_registry.get_adaln_modulation_op(device="cpu", hidden=4)
    assert isinstance(op, NativeAdaLNModulationOp)
    assert trace["selected_backend"] == OpBackend.PYTORCH_ADALN_MODULATION.name
    assert trace["fallback"] is False
    assert trace["accumulator_dtype"] == "fp32"
    assert trace["split_k"] is False and trace["stream_k"] is False


def test_adaln_modulate_index_uses_explicit_native_fallback():
    x = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]], requires_grad=True)
    # First batch row is choice 0, second batch row is choice 1.
    modulation = torch.tensor(
        [
            [0.5, -0.5, 1.0, 0.0, 2.0, 3.0],
            [-1.0, 1.0, 0.0, 1.0, 4.0, 5.0],
        ],
        requires_grad=True,
    )
    modulate_index = torch.tensor([[0, 1]])

    op, trace = kernel_registry.get_adaln_modulation_op(
        device=x.device,
        hidden=x.shape[-1],
        modulate_index=modulate_index,
    )
    y, gate = op(x, modulation, modulate_index=modulate_index, eps=0.0)

    torch.testing.assert_close(y, torch.tensor([[[-1.5, 0.5], [-2.0, 3.0]]]))
    torch.testing.assert_close(gate, torch.tensor([[[2.0, 3.0], [4.0, 5.0]]]))
    (y.sum() + gate.sum()).backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(
        modulation.grad,
        torch.tensor([[1.0, 1.0, -1.0, 1.0, 1.0, 1.0]]).repeat(2, 1),
    )
    assert isinstance(op, NativeAdaLNModulationOp)
    assert trace["selected_backend"] == OpBackend.PYTORCH_ADALN_MODULATION.name
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == "modulate_index_requires_select01"


def test_adaln_cuda_registry_reports_real_fallback():
    try:
        torch.cuda.init()
    except RuntimeError:
        pytest.skip("CUDA required")
    op, trace = kernel_registry.get_adaln_modulation_op(device="cuda", hidden=32)
    assert type(op).__name__ in {"CudaAdaLNModulationOp", "TritonAdaLNModulationOp"}
    if type(op).__name__ == "TritonAdaLNModulationOp":
        assert trace["fallback"] is True
        assert "CUDA_ADALN_MODULATION" in trace["rejected_backends"]
    else:
        assert trace["fallback"] is False


def test_adaln_registry_skips_cuda_for_large_hidden():
    op, trace = kernel_registry.get_adaln_modulation_op(device="cuda", hidden=5000)
    assert type(op).__name__ == "TritonAdaLNModulationOp"
    assert trace["fallback"] is True
    assert "CUDA_ADALN_MODULATION: H > 4096" in trace["rejected_backends"]


def test_adaln_modulation_matches_independent_layer_norm_autograd():
    torch.manual_seed(386)
    x = torch.randn(2, 3, 7, requires_grad=True)
    modulation = torch.randn(2, 21, requires_grad=True)
    upstream = torch.randn_like(x)
    gate_upstream = torch.randn(2, 1, 7)
    y, gate = NativeAdaLNModulationOp()(x, modulation)
    ((y * upstream).sum() + (gate * gate_upstream).sum()).backward()

    x_ref = x.detach().clone().requires_grad_()
    modulation_ref = modulation.detach().clone().requires_grad_()
    shift, scale, gate_ref = modulation_ref.chunk(3, dim=-1)
    y_ref = torch.nn.functional.layer_norm(x_ref, (7,), eps=1e-6)
    y_ref = y_ref * (1 + scale[:, None, :]) + shift[:, None, :]
    ((y_ref * upstream).sum() + (gate_ref[:, None, :] * gate_upstream).sum()).backward()
    torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(modulation.grad, modulation_ref.grad, atol=1e-6, rtol=1e-6)


def test_adaln_modulation_batch_position_does_not_change_bytes():
    torch.manual_seed(387)
    row = torch.randn(1, 5, 17)
    mod = torch.randn(1, 51)
    op = NativeAdaLNModulationOp()
    expected = op(row, mod)[0]
    for position in range(3):
        batch = torch.randn(3, 5, 17)
        batch_mod = torch.randn(3, 51)
        batch[position] = row[0]
        batch_mod[position] = mod[0]
        actual = op(batch, batch_mod)[0][position]
        assert torch.equal(actual, expected[0])


def test_adaln_modulation_bf16_cast_and_backward():
    torch.manual_seed(389)
    x = torch.randn(1, 4, 9, dtype=torch.bfloat16, requires_grad=True)
    modulation = torch.randn(1, 27, dtype=torch.bfloat16, requires_grad=True)
    y, gate = NativeAdaLNModulationOp()(x, modulation)
    assert y.dtype == gate.dtype == torch.bfloat16
    (y.float().sum() + gate.float().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert modulation.grad is not None and torch.isfinite(modulation.grad).all()
    assert torch.equal(modulation.grad[0, 18:], torch.ones(9, dtype=torch.bfloat16))


def test_adaln_modulation_accepts_qwen_six_chunk_view():
    x = torch.randn(2, 3, 8)
    full = torch.randn(2, 48, requires_grad=True)
    modulation = full.chunk(2, dim=-1)[0]
    assert not modulation.is_contiguous()
    y, gate = NativeAdaLNModulationOp()(x, modulation)
    (y.sum() + gate.sum()).backward()
    assert full.grad is not None
    assert torch.equal(full.grad[:, 24:], torch.zeros_like(full.grad[:, 24:]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_adaln_modulation_forward_backward_and_batch_position(dtype):
    try:
        torch.cuda.init()
    except RuntimeError:
        pytest.skip("CUDA required")
    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    torch.manual_seed(388)
    x = torch.randn(2, 3, 32, device="cuda", dtype=dtype, requires_grad=True)
    modulation = torch.randn(2, 96, device="cuda", dtype=dtype, requires_grad=True)
    upstream = torch.randn_like(x)
    gate_upstream = torch.randn(2, 1, 32, device="cuda", dtype=dtype)
    y, gate = TritonAdaLNModulationOp()(x, modulation)
    ((y * upstream).sum() + (gate * gate_upstream).sum()).backward()
    x_ref = x.detach().clone().requires_grad_()
    mod_ref = modulation.detach().clone().requires_grad_()
    y_ref, gate_ref = NativeAdaLNModulationOp()(x_ref, mod_ref)
    ((y_ref * upstream).sum() + (gate_ref * gate_upstream).sum()).backward()
    assert torch.equal(y, y_ref)
    assert torch.equal(x.grad, x_ref.grad)
    assert torch.equal(modulation.grad, mod_ref.grad)

    x_new = torch.cat([torch.randn_like(x[:1]), x[:1]], dim=0).detach().requires_grad_()
    mod_new = (
        torch.cat([torch.randn_like(modulation[:1]), modulation[:1]], dim=0)
        .detach()
        .requires_grad_()
    )
    y_new, gate_new = TritonAdaLNModulationOp()(x_new, mod_new)
    assert torch.equal(y_new[1], y[0])
    assert torch.equal(gate_new[1], gate[0])
    upstream_new = torch.zeros_like(y_new)
    upstream_new[1] = upstream[0]
    gate_upstream_new = torch.zeros_like(gate_new)
    gate_upstream_new[1] = gate_upstream[0]
    torch.autograd.backward((y_new, gate_new), (upstream_new, gate_upstream_new))
    assert torch.equal(x_new.grad[1], x.grad[0])
    assert torch.equal(mod_new.grad[1], modulation.grad[0])


def test_triton_irregular_hidden_and_bf16_tie_rounding():
    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    try:
        torch.cuda.init()
    except RuntimeError:
        pytest.skip("CUDA required")
    for hidden, seq, dtype, seed in (
        (7, 3, torch.float32, 386),
        (3072, 64, torch.bfloat16, 391),
    ):
        torch.manual_seed(seed)
        x = torch.randn(1, seq, hidden, device="cuda", dtype=dtype, requires_grad=True)
        modulation = torch.randn(1, 3 * hidden, device="cuda", dtype=dtype, requires_grad=True)
        y, gate = TritonAdaLNModulationOp()(x, modulation)
        (y.float().sum() + gate.float().sum()).backward()

        x_cpu = x.detach().cpu().requires_grad_()
        modulation_cpu = modulation.detach().cpu().requires_grad_()
        y_cpu, gate_cpu = NativeAdaLNModulationOp()(x_cpu, modulation_cpu)
        (y_cpu.float().sum() + gate_cpu.float().sum()).backward()
        assert torch.equal(y.cpu(), y_cpu)
        assert torch.equal(x.grad.cpu(), x_cpu.grad)
        assert torch.equal(modulation.grad.cpu(), modulation_cpu.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_gpu_padding_and_batch_size_do_not_change_valid_bytes(dtype):
    from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    try:
        torch.cuda.init()
    except RuntimeError:
        pytest.skip("CUDA required")
    torch.manual_seed(392)
    x = torch.randn(1, 3, 32, device="cuda", dtype=dtype)
    modulation = torch.randn(1, 96, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)
    dgate = torch.randn(1, 1, 32, device="cuda", dtype=dtype)
    ops = [TritonAdaLNModulationOp()]
    try:
        ops.append(CudaAdaLNModulationOp())
    except RuntimeError:
        pass

    for op in ops:
        baseline_x = x.detach().requires_grad_()
        baseline_modulation = modulation.detach().requires_grad_()
        baseline_y, baseline_gate = op(baseline_x, baseline_modulation)
        torch.autograd.backward((baseline_y, baseline_gate), (dy, dgate))

        for batch, seq in ((1, 5), (3, 3), (3, 5)):
            padded_x = torch.randn(batch, seq, 32, device="cuda", dtype=dtype)
            padded_modulation = torch.randn(batch, 96, device="cuda", dtype=dtype)
            padded_dy = torch.zeros_like(padded_x)
            padded_dgate = torch.zeros(batch, 1, 32, device="cuda", dtype=dtype)
            padded_x[-1, :3] = x[0]
            padded_modulation[-1] = modulation[0]
            padded_dy[-1, :3] = dy[0]
            padded_dgate[-1] = dgate[0]
            padded_x.requires_grad_()
            padded_modulation.requires_grad_()
            y, gate = op(padded_x, padded_modulation)
            torch.autograd.backward((y, gate), (padded_dy, padded_dgate))
            assert torch.equal(y[-1, :3], baseline_y[0])
            assert torch.equal(gate[-1], baseline_gate[0])
            assert torch.equal(padded_x.grad[-1, :3], baseline_x.grad[0])
            assert torch.equal(padded_modulation.grad[-1], baseline_modulation.grad[0])


def test_cuda_adaln_bit_equality_harness():
    from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

    try:
        torch.cuda.init()
        cuda_op = CudaAdaLNModulationOp()
    except RuntimeError:
        pytest.skip("compiled CUDA AdaLN extension required")
    torch.manual_seed(390)
    x = torch.randn(1, 3, 32, requires_grad=True)
    modulation = torch.randn(1, 96, requires_grad=True)
    upstream = torch.randn_like(x)
    gate_upstream = torch.randn(1, 1, 32)
    cpu_y, cpu_gate = NativeAdaLNModulationOp()(x, modulation)
    ((cpu_y * upstream).sum() + (cpu_gate * gate_upstream).sum()).backward()

    x_gpu = x.detach().cuda().requires_grad_()
    mod_gpu = modulation.detach().cuda().requires_grad_()
    cuda_y, cuda_gate = cuda_op(x_gpu, mod_gpu)
    ((cuda_y * upstream.cuda()).sum() + (cuda_gate * gate_upstream.cuda()).sum()).backward()
    assert torch.equal(cuda_y.cpu(), cpu_y)
    assert torch.equal(cuda_gate.cpu(), cpu_gate)
    assert torch.equal(x_gpu.grad.cpu(), x.grad)
    assert torch.equal(mod_gpu.grad.cpu(), modulation.grad)


@pytest.mark.parametrize("seq", [4096, 6889, 6032])
@pytest.mark.skipif(
    os.environ.get("RLK_ADALN_REAL_SHAPES") != "1",
    reason="set RLK_ADALN_REAL_SHAPES=1 for Qwen-Image resolution coverage",
)
def test_adaln_real_image_shapes_forward_backward(seq):
    try:
        torch.cuda.init()
    except RuntimeError:
        pytest.skip("CUDA required")
    op, trace = kernel_registry.get_adaln_modulation_op(device="cuda", hidden=3072)
    if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
        assert trace["selected_backend"] == OpBackend.CUDA_ADALN_MODULATION.name
    assert trace["selected_backend"] in {
        OpBackend.CUDA_ADALN_MODULATION.name,
        OpBackend.TRITON_ADALN_MODULATION.name,
    }
    x = torch.randn(1, seq, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    modulation = torch.randn(1, 9216, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y, gate = op(x, modulation)
    assert y.shape == x.shape and gate.shape == (1, 1, 3072)
    (y.float().sum() + gate.float().sum()).backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(modulation.grad).all()

    cpu_x = x.detach().cpu().requires_grad_()
    cpu_modulation = modulation.detach().cpu().requires_grad_()
    cpu_y, cpu_gate = NativeAdaLNModulationOp()(cpu_x, cpu_modulation)
    (cpu_y.float().sum() + cpu_gate.float().sum()).backward()
    assert torch.equal(y.cpu(), cpu_y)
    assert torch.equal(gate.cpu(), cpu_gate)
    assert torch.equal(x.grad.cpu(), cpu_x.grad)
    assert torch.equal(modulation.grad.cpu(), cpu_modulation.grad)

    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    triton_x = x.detach().clone().requires_grad_()
    triton_modulation = modulation.detach().clone().requires_grad_()
    triton_y, triton_gate = TritonAdaLNModulationOp()(triton_x, triton_modulation)
    (triton_y.float().sum() + triton_gate.float().sum()).backward()
    assert torch.equal(triton_y, y)
    assert torch.equal(triton_gate, gate)
    assert torch.equal(triton_x.grad, x.grad)
    assert torch.equal(triton_modulation.grad, modulation.grad)
