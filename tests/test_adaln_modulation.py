import os

import pytest
import torch

from rl_engine.kernels.ops.pytorch.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.kernels.registry import OpBackend, kernel_registry


def _assert_bytes_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual_bytes = actual.detach().cpu().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


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
        _assert_bytes_equal(actual, expected[0])


def test_adaln_modulation_bf16_cast_and_backward():
    torch.manual_seed(389)
    x = torch.randn(1, 4, 9, dtype=torch.bfloat16, requires_grad=True)
    modulation = torch.randn(1, 27, dtype=torch.bfloat16, requires_grad=True)
    y, gate = NativeAdaLNModulationOp()(x, modulation)
    assert y.dtype == gate.dtype == torch.bfloat16
    (y.float().sum() + gate.float().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert modulation.grad is not None and torch.isfinite(modulation.grad).all()
    _assert_bytes_equal(modulation.grad[0, 18:], torch.ones(9, dtype=torch.bfloat16))


def test_adaln_modulation_accepts_qwen_six_chunk_view():
    x = torch.randn(2, 3, 8)
    full = torch.randn(2, 48, requires_grad=True)
    modulation = full.chunk(2, dim=-1)[0]
    assert not modulation.is_contiguous()
    y, gate = NativeAdaLNModulationOp()(x, modulation)
    (y.sum() + gate.sum()).backward()
    assert full.grad is not None
    _assert_bytes_equal(full.grad[:, 24:], torch.zeros_like(full.grad[:, 24:]))


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
    _assert_bytes_equal(y, y_ref)
    _assert_bytes_equal(x.grad, x_ref.grad)
    _assert_bytes_equal(modulation.grad, mod_ref.grad)

    x_new = torch.cat([torch.randn_like(x[:1]), x[:1]], dim=0).detach().requires_grad_()
    mod_new = (
        torch.cat([torch.randn_like(modulation[:1]), modulation[:1]], dim=0)
        .detach()
        .requires_grad_()
    )
    y_new, gate_new = TritonAdaLNModulationOp()(x_new, mod_new)
    _assert_bytes_equal(y_new[1], y[0])
    _assert_bytes_equal(gate_new[1], gate[0])
    upstream_new = torch.zeros_like(y_new)
    upstream_new[1] = upstream[0]
    gate_upstream_new = torch.zeros_like(gate_new)
    gate_upstream_new[1] = gate_upstream[0]
    torch.autograd.backward((y_new, gate_new), (upstream_new, gate_upstream_new))
    _assert_bytes_equal(x_new.grad[1], x.grad[0])
    _assert_bytes_equal(mod_new.grad[1], modulation.grad[0])


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
        _assert_bytes_equal(y.cpu(), y_cpu)
        _assert_bytes_equal(x.grad.cpu(), x_cpu.grad)
        _assert_bytes_equal(modulation.grad.cpu(), modulation_cpu.grad)


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
            _assert_bytes_equal(y[-1, :3], baseline_y[0])
            _assert_bytes_equal(gate[-1], baseline_gate[0])
            _assert_bytes_equal(padded_x.grad[-1, :3], baseline_x.grad[0])
            _assert_bytes_equal(padded_modulation.grad[-1], baseline_modulation.grad[0])


@pytest.mark.parametrize("hidden", [32, 64, 3072])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_adaln_bit_equality_harness(hidden, dtype):
    from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp

    try:
        torch.cuda.init()
        cuda_op = CudaAdaLNModulationOp()
    except RuntimeError:
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")
    torch.manual_seed(390)
    x = torch.randn(1, 3, hidden, dtype=dtype, requires_grad=True)
    modulation = torch.randn(1, 3 * hidden, dtype=dtype, requires_grad=True)
    upstream = torch.randn_like(x)
    gate_upstream = torch.randn(1, 1, hidden, dtype=dtype)
    cpu_y, cpu_gate = NativeAdaLNModulationOp()(x, modulation)
    ((cpu_y * upstream).sum() + (cpu_gate * gate_upstream).sum()).backward()

    x_gpu = x.detach().cuda().requires_grad_()
    mod_gpu = modulation.detach().cuda().requires_grad_()
    cuda_y, cuda_gate = cuda_op(x_gpu, mod_gpu)
    ((cuda_y * upstream.cuda()).sum() + (cuda_gate * gate_upstream.cuda()).sum()).backward()
    _assert_bytes_equal(cuda_y.cpu(), cpu_y)
    _assert_bytes_equal(cuda_gate.cpu(), cpu_gate)
    _assert_bytes_equal(x_gpu.grad.cpu(), x.grad)
    _assert_bytes_equal(mod_gpu.grad.cpu(), modulation.grad)


@pytest.mark.parametrize("seq", [4096, 6889, 6032])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.skipif(
    os.environ.get("RLK_ADALN_REAL_SHAPES") != "1",
    reason="set RLK_ADALN_REAL_SHAPES=1 for Qwen-Image resolution coverage",
)
def test_adaln_real_image_shapes_forward_backward(seq, dtype):
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
    torch.manual_seed(396 + seq)
    x = torch.randn(1, seq, 3072, device="cuda", dtype=dtype, requires_grad=True)
    modulation = torch.randn(1, 9216, device="cuda", dtype=dtype, requires_grad=True)
    y, gate = op(x, modulation)
    assert y.shape == x.shape and gate.shape == (1, 1, 3072)
    dy, dg = torch.randn_like(y), torch.randn_like(gate)
    torch.autograd.backward((y, gate), (dy, dg))
    assert torch.isfinite(x.grad).all() and torch.isfinite(modulation.grad).all()

    cpu_x = x.detach().cpu().requires_grad_()
    cpu_modulation = modulation.detach().cpu().requires_grad_()
    cpu_y, cpu_gate = NativeAdaLNModulationOp()(cpu_x, cpu_modulation)
    torch.autograd.backward((cpu_y, cpu_gate), (dy.cpu(), dg.cpu()))
    _assert_bytes_equal(y.cpu(), cpu_y)
    _assert_bytes_equal(gate.cpu(), cpu_gate)
    _assert_bytes_equal(x.grad.cpu(), cpu_x.grad)
    _assert_bytes_equal(modulation.grad.cpu(), cpu_modulation.grad)

    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    triton_x = x.detach().clone().requires_grad_()
    triton_modulation = modulation.detach().clone().requires_grad_()
    triton_y, triton_gate = TritonAdaLNModulationOp()(triton_x, triton_modulation)
    torch.autograd.backward((triton_y, triton_gate), (dy, dg))
    _assert_bytes_equal(triton_y, y)
    _assert_bytes_equal(triton_gate, gate)
    _assert_bytes_equal(triton_x.grad, x.grad)
    _assert_bytes_equal(triton_modulation.grad, modulation.grad)


def test_adaln_byte_comparison_distinguishes_signed_zero():
    with pytest.raises(AssertionError):
        _assert_bytes_equal(torch.tensor([0.0]), torch.tensor([-0.0]))


@pytest.mark.parametrize("backend", ["cuda", "triton"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_launch_geometry_preserves_bytes(backend, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    if backend == "cuda":
        try:
            ops = [CudaAdaLNModulationOp(threads=n) for n in (128, 256, 512)]
        except RuntimeError:
            if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
                raise
            pytest.skip("compiled CUDA AdaLN extension required")
    else:
        ops = [
            TritonAdaLNModulationOp(num_warps=w, reduction_tile=t)
            for w in (4, 8) for t in (64, 128, 256)
        ]
    torch.manual_seed(393)
    x = torch.randn(2, 5, 3072, dtype=dtype)
    modulation = torch.randn(2, 9216, dtype=dtype)
    dy = torch.randn_like(x)
    dg = torch.randn(2, 1, 3072, dtype=dtype)
    cpu_x = x.clone().requires_grad_()
    cpu_modulation = modulation.clone().requires_grad_()
    expected_y, expected_gate = NativeAdaLNModulationOp()(cpu_x, cpu_modulation)
    torch.autograd.backward((expected_y, expected_gate), (dy, dg))
    for op in ops:
        gpu_x = x.cuda().requires_grad_()
        gpu_modulation = modulation.cuda().requires_grad_()
        y, gate = op(gpu_x, gpu_modulation)
        torch.autograd.backward((y, gate), (dy.cuda(), dg.cuda()))
        for actual, expected in (
            (y, expected_y), (gate, expected_gate),
            (gpu_x.grad, cpu_x.grad), (gpu_modulation.grad, cpu_modulation.grad),
        ):
            _assert_bytes_equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("broadcast", [False, True])
def test_adaln_indexed_matches_independent_autograd(device, dtype, broadcast):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(394)
    x = torch.randn(2, 4, 7, device=device, dtype=dtype, requires_grad=True)
    modulation = torch.randn(4, 21, device=device, dtype=dtype, requires_grad=True)
    index = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0]], device=device)
    if broadcast:
        index = index[:1]
    dy, dg = torch.randn_like(x), torch.randn_like(x)
    op, trace = kernel_registry.get_adaln_modulation_op(
        device=device, hidden=7, modulate_index=index
    )
    y, gate = op(x, modulation, modulate_index=index)
    torch.autograd.backward((y, gate), (dy, dg))

    ref_x = x.detach().clone().requires_grad_()
    ref_modulation = modulation.detach().clone().requires_grad_()
    # Independent row indexing, not the implementation's split-and-where path.
    selected = ref_modulation.float()[torch.arange(2, device=device)[:, None] + index * 2]
    shift, scale, ref_gate = selected.chunk(3, dim=-1)
    ref_y = (
        torch.nn.functional.layer_norm(ref_x.float(), (7,), eps=1e-6) * (1 + scale) + shift
    ).to(dtype)
    ref_gate = ref_gate.to(dtype)
    torch.autograd.backward((ref_y, ref_gate), (dy, dg))
    for actual, expected in (
        (y, ref_y), (x.grad, ref_x.grad), (modulation.grad, ref_modulation.grad)
    ):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    _assert_bytes_equal(gate, ref_gate)
    assert x.grad.abs().max() > 0
    assert trace["fallback"] is True
    assert trace["fallback_reason"] == "modulate_index_requires_select01"
    assert "framework_select01_backward" in trace["reduction_order"]


@pytest.mark.parametrize(
    "index, error",
    [
        (torch.tensor([[0, 2]]), ValueError),
        (torch.tensor([[-1, 0]]), ValueError),
        (torch.tensor([[0.0, 1.0]]), TypeError),
        (torch.tensor([0, 1]), ValueError),
    ],
)
def test_adaln_indexed_rejects_invalid_routing(index, error):
    with pytest.raises(error):
        NativeAdaLNModulationOp()(
            torch.ones(1, 2, 7), torch.ones(2, 21), modulate_index=index
        )


@pytest.mark.parametrize("backend", ["pytorch", "triton", "cuda"])
@pytest.mark.parametrize("hidden", [7, 3072])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_shared_matches_independent_autograd(backend, hidden, dtype):
    if backend != "pytorch" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from rl_engine.kernels.ops.cuda.norm.adaln_modulation import CudaAdaLNModulationOp
    from rl_engine.kernels.ops.triton.norm.adaln_modulation import TritonAdaLNModulationOp

    ops = {"pytorch": NativeAdaLNModulationOp, "triton": TritonAdaLNModulationOp,
           "cuda": CudaAdaLNModulationOp}
    if backend == "cuda":
        try:
            op = ops[backend]()
        except RuntimeError:
            if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
                raise
            pytest.skip("compiled CUDA AdaLN extension required")
    else:
        op = ops[backend]()
    device = "cpu" if backend == "pytorch" else "cuda"
    torch.manual_seed(395)
    x = torch.randn(2, 3, hidden, device=device, dtype=dtype, requires_grad=True)
    modulation = torch.randn(2, 3 * hidden, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    dg = torch.randn(2, 1, hidden, device=device, dtype=dtype)
    y, gate = op(x, modulation)
    torch.autograd.backward((y, gate), (dy, dg))

    ref_x = x.detach().cpu().requires_grad_()
    ref_modulation = modulation.detach().cpu().requires_grad_()
    shift, scale, ref_gate = ref_modulation.float().chunk(3, dim=-1)
    ref_y = (
        torch.nn.functional.layer_norm(ref_x.float(), (hidden,), eps=1e-6)
        * (1 + scale[:, None, :]) + shift[:, None, :]
    ).to(dtype)
    ref_gate = ref_gate[:, None, :].to(dtype)
    torch.autograd.backward((ref_y, ref_gate), (dy.cpu(), dg.cpu()))
    for actual, expected in (
        (y, ref_y), (x.grad, ref_x.grad), (modulation.grad, ref_modulation.grad)
    ):
        torch.testing.assert_close(
            actual.cpu(), expected,
            atol=2e-5 if dtype == torch.float32 else 0.02,
            rtol=2e-5 if dtype == torch.float32 else 0.016,
        )
    _assert_bytes_equal(gate, ref_gate)
