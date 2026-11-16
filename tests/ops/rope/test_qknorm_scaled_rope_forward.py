# SPDX-License-Identifier: Apache-2.0
"""F2 fused forward/backward. RL_KERNEL_F2_TEST_JIT=1 enables an isolated CUDA build."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rl_engine.ops.rope.qknorm_scaled_rope import qknorm_scaled_rope, qknorm_scaled_rope_forward
from rl_engine.ops.rope.rope_table import build_shared_rope_table as build_rope_table
from rl_engine.reference.rope.qknorm_scaled_rope import (
    fixed_sum_128,
    fp64_math,
    qknorm_scaled_rope_backward_reference,
    qknorm_scaled_rope_fixed_tree_reference,
    qknorm_scaled_rope_fp64_reference,
)


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("F2 GPU forward requires CUDA/ROCm")
    if torch.version.hip is None and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("F2 BF16 requires SM80+")


@pytest.fixture(scope="module")
def cuda_extension():
    _require_gpu()
    if torch.version.hip is not None:
        pytest.skip("CUDA implementation is NVIDIA-only")
    from rl_engine.backends import extension

    if os.environ.get("RL_KERNEL_F2_TEST_JIT") == "1":
        from torch.utils.cpp_extension import load_inline

        source = Path(__file__).resolve().parents[3] / "csrc/cuda/rope/qknorm_scaled_rope.cu"
        declaration = """
        std::vector<torch::Tensor> qknorm_scaled_rope_forward_cuda(
            torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>,
            c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, bool);
        std::vector<torch::Tensor> qknorm_scaled_rope_backward_cuda(
            torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
            torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>,
            c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, bool);
        """
        module = load_inline(
            name="rlk_f2_forward_test",
            cpp_sources=declaration,
            cuda_sources=source.read_text(),
            functions=["qknorm_scaled_rope_forward_cuda", "qknorm_scaled_rope_backward_cuda"],
            # Use the installed PyTorch's default standard (2.14 requires C++20).
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ],
        )
        replacement = SimpleNamespace(
            qknorm_scaled_rope_forward=module.qknorm_scaled_rope_forward_cuda,
            qknorm_scaled_rope_backward=module.qknorm_scaled_rope_backward_cuda,
        )
        with patch.object(extension, "_C", replacement):
            yield replacement
    else:
        if extension._C is None or not hasattr(extension._C, "qknorm_scaled_rope_forward"):
            pytest.skip("build rl_engine._C or set RL_KERNEL_F2_TEST_JIT=1")
        yield extension._C


@pytest.fixture(params=["cuda", "triton"])
def backend(request):
    _require_gpu()
    if request.param == "cuda":
        request.getfixturevalue("cuda_extension")
    else:
        pytest.importorskip("triton")
    return request.param


def _inputs(tokens, hq=32, hk=2):
    gen = torch.Generator(device="cuda").manual_seed(501)
    q = torch.randn(tokens, hq, 128, device="cuda", generator=gen).bfloat16()
    k = torch.randn(tokens, hk, 128, device="cuda", generator=gen).bfloat16()
    if tokens:
        q[0, 0, :4] = torch.tensor([-0.0, 0.0, 1e-30, -1e-30], device="cuda").bfloat16()
    pos = (torch.arange(tokens, device="cuda", dtype=torch.int64) * 17 + 2047) % 131072
    c, s = build_rope_table(131072, device="cuda")
    return q, k, pos, c, s


def _assert_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(a.contiguous().view(dtype), b.contiguous().view(dtype))


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("tokens,hq,hk", [(0, 32, 2), (1, 32, 2), (33, 32, 2), (7, 4, 1)])
def test_forward_matches_fixed_tree_and_saved_rstd(backend, local, tokens, hq, hk):
    q, k, p, c, s = _inputs(tokens, hq, hk)
    before = q.clone(), k.clone()
    actual = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend=backend)
    expected = qknorm_scaled_rope_fixed_tree_reference(q, k, p, c, s, is_local=local)
    for out, ref in zip(actual[:2], expected):
        _assert_bits(out, ref)
    for x, r in zip((q, k), actual[2:]):
        v = fixed_sum_128(x.float() * x.float()) * (1 / 128) + 1e-5
        _assert_bits(r, v.pow(-0.5).squeeze(-1))
    _assert_bits(q, before[0])
    _assert_bits(k, before[1])


@pytest.mark.parametrize("local", [False, True])
def test_forward_partition_reorder_and_fp64(backend, local):
    q, k, p, c, s = _inputs(7)
    p[:] = torch.tensor([0, 1, 2047, 2048, 2049, 65535, 131071], device="cuda")
    run = lambda q, k, p: qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend=backend)
    full = run(q, k, p)
    parts = [run(q[a:b], k[a:b], p[a:b]) for a, b in [(0, 1), (1, 4), (4, 7)]]
    order = torch.tensor([6, 1, 4, 0, 2, 5, 3], device="cuda")
    perm = run(q[order], k[order], p[order])
    for i in range(4):
        _assert_bits(full[i], torch.cat([part[i] for part in parts]))
        _assert_bits(full[i][order], perm[i])
    fp64 = qknorm_scaled_rope_fp64_reference(q, k, p, is_local=local)
    for y, ref in zip(full[:2], fp64):
        # Same provisional reference smoke bound, not final operator acceptance.
        assert torch.linalg.vector_norm(y.double() - ref) / torch.linalg.vector_norm(ref) < 0.02


def test_cuda_triton_direct_parity(cuda_extension):
    pytest.importorskip("triton")
    q, k, p, c, s = _inputs(129)
    for local in (False, True):
        cuda = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend="cuda")
        triton = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend="triton")
        for a, b in zip(cuda, triton):
            _assert_bits(a, b)


def test_rejects_out_of_range_and_training(backend):
    q, k, p, c, s = _inputs(1)
    p[0] = 131072
    with pytest.raises(ValueError, match="outside"):
        qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)
    q.requires_grad_()
    with pytest.raises(RuntimeError, match="backward"):
        qknorm_scaled_rope_forward(q, k, is_local=False, backend=backend)
    with torch.no_grad():
        assert (
            qknorm_scaled_rope_forward(q, k, is_local=False, backend=backend).query.shape == q.shape
        )


def test_cuda_binding_checks_bounds_without_python_wrapper(cuda_extension):
    q, k, p, c, s = _inputs(1)
    p[0] = -1
    with pytest.raises(RuntimeError, match="outside"):
        cuda_extension.qknorm_scaled_rope_forward(q, k, p, c, s, True)


def test_forward_fails_closed_on_cpu_and_unknown_backend():
    q = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="backend"):
        qknorm_scaled_rope_forward(q, q, is_local=False, backend="auto")
    with pytest.raises(ValueError, match="GPU"):
        qknorm_scaled_rope_forward(q, q, is_local=False, backend="triton")
    q.requires_grad_()
    with pytest.raises(RuntimeError, match="backward"):
        qknorm_scaled_rope_forward(q, q, is_local=False, backend="cuda")


def _upstream(q, k):
    gen = torch.Generator(device=q.device).manual_seed(558)
    # Noncontiguous upstream gradients exercise autograd normalization of layout.
    return tuple(
        torch.randn(*x.shape[:-1], 256, generator=gen, device=x.device).bfloat16()[..., ::2]
        for x in (q, k)
    )


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("tokens,hq,hk", [(0, 32, 2), (1, 32, 2), (7, 4, 1)])
def test_backward_declared_vjp_and_fp64(backend, local, tokens, hq, hk):
    q, k, p, c, s = _inputs(tokens, hq, hk)
    if tokens:
        q[0, 0] = 0
        k[0, 0] = 1e-4
    saved = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend=backend)
    q.requires_grad_()
    k.requires_grad_()
    c.requires_grad_()
    s.requires_grad_()
    actual = qknorm_scaled_rope(q, k, p, c, s, is_local=local, backend=backend)
    upstream = _upstream(q, k)
    grads_all = torch.autograd.grad(actual, (q, k, c, s), grad_outputs=upstream, allow_unused=True)
    assert grads_all[2:] == (None, None)
    expected = qknorm_scaled_rope_backward_reference(
        q.detach(), k.detach(), *upstream, saved.query_rstd, saved.key_rstd, p, c, s, is_local=local
    )
    for g, ref in zip(grads_all[:2], expected):
        _assert_bits(g, ref)
    if tokens:
        q64 = q.detach().double().requires_grad_()
        k64 = k.detach().double().requires_grad_()
        math = fp64_math(q64, k64, p, is_local=local)
        grads64 = torch.autograd.grad(
            math, (q64, k64), grad_outputs=tuple(g.double() for g in upstream)
        )
        # Check each head, so high near-zero-input gradients cannot hide other heads.
        for g, ref in zip(grads_all[:2], grads64):
            err = torch.linalg.vector_norm(g.double() - ref, dim=-1)
            denom = torch.linalg.vector_norm(ref, dim=-1).clamp_min(1e-30)
            assert torch.all(err / denom < 0.03)


@pytest.mark.parametrize("local", [False, True])
def test_backward_partition_permutation_and_repeat(backend, local):
    q, k, p, c, s = _inputs(9)
    gq, gk = _upstream(q, k)

    def run(q, k, p, gq, gk):
        q = q.detach().clone().requires_grad_()
        k = k.detach().clone().requires_grad_()
        out = qknorm_scaled_rope(q, k, p, c, s, is_local=local, backend=backend)
        return torch.autograd.grad(out, (q, k), grad_outputs=(gq, gk))

    full = run(q, k, p, gq, gk)
    repeated = run(q, k, p, gq, gk)
    parts = [run(q[a:b], k[a:b], p[a:b], gq[a:b], gk[a:b]) for a, b in ((0, 1), (1, 5), (5, 9))]
    order = torch.tensor([8, 0, 7, 1, 6, 2, 5, 3, 4], device=q.device)
    perm = run(q[order], k[order], p[order], gq[order], gk[order])
    for i in range(2):
        _assert_bits(full[i], repeated[i])
        _assert_bits(full[i], torch.cat([part[i] for part in parts]))
        _assert_bits(full[i][order], perm[i])


@pytest.mark.parametrize("local", [False, True])
def test_backward_cuda_triton_parity(cuda_extension, local):
    pytest.importorskip("triton")
    q, k, p, c, s = _inputs(129)
    q.requires_grad_()
    k.requires_grad_()
    gradients = []
    for name in ("cuda", "triton"):
        outputs = qknorm_scaled_rope(q, k, p, c, s, is_local=local, backend=name)
        gradients.append(torch.autograd.grad(outputs, (q, k), grad_outputs=_upstream(q, k)))
    for a, b in zip(*gradients):
        _assert_bits(a, b)


@pytest.mark.parametrize("branch", [0, 1])
def test_unused_output_and_no_higher_derivatives(backend, branch):
    q, k, p, c, s = _inputs(1)
    q.requires_grad_()
    k.requires_grad_()
    outputs = qknorm_scaled_rope(q, k, p, c, s, is_local=True, backend=backend)
    grads = torch.autograd.grad(
        outputs[branch].float().sum(), (q, k), allow_unused=True, create_graph=True
    )
    assert grads[1 - branch] is None
    assert grads[branch] is not None
    with pytest.raises(RuntimeError):
        torch.autograd.grad(grads[branch].float().sum(), (q, k)[branch])


def test_saved_input_mutation_rejected(backend):
    q, k, p, c, s = _inputs(1)
    q.requires_grad_()
    result = qknorm_scaled_rope(q, k, p, c, s, is_local=True, backend=backend)
    with torch.no_grad():
        q.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result[0].float().sum().backward()


def test_cuda_backward_rejects_bad_saved_state(cuda_extension):
    q, k, p, c, s = _inputs(1)
    saved = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend="cuda")
    gq, gk = (g.contiguous() for g in _upstream(q, k))
    with pytest.raises(RuntimeError, match="rstd"):
        cuda_extension.qknorm_scaled_rope_backward(
            q, k, gq, gk, saved.query_rstd.bfloat16(), saved.key_rstd, p, c, s, True
        )


@pytest.mark.parametrize("local", [False, True])
def test_dynamic_range_and_structured_rows(backend, local):
    # Stay within finite FP32 square/sum arithmetic; overflow policy is separate.
    q, k, p, c, s = _inputs(257, 4, 1)
    scales = torch.exp2(torch.linspace(-60, 60, 257, device=q.device))
    for x in (q, k):
        x.copy_((x.float() * scales[:, None, None]).bfloat16())
        x[0] = 0
        x[1] = 1e-30
        x[2] = 0
        x[2, :, 0] = 1e10
        x[3, :, ::2] = 1
        x[3, :, 1::2] = -1
    p[:] = 131071  # Repeated positions must not couple independent rows.
    actual = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=local, backend=backend)
    expected = qknorm_scaled_rope_fixed_tree_reference(q, k, p, c, s, is_local=local)
    for out, ref in zip(actual[:2], expected):
        assert torch.isfinite(out).all()
        _assert_bits(out, ref)
    for x, r in zip((q, k), actual[2:]):
        variance = fixed_sum_128(x.float() * x.float()) * (1 / 128) + 1e-5
        _assert_bits(r, variance.pow(-0.5).squeeze(-1))

    upstream = tuple(g.contiguous() for g in _upstream(q, k))
    for g in upstream:
        # Equal gradients against alternating +/- input exercise cancellation.
        g[3] = 1
    expected_grad = qknorm_scaled_rope_backward_reference(
        q, k, *upstream, actual.query_rstd, actual.key_rstd, p, c, s, is_local=local
    )
    q.requires_grad_()
    k.requires_grad_()
    out = qknorm_scaled_rope(q, k, p, c, s, is_local=local, backend=backend)
    gradients = torch.autograd.grad(out, (q, k), grad_outputs=upstream)
    for grad, ref in zip(gradients, expected_grad):
        assert torch.isfinite(grad).all()
        _assert_bits(grad, ref)


@pytest.mark.parametrize("invalid", [-1, 131072])
def test_positions_revalidated_after_success(backend, invalid):
    q, k, p, c, s = _inputs(1)
    qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)
    p[0] = invalid
    with pytest.raises(ValueError, match="outside"):
        qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)


def test_nonempty_input_rejects_empty_table(backend):
    q, k, p, c, s = _inputs(1)
    c, s = build_rope_table(0, device="cuda")
    with pytest.raises(ValueError, match="outside"):
        qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)


def test_cuda_wrapper_preserves_unrelated_errors(cuda_extension):
    from rl_engine.backends import extension

    def fail(*args):
        raise RuntimeError("synthetic unrelated CUDA failure")

    q, k, p, c, s = _inputs(1)
    with patch.object(extension, "_C", SimpleNamespace(qknorm_scaled_rope_forward=fail)):
        with pytest.raises(RuntimeError, match="synthetic unrelated"):
            qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend="cuda")


@pytest.mark.parametrize("bad_index", [255, 256, 1024])
@pytest.mark.parametrize("bad_position", [-(2**63), 2**63 - 1])
def test_fused_bounds_scan_rejects_chunk_edges(backend, bad_index, bad_position):
    q, k, p, c, s = _inputs(1025, 1, 1)
    # A tail outside the first block must be checked without narrowing int64.
    p[bad_index] = bad_position
    with pytest.raises(ValueError, match="outside"):
        qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)


def test_triton_bounds_scan_large_and_empty():
    _require_gpu()
    pytest.importorskip("triton")
    from rl_engine.backends.shared.triton.rope.qknorm_scaled_rope import _positions_outside_table

    p = torch.full((131073,), 131071, dtype=torch.int64, device="cuda")
    assert not _positions_outside_table(p, 131072)
    p[-1] = 131072
    assert _positions_outside_table(p, 131072)
    p[-1] = -1
    assert _positions_outside_table(p, 131072)
    assert not _positions_outside_table(p[:0], 0)


def test_native_backward_rechecks_positions(cuda_extension):
    q, k, p, c, s = _inputs(257, 1, 1)
    saved = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend="cuda")
    p[-1] = 2**63 - 1
    with pytest.raises(RuntimeError, match="outside"):
        cuda_extension.qknorm_scaled_rope_backward(
            q, k, torch.ones_like(q), torch.ones_like(k), *saved[2:], p, c, s, True
        )


@pytest.mark.parametrize("exponent", [50, 55, 60])
def test_backward_large_constant_input_fp64(backend, exponent):
    q, k, p, c, s = _inputs(1, 1, 1)
    q.fill_(2.0**exponent)
    k.fill_(2.0**exponent)
    q.requires_grad_()
    k.requires_grad_()
    g = torch.zeros_like(k)
    g[..., :64] = 1
    out = qknorm_scaled_rope(q, k, is_local=False, backend=backend)
    actual = torch.autograd.grad(out[1], k, g)[0]
    k64 = k.detach().double().requires_grad_()
    ref = torch.autograd.grad(
        fp64_math(q.detach().double(), k64, None, is_local=False)[1], k64, g.double()
    )[0]
    expected = ref.bfloat16()
    _assert_bits(actual, expected)
    assert torch.linalg.vector_norm(actual.double() - ref) / torch.linalg.vector_norm(ref) < 1e-6


@pytest.mark.parametrize("local", [False, True])
def test_backward_dynamic_range_independent_fp64(backend, local):
    q, k, p, c, s = _inputs(25, 4, 1)
    scales = torch.exp2(torch.linspace(-60, 60, 25, device=q.device))[:, None, None]
    q = (q.float() * scales).bfloat16().requires_grad_()
    k = (k.float() * scales).bfloat16().requires_grad_()
    upstream = _upstream(q, k)
    out = qknorm_scaled_rope(q, k, p, c, s, is_local=local, backend=backend)
    actual = torch.autograd.grad(out, (q, k), grad_outputs=upstream)
    q64, k64 = q.detach().double().requires_grad_(), k.detach().double().requires_grad_()
    refs = torch.autograd.grad(
        fp64_math(q64, k64, p, is_local=local),
        (q64, k64),
        grad_outputs=tuple(g.double() for g in upstream),
    )
    for grad, ref in zip(actual, refs):
        error = torch.linalg.vector_norm(grad.double() - ref, dim=-1)
        denom = torch.linalg.vector_norm(ref, dim=-1).clamp_min(1e-30)
        assert torch.all(error / denom < 0.03)


def test_shared_table_lifecycle_and_dispatch(backend):
    from rl_engine.ops.rope.rope_table import load_shared_rope_table, table_identity
    from rl_engine.reference.rope.qknorm_scaled_rope import build_rope_table as reference_table

    q, k, p, c, s = _inputs(2, 4, 1)
    identity = table_identity(c, s)
    # Preserve the existing arithmetic/table bits; this change governs ownership.
    rc, rs = reference_table(131072, device="cuda")
    _assert_bits(c, rc)
    _assert_bits(s, rs)
    loaded_c, loaded_s = load_shared_rope_table(c.cpu(), s.cpu(), expected=identity, device="cuda")
    assert table_identity(loaded_c, loaded_s) == identity
    original = qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)
    loaded = qknorm_scaled_rope_forward(q, k, p, loaded_c, loaded_s, is_local=True, backend=backend)
    for a, b in zip(original, loaded):
        _assert_bits(a, b)
    # Cached identity validation must never transfer/hash contents at dispatch.
    with patch("rl_engine.ops.rope.rope_table._digest", side_effect=AssertionError("rehash")):
        qknorm_scaled_rope_forward(q, k, p, c, s, is_local=True, backend=backend)
    with pytest.raises(ValueError, match="unverified"):
        qknorm_scaled_rope_forward(q, k, p, rc, rs, is_local=True, backend=backend)
    with pytest.raises(ValueError, match="unverified"):
        qknorm_scaled_rope_forward(q, k, p, c, loaded_s, is_local=True, backend=backend)
    loaded_c[0, 0] = 0
    with pytest.raises(ValueError, match="modified"):
        qknorm_scaled_rope_forward(q, k, p, loaded_c, loaded_s, is_local=True, backend=backend)


def test_shared_table_build_inference_mode():
    _require_gpu()
    from rl_engine.ops.rope.rope_table import table_identity

    with torch.inference_mode():
        c, s = build_rope_table(8, device="cuda")
        assert table_identity(c, s).length == 8
    assert not c.is_inference()
