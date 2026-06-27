# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import queue
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from rl_engine.executors.deepspeed_trainer import _EmbeddingLMHeadModel, _safe_token_ids
from rl_engine.kernels.ops.pytorch.loss.linear_logp import (
    NativeLinearLogpOp,
    chunked_linear_logp_backward,
)
from rl_engine.testing import selected_logprobs_reference

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton linear log-prob requires a CUDA device and Triton.",
)


def _sm90_available():
    """SM90 forward needs a Hopper GPU and the kernel compiled into the extension."""
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

        if not (_EXT_AVAILABLE and hasattr(_C, "fused_linear_logp_sm90")):
            return False
    except Exception:  # pragma: no cover
        return False
    return torch.cuda.get_device_capability()[0] == 9


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="Fused linear log-prob SM90 kernel requires a Hopper (sm_90) GPU with the "
    "extension built KERNEL_ALIGN_FORCE_SM90=1.",
)


def _gloo_available():
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


requires_gloo = pytest.mark.skipif(
    not _gloo_available(),
    reason="tensor-parallel linear_logp CPU test requires torch.distributed Gloo.",
)


def _tp_linear_logp_gloo_worker(rank, world_size, init_method, result_queue):
    try:
        import torch.distributed as dist

        torch.set_num_threads(1)
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )

        torch.manual_seed(2026)
        n, d, vocab = 8, 5, 16
        boundaries = [0, 3, 7, 12, vocab]
        start = boundaries[rank]
        end = boundaries[rank + 1]

        hidden_base = torch.randn(n, d)
        weight_full = torch.randn(vocab, d)
        bias_full = torch.randn(vocab)
        target = torch.tensor([0, 2, 3, 6, 7, 11, 12, 15], dtype=torch.long)
        grad_out = torch.randn(n)
        op = NativeLinearLogpOp()

        ref_hidden = hidden_base.detach().clone().requires_grad_(True)
        ref_weight = weight_full.detach().clone().requires_grad_(True)
        ref_bias = bias_full.detach().clone().requires_grad_(True)
        ref_out = op(ref_hidden, ref_weight, target, ref_bias)
        ref_out.backward(grad_out)

        tp_hidden = hidden_base.detach().clone().requires_grad_(True)
        local_weight = weight_full[start:end].detach().clone().requires_grad_(True)
        local_bias = bias_full[start:end].detach().clone().requires_grad_(True)
        tp_out = op(
            tp_hidden,
            local_weight,
            target,
            local_bias,
            tp_group=dist.group.WORLD,
            vocab_start_index=start,
            global_vocab_size=vocab,
        )
        tp_out.backward(grad_out)

        result_queue.put(
            {
                "ok": True,
                "rank": rank,
                "out": float((tp_out - ref_out).abs().max().item()),
                "hidden_grad": float((tp_hidden.grad - ref_hidden.grad).abs().max().item()),
                "weight_grad": float(
                    (local_weight.grad - ref_weight.grad[start:end]).abs().max().item()
                ),
                "bias_grad": float((local_bias.grad - ref_bias.grad[start:end]).abs().max().item()),
            }
        )
    except Exception:  # pragma: no cover - forwarded to parent process
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


# SM90 forward needs bf16 and a hidden dim that is a multiple of the kernel's K
# slice (32); N / V are deliberately left unaligned to the 64-wide tiles.
_SM90_N = 96
_SM90_D = 128
_SM90_V = 500


def _sm90_inputs(seed, *, bias=True, dtype=torch.bfloat16, lead=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lead = lead or (_SM90_N,)
    hidden = torch.randn(*lead, _SM90_D, generator=gen, device="cuda", dtype=dtype)
    weight = torch.randn(_SM90_V, _SM90_D, generator=gen, device="cuda", dtype=dtype)
    bias_t = torch.randn(_SM90_V, generator=gen, device="cuda", dtype=dtype) if bias else None
    target = torch.randint(0, _SM90_V, lead, generator=gen, device="cuda")
    return hidden, weight, target, bias_t


# Deliberately non-multiples of the kernel block sizes (32 / 64 / 64).
_N = 40
_D = 80
_V = 300


def _inputs(seed, *, device, dtype=torch.float32, bias=True, lead=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    lead = lead or (_N,)
    hidden = torch.randn(*lead, _D, generator=gen, device=device, dtype=dtype)
    weight = torch.randn(_V, _D, generator=gen, device=device, dtype=dtype)
    bias_t = torch.randn(_V, generator=gen, device=device, dtype=dtype) if bias else None
    target = torch.randint(0, _V, lead, generator=gen, device=device)
    return hidden, weight, target, bias_t


def _manual_reference(hidden, weight, target, bias):
    """The semantic definition: materialize logits, log_softmax, gather."""
    logits = torch.nn.functional.linear(
        hidden.float(), weight.float(), None if bias is None else bias.float()
    )
    logp = torch.log_softmax(logits, dim=-1)
    idx = target.reshape(-1).long()
    sel = logp.reshape(-1, logp.size(-1)).gather(-1, idx.unsqueeze(1)).squeeze(1)
    return sel.reshape(target.shape)


def _layout_inputs(base_hidden, base_target, base_mask, order, lead_shape):
    order_t = torch.tensor(order, dtype=torch.long)
    hidden = base_hidden.index_select(0, order_t).reshape(*lead_shape, base_hidden.size(-1))
    target = base_target.index_select(0, order_t).reshape(*lead_shape)
    mask = base_mask.index_select(0, order_t).reshape(*lead_shape)
    masked_target = target.masked_fill(~mask, -100)
    return hidden, masked_target, mask


def _recover_canonical_rows(layout_values, order):
    flat = layout_values.reshape(
        layout_values.shape[0] * layout_values.shape[1], *layout_values.shape[2:]
    )
    recovered = torch.empty_like(flat)
    recovered[torch.tensor(order, dtype=torch.long)] = flat
    return recovered


def _run_chunked_backward(hidden, weight, target, bias, grad_out, *, chunk_elems):
    return chunked_linear_logp_backward(
        grad_out,
        hidden.reshape(-1, hidden.size(-1)).contiguous(),
        weight,
        target.reshape(-1).contiguous(),
        hidden.reshape(-1, hidden.size(-1)).contiguous() if bias is None else bias,
        has_bias=bias is not None,
        lead_shape=target.shape,
        hidden_dtype=hidden.dtype,
        weight_dtype=weight.dtype,
        bias_dtype=None if bias is None else bias.dtype,
        chunk_elems=chunk_elems,
    )


def _run_autograd_linear_logp(hidden, weight, target, bias, grad_out):
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True) if bias is not None else None
    NativeLinearLogpOp()(h, w, target, b).backward(grad_out)
    return h.grad, w.grad, (None if b is None else b.grad)


def test_native_matches_manual_reference():
    native = NativeLinearLogpOp()
    hidden, weight, target, bias = _inputs(0, device="cpu")
    out = native(hidden, weight, target, bias)
    ref = _manual_reference(hidden, weight, target, bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=1e-5)


def test_linear_logp_handoff_matches_masked_reference_across_layouts():
    torch.manual_seed(2026)
    op = NativeLinearLogpOp()
    base_hidden = torch.randn(6, 5)
    weight = torch.randn(17, 5)
    bias = torch.randn(17)
    base_target = torch.tensor([3, 7, 1, 9, 4, 6], dtype=torch.long)
    base_mask = torch.tensor([True, False, True, True, False, True], dtype=torch.bool)
    layouts = [
        ((2, 3), [0, 1, 2, 3, 4, 5]),
        ((3, 2), [5, 1, 3, 0, 4, 2]),
        ((1, 6), [2, 4, 1, 5, 0, 3]),
    ]

    canonical = None
    for lead_shape, order in layouts:
        hidden, target, mask = _layout_inputs(
            base_hidden, base_target, base_mask, order, lead_shape
        )
        actual = op(hidden, weight, _safe_token_ids(target, mask), bias).masked_fill(~mask, 0.0)
        logits = torch.nn.functional.linear(hidden.float(), weight.float(), bias.float())
        expected = selected_logprobs_reference(logits, target, mask=mask)
        recovered = _recover_canonical_rows(actual.unsqueeze(-1), order).squeeze(-1)

        assert torch.allclose(actual, expected, atol=1e-5)
        if canonical is None:
            canonical = recovered
        else:
            assert torch.allclose(recovered, canonical, atol=1e-6)


@pytest.mark.parametrize("use_bias", [True, False])
def test_chunked_linear_logp_backward_matches_autograd_and_layout_invariance(use_bias):
    torch.manual_seed(2027)
    weight = torch.randn(19, 7)
    bias = torch.randn(19) if use_bias else None
    base_hidden = torch.randn(6, 7)
    base_target = torch.tensor([1, 7, 3, 5, 0, 9], dtype=torch.long)
    base_mask = torch.tensor([True, False, True, True, False, True], dtype=torch.bool)
    base_grad = torch.tensor([0.5, 0.0, -1.25, 0.75, 0.0, 1.5], dtype=torch.float32)
    layouts = [
        ((2, 3), [0, 1, 2, 3, 4, 5]),
        ((3, 2), [5, 2, 1, 0, 4, 3]),
    ]

    canonical_hidden_grad = None
    canonical_weight_grad = None
    canonical_bias_grad = None
    chunk_elems = weight.size(0) * 2

    for lead_shape, order in layouts:
        hidden, target, mask = _layout_inputs(
            base_hidden, base_target, base_mask, order, lead_shape
        )
        safe_target = _safe_token_ids(target, mask)
        grad_out = base_grad[torch.tensor(order, dtype=torch.long)].reshape(lead_shape)
        grad_out = grad_out.masked_fill(~mask, 0.0)

        grad_hidden, grad_weight, grad_bias = _run_chunked_backward(
            hidden,
            weight,
            safe_target,
            bias,
            grad_out,
            chunk_elems=chunk_elems,
        )
        ref_hidden, ref_weight, ref_bias = _run_autograd_linear_logp(
            hidden,
            weight,
            safe_target,
            bias,
            grad_out,
        )
        recovered_hidden = _recover_canonical_rows(grad_hidden, order)

        assert torch.allclose(grad_hidden, ref_hidden, atol=1e-5)
        assert torch.allclose(grad_weight, ref_weight, atol=1e-5)
        if use_bias:
            assert torch.allclose(grad_bias, ref_bias, atol=1e-5)

        if canonical_hidden_grad is None:
            canonical_hidden_grad = recovered_hidden
            canonical_weight_grad = grad_weight
            canonical_bias_grad = grad_bias
        else:
            assert torch.allclose(recovered_hidden, canonical_hidden_grad, atol=1e-6)
            assert torch.allclose(grad_weight, canonical_weight_grad, atol=1e-6)
            if use_bias:
                assert torch.allclose(grad_bias, canonical_bias_grad, atol=1e-6)


def test_tied_embedding_lm_head_shared_gradient_is_layout_invariant():
    torch.manual_seed(2028)
    model = _EmbeddingLMHeadModel(vocab_size=13, hidden_dim=6, bias=False, tie_weights=True)
    op = NativeLinearLogpOp()
    base_input_ids = torch.tensor([2, 5, 1, 5, 2, 3], dtype=torch.long)
    base_target = torch.tensor([4, 1, 0, 2, 6, 3], dtype=torch.long)
    base_mask = torch.tensor([True, False, True, True, False, True], dtype=torch.bool)
    base_upstream = torch.tensor([0.75, 0.0, -1.25, 0.5, 0.0, 1.0], dtype=torch.float32)
    layouts = [
        ((2, 3), [0, 1, 2, 3, 4, 5]),
        ((3, 2), [5, 2, 1, 0, 4, 3]),
    ]

    assert model.lm_head.weight is model.embedding.weight
    canonical_logps = None
    canonical_grad = None

    for lead_shape, order in layouts:
        order_t = torch.tensor(order, dtype=torch.long)
        input_ids = base_input_ids.index_select(0, order_t).reshape(lead_shape)
        target = base_target.index_select(0, order_t).reshape(lead_shape)
        mask = base_mask.index_select(0, order_t).reshape(lead_shape)
        masked_target = target.masked_fill(~mask, -100)
        upstream = (
            base_upstream.index_select(0, order_t).reshape(lead_shape).masked_fill(~mask, 0.0)
        )

        model.zero_grad(set_to_none=True)
        hidden = model(input_ids)
        logps = op(
            hidden, model.lm_head.weight, _safe_token_ids(masked_target, mask), model.lm_head.bias
        )
        logps = logps.masked_fill(~mask, 0.0)
        logits = torch.nn.functional.linear(hidden.float(), model.lm_head.weight.float(), None)
        expected = selected_logprobs_reference(logits, masked_target, mask=mask)
        (logps * upstream).sum().backward()

        recovered_logps = _recover_canonical_rows(logps.unsqueeze(-1), order).squeeze(-1)
        shared_grad = model.embedding.weight.grad.detach().clone()

        assert torch.allclose(logps, expected, atol=1e-5)
        if canonical_logps is None:
            canonical_logps = recovered_logps
            canonical_grad = shared_grad
        else:
            assert torch.allclose(recovered_logps, canonical_logps, atol=1e-6)
            assert torch.allclose(shared_grad, canonical_grad, atol=1e-6)


def test_native_rejects_shape_mismatch():
    native = NativeLinearLogpOp()
    hidden, weight, _, bias = _inputs(0, device="cpu")
    with pytest.raises(ValueError):
        native(hidden, weight, torch.zeros(_N + 1, dtype=torch.long), bias)


def test_tensor_parallel_metadata_requires_multi_rank_group():
    native = NativeLinearLogpOp()
    hidden, weight, target, bias = _inputs(0, device="cpu")
    with pytest.raises(ValueError, match="vocab_start_index requires"):
        native(hidden, weight, target, bias, vocab_start_index=4)
    with pytest.raises(ValueError, match="global_vocab_size differs"):
        native(hidden, weight, target, bias, global_vocab_size=weight.size(0) + 1)


@requires_gloo
def test_native_tensor_parallel_matches_full_reference_cpu_gloo_4_ranks():
    ctx = mp.get_context("spawn")
    world_size = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "gloo_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_tp_linear_logp_gloo_worker,
                args=(rank, world_size, init_method, result_queue),
            )
            for rank in range(world_size)
        ]

        for process in processes:
            process.start()

        results = []
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=45))
        except queue.Empty:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail("timed out waiting for tensor-parallel Gloo workers")
        finally:
            for process in processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.terminate()

    sorted_results = sorted(results, key=lambda item: item["rank"])
    for result in sorted_results:
        assert result["ok"], result.get("traceback")
    for process in processes:
        assert process.exitcode == 0
    for result in sorted_results:
        assert result["out"] < 1e-5
        assert result["hidden_grad"] < 1e-5
        assert result["weight_grad"] < 1e-5
        assert result["bias_grad"] < 1e-5


@requires_triton_cuda
def test_triton_forward_matches_native_fp32():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(1, device="cuda")
    ref = native(hidden, weight, target, bias)
    out = trit(hidden, weight, target, bias)
    assert torch.allclose(out, ref, atol=1e-3)


@requires_triton_cuda
def test_triton_forward_matches_native_bf16():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(2, device="cuda", dtype=torch.bfloat16)
    # The kernel accumulates in fp32, so the oracle uses the fp32-upcast inputs.
    ref = native(hidden.float(), weight.float(), target, bias.float())
    out = trit(hidden, weight, target, bias)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_triton_cuda
@pytest.mark.parametrize("use_bias", [True, False])
def test_triton_backward_matches_native(use_bias):
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(3, device="cuda", bias=use_bias)
    grad_out = torch.randn(_N, device="cuda")

    def run(op, h, w, b):
        h = h.detach().clone().requires_grad_(True)
        w = w.detach().clone().requires_grad_(True)
        b = b.detach().clone().requires_grad_(True) if b is not None else None
        op(h, w, target, b).backward(grad_out)
        return h.grad, w.grad, (b.grad if b is not None else None)

    th, tw, tb = run(trit, hidden, weight, bias)
    nh, nw, nb = run(native, hidden, weight, bias)
    assert torch.allclose(th, nh, atol=2e-3)
    assert torch.allclose(tw, nw, atol=2e-3)
    if use_bias:
        assert torch.allclose(tb, nb, atol=2e-3)


@requires_triton_cuda
def test_triton_gradients_flow_to_inputs_only():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    trit = TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(4, device="cuda")
    hidden = hidden.requires_grad_(True)
    weight = weight.requires_grad_(True)
    bias = bias.requires_grad_(True)
    trit(hidden, weight, target, bias).sum().backward()
    assert hidden.grad is not None and weight.grad is not None and bias.grad is not None
    assert target.grad is None  # integer targets are non-differentiable


@requires_triton_cuda
def test_triton_preserves_leading_shape():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    native, trit = NativeLinearLogpOp(), TritonLinearLogpOp()
    hidden, weight, target, bias = _inputs(5, device="cuda", lead=(4, 7))
    out = trit(hidden, weight, target, bias)
    assert out.shape == (4, 7)
    assert torch.allclose(out, native(hidden, weight, target, bias), atol=1e-3)


@requires_triton_cuda
def test_triton_large_vocab_smoke():
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    trit = TritonLinearLogpOp()
    hidden = torch.randn(8, 64, device="cuda")
    weight = torch.randn(50257, 64, device="cuda")
    target = torch.randint(0, 50257, (8,), device="cuda")
    out = trit(hidden, weight, target)
    assert out.shape == (8,) and torch.isfinite(out).all()


@requires_sm90
def test_sm90_forward_matches_native_bf16():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(11)
    # The kernel matmul accumulates in fp32 (tensor cores), so the oracle uses the
    # fp32-upcast inputs -- like the Triton bf16 test.
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, bias.float())
    out = sm90(hidden, weight, target, bias)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
def test_sm90_forward_no_bias():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, _ = _sm90_inputs(12, bias=False)
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, None)
    out = sm90(hidden, weight, target)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
@pytest.mark.parametrize("use_bias", [True, False])
def test_sm90_forward_backward_matches_triton(use_bias):
    # The SM90 forward is fp32-accurate and the backward reuses the same
    # deterministic chunked path as the Triton op, so both match very tightly.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op
    from rl_engine.kernels.ops.triton.loss.linear_logp import TritonLinearLogpOp

    sm90, trit = FusedLinearLogpSM90Op(), TritonLinearLogpOp()
    hidden, weight, target, bias = _sm90_inputs(13, bias=use_bias)
    grad_out = torch.randn(_SM90_N, device="cuda")

    def run(op):
        h = hidden.detach().clone().requires_grad_(True)
        w = weight.detach().clone().requires_grad_(True)
        b = bias.detach().clone().requires_grad_(True) if bias is not None else None
        out = op(h, w, target, b)
        out.backward(grad_out)
        return out.detach(), h.grad, w.grad, (b.grad if b is not None else None)

    so, sh, sw, sb = run(sm90)
    to, th, tw, tb = run(trit)
    assert torch.allclose(so, to, atol=1e-3)
    assert torch.allclose(sh, th, atol=2e-3)
    assert torch.allclose(sw, tw, atol=2e-3)
    if use_bias:
        assert torch.allclose(sb, tb, atol=2e-3)


@requires_sm90
def test_sm90_preserves_leading_shape():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(14, lead=(6, 5))
    out = sm90(hidden, weight, target, bias)
    assert out.shape == (6, 5)
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, bias.float())
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
def test_sm90_large_vocab_smoke():
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden = torch.randn(40, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(50257, 256, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, 50257, (40,), device="cuda")
    out = sm90(hidden, weight, target)
    assert out.shape == (40,) and torch.isfinite(out).all()


@requires_sm90
def test_sm90_falls_back_for_unsupported_inputs():
    # fp32 inputs and a hidden dim not divisible by the kernel's K slice are not
    # handled by the compiled forward; the op must fall back instead of erroring.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()

    fp32 = _sm90_inputs(15, dtype=torch.float32)
    out = sm90(*fp32)
    ref = NativeLinearLogpOp()(*fp32)
    assert torch.allclose(out, ref, atol=1e-3)

    # bf16 but D=80 (not a multiple of 32) -> fallback path.
    gen = torch.Generator(device="cuda").manual_seed(16)
    hidden = torch.randn(40, 80, device="cuda", dtype=torch.bfloat16, generator=gen)
    weight = torch.randn(300, 80, device="cuda", dtype=torch.bfloat16, generator=gen)
    target = torch.randint(0, 300, (40,), device="cuda", generator=gen)
    out = sm90(hidden, weight, target)
    ref = NativeLinearLogpOp()(hidden.float(), weight.float(), target, None)
    assert torch.allclose(out, ref, atol=2e-2)


@requires_sm90
def test_sm90_rejects_bad_target_and_bias():
    # Shape/device mismatches must be a clean error, not a CUDA illegal access.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(17)

    with pytest.raises((ValueError, RuntimeError)):  # weight on the wrong device
        sm90(hidden, weight.cpu(), target, bias)
    with pytest.raises((ValueError, RuntimeError)):  # wrong target length
        sm90(hidden, weight, target[:-1], bias)
    with pytest.raises((ValueError, RuntimeError)):  # wrong bias length
        sm90(hidden, weight, target, bias[:-1])
    with pytest.raises((ValueError, RuntimeError)):  # bias on the wrong device
        sm90(hidden, weight, target, bias.cpu())


@requires_sm90
def test_sm90_rejects_out_of_range_target():
    # Padding (-100) / out-of-vocab ids must error, not silently corrupt fwd/bwd.
    from rl_engine.kernels.ops.cuda.loss.linear_logp import FusedLinearLogpSM90Op

    sm90 = FusedLinearLogpSM90Op()
    hidden, weight, target, bias = _sm90_inputs(18)

    pad = target.clone()
    pad[0] = -100  # typical ignore_index
    with pytest.raises(ValueError):
        sm90(hidden, weight, pad, bias)

    oob = target.clone()
    oob[1] = _SM90_V  # == V, one past the last valid id
    with pytest.raises(ValueError):
        sm90(hidden, weight, oob, bias)

    # A valid target (all in [0, V)) still works.
    out = sm90(hidden, weight, target, bias)
    assert out.shape == target.shape and torch.isfinite(out).all()


def test_sm90_tp_metadata_prefers_sm90_tp_helper(monkeypatch):
    from rl_engine.kernels.ops.cuda.loss import linear_logp as cuda_linear_logp

    op = object.__new__(cuda_linear_logp.FusedLinearLogpSM90Op)
    hidden = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    target = torch.tensor([3, 5])
    sentinel = torch.full((2,), 7.0)
    tp_group = object()
    calls = {}

    monkeypatch.setattr(
        cuda_linear_logp,
        "should_use_tensor_parallel_linear_logp",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(cuda_linear_logp, "_sm90_supported", lambda h, w: True)

    def fake_sm90_tp(hidden_arg, weight_arg, target_arg, bias_arg, **kwargs):
        calls["sm90_tp"] = (hidden_arg, weight_arg, target_arg, bias_arg, kwargs)
        return sentinel

    def forbidden_portable_tp(*args, **kwargs):
        raise AssertionError("portable TP path should not run when SM90 TP is available")

    monkeypatch.setattr(cuda_linear_logp, "_sm90_tensor_parallel_linear_logp", fake_sm90_tp)
    monkeypatch.setattr(cuda_linear_logp, "tensor_parallel_linear_logp", forbidden_portable_tp)

    out = op(
        hidden,
        weight,
        target,
        tp_group=tp_group,
        vocab_start_index=3,
        global_vocab_size=6,
    )

    assert out is sentinel
    assert calls["sm90_tp"][0] is hidden
    assert calls["sm90_tp"][1] is weight
    assert calls["sm90_tp"][2] is target
    assert calls["sm90_tp"][4] == {
        "tp_group": tp_group,
        "vocab_start_index": 3,
        "global_vocab_size": 6,
    }


def test_sm90_tp_metadata_falls_back_to_portable_tp_when_sm90_unsupported(monkeypatch):
    from rl_engine.kernels.ops.cuda.loss import linear_logp as cuda_linear_logp

    op = object.__new__(cuda_linear_logp.FusedLinearLogpSM90Op)
    hidden = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    target = torch.tensor([3, 5])
    sentinel = torch.full((2,), 11.0)

    monkeypatch.setattr(
        cuda_linear_logp,
        "should_use_tensor_parallel_linear_logp",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(cuda_linear_logp, "_sm90_supported", lambda h, w: False)
    monkeypatch.setattr(
        cuda_linear_logp,
        "_sm90_tensor_parallel_linear_logp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("SM90 TP helper should not run for unsupported inputs")
        ),
    )
    monkeypatch.setattr(
        cuda_linear_logp,
        "tensor_parallel_linear_logp",
        lambda *args, **kwargs: sentinel,
    )

    out = op(
        hidden,
        weight,
        target,
        tp_group=object(),
        vocab_start_index=3,
        global_vocab_size=6,
    )

    assert out is sentinel


def test_registry_dispatch_matches_native():
    from rl_engine.kernels.registry import kernel_registry
    from rl_engine.platforms.device import device_ctx

    op = kernel_registry.get_op("linear_logp")
    device = device_ctx.device if device_ctx.device_type == "cuda" else "cpu"
    hidden, weight, target, bias = _inputs(6, device=device)
    out = op(hidden, weight, target, bias)
    ref = NativeLinearLogpOp()(hidden, weight, target, bias)
    assert torch.allclose(out.cpu(), ref.cpu(), atol=1e-3)
