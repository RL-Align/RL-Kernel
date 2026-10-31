# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import contextlib

import pytest
import torch

from rl_engine.reference.embedding.lm_head import NativeLMHeadOp

_VOCAB = 128  # shrunk; real value: _GEMMA_VOCAB
_HIDDEN = 5376  # Gemma 4 31B hidden_size, kept real for representative drift.

# Real Gemma 4 31B dims, from rl_engine/models/gemma/gemma4_manifest.json.
_GEMMA_VOCAB = 262144
_GEMMA_HIDDEN = 5376

# Axis-B: max abs error as a fraction of the output peak magnitude. Calibrated
# at hidden=5376 (bf16 ~0.27% of peak, fp16 ~0.04%) with headroom.
_DTYPE_REL_PEAK = {torch.bfloat16: 1.0e-2, torch.float16: 2.0e-3}


def _cpu_fp16_matmul_supported() -> bool:
    """Probe whether this CPU backend implements float16 matmul."""
    try:
        _ = torch.ones(2, 2, dtype=torch.float16) @ torch.ones(2, 2, dtype=torch.float16)
        return True
    except RuntimeError:
        return False


_FP32 = pytest.param(torch.float32, id="fp32")
_BF16 = pytest.param(torch.bfloat16, id="bf16")
_FP16_IF_CPU_MATMUL_SUPPORTED = pytest.param(
    torch.float16,
    marks=pytest.mark.skipif(
        not _cpu_fp16_matmul_supported(),
        reason="CPU float16 matmul unsupported on this backend",
    ),
    id="fp16",
)
_DTYPES_AXIS_B = (_BF16, _FP16_IF_CPU_MATMUL_SUPPORTED)
_DTYPES_AXIS_A = (_FP32, _BF16, _FP16_IF_CPU_MATMUL_SUPPORTED)


@contextlib.contextmanager
def _single_thread():
    """Pin CPU GEMM to one thread so the K reduction order is M-independent."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def _rand_hidden(batch, seq, hidden=_HIDDEN, *, seed, dtype=torch.float32):
    """Fixed-seed random hidden-state tensor for reproducibility."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, hidden, generator=generator, dtype=dtype)


def _rand_weight(vocab=_VOCAB, hidden=_HIDDEN, *, seed, dtype=torch.float32):
    """Fixed-seed random lm_head weight tensor for reproducibility."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=generator, dtype=dtype)


def _fixed_k_reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Row-by-row GEMV: one fixed-K dot product per output logit."""
    flat_hidden = hidden.reshape(-1, hidden.size(-1))
    if flat_hidden.size(0) == 0:
        flat_out = flat_hidden @ weight.t()
    else:
        flat_out = torch.stack([torch.mv(weight, row) for row in flat_hidden], dim=0)
    return flat_out.reshape(*hidden.shape[:-1], weight.size(0))


def _enough_gpu_memory(num_bytes: int) -> bool:
    """Return True only if CUDA is present and has free memory with headroom."""
    if not torch.cuda.is_available():
        return False
    try:
        free, _ = torch.cuda.mem_get_info()
    except RuntimeError:
        return False
    return free > int(num_bytes * 1.5)


def _sm90_lm_head_available() -> bool:
    """True only on Hopper with the SM90 lm_head kernel compiled in."""
    if not torch.cuda.is_available():
        return False
    try:
        from rl_engine.backends.extension import _C, _EXT_AVAILABLE

        return (
            _EXT_AVAILABLE
            and hasattr(_C, "lm_head_sm90_forward")
            and torch.cuda.get_device_capability()[0] == 9
        )
    except Exception:
        return False


requires_sm90_lm_head = pytest.mark.skipif(
    not _sm90_lm_head_available(),
    reason=("SM90 lm_head kernel requires an H100/Hopper-class GPU and KERNEL_ALIGN_FORCE_SM90=1."),
)


def test_native_fp32_matches_fixed_k_reference():
    hidden = _rand_hidden(2, 4, seed=1)
    weight = _rand_weight(seed=1)
    ref = _fixed_k_reference(hidden, weight)
    assert torch.equal(NativeLMHeadOp().forward_fp32(hidden, weight), ref)
    assert torch.equal(NativeLMHeadOp().forward(hidden, weight), ref)


@pytest.mark.parametrize("dtype", _DTYPES_AXIS_B)
def test_native_dtype_path_accuracy(dtype: torch.dtype):
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 4, seed=2)
    weight = _rand_weight(seed=3)

    ref = op.forward_fp32(hidden, weight)
    cand = op.forward(hidden.to(dtype), weight.to(dtype))
    assert cand.dtype == dtype

    err = (cand.float() - ref).abs()
    peak = ref.abs().max()
    max_abs, mean_abs = err.max().item(), err.mean().item()
    print(f"\n[lm_head {dtype}] max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} peak={peak:.4g}")
    assert max_abs <= _DTYPE_REL_PEAK[dtype] * peak.item()


@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
def test_batch_invariance_slice(dtype: torch.dtype):
    op = NativeLMHeadOp()
    hidden = _rand_hidden(8, 32, seed=4).to(dtype)
    weight = _rand_weight(seed=4).to(dtype)
    with _single_thread():
        full = op.forward(hidden, weight)
        assert torch.equal(op.forward(hidden[:1], weight), full[:1])
        assert torch.equal(op.forward(hidden[3:5], weight), full[3:5])


@pytest.mark.parametrize("dtype", _DTYPES_AXIS_A)
def test_batch_invariance_with_padding(dtype: torch.dtype):
    op = NativeLMHeadOp()
    weight = _rand_weight(seed=5).to(dtype)
    real = _rand_hidden(4, 10, seed=5).to(dtype)
    pad = _rand_hidden(4, 6, seed=99).to(dtype)
    padded = torch.cat([real, pad], dim=1)
    with _single_thread():
        assert torch.equal(op.forward(padded, weight)[:, :10], op.forward(real, weight))


def test_tied_embedding_weight_forward_and_grad():
    op = NativeLMHeadOp()
    hidden = _rand_hidden(2, 4, seed=6)
    emb = torch.nn.Embedding.from_pretrained(_rand_weight(seed=7), freeze=False)
    untied = emb.weight.detach().clone().requires_grad_(True)
    before = emb.weight.detach().clone()

    # Forward
    out_tied = op.forward_fp32(hidden, emb.weight)
    out_untied = op.forward_fp32(hidden, untied)
    assert torch.equal(out_tied, out_untied)
    assert torch.equal(emb.weight.detach(), before)

    # Backward
    generator = torch.Generator().manual_seed(6)
    dy = torch.randn(out_tied.shape, generator=generator)
    out_tied.backward(dy)
    out_untied.backward(dy)
    assert torch.equal(emb.weight.grad, untied.grad)


@pytest.mark.skipif(
    not _enough_gpu_memory(_GEMMA_VOCAB * _GEMMA_HIDDEN * 4),
    reason="needs a CUDA GPU with room for the ~5.6 GB fp32 Gemma lm_head weight",
)
def test_native_gemma4_real_shape():
    op = NativeLMHeadOp()
    generator = torch.Generator(device="cuda").manual_seed(8)
    hidden = torch.randn(2, 16, _GEMMA_HIDDEN, generator=generator, device="cuda")
    weight = torch.randn(_GEMMA_VOCAB, _GEMMA_HIDDEN, generator=generator, device="cuda")

    out = op.forward_fp32(hidden, weight)
    assert out.shape == (2, 16, _GEMMA_VOCAB)
    assert torch.equal(out[:1], op.forward_fp32(hidden[:1], weight))


@requires_sm90_lm_head
def test_sm90_gemma4_beyond_int32_grid():
    from rl_engine.backends.cuda.embedding.lm_head import SM90LMHeadOp

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(9)
    hidden = torch.randn(
        1, 8193, _GEMMA_HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        _GEMMA_VOCAB, _GEMMA_HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    op = SM90LMHeadOp()

    out = op.forward(hidden, weight)
    assert out.shape == (1, 8193, _GEMMA_VOCAB)

    ref = NativeLMHeadOp().forward_fp32(hidden[:, -2:], weight)
    err = (out[:, -2:].float() - ref).abs()
    peak = ref.abs().max()
    max_abs, mean_abs = err.max().item(), err.mean().item()
    print(f"\n[sm90 bf16 tail] max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} peak={peak:.4g}")
    assert max_abs <= _DTYPE_REL_PEAK[torch.bfloat16] * peak.item()

    assert torch.equal(op.forward(hidden[:, -1:], weight), out[:, -1:])


def test_registry_prefers_triton_lm_head_on_rocm():
    from rl_engine.runtime.registry import OpBackend, kernel_registry

    assert kernel_registry._priority_map["rocm"]["lm_head"] == [
        OpBackend.TRITON_LM_HEAD,
        OpBackend.PYTORCH_NATIVE_LM_HEAD,
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU")
def test_triton_gemma4_beyond_int32_output():
    # Import inside the test: on a GPU host a missing/broken Triton must fail, not skip.
    from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(10)
    hidden = torch.randn(
        1, 8193, _GEMMA_HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(
        _GEMMA_VOCAB, _GEMMA_HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    op = TritonLMHeadOp()

    out = op.forward(hidden, weight)
    assert out.shape == (1, 8193, _GEMMA_VOCAB)

    ref = NativeLMHeadOp().forward_fp32(hidden[:, -2:], weight)
    err = (out[:, -2:].float() - ref).abs()
    peak = ref.abs().max()
    max_abs, mean_abs = err.max().item(), err.mean().item()
    print(f"\n[triton bf16 tail] max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} peak={peak:.4g}")
    assert max_abs <= _DTYPE_REL_PEAK[torch.bfloat16] * peak.item()

    assert torch.equal(op.forward(hidden[:, -1:], weight), out[:, -1:])


@pytest.mark.skipif(
    not _sm90_lm_head_available() or torch.cuda.device_count() < 2,
    reason="needs two H100/Hopper-class GPUs with the SM90 lm_head kernel.",
)
def test_sm90_runs_on_non_default_device():
    from rl_engine.backends.cuda.embedding.lm_head import SM90LMHeadOp

    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(11)
    hidden_0 = torch.randn(
        2, 4, _GEMMA_HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    weight_0 = torch.randn(
        _VOCAB, _GEMMA_HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    hidden_1 = hidden_0.to("cuda:1")
    weight_1 = weight_0.to("cuda:1")

    op = SM90LMHeadOp()
    out_0 = op.forward(hidden_0, weight_0)
    out_1 = op.forward(hidden_1, weight_1)
    assert torch.cuda.current_device() == 0
    assert out_1.device == torch.device("cuda:1")
    assert torch.equal(out_1.to("cuda:0"), out_0)
