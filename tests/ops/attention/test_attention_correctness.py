# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import os
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.attention.flash_attn import (
    StrictRocmAiterCKAttentionCore,
    StrictRocmAttentionUnavailable,
)
from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
)

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None


DTYPE_CASES = [
    pytest.param(torch.float16, 1e-3, 1e-3, id="fp16"),
    pytest.param(torch.bfloat16, 2e-2, 2e-2, id="bf16"),
]

AVAILABILITY_ERRORS = (ImportError, ModuleNotFoundError, OSError, RuntimeError)

ATTENTION_SHAPES = [
    pytest.param(1, 128, 4, 64, id="b1-s128-h4-d64"),
    pytest.param(2, 256, 8, 128, id="b2-s256-h8-d128"),
    pytest.param(1, 512, 8, 256, id="b1-s512-h8-d256"),
    pytest.param(1, 1024, 16, 64, id="b1-s1024-h16-d64"),
    pytest.param(1, 2048, 16, 64, id="b1-s2048-h16-d64"),
]


@contextmanager
def sdpa_math_backend():
    if sdpa_kernel is not None and SDPBackend is not None:
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return

    if hasattr(torch.backends.cuda, "sdp_kernel"):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        ):
            yield
        return

    raise RuntimeError("PyTorch SDPA backend selector is unavailable")


def should_print_attention_diff():
    return os.getenv("PRINT_ATTENTION_DIFF", "").lower() in {"1", "true", "yes"}


def pytorch_sdpa_reference(q, k, v, *, causal, softmax_scale):
    """
    Compute a PyTorch SDPA math-backend reference for FlashAttention-layout inputs.

    The reference runs in fp32 and validates each low-precision backend against
    that golden value. It does not guarantee CUDA and ROCm backends are
    numerically identical to each other.

    Inputs use FlashAttention layout: (batch, seqlen, nheads, headdim).
    PyTorch SDPA uses: (batch, nheads, seqlen, headdim), so this helper
    transposes before and after the reference call.
    """
    q_ref = q.transpose(1, 2).contiguous()
    k_ref = k.transpose(1, 2).contiguous()
    v_ref = v.transpose(1, 2).contiguous()

    with sdpa_math_backend():
        expected = F.scaled_dot_product_attention(
            q_ref.float(),
            k_ref.float(),
            v_ref.float(),
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )

    return expected.transpose(1, 2).contiguous()


def make_qkv(batch, seqlen, nheads, headdim, device, dtype, nheads_k=None):
    torch.manual_seed(0)
    if nheads_k is None:
        nheads_k = nheads
    q = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen, nheads_k, headdim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen, nheads_k, headdim, device=device, dtype=dtype)
    return q, k, v


def describe_attention_diff(actual, expected, *, dtype, atol, rtol, causal, softmax_scale):
    actual_f = actual.float()
    expected_f = expected.float()
    abs_diff = (actual_f - expected_f).abs()
    rel_diff = abs_diff / expected_f.abs().clamp_min(1e-12)

    return (
        "FlashAttention vs PyTorch SDPA math backend diff: "
        f"dtype={dtype}, shape={tuple(actual.shape)}, causal={causal}, "
        f"softmax_scale={softmax_scale}, atol={atol}, rtol={rtol}, "
        f"max_abs_diff={abs_diff.max().item():.6g}, "
        f"mean_abs_diff={abs_diff.mean().item():.6g}, "
        f"max_rel_diff={rel_diff.max().item():.6g}"
    )


def print_attention_diff(actual, expected, *, dtype, atol, rtol, causal, softmax_scale):
    if should_print_attention_diff():
        print(
            describe_attention_diff(
                actual,
                expected,
                dtype=dtype,
                atol=atol,
                rtol=rtol,
                causal=causal,
                softmax_scale=softmax_scale,
            )
        )


def is_cuda_platform():
    return torch.cuda.is_available() and torch.version.hip is None


def is_rocm_platform():
    return torch.cuda.is_available() and torch.version.hip is not None


def cuda_flash_attention_availability():
    if not torch.cuda.is_available():
        return (
            False,
            "CUDA is not available, check CUDA device, driver/runtime compatibility, "
            "and torch CUDA build",
        )
    if not is_cuda_platform():
        return False, "current torch build is not CUDA platform"
    try:
        from rl_engine.backends.cuda.attention.flash_attn import FlashAttentionOp

        FlashAttentionOp()
    except AVAILABILITY_ERRORS as exc:
        return False, f"CUDA FlashAttentionOp is unavailable: {exc}"
    return True, ""


def rocm_flash_attention_availability():
    if not torch.cuda.is_available():
        return (
            False,
            "ROCm is not available, check AMD GPU device, driver/runtime compatibility, "
            "and torch ROCm build",
        )
    if not is_rocm_platform():
        return False, "current torch build is not ROCm platform"
    try:
        from rl_engine.backends.rocm.attention.flash_attn import RocmFlashAttentionOp

        RocmFlashAttentionOp()
    except AVAILABILITY_ERRORS as exc:
        return False, f"ROCm FlashAttentionOp is unavailable: {exc}"
    return True, ""


def native_attention_availability():
    if not torch.cuda.is_available():
        return False, "CUDA/ROCm GPU is not available"
    try:
        from rl_engine.reference.attention import NativeAttentionOp

        NativeAttentionOp()
    except AVAILABILITY_ERRORS as exc:
        return False, f"NativeAttentionOp is unavailable: {exc}"
    return True, ""


def assert_flash_attention_matches_sdpa(
    op,
    dtype,
    atol,
    rtol,
    causal,
    use_explicit_scale,
    batch,
    seqlen,
    nheads,
    headdim,
):
    # PyTorch exposes both NVIDIA CUDA and AMD ROCm GPUs through the "cuda" device API.
    device = torch.device("cuda")
    q, k, v = make_qkv(batch, seqlen, nheads, headdim, device, dtype)
    softmax_scale = (1.0 / headdim**0.5) if use_explicit_scale else None

    actual = op(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )

    expected = pytorch_sdpa_reference(q, k, v, causal=causal, softmax_scale=softmax_scale)
    print_attention_diff(
        actual,
        expected,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=atol,
        rtol=rtol,
        msg=describe_attention_diff(
            actual,
            expected,
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            causal=causal,
            softmax_scale=softmax_scale,
        ),
    )


@pytest.mark.parametrize(("dtype", "atol", "rtol"), DTYPE_CASES)
@pytest.mark.parametrize(
    "causal",
    (pytest.param(False, id="noncausal"), pytest.param(True, id="causal")),
)
@pytest.mark.parametrize(
    "use_explicit_scale",
    (pytest.param(False, id="default-scale"), pytest.param(True, id="explicit-scale")),
)
@pytest.mark.parametrize(("batch", "seqlen", "nheads", "headdim"), ATTENTION_SHAPES)
def test_cuda_flash_attention_matches_sdpa(
    dtype,
    atol,
    rtol,
    causal,
    use_explicit_scale,
    batch,
    seqlen,
    nheads,
    headdim,
):
    available, reason = cuda_flash_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.backends.cuda.attention.flash_attn import FlashAttentionOp

    assert_flash_attention_matches_sdpa(
        FlashAttentionOp(),
        dtype,
        atol,
        rtol,
        causal,
        use_explicit_scale,
        batch,
        seqlen,
        nheads,
        headdim,
    )


@pytest.mark.parametrize(("dtype", "atol", "rtol"), DTYPE_CASES)
@pytest.mark.parametrize(
    "causal",
    (pytest.param(False, id="noncausal"), pytest.param(True, id="causal")),
)
@pytest.mark.parametrize(
    "use_explicit_scale",
    (pytest.param(False, id="default-scale"), pytest.param(True, id="explicit-scale")),
)
@pytest.mark.parametrize(("batch", "seqlen", "nheads", "headdim"), ATTENTION_SHAPES)
def test_rocm_flash_attention_matches_sdpa(
    dtype,
    atol,
    rtol,
    causal,
    use_explicit_scale,
    batch,
    seqlen,
    nheads,
    headdim,
):
    available, reason = rocm_flash_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.backends.rocm.attention.flash_attn import RocmFlashAttentionOp

    assert_flash_attention_matches_sdpa(
        RocmFlashAttentionOp(),
        dtype,
        atol,
        rtol,
        causal,
        use_explicit_scale,
        batch,
        seqlen,
        nheads,
        headdim,
    )


@pytest.mark.parametrize(
    "causal",
    (pytest.param(False, id="noncausal"), pytest.param(True, id="causal")),
)
def test_rocm_flash_attention_rejects_unsupported_head_dim(causal):
    available, reason = rocm_flash_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.backends.rocm.attention.flash_attn import RocmFlashAttentionOp

    q, k, v = make_qkv(
        batch=1,
        seqlen=64,
        nheads=2,
        headdim=513,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    with pytest.raises(NotImplementedError, match="head_dim <= 512"):
        RocmFlashAttentionOp()(q, k, v, causal=causal)


@pytest.mark.parametrize(("dtype", "atol", "rtol"), DTYPE_CASES)
@pytest.mark.parametrize(
    "causal",
    (pytest.param(False, id="noncausal"), pytest.param(True, id="causal")),
)
@pytest.mark.parametrize(
    "use_explicit_scale",
    (pytest.param(False, id="default-scale"), pytest.param(True, id="explicit-scale")),
)
@pytest.mark.parametrize(("batch", "seqlen", "nheads", "headdim"), ATTENTION_SHAPES)
def test_native_attention_matches_sdpa(
    dtype,
    atol,
    rtol,
    causal,
    use_explicit_scale,
    batch,
    seqlen,
    nheads,
    headdim,
):
    available, reason = native_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.reference.attention import NativeAttentionOp

    assert_flash_attention_matches_sdpa(
        NativeAttentionOp(),
        dtype,
        atol,
        rtol,
        causal,
        use_explicit_scale,
        batch,
        seqlen,
        nheads,
        headdim,
    )


@pytest.mark.parametrize(
    ("nheads", "nheads_k"),
    (pytest.param(8, 4, id="gqa"), pytest.param(8, 1, id="mqa")),
)
@pytest.mark.parametrize(
    "causal",
    (pytest.param(False, id="noncausal"), pytest.param(True, id="causal")),
)
def test_native_attention_supports_gqa_mqa(nheads, nheads_k, causal):
    available, reason = native_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.reference.attention import NativeAttentionOp

    device = torch.device("cuda")
    dtype = torch.float16
    q, k, v = make_qkv(
        batch=1,
        seqlen=128,
        nheads=nheads,
        nheads_k=nheads_k,
        headdim=64,
        device=device,
        dtype=dtype,
    )

    actual = NativeAttentionOp()(q, k, v, dropout_p=0.0, causal=causal)
    repeat = nheads // nheads_k
    expected = pytorch_sdpa_reference(
        q,
        k.repeat_interleave(repeat, dim=2),
        v.repeat_interleave(repeat, dim=2),
        causal=causal,
        softmax_scale=None,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)


def test_native_attention_rejects_invalid_gqa_head_ratio():
    available, reason = native_attention_availability()
    if not available:
        pytest.skip(reason)

    from rl_engine.reference.attention import NativeAttentionOp

    q, k, v = make_qkv(
        batch=1,
        seqlen=128,
        nheads=10,
        nheads_k=3,
        headdim=64,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    with pytest.raises(ValueError, match="q heads must be divisible"):
        NativeAttentionOp()(q, k, v)


def test_strict_rocm_aiter_ck_core_fixes_forward_and_backward_contract(monkeypatch):
    calls = []

    def fake_fwd(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_left,
        window_right,
        sink_size,
        return_lse,
        return_dropout_mask,
    ):
        calls.append(
            (
                "forward",
                dropout_p,
                softmax_scale,
                causal,
                window_left,
                window_right,
                sink_size,
                return_lse,
                return_dropout_mask,
            )
        )
        return (
            q.clone(),
            torch.zeros(q.size(0), q.size(2), q.size(1), dtype=torch.float32),
            torch.empty(0),
            torch.zeros(2, dtype=torch.int64),
        )

    def fake_bwd(
        dout,
        q,
        k,
        v,
        out,
        lse,
        dropout_p,
        softmax_scale,
        causal,
        window_left,
        window_right,
        deterministic,
        **kwargs,
    ):
        calls.append(
            (
                "backward",
                dropout_p,
                softmax_scale,
                causal,
                window_left,
                window_right,
                deterministic,
                kwargs["rng_state"].shape,
            )
        )
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v), torch.empty(0)

    core = StrictRocmAiterCKAttentionCore(
        _mha_fwd=fake_fwd,
        _mha_bwd=fake_bwd,
        _source_sha256="a" * 64,
    )
    monkeypatch.setattr(core, "_validate_inputs", lambda *_args: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Props", (), {"name": "test-gpu", "gcnArchName": "gfx-test"})(),
    )
    q = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
    result = core.forward_with_lse(
        q,
        k,
        v,
        causal=True,
        scale=0.125,
        query_position_ids=torch.tensor([[1, 2]]),
        key_position_ids=torch.tensor([[0, 1, 2]]),
    )
    result.out.float().sum().backward()

    assert calls == [
        ("forward", 0.0, 0.125, True, -1, -1, 0, True, False),
        ("backward", 0.0, 0.125, True, -1, -1, True, torch.Size([2])),
    ]
    assert result.out.shape == q.shape
    assert result.lse.shape == q.shape[:3]
    assert result.lse.dtype is torch.float32
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_ROCM_SCHEDULE_ID
    assert result.provenance["attention_backend"] == "aiter.rocm.ck_dense_mha"
    assert result.provenance["split_kv_control"] == "dense_non_split_api"
    assert result.provenance["num_splits"] == 1
    assert result.provenance["deterministic_backward"] is True
    assert result.provenance["aiter_source_sha256"] == "a" * 64
    assert q.grad is not None and k.grad is not None and v.grad is not None


def test_strict_rocm_aiter_ck_direct_decode_uses_callers_output(monkeypatch):
    seen_out = None

    def fake_fwd(q, k, v, *_args, out=None):
        nonlocal seen_out
        seen_out = out
        out.copy_(q)
        return (
            out,
            torch.zeros(q.size(0), q.size(2), q.size(1), dtype=torch.float32),
            torch.empty(0),
            torch.zeros(2, dtype=torch.int64),
        )

    core = StrictRocmAiterCKAttentionCore(
        _mha_fwd=fake_fwd,
        _mha_bwd=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(core, "_validate_bshd_inputs", lambda *_args: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Props", (), {"name": "test-gpu", "gcnArchName": "gfx-test"})(),
    )
    q = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 7, 8, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out = torch.empty_like(q)

    with torch.no_grad():
        result = core.forward_bshd_with_lse(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            out=out,
            causal=False,
            scale=0.125,
        )

    assert seen_out is not None and seen_out.data_ptr() == out.data_ptr()
    assert result.out is out
    assert torch.equal(out, q)
    assert result.provenance["input_layout"] == "bshd_direct"
    assert result.provenance["direct_output"] is True


def test_strict_rocm_aiter_ck_reuses_immutable_provenance_inputs(monkeypatch):
    def fake_fwd(q, k, v, *_args, out=None):
        out.copy_(q)
        return (
            out,
            torch.zeros(q.size(0), q.size(2), q.size(1), dtype=torch.float32),
            torch.empty(0),
            torch.zeros(2, dtype=torch.int64),
        )

    core = StrictRocmAiterCKAttentionCore(
        _mha_fwd=fake_fwd,
        _mha_bwd=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(core, "_validate_bshd_inputs", lambda *_args: None)
    device_lookups = 0

    def fake_device_properties(_device):
        nonlocal device_lookups
        device_lookups += 1
        return type("Props", (), {"name": "test-gpu", "gcnArchName": "gfx-test"})()

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_device_properties)
    split_resolves = 0
    split_type = type(core.split_kv)
    original_resolve = split_type.resolve

    def counted_resolve(self, total_kv_tokens, *, backend):
        nonlocal split_resolves
        split_resolves += 1
        return original_resolve(self, total_kv_tokens, backend=backend)

    monkeypatch.setattr(split_type, "resolve", counted_resolve)
    q = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)

    def run(kv_tokens):
        k = torch.randn(1, 1, kv_tokens, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            return core.forward_bshd_with_lse(
                q.transpose(1, 2),
                k.transpose(1, 2),
                k.clone().transpose(1, 2),
                out=torch.empty_like(q),
                causal=False,
                scale=None,
            )

    first = run(7)
    repeated = run(7)
    changed_length = run(9)
    restored_length = run(7)

    assert device_lookups == 1
    assert split_resolves == 3
    assert first.provenance == repeated.provenance
    assert first.provenance is not repeated.provenance
    assert first.provenance["split_kv"] is not repeated.provenance["split_kv"]
    first_boundaries = first.provenance["split_kv"]["actual_split_boundaries"]
    repeated_boundaries = repeated.provenance["split_kv"]["actual_split_boundaries"]
    assert first_boundaries is not repeated_boundaries
    assert first_boundaries[0] is not repeated_boundaries[0]
    assert changed_length.provenance["split_kv"]["actual_split_boundaries"] == [[0, 9]]
    assert restored_length.provenance["split_kv"]["actual_split_boundaries"] == [[0, 7]]
    first_boundaries[0][0] = 3
    assert repeated_boundaries == [[0, 7]]
    after_mutation = run(7)
    assert after_mutation.provenance["split_kv"]["actual_split_boundaries"] == [[0, 7]]
    assert split_resolves == 3

    assert core._device_description(torch.device("cuda:1")) == ("test-gpu", "gfx-test")
    assert core._device_description(torch.device("cuda:1")) == ("test-gpu", "gfx-test")
    assert device_lookups == 2


def test_strict_rocm_aiter_ck_direct_decode_rejects_ignored_output(monkeypatch):
    def fake_fwd(q, k, v, *_args, out=None):
        return (
            q.clone(),
            torch.zeros(q.size(0), q.size(2), q.size(1), dtype=torch.float32),
            torch.empty(0),
            torch.zeros(2, dtype=torch.int64),
        )

    core = StrictRocmAiterCKAttentionCore(
        _mha_fwd=fake_fwd,
        _mha_bwd=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(core, "_validate_bshd_inputs", lambda *_args: None)
    q = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 7, 8, dtype=torch.bfloat16)
    out = torch.empty_like(q)

    with (
        torch.no_grad(),
        pytest.raises(
            StrictRocmAttentionUnavailable,
            match="strict decode output buffer",
        ),
    ):
        core.forward_bshd_with_lse(
            q.transpose(1, 2),
            k.transpose(1, 2),
            k.transpose(1, 2),
            out=out,
            causal=False,
            scale=None,
        )


def test_strict_rocm_aiter_ck_core_rejects_non_fp32_lse(monkeypatch):
    def fake_fwd(q, k, v, *_args):
        return (
            q,
            torch.zeros(q.size(0), q.size(2), q.size(1), dtype=q.dtype),
            torch.empty(0),
            torch.empty(2),
        )

    core = StrictRocmAiterCKAttentionCore(
        _mha_fwd=fake_fwd,
        _mha_bwd=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(core, "_validate_inputs", lambda *_args: None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: type("Props", (), {"name": "test-gpu", "gcnArchName": "gfx-test"})(),
    )
    q = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)

    with pytest.raises(StrictRocmAttentionUnavailable, match="FP32 LSE"):
        core.forward_with_lse(
            q,
            k,
            k,
            causal=True,
            query_position_ids=torch.tensor([[0]]),
            key_position_ids=torch.tensor([[0]]),
        )
