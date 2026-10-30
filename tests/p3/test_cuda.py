"""Opt-in real H100 tests. P3_RUN_GPU=1 turns hardware absence into failure."""

import os

import numpy as np
import pytest
import torch

from rl_engine.p3 import bitmath
from rl_engine.p3.contract import P3OpCtxHost, P3Verdict
from rl_engine.p3.oracle import stable_topk6
from rl_engine.p3.provider import InvocationAllocator
from rl_engine.p3.stable_topk6 import CudaTopKProvider, device_bitmath

pytestmark = [
    pytest.mark.cuda_only,
    pytest.mark.skipif(os.getenv("P3_RUN_GPU") != "1", reason="set P3_RUN_GPU=1 on H100"),
]


def context(tmp_path, active):
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)
    return P3OpCtxHost(
        "gpu-test",
        "cuda",
        0,
        active,
        backend_tag="cuda",
        allocator=InvocationAllocator(tmp_path / "journal", "gpu-test", "cuda", 0),
    )


@pytest.mark.parametrize("block", [1, 32, 64, 128])
def test_cuda_total_order_repeat_padding_and_real_stream(tmp_path, block):
    rng = np.random.default_rng(991)
    q = rng.integers(-7, 8, size=(129, 256)).astype(np.float32)
    q[0] = 0
    q[1, ::2] = np.float32(-0.0)
    q[2, 55] = np.nextafter(np.float32(7), np.float32(np.inf))
    q[-1] = np.nan
    active = np.ones(129, np.bool_)
    active[-1] = False
    ctx = context(tmp_path, active)
    ctx.stream = torch.cuda.Stream()
    source = torch.as_tensor(q, device="cuda:0")
    result = CudaTopKProvider().stable_topk6_fwd(ctx, source, block_size=block)
    assert result.verdict == P3Verdict.PASS, result.provenance
    expected = stable_topk6(q[:-1])
    ids = result.payload["ids"].cpu().numpy()
    assert np.array_equal(ids[:-1], expected)
    assert np.array_equal(ids[-1], np.arange(6))
    previous_invocation = ctx.invocation_id
    repeat = CudaTopKProvider().stable_topk6_fwd(ctx, source, block_size=block)
    assert repeat.verdict == P3Verdict.PASS
    assert np.array_equal(repeat.payload["ids"].cpu().numpy(), ids)
    assert ctx.invocation_id == previous_invocation + 1
    assert result.provenance["actual_binary_arch"] == 90
    assert result.provenance["actual_backend"] == "cuda"
    assert "fast-math=false" in result.provenance["build_flags"]


@pytest.mark.parametrize("op", ["exp", "log1p", "softplus", "sigmoid", "sqrt", "bf16"])
def test_host_device_bitmath_exact_bytes(op):
    assert torch.cuda.is_available()
    values = np.unique(
        np.concatenate(
            [
                np.linspace(-110, 88, 4096, dtype=np.float32),
                np.array(
                    [
                        -104,
                        -100,
                        -80,
                        -20,
                        -0.0,
                        0,
                        1e-40,
                        1e-20,
                        0.5,
                        20,
                        np.nextafter(np.float32(20), np.float32(21)),
                        80,
                    ],
                    np.float32,
                ),
            ]
        )
    )
    if op in ("sqrt", "log1p"):
        values = np.abs(values)
    cpu = bitmath.apply(values, op)
    gpu = device_bitmath(torch.as_tensor(values, device="cuda:0"), op).cpu().numpy()
    assert np.array_equal(cpu.view(np.uint32), gpu.view(np.uint32)), op


def test_cuda_nonfinite_zero_active_and_no_cpu_fallback(tmp_path):
    ctx = context(tmp_path, np.ones(1, np.bool_))
    q = torch.full((1, 256), float("nan"), device="cuda:0")
    result = CudaTopKProvider().stable_topk6_fwd(ctx, q)
    assert result.verdict == P3Verdict.NON_FINITE and result.payload is None
    zero = context(tmp_path, np.zeros(1, np.bool_))
    result = CudaTopKProvider().stable_topk6_fwd(zero, q)
    assert result.verdict == P3Verdict.ZERO_ACTIVE_TOKENS and zero.invocation_id == 0
    cpu = CudaTopKProvider().stable_topk6_fwd(ctx, torch.zeros((1, 256)))
    assert cpu.verdict == P3Verdict.UNSUPPORTED_CAPABILITY and cpu.payload is None
