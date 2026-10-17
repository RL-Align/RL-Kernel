# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from types import SimpleNamespace

import pytest
import torch

from rl_engine.kernels.ops import base
from rl_engine.kernels.dsv4.attention import cuda_runtime
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.fixtures.catalog import make_oproj_case
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.dsv4.attention.o_proj import rope_consumer
from rl_engine.kernels.dsv4.attention.o_proj.rope_consumer import apply_gptj_interleaved_partial


def test_neox_variant_rejected():
    case = make_oproj_case("neox", tokens=1)
    with pytest.raises(DSv4FailClosedError) as exc:
        apply_gptj_interleaved_partial(
            case.o, case.cos, case.sin, inverse=True, variant="neox_split_half"
        )
    assert exc.value.status is DSv4Status.INVALID_ROPE_VARIANT


def test_inplace_rope_rejected():
    case = make_oproj_case("inplace", tokens=1)
    with pytest.raises(DSv4FailClosedError) as exc:
        apply_gptj_interleaved_partial(case.o, case.cos, case.sin, inverse=True, inplace=True)
    assert exc.value.status is DSv4Status.INVALID_ROPE_VARIANT


def test_det_gemm_backend_without_gpu_is_unsupported():
    case = make_oproj_case("cpu", tokens=1)
    op = OProjGroupedOp(backend="det_gemm")
    with pytest.raises(DSv4FailClosedError) as exc:
        op.forward_fp32(case.o, case.w_a, case.w_b, case.cos, case.sin)
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY


@pytest.mark.parametrize("index", range(5))
def test_forward_mixed_devices_rejected_before_backend_load(monkeypatch, index):
    inputs = [torch.empty(0) for _ in range(5)]
    inputs[index] = torch.empty(0, device="meta")

    def unexpected_backend(*args):
        pytest.fail("backend loaded before device validation")

    op = OProjGroupedOp(backend="det_gemm")
    monkeypatch.setattr(op, "_linear", unexpected_backend)
    with pytest.raises(DSv4FailClosedError, match="same device") as exc:
        op.forward(*inputs)
    assert exc.value.status is DSv4Status.SCHEMA_MISMATCH


@pytest.mark.parametrize("field", ["d_y", "o_tilde", "z_groups", "z", "w_a", "w_b", "cos", "sin"])
def test_backward_mixed_devices_rejected_before_backend_load(monkeypatch, field):
    tensor = torch.empty(0)
    saved = SimpleNamespace(
        o=tensor, o_tilde=tensor, z_groups=(tensor,), z=tensor, w_a=tensor, w_b=tensor
    )
    inputs = {"d_y": tensor, "cos": tensor, "sin": tensor}
    peer = torch.empty(0, device="meta")
    if field in inputs:
        inputs[field] = peer
    else:
        setattr(saved, field, (peer,) if field == "z_groups" else peer)

    def unexpected_backend(*args):
        pytest.fail("backend loaded before device validation")

    op = OProjGroupedOp(backend="det_gemm")
    monkeypatch.setattr(op, "_linear", unexpected_backend)
    with pytest.raises(DSv4FailClosedError, match="same device") as exc:
        op.backward(inputs["d_y"], saved, inputs["cos"], inputs["sin"])
    assert exc.value.status is DSv4Status.SCHEMA_MISMATCH


@pytest.mark.parametrize(
    "missing", ["det_gemm_fwd_rhs_transposed", "det_gemm_fwd", "det_gemm_db_transposed"]
)
def test_partial_det_gemm_after_jit_failure_is_unsupported(monkeypatch, missing):
    from rl_engine.kernels.ops.cuda.matmul import det_gemm

    native = SimpleNamespace(
        det_gemm_fwd_rhs_transposed=object(),
        det_gemm_fwd=object(),
        det_gemm_db_transposed=object(),
    )
    delattr(native, missing)

    def failed_jit():
        raise RuntimeError("JIT unavailable")

    monkeypatch.setattr(cuda_runtime, "ensure_native_kernels", failed_jit)
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(det_gemm, "_C", native)
    monkeypatch.setattr(det_gemm, "_EXT_AVAILABLE", True)
    with pytest.raises(DSv4FailClosedError) as exc:
        OProjGroupedOp(backend="det_gemm")._linear(torch.empty(0))
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY


def test_complete_strict_det_gemm_remains_available_after_jit_failure(monkeypatch):
    from rl_engine.kernels.ops.cuda.matmul import det_gemm

    native = SimpleNamespace(
        det_gemm_fwd_rhs_transposed=object(),
        det_gemm_fwd=object(),
        det_gemm_db_transposed=object(),
        det_gemm_sm90_compiled=lambda: True,
    )

    def failed_jit():
        raise RuntimeError("JIT unavailable")

    monkeypatch.setattr(cuda_runtime, "ensure_native_kernels", failed_jit)
    monkeypatch.setattr(base, "_C", native)
    monkeypatch.setattr(base, "_EXT_AVAILABLE", True)
    monkeypatch.setattr(det_gemm, "_C", native)
    monkeypatch.setattr(det_gemm, "_EXT_AVAILABLE", True)
    monkeypatch.setenv("RL_KERNEL_DET_GEMM_BACKEND", "sm90")
    linear, backend, _ = OProjGroupedOp(backend="det_gemm")._linear(torch.empty(0))
    assert callable(linear)
    assert backend == "det_gemm"


@pytest.mark.parametrize("inverse", [False, True])
def test_t02_alias_is_identity_drift(monkeypatch, inverse):
    x = torch.zeros(2, 64, 512)
    cos = torch.ones(2, 32)
    sin = torch.zeros_like(cos)

    def aliased_t02(value, *_args, **kwargs):
        assert kwargs["inplace"] is False
        return value.view_as(value)

    monkeypatch.setattr(rope_consumer, "_load_t02", lambda: aliased_t02)
    with pytest.raises(DSv4FailClosedError) as exc:
        apply_gptj_interleaved_partial(x, cos, sin, inverse=inverse)
    assert exc.value.status is DSv4Status.IDENTITY_DRIFT
