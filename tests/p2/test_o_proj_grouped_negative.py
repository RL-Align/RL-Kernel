# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.fixtures.catalog import make_oproj_case
from rl_engine.kernels.p2.o_proj.o_proj_grouped import OProjGroupedOp
from rl_engine.kernels.p2.o_proj import rope_consumer
from rl_engine.kernels.p2.o_proj.rope_consumer import apply_gptj_interleaved_partial


def test_neox_variant_rejected():
    case = make_oproj_case("neox", tokens=1)
    with pytest.raises(P2FailClosedError) as exc:
        apply_gptj_interleaved_partial(
            case.o, case.cos, case.sin, inverse=True, variant="neox_split_half"
        )
    assert exc.value.status is P2Status.INVALID_ROPE_VARIANT


def test_inplace_rope_rejected():
    case = make_oproj_case("inplace", tokens=1)
    with pytest.raises(P2FailClosedError) as exc:
        apply_gptj_interleaved_partial(case.o, case.cos, case.sin, inverse=True, inplace=True)
    assert exc.value.status is P2Status.INVALID_ROPE_VARIANT


def test_det_gemm_backend_without_gpu_is_unsupported():
    case = make_oproj_case("cpu", tokens=1)
    op = OProjGroupedOp(backend="det_gemm")
    with pytest.raises(P2FailClosedError) as exc:
        op.forward_fp32(case.o, case.w_a, case.w_b, case.cos, case.sin)
    assert exc.value.status is P2Status.UNSUPPORTED_CAPABILITY


@pytest.mark.parametrize("inverse", [False, True])
def test_t02_alias_is_identity_drift(monkeypatch, inverse):
    x = torch.zeros(2, 64, 512)
    cos = torch.ones(2, 32)
    sin = torch.zeros_like(cos)

    def aliased_t02(value, *_args, **kwargs):
        assert kwargs["inplace"] is False
        return value.view_as(value)

    monkeypatch.setattr(rope_consumer, "_load_t02", lambda: aliased_t02)
    with pytest.raises(P2FailClosedError) as exc:
        apply_gptj_interleaved_partial(x, cos, sin, inverse=inverse)
    assert exc.value.status is P2Status.IDENTITY_DRIFT
