# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.fixtures.catalog import make_attn_case
from rl_engine.kernels.dsv4.attention.state_gate import (
    StateGateVerdict,
    require_state_gate,
    synthetic_pass_verdict,
)


def test_missing_verdict_is_not_pass():
    with pytest.raises(DSv4FailClosedError) as exc:
        require_state_gate(None)
    assert exc.value.status is DSv4Status.STATE_BYTES_MISMATCH


def test_failed_state_stops_attention_compare():
    case = make_attn_case(
        "state_fail",
        layer_type="C0",
        tokens=1,
        n_compressed=0,
        n_recent=2,
        state_status="STATE_BYTES_MISMATCH",
    )
    op = MqaJointAttentionSinkOp(backend="oracle")
    with pytest.raises(DSv4FailClosedError) as exc:
        op.forward(
            case.q, case.k, case.v, case.sink, case.plan,
            compare=True, state_gate=case.state_gate,
        )
    assert exc.value.status is DSv4Status.STATE_BYTES_MISMATCH


def test_synthetic_pass_allows_compare():
    case = make_attn_case("state_ok", layer_type="C0", tokens=1, n_compressed=0, n_recent=2)
    assert case.state_gate.source == "synthetic_recorded"
    op = MqaJointAttentionSinkOp(backend="oracle")
    result = op.forward_fp32(
        case.q, case.k, case.v, case.sink, case.plan, compare=True, state_gate=case.state_gate
    )
    assert result.o.shape == (1, 64, 512)
    assert torch.isfinite(result.o).all()


def test_pass_helper_status():
    verdict = synthetic_pass_verdict(tag="x")
    assert require_state_gate(verdict).status == DSv4Status.PASS.value
    failed = StateGateVerdict(status="BYTE_MISMATCH", source="synthetic_recorded")
    with pytest.raises(DSv4FailClosedError):
        require_state_gate(failed)
