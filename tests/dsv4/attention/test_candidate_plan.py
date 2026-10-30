# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status


def _kv(n: int):
    k = torch.zeros(n, 512)
    v = torch.zeros(n, 512)
    return k, v


def test_compressed_then_recent_kind_ok():
    valid = torch.ones(6, dtype=torch.bool)
    kind = torch.tensor([0, 0, 0, 1, 1, 1])
    plan = CandidatePlan(layer_type="C4", n_compressed=3, n_recent=3, valid=valid, kind=kind)
    plan.validate_for_kv(*_kv(6))
    assert plan.attention_kind.value == "CSA"


def test_c0_rejects_compressed_rows():
    plan = CandidatePlan(
        layer_type="C0", n_compressed=1, n_recent=1, valid=torch.ones(2, dtype=torch.bool)
    )
    with pytest.raises(DSv4FailClosedError) as exc:
        plan.validate_for_kv(*_kv(2))
    assert exc.value.status is DSv4Status.INVALID_COMPRESSION_PLAN


def test_two_softmax_mode_is_fail_closed():
    plan = CandidatePlan(
        layer_type="C0",
        n_compressed=0,
        n_recent=2,
        valid=torch.ones(2, dtype=torch.bool),
        softmax_mode="two_softmax_merge",
    )
    with pytest.raises(DSv4FailClosedError) as exc:
        plan.validate_for_kv(*_kv(2))
    assert exc.value.status is DSv4Status.MULTIPLE_SOFTMAX_DENOMINATORS


def test_split_kv_is_fail_closed():
    plan = CandidatePlan(
        layer_type="C0",
        n_compressed=0,
        n_recent=2,
        valid=torch.ones(2, dtype=torch.bool),
        num_splits=2,
    )
    with pytest.raises(DSv4FailClosedError) as exc:
        plan.validate_for_kv(*_kv(2))
    assert exc.value.status is DSv4Status.FORBIDDEN_SPLIT_REDUCTION


def test_sink_has_v_is_fail_closed():
    plan = CandidatePlan(
        layer_type="C0",
        n_compressed=0,
        n_recent=1,
        valid=torch.ones(1, dtype=torch.bool),
        sink_has_v=True,
    )
    with pytest.raises(DSv4FailClosedError) as exc:
        plan.validate_for_kv(*_kv(1))
    assert exc.value.status is DSv4Status.INVALID_SINK_SEMANTICS


def test_kind_order_must_be_prefix_then_recent():
    kind = torch.tensor([1, 0, 0, 1])
    plan = CandidatePlan(
        layer_type="C4",
        n_compressed=2,
        n_recent=2,
        valid=torch.ones(4, dtype=torch.bool),
        kind=kind,
    )
    with pytest.raises(DSv4FailClosedError) as exc:
        plan.validate_for_kv(*_kv(4))
    assert exc.value.status is DSv4Status.INVALID_CANDIDATE_ORDER


def test_illegal_layer_type():
    with pytest.raises(DSv4FailClosedError) as exc:
        CandidatePlan(
            layer_type="C2", n_compressed=0, n_recent=1, valid=torch.ones(1, dtype=torch.bool)
        )
    assert exc.value.status is DSv4Status.INVALID_LAYER_TYPE
