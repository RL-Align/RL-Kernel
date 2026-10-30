# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared CPU harness for T05 gate tests: real operator/provider, strict-kernel double.

Test modules declare their own thin pytest fixtures around these helpers so
that fixtures are never imported across modules.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any, Iterator, Optional

import pytest
from p7_doubles import t02_sampling
from p7_doubles.fixtures import M10Fixture, build_fixtures
from p7_doubles.strict_linear_logp import CpuStrictLinearLogpDouble

import rl_engine.integrations.megatron_training_readback as readback
from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate
from rl_engine.alignment.cross_config.adapters.megatron_training import (
    PreUpdateWeightLock,
    ShardDigest,
    T05ContractError,
)
from rl_engine.integrations import framework_operators
from rl_engine.integrations.engines.train.megatron import operators as megatron_operators
from rl_engine.integrations.framework_operators import MegatronLogpOperator
from rl_engine.integrations.megatron_training_readback import (
    TrainingCallSnapshot,
    TrainingLogpObserver,
)
from rl_engine.integrations.vime.linear_logp_provider import _provider_impl

FIXTURES = build_fixtures()
TP_GROUP = object()


@contextlib.contextmanager
def strict_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[TrainingLogpObserver]:
    """Strict CPU route (device check bypassed) with an observer on the real operator class."""

    monkeypatch.setenv("VIME_RL_KERNEL_STRICT", "1")
    monkeypatch.delenv("RL_KERNEL_MODE", raising=False)
    monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "R/R")
    monkeypatch.setattr(framework_operators, "_require_nvidia_cuda", lambda tensor, module: None)
    # MegatronLogpOperator binds _require_nvidia_cuda at import time in its own module.
    monkeypatch.setattr(megatron_operators, "_require_nvidia_cuda", lambda tensor, module: None)
    instance = TrainingLogpObserver()
    instance.install()
    try:
        yield instance
    finally:
        instance.uninstall()


def patch_tp2(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Report a TP group of size 2 for ``TP_GROUP``; set ``.rank`` on the returned handle."""

    handle = SimpleNamespace(rank=0)

    def coordinates(group: Any) -> dict[str, Any]:
        if group is TP_GROUP:
            return {"group": "present", "rank": handle.rank, "world_size": 2}
        return {"group": None}

    monkeypatch.setattr(readback, "_group_coordinates", coordinates)
    return handle


def run(
    observer: TrainingLogpObserver, request: Any, wrapper: Optional[Any] = None
) -> TrainingCallSnapshot:
    operator = MegatronLogpOperator(
        _provider_impl, linear_logp=wrapper or CpuStrictLinearLogpDouble()
    )
    operator(request)
    return observer.snapshots[-1]


def make_context(fixture: M10Fixture, *, cp_rank: int = 0, **overrides: Any) -> Any:
    shards = tuple(
        ShardDigest.from_tensor(
            fixture.lm_head_weight[start:end], tp_rank=rank, vocab_start=start, vocab_end=end
        )
        for rank, (start, end) in enumerate(fixture.vocab_shards)
    )
    rows = fixture.cp_local_rows(cp_rank)
    values: dict[str, Any] = dict(
        checkpoint_id="ckpt-0",
        weight_id="w-0",
        weight_lock=PreUpdateWeightLock("ckpt-0", "w-0", shards),
        row_layout=gate.RowLayout(
            sample_ids=tuple(f"s{i}" for i in range(len(fixture.sample_lengths))),
            sample_lengths=fixture.sample_lengths,
        ),
        active_token_mask=fixture.active_mask[list(rows)].clone(),
        lm_head_lora=False,
        sampling_record=t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
    )
    values.update(overrides)
    return gate.TrainingCallContext(**values)


def expect(status: str, snapshot: Any, context: Any, field: Optional[str] = None) -> Any:
    """Assert the gate fails with ``status`` at ``field`` (prefix), not at an unrelated check."""

    with pytest.raises(T05ContractError) as info:
        gate.validate_training_call(snapshot, context)
    assert info.value.status == status, str(info.value)
    if field is not None:
        assert info.value.field_name.startswith(field), str(info.value)
    return info.value
