# SPDX-License-Identifier: Apache-2.0
"""INK-55, INK-57, INK-58, INK-60 and INK-74 runtime contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.runtime_contracts import (
    CheckpointExport,
    RuntimeContractError,
    RuntimeGate,
    RouteReplayRecord,
    VersionedWeightSync,
    WeightKeyMapping,
    build_miles_native_spec,
)


def test_checkpoint_mapping_is_one_to_one_and_fingerprinted():
    mapping = WeightKeyMapping((("actor.", "model."),))
    export = CheckpointExport("miles-r1", {"actor.layer.weight": torch.ones(2, 3)})
    converted = export.convert(mapping)
    assert tuple(converted) == ("model.layer.weight",)
    assert converted["model.layer.weight"].dtype == torch.bfloat16
    assert export.fingerprint(mapping) == export.fingerprint(mapping)
    with pytest.raises(RuntimeContractError, match="exactly one"):
        WeightKeyMapping((("actor.", "model."), ("actor.layer.", "model.layer."))).map_key(
            "actor.layer.weight"
        )


def test_versioned_sync_resets_cache_before_replay():
    events = []
    sync = VersionedWeightSync(
        reset_cache=lambda: events.append("reset"),
        replay_routes=lambda routes: events.append(("replay", len(routes))),
    )
    sync.publish(version=1, fingerprint="abc")
    sync.replay([RouteReplayRecord(3, (1, 2), (0.7, 0.3), 12)])
    assert events == ["reset", ("replay", 1)]
    with pytest.raises(RuntimeContractError, match="increase"):
        sync.publish(version=1, fingerprint="def")
    with pytest.raises(RuntimeContractError, match="publish"):
        VersionedWeightSync(reset_cache=lambda: None, replay_routes=lambda _: None).replay([])


def test_runtime_gate_reports_missing_external_environment():
    report = RuntimeGate(
        "miles", required_modules=("module_that_does_not_exist",), required_env=("MISSING",)
    ).probe()
    assert report.status == "gated"
    assert not report.ready
    assert "missing module" in report.blockers[0]
    assert "missing environment" in report.blockers[1]


def test_miles_run_spec_is_platform_explicit():
    cuda = build_miles_native_spec(platform="cuda", strict=True)
    rocm = build_miles_native_spec(platform="rocm")
    assert cuda.command[-1] == "native"
    assert cuda.environment["RL_KERNEL_STRICT"] == "true"
    assert rocm.environment["RL_KERNEL_PLATFORM"] == "rocm"
    with pytest.raises(RuntimeContractError, match="cuda or rocm"):
        build_miles_native_spec(platform="cpu")
