# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from rl_engine.bi import Benchmark, RuntimeContext, runtime
from rl_engine.bi.builtin import QWEN3_8B, builtin_catalog
from rl_engine.bi.catalog import canonical, fingerprint


@pytest.fixture
def context():
    return RuntimeContext(
        model_id="qwen3-8b",
        framework="vime",
        platform="cuda",
        hardware=("NVIDIA H100:sm90:80000000000",) * 8,
        topology=(8, 4, 2, 4, 1),
        model_config=QWEN3_8B.config_match,
        runtime=canonical({"torch": "pinned"}),
        workload=canonical({"dtype": "bf16", "batch": 8}),
        route_environment="{}",
    )


def candidate_catalog(context, *, reference_seconds=12.0, candidate_seconds=10.0):
    catalog = builtin_catalog()
    reference, _ = catalog.select(context)
    candidate = replace(
        reference,
        plan_id="qwen3-8b.vime.cuda.reviewed-upstream.v1",
        reference=False,
        environment=tuple(
            (key, "reviewed-fast" if key == "RL_KERNEL_DET_GEMM_BACKEND" else value)
            for key, value in reference.environment
        ),
        evidence=("test-only/WS1-WS2-E2E.json",),
    )
    catalog.register_plan(candidate)
    benchmark = Benchmark(
        context_digest=context.digest,
        report="test-only/comparison.json",
        timings=((reference.digest, reference_seconds), (candidate.digest, candidate_seconds)),
        ws1_passed=True,
        ws2_passed=True,
        e2e_mismatch_count=0,
        e2e_max_abs_diff=0.0,
        compared_elements=4096,
    )
    return catalog, reference, candidate, benchmark


@pytest.mark.parametrize(
    "seconds,winner", [(10, "candidate"), (12, "reference"), (14, "reference")]
)
def test_selects_only_comparable_e2e_winner(context, seconds, winner):
    catalog, reference, candidate, benchmark = candidate_catalog(context, candidate_seconds=seconds)
    catalog.register_benchmark(benchmark)
    selected, reason = catalog.select(context)
    assert selected == {"reference": reference, "candidate": candidate}[winner]
    assert benchmark.report in reason


def test_no_evidence_does_not_activate_candidate(context):
    catalog, reference, _, _ = candidate_catalog(context)
    assert catalog.select(context)[0] == reference


@pytest.mark.parametrize(
    "change",
    [
        {"workload": '{"batch":16}'},
        {"runtime": '{"torch":"upgraded"}'},
        {"hardware": ("NVIDIA H200",) * 8},
        {"topology": (8, 2, 4, 4, 1)},
        {"model_config": '{"rope_theta":500000}'},
        {"route_environment": '{"RL_KERNEL_DET_GEMM_BACKEND":"changed"}'},
    ],
)
def test_changed_context_cannot_reuse_performance_evidence(context, change):
    catalog, reference, _, benchmark = candidate_catalog(context)
    catalog.register_benchmark(benchmark)
    assert catalog.select(replace(context, **change))[0] == reference


@pytest.mark.parametrize(
    "change",
    [
        {"ws1_passed": False},
        {"ws2_passed": False},
        {"e2e_mismatch_count": 1},
        {"e2e_max_abs_diff": 1e-10},
        {"compared_elements": 0},
        {"report": ""},
    ],
)
def test_rejects_incomplete_or_failed_consistency_evidence(context, change):
    catalog, _, _, benchmark = candidate_catalog(context)
    with pytest.raises(ValueError, match="equality evidence"):
        catalog.register_benchmark(replace(benchmark, **change))


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_timings(context, seconds):
    catalog, _, _, benchmark = candidate_catalog(context, candidate_seconds=seconds)
    with pytest.raises(ValueError, match="finite and positive"):
        catalog.register_benchmark(benchmark)


def test_modified_candidate_invalidates_evidence(context):
    catalog, _, candidate, benchmark = candidate_catalog(context)
    catalog.plans[candidate.plan_id] = replace(candidate, environment=())
    with pytest.raises(ValueError, match="modified execution plan"):
        catalog.register_benchmark(benchmark)


def test_cannot_mix_benchmark_cohorts(context):
    catalog, _, _, benchmark = candidate_catalog(context)
    catalog.register_benchmark(benchmark)
    with pytest.raises(ValueError, match="cohorts"):
        catalog.register_benchmark(replace(benchmark, report="another-run.json"))


def test_model_detection_uses_config_not_directory_name():
    catalog = builtin_catalog()
    config = json.loads(QWEN3_8B.config_match)
    assert catalog.identify(config) == "qwen3-8b"
    for change in ({"hidden_size": 2048}, {"model_type": "gemma"}, {"vocab_size": 128000}):
        with pytest.raises(ValueError, match="supported model"):
            catalog.identify({**config, **change})


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_ACTIVE", None)
    monkeypatch.delenv(runtime.PLAN_ENV, raising=False)
    monkeypatch.setenv("RL_KERNEL_BI", "1")
    config = json.loads(QWEN3_8B.config_match)
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(runtime, "runtime_identity", lambda repos: {"runtime": "pinned"})
    monkeypatch.setattr(
        runtime, "hardware_identity", lambda: ("cuda", ("NVIDIA H100:sm90:80GB",) * 8)
    )
    kwargs = dict(
        model_root=tmp_path,
        platform="cuda",
        topology=(8, 4, 2, 4, 1),
        workload={"dtype": "bf16", "batch": 8},
        repositories={},
        caller_environment={"RL_KERNEL_BI": "1"},
    )
    return {}, kwargs


def install_environment(monkeypatch, environment):
    for key, value in environment.items():
        monkeypatch.setenv(key, value)


def test_off_is_a_noop_without_model_or_gpu(launch):
    env, kwargs = launch
    kwargs["model_root"] = kwargs["model_root"] / "does-not-exist"
    kwargs["caller_environment"] = {}
    assert runtime.prepare_vime_environment(env, **kwargs) is None
    assert env == {}


@pytest.mark.parametrize("value", ["", "yes", "typo", "2"])
def test_flag_typos_fail(value):
    with pytest.raises(ValueError, match="RL_KERNEL_BI"):
        runtime.enabled({"RL_KERNEL_BI": value})


@pytest.mark.parametrize("conflict", [{"RL_KERNEL_MODE": "auto"}, {"RL_KERNEL_FFN_CASE": "P/P"}])
def test_conflicting_user_settings_fail(launch, conflict):
    env, kwargs = launch
    kwargs["caller_environment"].update(conflict)
    with pytest.raises(ValueError):
        runtime.prepare_vime_environment(env, **kwargs)
    assert env == {}


def test_native_runner_cannot_claim_bi(launch):
    env, kwargs = launch
    env["RL_KERNEL_ATTENTION_CASE"] = "P/P"
    with pytest.raises(ValueError, match="native/mixed"):
        runtime.prepare_vime_environment(env, **kwargs)


@pytest.mark.parametrize("change", ["topology", "gpu", "model", "quantization", "adapter"])
def test_unsupported_scope_fails_before_export(launch, monkeypatch, change):
    env, kwargs = launch
    if change == "topology":
        kwargs["topology"] = (8, 2, 4, 4, 1)
    elif change == "gpu":
        monkeypatch.setattr(runtime, "hardware_identity", lambda: ("cuda", ("NVIDIA H200",) * 8))
    elif change in {"model", "quantization"}:
        config = json.loads(QWEN3_8B.config_match)
        config.update(
            {"model_type": "gemma"} if change == "model" else {"quantization_config": {"bits": 4}}
        )
        (kwargs["model_root"] / "config.json").write_text(json.dumps(config))
    else:
        catalog = builtin_catalog()
        for key, plan in list(catalog.plans.items()):
            catalog.plans[key] = replace(plan, adapter="unimplemented")
        monkeypatch.setattr(runtime, "builtin_catalog", lambda: catalog)
    with pytest.raises(ValueError):
        runtime.prepare_vime_environment(env, **kwargs)
    assert env == {}


def test_plan_reaches_workers_and_readback(launch, monkeypatch):
    env, kwargs = launch
    record = runtime.prepare_vime_environment(env, **kwargs)
    install_environment(monkeypatch, env)
    assert runtime.verify_worker_plan(check_hardware=True)["digest"] == record["digest"]
    assert runtime.active_plan_readback()["plan_id"].endswith("cuda.reference.v1")
    # Ray worker sees a subset of the parent's GPUs.
    monkeypatch.setattr(runtime, "hardware_identity", lambda: ("cuda", ("NVIDIA H100:sm90:80GB",)))
    runtime.verify_worker_plan(check_hardware=True)


def test_rocm_selects_rocm_reference(launch, monkeypatch):
    env, kwargs = launch
    kwargs["platform"] = "rocm"
    monkeypatch.setattr(
        runtime, "hardware_identity", lambda: ("rocm", ("AMD MI300X:gfx942:192GB",) * 8)
    )
    record = runtime.prepare_vime_environment(env, **kwargs)
    install_environment(monkeypatch, env)
    assert runtime.verify_worker_plan(check_hardware=True)["plan_id"].endswith("rocm.reference.v1")
    assert record["context"]["platform"] == "rocm"


def test_upstream_candidate_changes_actual_exported_route(launch, monkeypatch):
    env, kwargs = launch
    reference_record = runtime.prepare_vime_environment(env, **kwargs)
    context = RuntimeContext(**reference_record["context"])
    catalog, _, candidate, benchmark = candidate_catalog(context)
    catalog.register_benchmark(benchmark)
    monkeypatch.setattr(runtime, "builtin_catalog", lambda: catalog)
    env.clear()
    record = runtime.prepare_vime_environment(env, **kwargs)
    assert record["plan_id"] == candidate.plan_id
    assert env["RL_KERNEL_DET_GEMM_BACKEND"] == "reviewed-fast"
    install_environment(monkeypatch, env)
    assert runtime.verify_worker_plan()["plan_digest"] == candidate.digest


def test_frontend_plugin_verification_does_not_initialize_cuda(launch, monkeypatch):
    env, kwargs = launch
    runtime.prepare_vime_environment(env, **kwargs)
    install_environment(monkeypatch, env)

    def forbidden():
        raise AssertionError("frontend must not initialize CUDA")

    monkeypatch.setattr(runtime, "hardware_identity", forbidden)
    runtime.verify_worker_plan()


@pytest.mark.parametrize("change", ["env", "model", "runtime", "hardware", "digest"])
def test_worker_rejects_changed_launch_contract(launch, monkeypatch, change):
    env, kwargs = launch
    record = runtime.prepare_vime_environment(env, **kwargs)
    install_environment(monkeypatch, env)
    if change == "env":
        monkeypatch.setenv("RL_KERNEL_LOGP_CASE", "P/P")
    elif change == "model":
        (kwargs["model_root"] / "config.json").write_text("{}")
    elif change == "runtime":
        monkeypatch.setattr(runtime, "runtime_identity", lambda repos: {"runtime": "upgraded"})
    elif change == "hardware":
        monkeypatch.setattr(runtime, "hardware_identity", lambda: ("cuda", ("NVIDIA H200",)))
    else:
        record["plan_digest"] = "changed"
        monkeypatch.setenv(runtime.PLAN_ENV, canonical(record))
    with pytest.raises((RuntimeError, ValueError)):
        runtime.verify_worker_plan(check_hardware=True)


def test_cannot_disable_or_hot_swap_active_plan(launch, monkeypatch):
    env, kwargs = launch
    record = runtime.prepare_vime_environment(env, **kwargs)
    install_environment(monkeypatch, env)
    runtime.verify_worker_plan()
    monkeypatch.setenv("RL_KERNEL_BI", "0")
    with pytest.raises(RuntimeError, match="disabled"):
        runtime.verify_worker_plan()
    monkeypatch.setenv("RL_KERNEL_BI", "1")
    record["reason"] = "changed"
    monkeypatch.setenv(runtime.PLAN_ENV, canonical(record))
    with pytest.raises(RuntimeError, match="restart"):
        runtime.verify_worker_plan()


def test_unprepared_one_flag_is_actionable(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_BI", "1")
    monkeypatch.delenv(runtime.PLAN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="rlk run"):
        runtime.verify_worker_plan()


def test_plan_envelope_cannot_override_reviewed_environment(launch, monkeypatch):
    env, kwargs = launch
    record = runtime.prepare_vime_environment(env, **kwargs)
    record["environment"]["RL_KERNEL_MODE"] = "off"
    record.pop("digest")
    record["digest"] = fingerprint(record)
    env[runtime.PLAN_ENV] = canonical(record)
    env["RL_KERNEL_MODE"] = "off"
    install_environment(monkeypatch, env)
    with pytest.raises(RuntimeError, match="catalog plan"):
        runtime.verify_worker_plan()
