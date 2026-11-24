# SPDX-License-Identifier: Apache-2.0
"""ROCm runtime gates (INK-55/57/58/60/74)."""

import importlib.util

import pytest
import torch

from rl_engine.models.inkling2.runtime_contracts import RuntimeContractError
from rl_engine.models.inkling2.runtime_contracts_rocm import (
    GATED_ARCH_REASON,
    build_miles_rocm_spec,
    device_archs,
    probe_external_modules,
    rocm_arch_gate,
    rocm_full_gate,
    run_device_sync_parity,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a ROCm build of torch and a HIP device",
)


def test_arch_gate_is_arch_aware():
    mi355 = rocm_arch_gate(("gfx950",))
    assert mi355.ready and not mi355.blockers
    mi300 = rocm_arch_gate(("gfx942:sramecc+:xnack-",))
    assert mi300.status == "gated" and mi300.blockers == (GATED_ARCH_REASON,)
    assert not rocm_arch_gate(("gfx90a",)).ready
    assert not rocm_arch_gate(("gfx950", "gfx942")).ready
    assert not rocm_arch_gate(()).ready


def test_local_device_gate_matches_readback():
    archs = device_archs()
    assert archs and all(a.startswith("gfx") for a in archs)
    report = rocm_arch_gate()
    assert report.evidence["archs"] == list(archs)
    if archs[0].startswith("gfx942"):
        assert GATED_ARCH_REASON in report.blockers
    full = rocm_full_gate()
    assert not full.ready  # miles/sglang/torchtitan and their env are not provisioned here


def test_external_modules_reported_honestly():
    report = probe_external_modules()
    for name, present in report.evidence["modules"].items():
        assert present == (importlib.util.find_spec(name) is not None)
        assert present or f"missing module {name}" in report.blockers
    assert report.ready == all(report.evidence["modules"].values())


def test_miles_rocm_spec_environment():
    spec = build_miles_rocm_spec(strict=True, visible_devices="4, 5,6")
    assert spec.platform == "rocm" and spec.strict
    env = spec.environment
    assert env["ROCM_QUICK_REDUCE_QUANTIZATION"] == "NONE"
    assert env["RL_KERNEL_ATTENTION_BACKEND"] == "triton"
    assert env["RL_KERNEL_PLATFORM"] == "rocm" and env["RL_KERNEL_STRICT"] == "true"
    assert env["HIP_VISIBLE_DEVICES"] == "4,5,6"
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert spec.command[-2:] == ("--attention-backend", "triton")
    assert '"HIP_VISIBLE_DEVICES": "4,5,6"' in spec.to_json()


def test_miles_rocm_spec_inherits_or_omits_hip_visible_devices():
    inherited = build_miles_rocm_spec(inherit_environment={"HIP_VISIBLE_DEVICES": "7"})
    assert inherited.environment["HIP_VISIBLE_DEVICES"] == "7"
    assert "HIP_VISIBLE_DEVICES" not in build_miles_rocm_spec(inherit_environment={}).environment
    explicit = build_miles_rocm_spec(
        visible_devices="1", inherit_environment={"HIP_VISIBLE_DEVICES": "7"}
    )
    assert explicit.environment["HIP_VISIBLE_DEVICES"] == "1"
    for bad in ("a", "1,1", "-1", "1,,2"):
        with pytest.raises(RuntimeContractError, match="HIP_VISIBLE_DEVICES"):
            build_miles_rocm_spec(visible_devices=bad)


def test_weight_sync_mapping_and_route_replay_on_hip_device_match_cpu():
    report = run_device_sync_parity("cuda")
    assert report.device == "cuda"
    assert report.cpu_fingerprint == report.device_fingerprint
    assert report.route_digest_cpu == report.route_digest_device
    assert report.equal
    assert report.events == ("reset", "replay")
    assert report.mapped_keys[0] == "model.layers.0.attn.q_proj.weight"
    again = run_device_sync_parity("cuda")
    assert again.device_fingerprint == report.device_fingerprint
    assert run_device_sync_parity("cuda", seed=1).device_fingerprint != report.device_fingerprint
