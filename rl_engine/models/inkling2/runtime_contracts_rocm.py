# SPDX-License-Identifier: Apache-2.0
"""ROCm counterpart of the Inkling 2 Mini runtime gates (INK-55/57/58/60/74).

ROCm-only.  Provides an architecture-aware gate (gfx950 eligible for the SGLang
Inkling recipe, gfx942/MI300X gated), the ROCm Miles run spec environment,
honest probes for the external modules (miles, sglang, torchtitan), and a
device-parity run of versioned weight sync, key mapping and route replay with
tensors on the HIP device whose fingerprints must equal the CPU ones.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.models.inkling2.runtime_contracts import (
    CheckpointExport,
    MilesRunSpec,
    RouteReplayRecord,
    RuntimeContractError,
    RuntimeGate,
    RuntimeGateReport,
    VersionedWeightSync,
    WeightKeyMapping,
    build_miles_native_spec,
)

RECIPE_ARCHES = ("gfx950",)
GATED_ARCH_REASON = (
    "SGLang Inkling2Loop recipe lists MI350X/MI355X only; MI300X needs a separate port"
)
EXTERNAL_MODULES = ("miles", "sglang", "torchtitan")


def base_arch(arch_name: str) -> str:
    """``gfx942:sramecc+:xnack-`` -> ``gfx942``."""
    return arch_name.split(":")[0]


def device_archs() -> tuple[str, ...]:
    if torch.version.hip is None or not torch.cuda.is_available():
        return ()
    return tuple(
        torch.cuda.get_device_properties(i).gcnArchName for i in range(torch.cuda.device_count())
    )


def rocm_arch_gate(archs: tuple[str, ...] | None = None) -> RuntimeGateReport:
    """Arch-aware gate: ``ready`` only when every visible device is gfx950."""
    archs = device_archs() if archs is None else archs
    evidence = {"archs": list(archs), "hip": torch.version.hip}
    if not archs:
        return RuntimeGateReport("INK-74-arch", "gated", ("no HIP device is visible",), evidence)
    families = sorted({base_arch(a) for a in archs})
    blockers = []
    for family in families:
        if family == "gfx942":
            blockers.append(GATED_ARCH_REASON)
        elif family not in RECIPE_ARCHES:
            blockers.append(f"architecture {family} is not covered by the Inkling recipe")
    if len(families) > 1:
        blockers.append(f"mixed architectures are not supported: {families}")
    return RuntimeGateReport(
        "INK-74-arch", "ready" if not blockers else "gated", tuple(blockers), evidence
    )


def probe_external_modules(names: tuple[str, ...] = EXTERNAL_MODULES) -> RuntimeGateReport:
    """Report miles/sglang/torchtitan honestly (missing -> gated, never faked)."""
    return RuntimeGate("INK-74-external", required_modules=names).probe()


def rocm_full_gate(archs: tuple[str, ...] | None = None) -> RuntimeGateReport:
    """Union of arch, external-module and Miles/SGLang environment gates."""
    arch = rocm_arch_gate(archs)
    external = probe_external_modules()
    env = RuntimeGate(
        "INK-74-env", required_env=("MILES_CHECKPOINT", "SGLANG_CONFIG"), require_rocm=True
    ).probe()
    blockers = arch.blockers + external.blockers + env.blockers
    evidence = {
        "arch": dict(arch.evidence),
        "modules": dict(external.evidence["modules"]),
        "environment": dict(env.evidence["environment"]),
    }
    return RuntimeGateReport(
        "INK-74", "ready" if not blockers else "gated", tuple(blockers), evidence
    )


ROCM_ENV = {
    "ROCM_QUICK_REDUCE_QUANTIZATION": "NONE",
    "RL_KERNEL_ATTENTION_BACKEND": "triton",
}


def _validate_visible_devices(value: str) -> str:
    parts = [p.strip() for p in value.split(",")]
    if not parts or any(not p.isdigit() for p in parts) or len(set(parts)) != len(parts):
        raise RuntimeContractError(f"HIP_VISIBLE_DEVICES must be unique indices: {value!r}")
    return ",".join(parts)


def build_miles_rocm_spec(
    *,
    strict: bool = False,
    visible_devices: str | None = None,
    inherit_environment: Mapping[str, str] | None = None,
) -> MilesRunSpec:
    """ROCm Miles spec: no quick-reduce quantization, Triton attention, HIP device list.

    ``visible_devices`` wins; otherwise ``HIP_VISIBLE_DEVICES`` is inherited from
    ``inherit_environment`` (default ``os.environ``) when set.  CUDA_VISIBLE_DEVICES
    is never emitted.
    """
    base = build_miles_native_spec(platform="rocm", strict=strict)
    source = os.environ if inherit_environment is None else inherit_environment
    devices = visible_devices if visible_devices is not None else source.get("HIP_VISIBLE_DEVICES")
    env = {**base.environment, **ROCM_ENV}
    if devices is not None and devices.strip():
        env["HIP_VISIBLE_DEVICES"] = _validate_visible_devices(devices)
    return dataclasses.replace(
        base,
        command=base.command + ("--attention-backend", "triton"),
        environment=env,
    )


@dataclass(frozen=True)
class ParityReport:
    device: str
    cpu_fingerprint: str
    device_fingerprint: str
    mapped_keys: tuple[str, ...]
    route_digest_cpu: str
    route_digest_device: str
    events: tuple[Any, ...] = field(default_factory=tuple)

    @property
    def equal(self) -> bool:
        return (
            self.cpu_fingerprint == self.device_fingerprint
            and self.route_digest_cpu == self.route_digest_device
        )


def _route_records(logits: torch.Tensor, top_k: int) -> list[RouteReplayRecord]:
    values, indices = torch.topk(logits, top_k, dim=-1)
    values, indices = values.float().cpu(), indices.cpu()
    return [
        RouteReplayRecord(
            token_id=token,
            expert_ids=tuple(int(e) for e in indices[token]),
            weights=tuple(float(w) for w in values[token]),
            visit_id=token % 3,
        )
        for token in range(logits.shape[0])
    ]


def _route_digest(records: list[RouteReplayRecord]) -> str:
    return hashlib.sha256(repr(records).encode()).hexdigest()


def run_device_sync_parity(
    device: str = "cuda", *, seed: int = 0, tokens: int = 64, experts: int = 128, top_k: int = 8
) -> ParityReport:
    """Run key mapping, fingerprint, versioned sync and route replay on ``device`` and CPU."""
    gen = torch.Generator().manual_seed(seed)
    cpu_state = {
        "actor.layers.0.attn.q_proj.weight": torch.randn(24 * 128, 2304, generator=gen),
        "actor.layers.0.moe.router.weight": torch.randn(experts, 2304, generator=gen),
        "actor.layers.0.moe.up.weight": torch.randn(1152, 2304, generator=gen).bfloat16(),
        "actor.embed.weight": torch.randn(257, 2304, generator=gen).half(),
    }
    logits = torch.randn(tokens, experts, generator=gen)
    mapping = WeightKeyMapping((("actor.", "model."),))

    def fingerprint(where: str) -> tuple[str, tuple[str, ...]]:
        state = {k: v.to(where) for k, v in cpu_state.items()}
        export = CheckpointExport("miles-r7", state)
        converted = export.convert(mapping)
        assert all(t.device.type == torch.device(where).type for t in converted.values())
        return export.fingerprint(mapping), tuple(converted)

    cpu_fp, keys = fingerprint("cpu")
    dev_fp, dev_keys = fingerprint(device)
    if keys != dev_keys:
        raise RuntimeContractError("mapped key order differs between CPU and device")

    def replay(where: str) -> tuple[str, list[Any]]:
        events: list[Any] = []
        replayed: list[RouteReplayRecord] = []
        sync = VersionedWeightSync(
            reset_cache=lambda: events.append("reset"),
            replay_routes=lambda routes: (replayed.extend(routes), events.append("replay")),
        )
        sync.publish(version=1, fingerprint=cpu_fp if where == "cpu" else dev_fp)
        sync.replay(_route_records(logits.to(where), top_k))
        return _route_digest(replayed), events

    cpu_digest, cpu_events = replay("cpu")
    dev_digest, dev_events = replay(device)
    if cpu_events != dev_events:
        raise RuntimeContractError("sync event order differs between CPU and device")
    return ParityReport(device, cpu_fp, dev_fp, keys, cpu_digest, dev_digest, tuple(dev_events))
