# SPDX-License-Identifier: Apache-2.0
"""ROCm counterpart of the Inkling 2 Mini DSpark/MTP profile (INK-73).

ROCm-only.  The speculative profile is optional: ``base_rocm_bf16_gate`` takes no
profile argument and reads only the HIP runtime, so enabling or disabling DSpark
cannot change it.  ``run_speculative_parity`` executes acceptance, EOS handling and
KV-cache rollback on the HIP device and compares to CPU bitwise (integer tokens and
pure copies, so exact equality is the contract).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rl_engine.models.inkling2.speculative_contracts import (
    SpeculativeContractError,
    SpeculativeProfile,
    accept_speculative_tokens,
    commit_speculative_cache,
)

# Inkling KV cache geometry: 8 KV heads, head dim 128.
KV_HEADS = 8
HEAD_DIM = 128


@dataclass(frozen=True)
class BaseRocmGate:
    hip: str | None
    device_count: int
    archs: tuple[str, ...]
    bf16_supported: bool
    bf16_device_op_ok: bool

    @property
    def ready(self) -> bool:
        return (
            self.hip is not None
            and self.device_count > 0
            and self.bf16_supported
            and self.bf16_device_op_ok
        )


def base_rocm_bf16_gate() -> BaseRocmGate:
    """Base ROCm BF16 gate.  Deliberately independent of any speculative profile."""
    hip = torch.version.hip
    if hip is None or not torch.cuda.is_available():
        return BaseRocmGate(hip, 0, (), False, False)
    count = torch.cuda.device_count()
    archs = tuple(torch.cuda.get_device_properties(i).gcnArchName for i in range(count))
    x = torch.arange(8, device="cuda", dtype=torch.float32).bfloat16()
    ok = bool(torch.equal((x + x).float().cpu(), (x.cpu() + x.cpu()).float()))
    return BaseRocmGate(hip, count, archs, bool(torch.cuda.is_bf16_supported()), ok)


@dataclass(frozen=True)
class ProfileGateReport:
    profile: str
    status: str  # "disabled" | "ready" | "gated"
    blockers: tuple[str, ...] = ()


def rocm_dspark_profile_gate(profile: SpeculativeProfile) -> ProfileGateReport:
    """Optional profile gate: a disabled profile never blocks; enabled needs base gate."""
    profile.validate()
    if not profile.enabled:
        return ProfileGateReport(profile.name, "disabled")
    if not base_rocm_bf16_gate().ready:
        return ProfileGateReport(profile.name, "gated", ("base ROCm BF16 gate is not ready",))
    return ProfileGateReport(profile.name, "ready")


@dataclass(frozen=True)
class SpeculativeParity:
    accepted_tokens_equal: bool
    accepted_lengths_equal: bool
    rollback_lengths_equal: bool
    eos_cut_equal: bool
    cache_commit_bitwise: bool
    cache_commit_shape: tuple[int, ...]
    device_run_to_run_equal: bool


def _cache_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.cpu().view(torch.uint8), b.cpu().view(torch.uint8))
    )


def run_speculative_parity(*, seed: int = 0, batch: int = 16, steps: int = 4) -> SpeculativeParity:
    if torch.version.hip is None or not torch.cuda.is_available():
        raise SpeculativeContractError("a ROCm build of torch with a HIP device is required")
    gen = torch.Generator().manual_seed(seed)
    eos = 2
    target = torch.randint(3, 201024, (batch, steps), generator=gen, dtype=torch.int64)
    draft = target.clone()
    # Row r diverges at step r % (steps + 1); a divergence at ``steps`` means fully accepted.
    for row in range(batch):
        cut = row % (steps + 1)
        if cut < steps:
            draft[row, cut] += 1
    # Plant an EOS at step 1 on rows 1, 6, 11, ... (re-aligned): accepted, then acceptance stops.
    for row in range(1, batch, 5):
        draft[row, 1] = eos
        target[row, 1] = eos

    cpu = accept_speculative_tokens(draft, target, eos_token_id=eos)
    dev = accept_speculative_tokens(draft.cuda(), target.cuda(), eos_token_id=eos)
    dev2 = accept_speculative_tokens(draft.cuda(), target.cuda(), eos_token_id=eos)
    plain_cpu = accept_speculative_tokens(draft, target)
    plain_dev = accept_speculative_tokens(draft.cuda(), target.cuda())

    cache = torch.randn(batch, steps, KV_HEADS, HEAD_DIM, generator=gen).bfloat16()
    cache_cpu = commit_speculative_cache(cache, cpu)
    cache_dev = commit_speculative_cache(cache.cuda(), dev)

    return SpeculativeParity(
        accepted_tokens_equal=torch.equal(cpu.accepted_tokens, dev.accepted_tokens.cpu()),
        accepted_lengths_equal=torch.equal(cpu.accepted_lengths, dev.accepted_lengths.cpu()),
        rollback_lengths_equal=torch.equal(cpu.rollback_lengths, dev.rollback_lengths.cpu()),
        eos_cut_equal=(
            torch.equal(plain_cpu.accepted_lengths, plain_dev.accepted_lengths.cpu())
            and bool((cpu.accepted_lengths <= plain_cpu.accepted_lengths).all())
            and bool((cpu.accepted_lengths < plain_cpu.accepted_lengths).any())
        ),
        cache_commit_bitwise=_cache_equal(cache_cpu, cache_dev),
        cache_commit_shape=tuple(cache_dev.shape),
        device_run_to_run_equal=(
            torch.equal(dev.accepted_tokens, dev2.accepted_tokens)
            and torch.equal(dev.accepted_lengths, dev2.accepted_lengths)
        ),
    )
