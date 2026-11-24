# SPDX-License-Identifier: Apache-2.0
"""ROCm DSpark/MTP profile parity (INK-73)."""

import pytest
import torch

from rl_engine.models.inkling2.speculative_contracts import (
    SpeculativeContractError,
    SpeculativeProfile,
    accept_speculative_tokens,
    commit_speculative_cache,
)
from rl_engine.models.inkling2.speculative_contracts_rocm import (
    base_rocm_bf16_gate,
    rocm_dspark_profile_gate,
    run_speculative_parity,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a ROCm build of torch and a HIP device",
)


def test_acceptance_eos_and_cache_rollback_match_cpu_on_hip():
    report = run_speculative_parity()
    assert report.accepted_tokens_equal
    assert report.accepted_lengths_equal
    assert report.rollback_lengths_equal
    assert report.eos_cut_equal
    assert report.cache_commit_bitwise
    assert report.device_run_to_run_equal
    assert report.cache_commit_shape[0] == 16 and report.cache_commit_shape[2:] == (8, 128)
    other = run_speculative_parity(seed=3, batch=7, steps=6)
    assert other.accepted_lengths_equal and other.cache_commit_bitwise


def test_acceptance_awkward_shapes_and_dtypes_on_hip():
    for dtype in (torch.int32, torch.int64):
        draft = torch.tensor([[7]], dtype=dtype)
        for target_value, expected in ((7, 1), (8, 0)):
            target = torch.tensor([[target_value]], dtype=dtype)
            cpu = accept_speculative_tokens(draft, target)
            dev = accept_speculative_tokens(draft.cuda(), target.cuda())
            assert int(dev.accepted_lengths) == int(cpu.accepted_lengths) == expected
            assert int(dev.rollback_lengths) == 1 - expected
    empty = accept_speculative_tokens(
        torch.empty(0, 4, dtype=torch.int64).cuda(), torch.empty(0, 4, dtype=torch.int64).cuda()
    )
    assert empty.accepted_lengths.numel() == 0
    with pytest.raises(SpeculativeContractError, match="shape and device"):
        accept_speculative_tokens(torch.zeros(1, 2, dtype=torch.int64).cuda(), torch.zeros(1, 2))


def test_cache_commit_zeroes_rejected_tail_on_device():
    draft = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
    target = torch.tensor([[1, 9, 3], [4, 5, 6]]).cuda()
    result = accept_speculative_tokens(draft, target)
    cache = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).bfloat16().cuda()
    committed = commit_speculative_cache(cache, result)
    assert committed.shape == (2, 3, 4)
    assert torch.equal(committed[0, 1:].float().cpu(), torch.zeros(2, 4))
    assert torch.equal(committed[1], cache[1])
    with pytest.raises(SpeculativeContractError, match="exceed"):
        commit_speculative_cache(cache[:, :2], result)


def test_profile_is_optional_and_does_not_affect_base_bf16_gate():
    before = base_rocm_bf16_gate()
    assert before.ready and before.hip == torch.version.hip
    assert before.device_count == torch.cuda.device_count()
    off = SpeculativeProfile()
    on = SpeculativeProfile(enabled=True)
    assert rocm_dspark_profile_gate(off).status == "disabled"
    assert rocm_dspark_profile_gate(on).status == "ready"
    run_speculative_parity()
    assert base_rocm_bf16_gate() == before
    with pytest.raises(SpeculativeContractError, match="rollback"):
        rocm_dspark_profile_gate(SpeculativeProfile(enabled=True, rollback_cache=False))
    assert base_rocm_bf16_gate() == before
