# SPDX-License-Identifier: Apache-2.0
"""INK-23/34/71 sequence contracts executed on the ROCm HIP device (MI300X)."""

import pytest
import torch

from rl_engine.models.inkling2.sequence_contracts import LongContextPlan, causal_position_mask
from rl_engine.models.inkling2.sequence_contracts_rocm import (
    GIB,
    InklingDims,
    RocmDeviceInfo,
    assess_long_context_plan,
    device_cpu_parity,
    estimate_long_context_memory,
    read_rocm_device,
    recompute_matches_plain_on_device,
    sequence_contract_results,
)

HIP = getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()
needs_hip = pytest.mark.skipif(not HIP, reason="ROCm GPU is required")


# ---------------------------------------------------------------- readback
@needs_hip
def test_device_readback_is_real():
    info = read_rocm_device()
    props = torch.cuda.get_device_properties(0)
    assert info.is_rocm and info.hip_version == torch.version.hip
    assert info.gcn_arch_name == props.gcnArchName and info.gcn_arch_name.startswith("gfx")
    assert info.total_memory_bytes == props.total_memory
    assert 0 < info.free_memory_bytes <= info.total_memory_bytes
    assert info.device_count == torch.cuda.device_count()


def test_readback_without_rocm_is_empty_and_gates_everything():
    none = RocmDeviceInfo(None, 0, "", "", 0, 0)
    assert not none.is_rocm
    reports = assess_long_context_plan(info=none, probe=False)
    assert [r.sequence_length for r in reports] == [4096, 32768, 131072, 1048576]
    assert all(r.status == "gated" and "no ROCm device" in r.reason for r in reports)


# ---------------------------------------------------------------- device parity
@needs_hip
def test_sequence_contracts_bitwise_parity_device_vs_cpu():
    assert device_cpu_parity("cuda") == ()


@needs_hip
def test_device_results_have_expected_values_and_are_run_to_run_identical():
    first = sequence_contract_results("cuda")
    second = sequence_contract_results("cuda")
    assert first.keys() == second.keys()
    for name in first:
        assert torch.equal(first[name], second[name]), name
    assert first["ranges_131077_4096"].shape == (33, 2)
    assert first["ranges_131077_4096"][-1].tolist() == [131072, 131077]
    assert first["ranges_1_4096"].tolist() == [[0, 1]]
    assert first["decode_pos_step4"][4].tolist() == [1048575, 1048576, 1048577, 1048578]
    assert first["packed_offsets"].tolist() == [0, 1, 6, 70, 135, 265]
    assert first["packed_position_ids"][:7].tolist() == [0, 0, 1, 2, 3, 4, 0]
    assert first["mask_window_1"].sum().item() == 7  # only key == query survives window=1
    assert first["schedule"][-1].tolist() == [4, 2, 14]
    # The local mask keeps exactly min(i + 1, 128) keys per query row.
    counts = first["mask_local_300"][0].sum(dim=1)
    assert counts.tolist() == [min(i + 1, 128) for i in range(300)]
    assert first["invariance_ok"].item() == 1


@needs_hip
def test_unpacked_rows_equal_original_rows_on_device():
    results = sequence_contract_results("cuda")
    packed = results["packed_values"]
    offsets = results["packed_offsets"].tolist()
    for i in range(5):
        assert torch.equal(results[f"unpacked_{i}"], packed[offsets[i] : offsets[i + 1]])


@needs_hip
def test_recompute_with_dropout_rng_matches_plain_run_on_device():
    assert recompute_matches_plain_on_device("cuda")


@needs_hip
def test_causal_mask_batch_row_invariance_on_device():
    pos = torch.arange(200, device="cuda")[None].repeat(3, 1)
    full = causal_position_mask(pos, pos, window=17)
    other = pos.clone()
    other[0] = other[0].flip(0)  # change an unrelated batch row
    changed = causal_position_mask(other, other, window=17)
    assert torch.equal(full[1:], changed[1:])
    assert not torch.equal(full[0], changed[0])


# ---------------------------------------------------------------- long context
def test_memory_formula_matches_hand_computation():
    dims = InklingDims(physical_layers=32)
    est = estimate_long_context_memory(4096, dims)
    assert est.kv_cache == 32 * 4096 * 2 * 8 * 128 * 2
    assert est.boundary_activations == 32 * 4096 * 2304 * 2
    assert est.layer_work == 4096 * (4 * 2304 + (24 + 16) * 128 + 3 * 8 * 1152) * 2
    assert est.mask_block == 4096 * 4096
    assert est.dense_mask == 4096 * 4096
    assert est.required == est.kv_cache + est.boundary_activations + est.layer_work + est.mask_block
    big = estimate_long_context_memory(1048576, dims)
    assert big.dense_mask == 1 << 40  # 1 TiB: dense mask materialization is impossible
    assert big.mask_block == 4096 * 1048576


@needs_hip
def test_formula_matches_real_allocations_at_4k():
    dims = InklingDims()
    est = estimate_long_context_memory(4096, dims)
    kv = torch.empty(
        dims.physical_layers,
        2,
        4096,
        dims.kv_heads,
        dims.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert kv.numel() * kv.element_size() == est.kv_cache
    pos = torch.arange(4096, device="cuda")[None]
    mask = causal_position_mask(pos, pos)
    assert mask.numel() * mask.element_size() == est.dense_mask
    del kv, mask


@needs_hip
def test_long_context_plan_on_real_device_is_honest():
    info = read_rocm_device()
    reports = assess_long_context_plan(info=info)
    assert [r.sequence_length for r in reports] == list(LongContextPlan().context_lengths)
    budget = int(info.free_memory_bytes * 0.90)
    for r in reports:
        assert r.budget_bytes == budget and r.assumed_layers == 32
        assert any("not a measurement" in n for n in r.notes)
        assert any("dense causal_position_mask" in n for n in r.notes)
        if r.estimate.required > budget:
            assert r.status == "gated" and "GiB" in r.reason
            assert r.min_devices_context_parallel >= 2 and not r.probe_allocated
        else:
            assert r.status == "ready" and r.probe_allocated
        assert r.ready == (r.status == "ready")
    by_len = {r.sequence_length: r for r in reports}
    # 4K always fits a data-center GPU; required memory is monotone in length.
    assert by_len[4096].ready
    required = [r.estimate.required for r in reports]
    assert required == sorted(required)
    # 1M context at 32 assumed layers is ~360 GiB: gated on a single MI300X (192 GB).
    if info.total_memory_bytes < 300 * GIB:
        assert not by_len[1048576].ready
        assert "context parallelism" in by_len[1048576].reason


def test_plan_status_follows_budget_with_synthetic_devices():
    small = RocmDeviceInfo("7.2", 1, "x", "gfx942", 8 * GIB, 8 * GIB)
    assert all(not r.ready for r in assess_long_context_plan(info=small, probe=False)[1:])
    huge = RocmDeviceInfo("7.2", 1, "x", "gfx942", 4096 * GIB, 4096 * GIB)
    reports = assess_long_context_plan(info=huge, probe=False)
    assert all(r.ready and not r.probe_allocated for r in reports)
    with pytest.raises(ValueError, match="safety_fraction"):
        assess_long_context_plan(info=huge, safety_fraction=0.0)
