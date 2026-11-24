# SPDX-License-Identifier: Apache-2.0
"""ROCm topology gates (INK-36/37/38/39) with a real multi-GPU RCCL check."""

import pytest
import torch

from rl_engine.models.inkling2.topology_contracts import InklingTopology
from rl_engine.models.inkling2.topology_contracts_rocm import (
    GATED_ARCH_REASON,
    RocmReadback,
    assess_deferred_parallelism_rocm,
    qualify_tp_ep_rocm,
    read_rocm_runtime,
    run_rccl_rank_order_check,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires a ROCm build of torch and a HIP device",
)


def _fake(arch, count=8):
    return RocmReadback("7.2", count, (arch,) * count, True, (2, 27, 7))


def test_runtime_readback_is_real():
    rb = read_rocm_runtime()
    assert rb.hip_version == torch.version.hip
    assert rb.device_count == torch.cuda.device_count() >= 1
    assert all(a.startswith("gfx") for a in rb.arch_names)
    assert rb.rccl_available and rb.nccl_version is not None
    assert torch.distributed.is_nccl_available()


def test_local_host_gating_follows_arch():
    rb = read_rocm_runtime()
    size = min(4, rb.device_count)
    if size not in (2, 4):
        pytest.skip("needs 2 or 4 visible devices for a TP2/TP4 topology")
    report = qualify_tp_ep_rocm(InklingTopology(tensor_parallel=size, world_size=size), rb)
    if rb.base_arch == "gfx942":
        assert report.status == "gated"
        assert any(GATED_ARCH_REASON in b for b in report.blockers)
    assert not report.ready or rb.base_arch == "gfx950"


def test_mi300x_is_gated_for_recipe():
    tp4 = InklingTopology(tensor_parallel=4, world_size=4)
    report = qualify_tp_ep_rocm(tp4, _fake("gfx942:sramecc+:xnack-"), expert_parallel=4)
    assert report.status == "gated" and not report.ready
    assert any("MI300X" in b for b in report.blockers)
    assert report.evidence.platform == "rocm"


def test_mi355x_tp4_ep4_qualifies_and_other_shapes_are_gated():
    rb = _fake("gfx950")
    tp4 = InklingTopology(tensor_parallel=4, world_size=4)
    assert qualify_tp_ep_rocm(tp4, rb, expert_parallel=4).ready
    assert qualify_tp_ep_rocm(tp4, rb, expert_parallel=2).status == "gated"
    tp2 = InklingTopology(tensor_parallel=2, world_size=2)
    assert qualify_tp_ep_rocm(tp2, rb, expert_parallel=2).status == "gated"
    assert qualify_tp_ep_rocm(tp4, _fake("gfx950", count=2), expert_parallel=4).status == "gated"
    no_rccl = RocmReadback("7.2", 8, ("gfx950",) * 8, False, None)
    assert qualify_tp_ep_rocm(tp4, no_rccl, expert_parallel=4).status == "gated"
    bad = InklingTopology(tensor_parallel=2, expert_parallel=2, world_size=2)
    assert qualify_tp_ep_rocm(bad, rb).status == "rejected"
    assert qualify_tp_ep_rocm(tp4, rb, expert_parallel=3).status == "gated"


def test_cp_sp_pp_remain_deferred_on_rocm():
    for item, topology in (
        ("INK-37", InklingTopology(context_parallel=2, world_size=2)),
        ("INK-38", InklingTopology(sequence_parallel=2, world_size=2)),
        ("INK-39", InklingTopology(pipeline_parallel=2, world_size=2)),
    ):
        report = assess_deferred_parallelism_rocm(item, topology)
        assert report.status == "gated"
        assert any("ROCm" in b for b in report.blockers) and len(report.blockers) >= 2


@pytest.mark.parametrize("world_size", [2, 4])
def test_rccl_all_gather_fixed_rank_order_sum_is_bitwise_identical(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} visible HIP devices (use GPUS=4,5,6,7)")
    out = run_rccl_rank_order_check(world_size, repeats=2)
    assert not out["errors"], out["errors"]
    assert sorted(out["digests"]) == list(range(world_size))
    for rank, digests in out["digests"].items():
        # identical across ranks and repeats, and equal to the CPU rank-order reference
        assert digests == out["cpu_reference"], rank
