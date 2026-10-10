# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.backends.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPCommunicationUnavailable,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
    RCCLAGRSAttentionCPCommunication,
)
from rl_engine.distributed.algorithms import collectives
from rl_engine.runtime.registry import KernelRegistry, OpBackend


class _FakeDist:
    class group:
        WORLD = "world-group"


def _plan(*, backend: str = "rccl_ag_rs") -> AttentionCPCommunicationPlan:
    return AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend=backend,  # type: ignore[arg-type]
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )


class _FakeRCCLTransport:
    world_size = 2

    def __init__(self) -> None:
        self.gather_calls = 0
        self.scatter_calls = 0

    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        self.gather_calls += 1
        return torch.cat((tensor, tensor), dim=0)

    def scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        self.scatter_calls += 1
        return tensor.chunk(self.world_size, dim=0)[0].contiguous()


def test_rccl_plan_reports_transport_only_runtime() -> None:
    provenance = _plan().provenance()

    assert provenance["cp_comm_runtime"] == "rccl"
    assert provenance["cp_comm_attention_numeric_reduction"] is False


def test_rccl_adapter_uses_root_scatter_and_reports_no_fusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeRCCLTransport()
    communication = RCCLAGRSAttentionCPCommunication(collective=transport)
    monkeypatch.setattr(torch.version, "hip", "test", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    plan = _plan()

    local_q = torch.zeros(1, 2, 1, 4, dtype=torch.bfloat16)
    assert communication.all_gather_query(local_q, plan).shape == (1, 2, 2, 4)

    full_out = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16)
    full_lse = torch.zeros(1, 2, 2, dtype=torch.float32)
    shard = communication.reduce_scatter_strict_result(full_out, full_lse, plan)

    assert shard.out.shape == (1, 2, 1, 4)
    assert shard.lse.shape == (1, 2, 1)
    assert transport.gather_calls == 1
    assert transport.scatter_calls == 2
    assert communication.transport_only is True
    assert communication.supports_async_overlap is False
    assert communication.supports_compute_communication_fusion is False


def test_rccl_adapter_fails_closed_outside_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    communication = RCCLAGRSAttentionCPCommunication(collective=_FakeRCCLTransport())
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(AttentionCPCommunicationUnavailable, match="ROCm device"):
        communication.all_gather_query(torch.zeros(1, 2, 1, 4), _plan())


def test_rccl_adapter_rejects_cuda_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    communication = RCCLAGRSAttentionCPCommunication(collective=_FakeRCCLTransport())
    monkeypatch.setattr(torch.version, "hip", "test", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(AttentionCPCommunicationUnavailable, match="rccl_ag_rs plan"):
        communication.all_gather_query(torch.zeros(1, 2, 1, 4), _plan(backend="cuda_ag_rs"))


def test_rccl_adapter_shares_the_cuda_collective_resolution() -> None:
    """ROCm must not own a second transport implementation.

    CUDA and ROCm run the same balanced rank tree only because both resolve
    their collective through ``collective_for_group``. A ROCm-side override
    would let the two reduction orders drift apart silently, so pin that the
    two adapters share one implementation.
    """

    assert (
        RCCLAGRSAttentionCPCommunication._get_collective
        is CUDAAGRSAttentionCPCommunication._get_collective
    )


def test_rccl_adapter_resolves_the_shared_deterministic_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = _FakeRCCLTransport()
    calls: list[Any] = []

    def _fake_collective_for_group(*, group: Any, device: Any) -> Any:
        calls.append((group, device))
        return resolved

    monkeypatch.setattr(collectives, "collective_for_group", _fake_collective_for_group)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    communication = RCCLAGRSAttentionCPCommunication(process_group="cp-group")
    monkeypatch.setattr(communication, "_dist", lambda: _FakeDist())

    assert communication._get_collective(_plan()) is resolved
    assert calls == [("cp-group", torch.device("cuda", 0))]


def test_rccl_adapter_world_sizes_match_the_shared_collective() -> None:
    capability = KernelRegistry()._attention_capabilities[OpBackend.ROCM_STRICT_ATTENTION]

    assert tuple(capability.cp_world_sizes) == collectives._SUPPORTED_WORLD_SIZES
