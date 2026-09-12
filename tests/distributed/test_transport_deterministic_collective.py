# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any

import pytest
import torch

import rl_engine.distributed as distributed
import rl_engine.distributed.collectives as collectives
import rl_engine.distributed.rocm_collectives as rocm_collectives
from rl_engine.distributed import (
    RCCLDeterministicCollective,
    TorchDistributedDeterministicCollective,
)


class _FakeDistributed:
    """Single-process model of rank-ordered AllGather transport."""

    def __init__(
        self,
        peer_inputs: list[torch.Tensor],
        *,
        rank: int = 0,
        backend: str = "gloo",
        peer_signatures: list[tuple[Any, ...]] | None = None,
        peer_capacities: list[int] | None = None,
    ) -> None:
        self.peer_inputs = peer_inputs
        self.rank = rank
        self.backend = backend
        self.peer_signatures = peer_signatures
        self.peer_capacities = peer_capacities
        self.into_tensor_calls = 0
        self.list_transport_calls = 0
        self.object_gather_calls = 0
        self.last_transport_output: torch.Tensor | None = None

    @property
    def tensor_transport_calls(self) -> int:
        return self.into_tensor_calls + self.list_transport_calls

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    def get_rank(self, *, group: Any) -> int:
        return self.rank

    def get_world_size(self, *, group: Any) -> int:
        return len(self.peer_inputs)

    def get_backend(self, group: Any) -> str:
        return self.backend

    def all_gather_object(self, output: list[Any], value: Any, *, group: Any) -> None:
        self.object_gather_calls += 1
        if isinstance(value, int):
            values = self.peer_capacities or [value] * len(self.peer_inputs)
        else:
            values = self.peer_signatures or [value] * len(self.peer_inputs)
        output[:] = values

    def all_gather_into_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        group: Any,
    ) -> None:
        self.into_tensor_calls += 1
        self.last_transport_output = output
        gathered = torch.cat([peer.reshape(-1) for peer in self.peer_inputs])
        output.copy_(gathered)

    def all_gather(
        self,
        output: list[torch.Tensor],
        input: torch.Tensor,
        *,
        group: Any,
    ) -> None:
        self.list_transport_calls += 1
        self.last_transport_output = output[0]._base
        for destination, peer in zip(output, self.peer_inputs, strict=True):
            destination.copy_(peer.reshape(-1))


def test_public_exports_use_canonical_collectives_module() -> None:
    assert distributed.DeterministicCollective is collectives.DeterministicCollective
    assert distributed.RCCLDeterministicCollective is collectives.RCCLDeterministicCollective
    assert (
        distributed.TorchDistributedDeterministicCollective
        is collectives.TorchDistributedDeterministicCollective
    )
    assert (
        distributed.create_deterministic_collective is collectives.create_deterministic_collective
    )


def test_cuda_collective_has_stable_backend_identity() -> None:
    assert collectives.DeterministicCollective.backend_id == "cuda_ipc_fixed_tree"


def _make_collective(
    monkeypatch: pytest.MonkeyPatch,
    peer_inputs: list[torch.Tensor],
    *,
    rank: int = 0,
    backend: str = "gloo",
    peer_signatures: list[tuple[Any, ...]] | None = None,
    peer_capacities: list[int] | None = None,
    max_size_bytes: int = 1024,
) -> tuple[TorchDistributedDeterministicCollective, _FakeDistributed]:
    fake_dist = _FakeDistributed(
        peer_inputs,
        rank=rank,
        backend=backend,
        peer_signatures=peer_signatures,
        peer_capacities=peer_capacities,
    )
    monkeypatch.setattr(rocm_collectives, "dist", fake_dist)
    collective = TorchDistributedDeterministicCollective(
        group=object(),
        device="cpu",
        max_size_bytes=max_size_bytes,
    )
    return collective, fake_dist


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_reduce_supports_fixed_world_sizes_and_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
    dtype: torch.dtype,
) -> None:
    peers = [torch.full((2, 3), rank + 1, dtype=dtype) for rank in range(world_size)]
    collective, fake_dist = _make_collective(monkeypatch, peers)

    provided = torch.empty_like(peers[0])
    returned = collective.all_reduce(peers[0], out=provided)

    assert returned is provided
    assert torch.equal(provided, torch.full_like(provided, world_size * (world_size + 1) // 2))
    assert fake_dist.tensor_transport_calls == (0 if world_size == 1 else 1)


def test_all_reduce_uses_balanced_not_rank_ordered_left_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Balanced: (1e20 + 1) + (-1e20 + 1) == 0 in FP32.
    # Left fold: ((1e20 + 1) + -1e20) + 1 == 1 in FP32.
    peers = [torch.tensor([value], dtype=torch.float32) for value in (1.0e20, 1.0, -1.0e20, 1.0)]
    collective, _ = _make_collective(monkeypatch, peers)

    output = collective.all_reduce(peers[0])

    assert torch.equal(output, torch.zeros_like(output))


def test_all_reduce_stages_before_writing_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = [torch.full((2, 3), rank + 1, dtype=torch.float32) for rank in range(4)]
    collective, _ = _make_collective(monkeypatch, peers)
    local = peers[0].clone()

    returned = collective.all_reduce(local, out=local)

    assert returned is local
    assert torch.equal(local, torch.full_like(local, 10))


def test_all_gather_is_rank_ordered_and_transport_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.arange(rank * 6, (rank + 1) * 6).reshape(2, 3) for rank in range(4)]
    collective, fake_dist = _make_collective(monkeypatch, peers, rank=2)

    output = collective.all_gather(peers[2])

    assert torch.equal(output, torch.cat(peers, dim=0))
    assert fake_dist.tensor_transport_calls == 1


def test_nccl_backend_uses_all_gather_into_tensor_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.full((2, 3), rank) for rank in range(2)]
    collective, fake_dist = _make_collective(monkeypatch, peers, backend="nccl")

    output = collective.all_gather(peers[0])

    assert torch.equal(output, torch.cat(peers, dim=0))
    assert fake_dist.into_tensor_calls == 1
    assert fake_dist.list_transport_calls == 0


def test_all_gather_writes_directly_to_provided_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.full((2, 3), rank) for rank in range(2)]
    collective, fake_dist = _make_collective(monkeypatch, peers, backend="nccl")
    provided = torch.empty(4, 3, dtype=peers[0].dtype)

    returned = collective.all_gather(peers[0], out=provided)

    assert returned is provided
    assert fake_dist.last_transport_output is not None
    assert fake_dist.last_transport_output.data_ptr() == provided.data_ptr()
    assert collective._workspace is None


def test_reduction_workspace_grows_once_and_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.full((2, 3), rank + 1, dtype=torch.float32) for rank in range(2)]
    collective, _ = _make_collective(monkeypatch, peers, backend="nccl")

    collective.all_reduce(peers[0])
    assert collective._workspace is not None
    assert collective.workspace_size_bytes == sum(peer.numel() for peer in peers) * 4
    first_pointer = collective._workspace.data_ptr()
    collective.all_reduce(peers[0])

    assert collective._workspace.data_ptr() == first_pointer
    collective.close()
    assert collective._workspace is None
    assert collective.workspace_size_bytes == 0


def test_matching_signature_is_validated_once_per_hot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.full((2, 3), rank + 1, dtype=torch.float32) for rank in range(2)]
    collective, fake_dist = _make_collective(monkeypatch, peers, backend="nccl")
    constructor_object_gathers = fake_dist.object_gather_calls

    collective.all_reduce(peers[0])
    collective.all_reduce(peers[0])

    assert fake_dist.object_gather_calls == constructor_object_gathers + 1


def test_latest_collective_api_can_skip_signature_handshakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.full((4, 2), rank + 1, dtype=torch.float32) for rank in range(2)]
    collective, fake_dist = _make_collective(monkeypatch, peers, backend="nccl")
    constructor_object_gathers = fake_dist.object_gather_calls

    collective.all_reduce(peers[0], validate_signature=False)
    collective.all_gather(peers[0], validate_signature=False)
    collective.reduce_scatter(peers[0], validate_signature=False)
    gathered = collective.all_gather_many(
        (peers[0], peers[0]),
        validate_signature=False,
    )
    fake_dist.peer_inputs = [torch.cat((peer, peer), dim=-1) for peer in peers]
    scattered = collective.reduce_scatter_many(
        (peers[0], peers[0]),
        validate_signature=False,
    )

    assert len(gathered) == 2
    assert len(scattered) == 2
    assert fake_dist.object_gather_calls == constructor_object_gathers


def test_reduce_scatter_reduces_then_selects_local_leading_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.full((8, 3), rank + 1, dtype=torch.bfloat16) for rank in range(4)]
    collective, _ = _make_collective(monkeypatch, peers, rank=2)

    output = collective.reduce_scatter(peers[2])

    expected_full = torch.full_like(peers[0], 10)
    assert torch.equal(output, expected_full.chunk(4, dim=0)[2])


def test_reduce_scatter_many_packs_lanes_and_transports_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each lane must retain its own balanced tree. A left fold would produce
    # one for lane 0 and two for lane 1, while the fixed tree produces zero for
    # both lanes in FP32.
    lane_values = [
        (
            1.0e20,
            1.0,
            -1.0e20,
            1.0,
        ),
        (
            1.0e20,
            2.0,
            -1.0e20,
            2.0,
        ),
    ]
    # Give each rank distinct values while preserving the cancellation pattern
    # in every row. The fake transport returns these packed rank inputs.
    lane_peers = [
        (
            torch.full((8, 1), lane_values[0][rank], dtype=torch.float32),
            torch.full((8, 1), lane_values[1][rank], dtype=torch.float32),
        )
        for rank in range(4)
    ]
    packed_peers = [torch.cat(lanes, dim=-1) for lanes in lane_peers]
    collective, fake_dist = _make_collective(monkeypatch, packed_peers, rank=2)

    local_lanes = lane_peers[2]
    outputs = (torch.empty(2, 1), torch.empty(2, 1))
    returned = collective.reduce_scatter_many(local_lanes, outs=outputs)

    assert returned[0] is outputs[0]
    assert returned[1] is outputs[1]
    assert torch.equal(outputs[0], torch.zeros_like(outputs[0]))
    assert torch.equal(outputs[1], torch.zeros_like(outputs[1]))
    assert fake_dist.tensor_transport_calls == 1


def test_reduce_scatter_many_rejects_oversized_packed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.ones(4, 2, dtype=torch.float32) for _ in range(2)]
    collective, fake_dist = _make_collective(
        monkeypatch,
        peers,
        max_size_bytes=32,
    )
    monkeypatch.setattr(rocm_collectives, "_PACKED_REDUCE_SCATTER_MAX_BYTES", 1024)

    with pytest.raises(ValueError, match="packed input requires"):
        collective.reduce_scatter_many((peers[0], peers[0]))
    assert fake_dist.tensor_transport_calls == 0


def test_reduce_scatter_many_uses_separate_calls_for_large_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.ones(4, 2, dtype=torch.float32) for _ in range(2)]
    collective, fake_dist = _make_collective(monkeypatch, peers)
    monkeypatch.setattr(rocm_collectives, "_PACKED_REDUCE_SCATTER_MAX_BYTES", 1)

    outputs = collective.reduce_scatter_many((peers[0], peers[0]))

    assert len(outputs) == 2
    assert all(torch.equal(output, torch.full_like(output, 2)) for output in outputs)
    assert fake_dist.tensor_transport_calls == 2


def test_matching_signature_is_checked_before_tensor_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.ones(2, 3), torch.ones(2, 3)]
    signatures = [
        ("all_reduce", (2, 3), "torch.float32", 6),
        ("reduce_scatter", (2, 3), "torch.float32", 6),
    ]
    collective, fake_dist = _make_collective(
        monkeypatch,
        peers,
        peer_signatures=signatures,
    )

    with pytest.raises(ValueError, match="matching shapes and dtypes"):
        collective.all_reduce(peers[0])
    assert fake_dist.tensor_transport_calls == 0


def test_capacity_must_match_on_every_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = [torch.ones(2, 3), torch.ones(2, 3)]

    with pytest.raises(ValueError, match="same max_size_bytes"):
        _make_collective(
            monkeypatch,
            peers,
            max_size_bytes=1024,
            peer_capacities=[1024, 2048],
        )


def test_validation_fails_closed_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    peers = [torch.ones(4, 2), torch.ones(4, 2)]
    collective, fake_dist = _make_collective(monkeypatch, peers, max_size_bytes=31)

    with pytest.raises(TypeError, match="float32, float16, and bfloat16"):
        collective.all_reduce(torch.ones(1, dtype=torch.int32))
    with pytest.raises(ValueError, match="max_size_bytes"):
        collective.all_reduce(peers[0])
    with pytest.raises(ValueError, match=r"input.size\(0\).+divisible"):
        collective.reduce_scatter(torch.ones(3, 2))
    with pytest.raises(ValueError, match="at least one dimension"):
        collective.all_gather(torch.tensor(1.0))
    assert fake_dist.tensor_transport_calls == 0


def test_lifecycle_is_idempotent_and_context_manager_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peers = [torch.ones(2, 3)]
    collective, _ = _make_collective(monkeypatch, peers)

    with collective as entered:
        assert entered is collective
        assert not collective.closed
        assert torch.equal(collective.all_reduce(peers[0]), peers[0])

    assert collective.closed
    collective.close()
    with pytest.raises(RuntimeError, match="closed"):
        collective.all_reduce(peers[0])


@pytest.mark.parametrize("world_size", [3, 16])
def test_unsupported_world_size_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    world_size: int,
) -> None:
    fake_dist = _FakeDistributed([torch.ones(1)] * world_size)
    monkeypatch.setattr(rocm_collectives, "dist", fake_dist)

    with pytest.raises(ValueError, match="world_size in"):
        TorchDistributedDeterministicCollective(group=object(), device="cpu")


def test_rccl_class_requires_rocm_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rocm_collectives.torch.version, "hip", None, raising=False)

    with pytest.raises(RuntimeError, match="ROCm PyTorch build"):
        RCCLDeterministicCollective(group=object(), device="cuda:0")


def test_rccl_class_requires_nccl_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dist = _FakeDistributed([torch.ones(1)], backend="gloo")
    monkeypatch.setattr(rocm_collectives, "dist", fake_dist)
    monkeypatch.setattr(rocm_collectives.torch.version, "hip", "6.3", raising=False)
    monkeypatch.setattr(rocm_collectives.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(rocm_collectives.torch.cuda, "current_device", lambda: 0)

    with pytest.raises(RuntimeError, match="NCCL process-group API"):
        RCCLDeterministicCollective(group=object(), device="cuda:0")


def test_rccl_class_rejects_cpu_before_process_group_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rocm_collectives.torch.version, "hip", "6.3", raising=False)
    monkeypatch.setattr(rocm_collectives.torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="ROCm device"):
        RCCLDeterministicCollective(group=object(), device="cpu")


def test_factory_dispatches_rocm_to_rccl(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def fake_rccl(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(collectives.torch.version, "hip", "6.3", raising=False)
    monkeypatch.setattr(collectives, "RCCLDeterministicCollective", fake_rccl)

    group = object()
    result = collectives.create_deterministic_collective(
        group=group,
        device="cuda:3",
        max_size_bytes=1234,
    )

    assert result is sentinel
    assert calls == [{"group": group, "device": "cuda:3", "max_size_bytes": 1234}]


def test_factory_preserves_existing_cuda_collective(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def fake_cuda(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(collectives.torch.version, "hip", None, raising=False)
    monkeypatch.setattr(collectives, "DeterministicCollective", fake_cuda)

    group = object()
    result = collectives.create_deterministic_collective(
        group=group,
        device="cuda:1",
        max_size_bytes=4321,
    )

    assert result is sentinel
    assert calls == [{"group": group, "device": "cuda:1", "max_size_bytes": 4321}]
