# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real multi-GPU topology checks for the ROCm-native Triton Qwen3 FFN.

PyTorch exposes RCCL through the ``nccl`` process-group backend. The FFN uses
ROCm-native Triton compute and a rank-ordered RCCL transport collective.
"""

from __future__ import annotations

import os
import queue
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

import rl_engine.kernels.ops.triton.ffn.ffn as ffn_module
from rl_engine.kernels.ops.triton.ffn import (
    QWEN3_8B_HIDDEN_SIZE,
    QWEN3_8B_INTERMEDIATE_SIZE,
    pack_qwen3_ffn_forward_weights,
    qwen3_ffn,
)

_IS_ROCM = getattr(torch.version, "hip", None) is not None
_EXTERNAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

# I=512 keeps every TP=1/2/4/8 shard aligned to the 32-wide GEMM K-tree and
# keeps the SM90 N dimension tile-aligned even at TP=8.
_TOKENS = 32
_HIDDEN = 64
_INTERMEDIATE = 512

_WORLD2_CONFIGS = (
    ("tp2", 2, 1, False),
    ("tp2_sp", 2, 1, True),
)
_WORLD4_CONFIGS = (
    ("tp4", 4, 1, False),
    ("tp2_cp2", 2, 2, False),
    ("tp2_cp2_sp", 2, 2, True),
)
_WORLD8_CONFIGS = (("tp8", 8, 1, False),)

pytestmark = pytest.mark.skipif(
    _EXTERNAL_WORLD_SIZE != 1,
    reason="topology tests own their local worker processes; run pytest directly",
)


def _has_topology_devices(count: int) -> bool:
    return bool(
        _IS_ROCM
        and torch.cuda.is_available()
        and torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
        and torch.cuda.device_count() >= count
    )


def _has_qwen3_8b_capacity(count: int) -> bool:
    if not _has_topology_devices(count):
        return False
    minimum_bytes = 8 * 1024**3
    try:
        return all(
            torch.cuda.get_device_properties(index).total_memory >= minimum_bytes
            for index in range(count)
        )
    except RuntimeError:
        return False


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02
    return value.to(device=device, dtype=torch.bfloat16)


def _make_inputs(
    token_count: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    return (
        _randn((token_count, hidden_size), seed=seed, device=device),
        _randn((intermediate_size, hidden_size), seed=seed + 1, device=device),
        _randn((intermediate_size, hidden_size), seed=seed + 2, device=device),
        _randn((hidden_size, intermediate_size), seed=seed + 3, device=device),
        _randn((token_count, hidden_size), seed=seed + 4, device=device),
    )


def _close_ffn_collectives() -> None:
    for collective in list(ffn_module._COLLECTIVES.values()):
        collective.close()
    ffn_module._COLLECTIVES.clear()


def _shard_ranges(
    rank: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
    token_count: int,
    intermediate_size: int,
) -> tuple[int, int, int, int]:
    if tp_size * cp_size <= rank:
        raise ValueError("rank lies outside the requested TP/CP mesh")
    if token_count % cp_size:
        raise ValueError("token count must be divisible by CP size")
    cp_tokens = token_count // cp_size
    if sequence_parallel and cp_tokens % tp_size:
        raise ValueError("each CP token shard must be divisible by TP size for SP")
    if intermediate_size % tp_size:
        raise ValueError("intermediate size must be divisible by TP size")

    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    local_tokens = cp_tokens // tp_size if sequence_parallel else cp_tokens
    token_start = cp_rank * cp_tokens
    if sequence_parallel:
        token_start += tp_rank * local_tokens
    token_end = token_start + local_tokens

    local_intermediate = intermediate_size // tp_size
    feature_start = tp_rank * local_intermediate
    feature_end = feature_start + local_intermediate
    return token_start, token_end, feature_start, feature_end


def _canonical(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    with torch.no_grad():
        inference = qwen3_ffn(hidden, gate, up, down)
    inputs = [value.detach().clone().requires_grad_(True) for value in (hidden, gate, up, down)]
    training = qwen3_ffn(*inputs)
    training.backward(grad_output)
    assert torch.equal(inference, training.detach()), "TP=1 train/infer forward mismatch"
    return inference, training, inputs


def _mesh_groups(
    dist: Any,
    tp_size: int,
    cp_size: int,
) -> tuple[list[Any], list[Any]]:
    world_size = dist.get_world_size()
    if tp_size * cp_size != world_size:
        raise ValueError("TP size times CP size must equal the process-group world size")
    if tp_size == world_size and cp_size == 1:
        return [dist.group.WORLD], []
    if cp_size == world_size and tp_size == 1:
        return [], [dist.group.WORLD]

    tp_groups = []
    if tp_size > 1:
        for cp_rank in range(cp_size):
            ranks = list(range(cp_rank * tp_size, (cp_rank + 1) * tp_size))
            tp_groups.append(dist.new_group(ranks=ranks))

    cp_groups = []
    if cp_size > 1:
        for tp_rank in range(tp_size):
            ranks = [cp_rank * tp_size + tp_rank for cp_rank in range(cp_size)]
            cp_groups.append(dist.new_group(ranks=ranks))
    return tp_groups, cp_groups


def _run_topology(
    rank: int,
    dist: Any,
    meshes: dict[tuple[int, int], tuple[list[Any], list[Any]]],
    *,
    name: str,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
    hidden: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    grad_output: torch.Tensor,
    inference_reference: torch.Tensor,
    training_reference: torch.Tensor,
    reference_inputs: list[torch.Tensor],
) -> None:
    mesh_key = (tp_size, cp_size)
    if mesh_key not in meshes:
        meshes[mesh_key] = _mesh_groups(dist, tp_size, cp_size)
    tp_groups, cp_groups = meshes[mesh_key]

    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    tp_group = tp_groups[cp_rank] if tp_size > 1 else None
    cp_group = cp_groups[tp_rank] if cp_size > 1 else None
    token_start, token_end, feature_start, feature_end = _shard_ranges(
        rank,
        tp_size=tp_size,
        cp_size=cp_size,
        sequence_parallel=sequence_parallel,
        token_count=hidden.size(0),
        intermediate_size=gate.size(0),
    )
    shard = (
        hidden[token_start:token_end].contiguous(),
        gate[feature_start:feature_end].contiguous(),
        up[feature_start:feature_end].contiguous(),
        down[:, feature_start:feature_end].contiguous(),
    )
    inference_forward_weights = pack_qwen3_ffn_forward_weights(*shard[1:])

    with torch.no_grad():
        inference = qwen3_ffn(
            *shard,
            forward_weights=inference_forward_weights,
            tp_group=tp_group,
            cp_group=cp_group,
            sequence_parallel=sequence_parallel,
        )
    inputs = [value.detach().clone().requires_grad_(True) for value in shard]
    training_forward_weights = pack_qwen3_ffn_forward_weights(*inputs[1:])
    training = qwen3_ffn(
        *inputs,
        forward_weights=training_forward_weights,
        tp_group=tp_group,
        cp_group=cp_group,
        sequence_parallel=sequence_parallel,
    )
    training.backward(grad_output[token_start:token_end].contiguous())

    expected_output = inference_reference[token_start:token_end]
    assert torch.equal(inference, training.detach()), f"{name}: train/infer mismatch"
    assert torch.equal(inference, expected_output), f"{name}: inference mismatch vs TP=1"
    assert torch.equal(
        training.detach(), training_reference.detach()[token_start:token_end]
    ), f"{name}: training forward mismatch vs TP=1"
    assert torch.equal(
        inputs[0].grad, reference_inputs[0].grad[token_start:token_end]
    ), f"{name}: hidden grad mismatch vs TP=1"

    expected_weight_grads = (
        (inputs[1].grad, reference_inputs[1].grad[feature_start:feature_end], "gate"),
        (inputs[2].grad, reference_inputs[2].grad[feature_start:feature_end], "up"),
        (
            inputs[3].grad,
            reference_inputs[3].grad[:, feature_start:feature_end],
            "down",
        ),
    )
    for actual, expected, label in expected_weight_grads:
        assert torch.equal(actual, expected), f"{name}: {label} weight grad mismatch vs TP=1"


def _topology_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue: Any,
    configs: tuple[tuple[str, int, int, bool], ...],
) -> None:
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=5),
        )
        device = torch.device("cuda", rank)
        hidden, gate, up, down, grad_output = _make_inputs(
            _TOKENS,
            _HIDDEN,
            _INTERMEDIATE,
            device,
            seed=600,
        )
        inference_reference, training_reference, reference_inputs = _canonical(
            hidden,
            gate,
            up,
            down,
            grad_output,
        )
        meshes: dict[tuple[int, int], tuple[list[Any], list[Any]]] = {}
        for name, tp_size, cp_size, sequence_parallel in configs:
            _run_topology(
                rank,
                dist,
                meshes,
                name=name,
                tp_size=tp_size,
                cp_size=cp_size,
                sequence_parallel=sequence_parallel,
                hidden=hidden,
                gate=gate,
                up=up,
                down=down,
                grad_output=grad_output,
                inference_reference=inference_reference,
                training_reference=training_reference,
                reference_inputs=reference_inputs,
            )
        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _qwen3_8b_tp2_worker(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue: Any,
) -> None:
    try:
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )
        device = torch.device("cuda", rank)
        hidden, gate, up, down, grad_output = _make_inputs(
            2,
            QWEN3_8B_HIDDEN_SIZE,
            QWEN3_8B_INTERMEDIATE_SIZE,
            device,
            seed=800,
        )
        inference_reference, training_reference, reference_inputs = _canonical(
            hidden,
            gate,
            up,
            down,
            grad_output,
        )
        _run_topology(
            rank,
            dist,
            {},
            name="qwen3_8b_tp2",
            tp_size=2,
            cp_size=1,
            sequence_parallel=False,
            hidden=hidden,
            gate=gate,
            up=up,
            down=down,
            grad_output=grad_output,
            inference_reference=inference_reference,
            training_reference=training_reference,
            reference_inputs=reference_inputs,
        )
        result_queue.put({"ok": True, "rank": rank})
    except Exception:  # pragma: no cover - forwarded to the parent process.
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        _close_ffn_collectives()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _spawn_workers(
    worker: Any,
    world_size: int,
    worker_args: tuple[Any, ...] = (),
    *,
    timeout_seconds: int,
) -> None:
    if not _has_topology_devices(world_size):
        platform = "ROCm/RCCL" if _IS_ROCM else "CUDA/NCCL"
        pytest.skip(
            f"requires {world_size} {platform} GPUs plus FFN and deterministic collective support"
        )

    # Single-GPU tests may have populated the parent process's caching
    # allocator on device 0.  Release unused blocks before spawned rank 0 owns
    # that device, which matters for the Qwen3-8B smoke case.
    torch.cuda.empty_cache()
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as temporary_directory:
        init_method = (Path(temporary_directory) / "nccl_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=worker,
                args=(rank, world_size, init_method, result_queue, *worker_args),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()

        results = []
        try:
            for _ in processes:
                result = result_queue.get(timeout=timeout_seconds)
                results.append(result)
                if not result["ok"]:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    break
        except queue.Empty:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail(f"timed out waiting for {world_size} FFN topology workers")
        finally:
            for process in processes:
                # RCCL teardown can take longer than ten seconds after every
                # worker has already reported a successful result, especially
                # for the eight-rank topology on ROCm hosts with unusable RDMA
                # interfaces.  Allow cleanup to finish instead of turning a
                # successful numerical check into a SIGTERM false failure.
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)
            result_queue.close()
            result_queue.join_thread()

    for result in sorted(results, key=lambda item: item["rank"]):
        assert result["ok"], result.get("traceback")
    for process in processes:
        assert process.exitcode == 0


def test_triton_qwen3_ffn_tp2_and_tp_sp_match_tp1_bitwise() -> None:
    _spawn_workers(
        _topology_worker,
        2,
        (_WORLD2_CONFIGS,),
        timeout_seconds=180,
    )


def test_triton_qwen3_ffn_tp4_tp_cp_and_tp_cp_sp_match_tp1_bitwise() -> None:
    _spawn_workers(
        _topology_worker,
        4,
        (_WORLD4_CONFIGS,),
        timeout_seconds=240,
    )


def test_triton_qwen3_ffn_tp8_matches_tp1_bitwise() -> None:
    _spawn_workers(
        _topology_worker,
        8,
        (_WORLD8_CONFIGS,),
        timeout_seconds=360,
    )


def test_triton_qwen3_8b_ffn_tp2_smoke_matches_tp1_bitwise() -> None:
    if os.environ.get("RL_KERNEL_SKIP_QWEN3_8B_TOPOLOGY") == "1":
        pytest.skip("Qwen3-8B topology smoke disabled by environment")
    if not _has_qwen3_8b_capacity(2):
        pytest.skip("Qwen3-8B TP=2 smoke requires two GPUs with at least 8 GiB each")
    _spawn_workers(
        _qwen3_8b_tp2_worker,
        2,
        timeout_seconds=600,
    )
