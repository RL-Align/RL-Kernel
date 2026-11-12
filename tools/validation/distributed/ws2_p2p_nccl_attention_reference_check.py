# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""NCCL correctness reference for issue #235 CP Attention communication.

Use two ranks for a CP-only diagnostic, four ranks for the formal TP=2/CP=2
target, or eight ranks for two independent TP=2/CP=2 replicas::

    torchrun --standalone --nproc-per-node=4 \
      tools/validation/distributed/ws2_p2p_nccl_attention_reference_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.backends.cuda.attention.cp_comm import (  # noqa: E402
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
    P2PNCCLAttentionCPCommunication,
    RCCLAGRSAttentionCPCommunication,
)
from rl_engine.backends.cuda.attention.flash_attn import StrictFlashAttention4Core  # noqa: E402
from rl_engine.backends.cuda.attention.flashinfer_paged_attention import (  # noqa: E402
    FlashInferPagedAttentionConfig,
    FlashInferQwen3PagedAttentionOp,
    _apply_strict_rope,
)
from rl_engine.contracts.operators.attention import (  # noqa: E402
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_RING_SCHEDULE_ID,
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
)
from rl_engine.reference.attention.cp_attention import (  # noqa: E402
    AttentionPartialState,
    DeterministicCPAttentionReferenceOp,
    merge_attention_partial_states,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--final-write-atol", type=float, default=2.0e-2)
    parser.add_argument(
        "--transport",
        choices=("p2p_nccl_reference", "cuda_ag_rs", "rccl_ag_rs"),
        default="p2p_nccl_reference",
        help="P2P is the reference; cuda_ag_rs and rccl_ag_rs are self-owned transports",
    )
    parser.add_argument(
        "--strict-shared-core",
        action="store_true",
        help="run AG(Q/K/V/positions) -> shared CUDA core -> RS(Out/LSE) with backward",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-rocm-matrix",
        action="store_true",
        help="run the complete 1/2/4/8-GPU ROCm acceptance matrix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rocm-attention"),
        help="directory for --run-rocm-matrix logs and JSON reports",
    )
    args = parser.parse_args(argv)
    if args.strict_shared_core and args.transport not in {"cuda_ag_rs", "rccl_ag_rs"}:
        parser.error("--strict-shared-core requires a self-owned AG/RS transport")
    if args.run_rocm_matrix and (args.strict_shared_core or args.output is not None):
        parser.error("--run-rocm-matrix cannot be combined with single-run output options")
    return args


def _run_rocm_matrix(output_dir: Path) -> int:
    if torch.version.hip is None or torch.cuda.device_count() < 8:
        raise RuntimeError("the formal acceptance matrix requires 8 visible ROCm GPUs")

    repo = Path(__file__).resolve().parents[3]
    output = (repo / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    commands: list[tuple[str, list[str]]] = [
        (
            "single_gpu",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/ops/attention/test_deterministic_attention_cuda.py",
            ],
        ),
        (
            "adapter_cp_contracts",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/backends/cuda/test_flashinfer_pr7_attention.py",
                "tests/distributed/cp/test_cp_attention.py",
                "tests/validation/common/test_attention_comparison.py",
            ],
        ),
    ]
    for transport, strict in (("p2p_nccl_reference", False), ("rccl_ag_rs", True)):
        for ranks in (2, 4, 8):
            name = f"{transport}_{ranks}r"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={ranks}",
                str(script),
                "--transport",
                transport,
                "--output",
                str(output / f"{name}.json"),
            ]
            if strict:
                command.append("--strict-shared-core")
            commands.append((name, command))

    steps: list[dict[str, object]] = []
    for name, command in commands:
        completed = subprocess.run(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (output / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
        steps.append({"name": name, "command": command, "returncode": completed.returncode})

    summary = {
        "schema_version": "ws2_rocm_attention_acceptance/v1",
        "git_commit": _current_git_commit(repo),
        "platform": "rocm",
        "torch": str(torch.__version__),
        "hip": str(torch.version.hip),
        "collective": list(torch.cuda.nccl.version()),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "steps": steps,
        "passed": all(step["returncode"] == 0 for step in steps),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def _current_git_commit(repo: Path) -> str:
    """Bind an acceptance artifact to the checkout that generated it."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("ROCm Attention acceptance requires a Git checkout") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_rocm_matrix:
        return _run_rocm_matrix(args.output_dir)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this check requires at least two visible CUDA/ROCm devices")
    dist.init_process_group("nccl", init_method="env://")
    try:
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if world_size not in {2, 4, 8}:
            raise RuntimeError("this check requires 2, 4, or 8 NCCL/RCCL ranks")
        if torch.cuda.device_count() < world_size:
            raise RuntimeError("this single-node check requires one visible GPU per NCCL rank")
        local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        cp_groups = [
            dist.new_group(ranks=[pair_start, pair_start + 1])
            for pair_start in range(0, world_size, 2)
        ]
        rank_in_replica = global_rank if world_size < 8 else global_rank % 4
        replica_index = 0 if world_size < 8 else global_rank // 4
        tp_rank = 0 if world_size == 2 else rank_in_replica // 2
        cp_rank = rank_in_replica % 2
        cp_group = cp_groups[global_rank // 2]
        result = run_check(
            args,
            global_rank=global_rank,
            tp_rank=tp_rank,
            cp_rank=cp_rank,
            replica_index=replica_index,
            cp_group=cp_group,
            device=device,
        )
        failures = torch.tensor(
            [0 if result["passed"] else 1],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(failures, op=dist.ReduceOp.SUM)
        result["global_failure_count"] = int(failures.item())
        reports: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(reports, result)
        if global_rank == 0:
            report = {
                "schema_version": f"ws2_{args.transport}_attention/v2",
                "git_commit": _current_git_commit(REPO_ROOT),
                "backend": str(dist.get_backend()),
                "platform": "rocm" if torch.version.hip is not None else "cuda",
                "torch_version": str(torch.__version__),
                "runtime_version": (
                    str(torch.version.hip)
                    if torch.version.hip is not None
                    else str(torch.version.cuda)
                ),
                "collective_version": list(torch.cuda.nccl.version()),
                "device_name": torch.cuda.get_device_name(0),
                "transport": args.transport,
                "world_size": world_size,
                "tp_world_size": 1 if world_size == 2 else 2,
                "cp_world_size": 2,
                "replica_count": 2 if world_size == 8 else 1,
                "global_failure_count": int(failures.item()),
                "ranks": reports,
            }
            serialized = json.dumps(report, indent=2, sort_keys=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(serialized + "\n", encoding="utf-8")
            print(serialized)
        return 0 if int(failures.item()) == 0 else 1
    finally:
        dist.destroy_process_group()


def run_check(
    args: argparse.Namespace,
    *,
    global_rank: int,
    tp_rank: int,
    cp_rank: int,
    replica_index: int,
    cp_group: Any,
    device: torch.device,
) -> dict[str, object]:
    if args.batch < 1:
        raise ValueError("batch must be positive")
    if args.seq_len < 2 or args.seq_len % 2 != 0:
        raise ValueError("seq_len must be positive and divisible by CP=2")
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2 for a bitwise stability check")
    if args.q_heads != 16 or args.kv_heads != 4 or args.head_dim != 128:
        raise ValueError("TP=2 Qwen3-8B local heads must be Hq=16, Hkv=4, D=128")
    for name in ("atol", "final_write_atol"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    generator = torch.Generator(device="cpu").manual_seed(args.seed + tp_rank + 100 * replica_index)
    shape_q = (args.batch, args.q_heads, args.seq_len, args.head_dim)
    shape_kv = (args.batch, args.kv_heads, args.seq_len, args.head_dim)
    q = torch.randn(shape_q, generator=generator, dtype=torch.bfloat16).to(device)
    k = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16).to(device)
    owner_ranges = ((0, args.seq_len // 2), (args.seq_len // 2, args.seq_len))
    blocks: list[AttentionCPBlockMetadata] = []
    for owner, (owner_start, owner_end) in enumerate(owner_ranges):
        for start in range(owner_start, owner_end, args.chunk_size):
            blocks.append(
                AttentionCPBlockMetadata(
                    global_block_index=len(blocks),
                    kv_block_start=start,
                    kv_block_end=min(start + args.chunk_size, owner_end),
                    owner_cp_rank=owner,
                    owner_tp_rank=tp_rank,
                )
            )
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=2,
            tp_rank=tp_rank,
            cp_world_size=2,
            cp_rank=cp_rank,
        ),
        backend=args.transport,
        status="implemented",
        expected_blocks=tuple(blocks),
        expected_kv_token_range=(0, args.seq_len),
        query_token_ranges=owner_ranges,
    )
    communication: (
        P2PNCCLAttentionCPCommunication
        | CUDAAGRSAttentionCPCommunication
        | RCCLAGRSAttentionCPCommunication
    )
    if args.transport == "p2p_nccl_reference":
        communication = P2PNCCLAttentionCPCommunication(process_group=cp_group)
    elif args.transport == "cuda_ag_rs":
        communication = CUDAAGRSAttentionCPCommunication(process_group=cp_group)
    else:
        communication = RCCLAGRSAttentionCPCommunication(process_group=cp_group)

    query_start, query_end = owner_ranges[cp_rank]
    q_local = q[:, :, query_start:query_end, :].contiguous()
    q_gathered = communication.all_gather_query(q_local, plan)
    query_ag_max_abs = float((q_gathered - q).abs().max().item())
    reference = DeterministicCPAttentionReferenceOp()
    local_states: list[AttentionCPPartialState] = []
    for block in reversed(blocks):
        if block.owner_cp_rank != cp_rank:
            continue
        state = reference.local_partial_state(
            q_gathered,
            k[:, :, block.kv_block_start : block.kv_block_end, :],
            v[:, :, block.kv_block_start : block.kv_block_end, :],
            q_start=0,
            k_start=block.kv_block_start,
            total_kv_len=args.seq_len,
            total_query_len=args.seq_len,
            causal=True,
        )
        local_states.append(AttentionCPPartialState(state.out, state.lse, block))

    def communicate() -> tuple[tuple[AttentionCPPartialState, ...], AttentionCPMergedState]:
        gathered_states = communication.all_gather_partial_states(tuple(local_states), plan)
        merged_state = merge_attention_partial_states(
            [
                AttentionPartialState(
                    state.out,
                    state.lse,
                    state.block.kv_block_start,
                    state.block.kv_block_end,
                )
                for state in gathered_states
            ]
        )
        local_state = communication.reduce_scatter_merged_state(
            AttentionCPMergedState(merged_state.out, merged_state.lse),
            plan,
        )
        return gathered_states, local_state

    gathered, local = communicate()
    gathered_indices = [state.block.global_block_index for state in gathered]
    repeat_query_bitwise = True
    repeat_out_bitwise = True
    repeat_lse_bitwise = True
    repeat_manifest_bitwise = True
    for _ in range(args.repeats - 1):
        repeated_q = communication.all_gather_query(q_local, plan)
        repeated_gathered, repeated_local = communicate()
        repeat_query_bitwise = repeat_query_bitwise and torch.equal(repeated_q, q_gathered)
        repeat_out_bitwise = repeat_out_bitwise and torch.equal(repeated_local.out, local.out)
        repeat_lse_bitwise = repeat_lse_bitwise and torch.equal(repeated_local.lse, local.lse)
        repeat_manifest_bitwise = (
            repeat_manifest_bitwise
            and [state.block.global_block_index for state in repeated_gathered] == gathered_indices
        )

    full_out, full_lse = reference.forward_fp32_with_lse(q, k, v, causal=True)
    start, end = owner_ranges[cp_rank]
    out_max_abs = float((local.out - full_out[:, :, start:end, :]).abs().max().item())
    lse_max_abs = float((local.lse - full_lse[:, :, start:end]).abs().max().item())
    final_out = local.out.to(q.dtype)
    expected_final_out = full_out[:, :, start:end, :].to(q.dtype)
    final_out_max_abs = float((final_out.float() - expected_final_out.float()).abs().max().item())
    strict_shared_core = (
        _run_strict_shared_core_check(
            args,
            plan=plan,
            communication=communication,
            q=q,
            k=k,
            v=v,
            owner_ranges=owner_ranges,
        )
        if args.strict_shared_core
        else {"executed": False, "passed": False}
    )
    passed = (
        gathered_indices == list(range(len(blocks)))
        and query_ag_max_abs == 0.0
        and repeat_query_bitwise
        and repeat_out_bitwise
        and repeat_lse_bitwise
        and repeat_manifest_bitwise
        and out_max_abs <= args.atol
        and lse_max_abs <= args.atol
        and final_out.dtype == q.dtype
        and final_out_max_abs <= args.final_write_atol
        and (not args.strict_shared_core or strict_shared_core["passed"] is True)
    )
    return {
        "rank": global_rank,
        "global_world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "tp_rank": tp_rank,
        "tp_world_size": 1 if dist.is_initialized() and dist.get_world_size() == 2 else 2,
        "cp_rank": cp_rank,
        "cp_world_size": 2,
        "replica_index": replica_index,
        "replica_count": 2 if dist.is_initialized() and dist.get_world_size() == 8 else 1,
        "device": str(device),
        "dtype": "bf16",
        "accum_dtype": "fp32",
        "downcast_at": "final_write",
        "final_output_dtype": str(final_out.dtype).removeprefix("torch."),
        "transport": args.transport,
        "protocol": "ag_query_local_kv_rs_out_lse",
        "strict_protocol": "ag_qkv_positions_shared_core_rs_out_lse",
        "query_ag": args.transport,
        "query_ag_max_abs": query_ag_max_abs,
        "query_range": [start, end],
        "expected_block_manifest": [block.provenance() for block in blocks],
        "local_block_indices": sorted(state.block.global_block_index for state in local_states),
        "gathered_block_indices": gathered_indices,
        "repeat_count": args.repeats,
        "repeat_query_bitwise": repeat_query_bitwise,
        "repeat_out_bitwise": repeat_out_bitwise,
        "repeat_lse_bitwise": repeat_lse_bitwise,
        "repeat_manifest_bitwise": repeat_manifest_bitwise,
        "out_max_abs": out_max_abs,
        "lse_max_abs": lse_max_abs,
        "final_out_max_abs": final_out_max_abs,
        "atol": args.atol,
        "final_write_atol": args.final_write_atol,
        "strict_shared_core": strict_shared_core,
        "passed": passed,
    }


def _run_strict_shared_core_check(
    args: argparse.Namespace,
    *,
    plan: AttentionCPCommunicationPlan,
    communication: (
        CUDAAGRSAttentionCPCommunication
        | RCCLAGRSAttentionCPCommunication
        | P2PNCCLAttentionCPCommunication
    ),
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    owner_ranges: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    """Exercise the complete differentiable self-owned communication path."""

    cp_rank = plan.parallel.cp_rank
    start, end = owner_ranges[cp_rank]
    q_local = q[:, :, start:end, :].detach().clone().requires_grad_()
    k_local = k[:, :, start:end, :].detach().clone().requires_grad_()
    v_local = v[:, :, start:end, :].detach().clone().requires_grad_()
    positions = torch.arange(args.seq_len, dtype=torch.long, device=q.device).expand(args.batch, -1)
    metadata = SimpleNamespace(
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
        query_position_ids=positions[:, start:end].contiguous(),
        key_position_ids=positions[:, start:end].contiguous(),
    )
    config = FlashInferPagedAttentionConfig(
        mode="prefill",
        cp_comm_plan=plan,
        require_cp_comm=True,
        strict_mode=True,
        cp_communication=communication,
    )
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=None)
    distributed = op(q_local, k_local, v_local, metadata, config=config)

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    rope = _strict_rope_op()
    q_ready = _apply_strict_rope(rope, q_ref, positions, config.rope.rope_theta)
    k_ready = _apply_strict_rope(rope, k_ref, positions, config.rope.rope_theta)
    reference = _strict_attention_reference_rows(
        q_ready,
        k_ready,
        v_ref,
        positions=positions,
        output_dtype=q.dtype,
    )
    out_ref = reference.out[:, :, start:end, :]
    lse_ref = reference.lse[:, :, start:end]

    dout = torch.randn(
        q.shape,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 10_000),
        dtype=q.dtype,
    ).to(q.device)
    (distributed.out.float() * dout[:, :, start:end, :].float()).sum().backward()
    (reference.out.float() * dout.float()).sum().backward()

    comparisons = {
        "out": (distributed.out, out_ref),
        "lse": (distributed.lse, lse_ref),
        "dq": (q_local.grad, q_ref.grad[:, :, start:end, :]),
        "dk": (k_local.grad, k_ref.grad[:, :, start:end, :]),
        "dv": (v_local.grad, v_ref.grad[:, :, start:end, :]),
    }
    bitwise = {
        name: left is not None and right is not None and torch.equal(left, right)
        for name, (left, right) in comparisons.items()
    }
    max_abs = {
        name: float((left.float() - right.float()).abs().max().item())
        for name, (left, right) in comparisons.items()
        if left is not None and right is not None
    }

    repeat_out_bitwise = True
    repeat_lse_bitwise = True
    for _ in range(args.repeats - 1):
        repeated = op(
            q_local.detach(),
            k_local.detach(),
            v_local.detach(),
            metadata,
            config=config,
        )
        repeat_out_bitwise = repeat_out_bitwise and torch.equal(repeated.out, distributed.out)
        repeat_lse_bitwise = repeat_lse_bitwise and torch.equal(repeated.lse, distributed.lse)

    provenance = distributed.provenance
    identity_errors = _strict_shared_core_identity_errors(
        provenance,
        transport=args.transport,
        is_rocm=torch.version.hip is not None,
    )
    return {
        "executed": True,
        "passed": (
            all(bitwise.values())
            and repeat_out_bitwise
            and repeat_lse_bitwise
            and not identity_errors
        ),
        "strict_core_id": provenance.get("strict_core_id"),
        "strict_schedule": provenance.get("strict_schedule"),
        "actual_backend": provenance.get("actual_backend"),
        "communication_backend": provenance.get("communication_backend"),
        "production_ready": provenance.get("production_ready"),
        "strict_mode": provenance.get("strict_mode"),
        "native_attention_arithmetic": provenance.get("native_attention_arithmetic"),
        "fallback": provenance.get("fallback"),
        "split_kv_policy": provenance.get("strict_split_kv"),
        "communication_autograd": provenance.get("strict_comm_autograd"),
        "strict_provenance": provenance,
        "identity_errors": identity_errors,
        "bitwise": bitwise,
        "max_abs": max_abs,
        "repeat_out_bitwise": repeat_out_bitwise,
        "repeat_lse_bitwise": repeat_lse_bitwise,
    }


def _strict_shared_core_identity_errors(
    provenance: dict[str, object],
    *,
    transport: str,
    is_rocm: bool,
) -> list[str]:
    """Return every strict-contract provenance mismatch for the rank report."""

    expected_core = (
        STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID if is_rocm else STRICT_ATTENTION_PRODUCTION_CORE_ID
    )
    expected_schedule = (
        STRICT_ATTENTION_ROCM_SCHEDULE_ID if is_rocm else STRICT_ATTENTION_FA4_SCHEDULE_ID
    )
    expected_backend = "aiter.rocm.ck_dense_mha" if is_rocm else "flash_attention_4.cute"
    expected_rope = "rlkernel.rocm.deterministic_rope" if is_rocm else "rlkernel.cuda.rope_sm90"
    expected_communication = "rccl_ag_rs" if is_rocm else "self_owned_cuda_ag_rs"
    required = {
        "strict_core_id": expected_core,
        "strict_schedule": expected_schedule,
        "attention_backend": expected_backend,
        "actual_backend": expected_backend,
        "rope_backend": expected_rope,
        "strict_mode": True,
        "native_attention_arithmetic": True,
        "num_splits": 1,
        "deterministic_backward": True,
        "reference_only": False,
        "fallback": False,
        "strict_split_kv": "disabled",
        "strict_comm_autograd": True,
        "communication_backend": expected_communication,
        "production_ready": True,
        "strict_full_qkv_all_gather": True,
        "strict_position_ids_all_gather": True,
        "compute_communication": "decoupled",
        "compute_schedule": STRICT_ATTENTION_RING_SCHEDULE_ID,
        "communication_overlap": "disabled",
        "ring_schedule_default": True,
        "ring_partial_arithmetic": False,
        "rope_fusion": False,
        "q_rope_state": "post_rope",
        "k_cache_rope_state": "post_rope",
    }
    errors = [
        f"{name}={provenance.get(name)!r}, expected {expected!r}"
        for name, expected in required.items()
        if provenance.get(name) != expected
    ]
    if is_rocm:
        if provenance.get("split_kv_control") != "dense_non_split_api":
            errors.append("split_kv_control must prove the AITER dense non-Split-K API")
        if provenance.get("aiter_api_source") != "aiter.ops.mha":
            errors.append("aiter_api_source must identify aiter.ops.mha")
        if not provenance.get("aiter_source_sha256"):
            errors.append("aiter_source_sha256 is missing")
    else:
        if provenance.get("fa_api_source") != "flash_attn.cute.interface":
            errors.append("fa_api_source must identify the FA4 CuTe API")
    return errors


def _strict_attention_core():
    if torch.version.hip is not None:
        from rl_engine.backends.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore

        return StrictRocmAiterCKAttentionCore()
    return StrictFlashAttention4Core()


def _strict_attention_reference_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    positions: torch.Tensor,
    output_dtype: torch.dtype,
) -> SimpleNamespace:
    """Run the production core with its one-logical-row execution contract."""

    core = _strict_attention_core()
    rows = [
        core.forward_with_lse(
            q[index : index + 1],
            k[index : index + 1],
            v[index : index + 1],
            causal=True,
            query_position_ids=positions[index : index + 1],
            key_position_ids=positions[index : index + 1],
            output_dtype=output_dtype,
        )
        for index in range(q.size(0))
    ]
    return SimpleNamespace(
        out=torch.cat([row.out for row in rows], dim=0),
        lse=torch.cat([row.lse for row in rows], dim=0),
    )


def _strict_rope_op():
    from rl_engine.backends.cuda.rope.rope import RocmDeterministicRoPEOp, RoPESM90Op

    if torch.version.hip is not None:
        return RocmDeterministicRoPEOp()
    return RoPESM90Op()


if __name__ == "__main__":
    raise SystemExit(main())
