# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""WS2 CP attention drift benchmark and report artifact generator.

This is the PR5 artifact path for issue #235. It is intentionally rank-aware
and torchrun-friendly, but the correctness surface remains the deterministic
PyTorch CP reference. The benchmark can run as a CPU smoke test on one process
or under torchrun; rank 0 writes the shared JSON report.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.reference.attention.cp_attention import (  # noqa: E402
    AttentionPartialState,
    DeterministicCPAttentionReferenceOp,
    build_reference_split_kv_runtime_plan_set,
    compare_cp_attention_backward,
    merge_attention_partial_states,
    split_kv_execution_plan_provenance,
)
from rl_engine.reference.definitions import selected_logprobs_reference  # noqa: E402
from rl_engine.reference.rope.rope import NativeRoPEOp  # noqa: E402

SCHEMA_VERSION = "ws2_cp_attention_drift/v2"
ISSUE = 235
PR = 5
DEFAULT_SEQ_LEN = 16
QWEN3_8B_HEADS = 32
QWEN3_8B_KV_HEADS = 8
QWEN3_8B_HEAD_DIM = 128
QWEN3_8B_ROPE_THETA = 1_000_000.0
TE_CONTEXT_PARALLEL_MODULE = (
    "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"
)
TE_SYMBOLS = (
    "flash_attn_fwd_softmax_lse_correction",
    "flash_attn_fwd_out_correction_init",
    "flash_attn_fwd_out_correction",
)


class TEContextParallelMergeAdapter:
    """Optional Transformer Engine CP merge oracle used only by PR5 reports."""

    def __init__(self, module: Any, *, version: str) -> None:
        self._module = module
        self.version = version

    @classmethod
    def probe(cls) -> tuple["TEContextParallelMergeAdapter | None", dict[str, object]]:
        status: dict[str, object] = {
            "te_available": False,
            "te_version": None,
            "te_module": TE_CONTEXT_PARALLEL_MODULE,
            "te_symbols": list(TE_SYMBOLS),
            "te_capability_probe": "unavailable",
            "te_signature_checked": False,
            "te_numeric_selftest": "not_run",
            "fallback": True,
            "fallback_reason": None,
        }
        try:
            module = importlib.import_module(TE_CONTEXT_PARALLEL_MODULE)
            version = _transformer_engine_version()
            missing = [name for name in TE_SYMBOLS if not hasattr(module, name)]
            if missing:
                status.update(
                    {
                        "te_version": version,
                        "te_capability_probe": "missing_symbols",
                        "fallback_reason": f"missing symbols: {', '.join(missing)}",
                    }
                )
                return None, status
            adapter = cls(module, version=version)
            status.update(
                {
                    "te_available": True,
                    "te_version": version,
                    "te_signature_checked": True,
                }
            )
            adapter._numeric_selftest()
        except (
            ImportError,
            OSError,
            RuntimeError,
            AttributeError,
            TypeError,
            AssertionError,
        ) as exc:
            status.update(
                {
                    "te_capability_probe": "failed",
                    "te_numeric_selftest": "failed",
                    "fallback_reason": str(exc),
                }
            )
            return None, status

        status.update(
            {
                "te_capability_probe": "passed",
                "te_numeric_selftest": "passed",
                "fallback": False,
                "fallback_reason": None,
            }
        )
        return adapter, status

    def merge(self, states: Sequence[AttentionPartialState]) -> AttentionPartialState:
        if not states:
            raise ValueError("at least one partial state is required")
        ordered = sorted(states, key=lambda item: (item.block_start, item.block_end))
        if len(ordered) == 1:
            state = ordered[0]
            return AttentionPartialState(
                out=state.out.float().clone(),
                lse=state.lse.float().clone(),
                block_start=state.block_start,
                block_end=state.block_end,
            )

        merged_lse = ordered[0].lse.float().clone()
        merged_out = ordered[0].out.float().clone()
        for state in ordered[1:]:
            next_lse = state.lse.float()
            previous_lse = merged_lse.clone()
            self._module.flash_attn_fwd_softmax_lse_correction(merged_lse, next_lse)
            merged_out = self._module.flash_attn_fwd_out_correction_init(
                merged_out,
                merged_lse,
                previous_lse,
                seq_dim=2,
            )
            self._module.flash_attn_fwd_out_correction(
                merged_out,
                state.out.float(),
                merged_lse,
                next_lse,
                seq_dim=2,
            )

        return AttentionPartialState(
            out=merged_out,
            lse=merged_lse,
            block_start=ordered[0].block_start,
            block_end=ordered[-1].block_end,
        )

    def _numeric_selftest(self) -> None:
        gen = torch.Generator().manual_seed(5)
        states = [
            AttentionPartialState(
                out=torch.randn(1, 2, 3, 4, generator=gen),
                lse=torch.randn(1, 2, 3, generator=gen),
                block_start=0,
                block_end=2,
            ),
            AttentionPartialState(
                out=torch.randn(1, 2, 3, 4, generator=gen),
                lse=torch.randn(1, 2, 3, generator=gen),
                block_start=2,
                block_end=5,
            ),
        ]
        ours = merge_attention_partial_states(states)
        te = self.merge(states)
        torch.testing.assert_close(te.lse, ours.lse, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(te.out, ours.out, atol=1.0e-6, rtol=0.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WS2 CP attention drift benchmark for issue #235 PR5."
    )
    parser.add_argument("--model", default="qwen3-8b", choices=["qwen3-8b"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=2355)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--tp-world-sizes", default="1,2")
    parser.add_argument("--cp-world-sizes", default="1,2")
    parser.add_argument(
        "--kv-chunk-sizes",
        default="none,4",
        help="Comma list such as 'none,4'. 'none' means full prefill.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny CPU-friendly shape.")
    parser.add_argument(
        "--include-backward",
        action="store_true",
        help="Include optional PR8 dq/dk/dv drift fields.",
    )
    parser.add_argument(
        "--include-dlogp",
        action="store_true",
        help=(
            "Project attention outputs through a deterministic synthetic lm_head "
            "and report active-token dlogp drift."
        ),
    )
    parser.add_argument(
        "--no-rope",
        action="store_false",
        dest="compose_rope",
        help="Disable the pre-attention RoPE composition step.",
    )
    parser.set_defaults(compose_rope=True)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--init-process-group",
        action="store_true",
        help="Initialize torch.distributed from torchrun env vars before benchmarking.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON artifact path.")
    parser.add_argument("--json", action="store_true", help="Print the JSON report on rank 0.")
    return parser.parse_args(argv)


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    rank_env = _rank_env()
    device = _resolve_device(args.device, rank_env)
    _validate_args(args)
    distributed = _maybe_init_process_group(args, device, rank_env)
    te_adapter, te_status = TEContextParallelMergeAdapter.probe()
    seq_len = 4 if args.smoke and args.seq_len == DEFAULT_SEQ_LEN else args.seq_len
    kv_chunk_sizes = _parse_kv_chunk_sizes(args.kv_chunk_sizes)
    if args.smoke and args.kv_chunk_sizes == "none,4":
        kv_chunk_sizes = (None, 1)

    cases: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "report_family": "ws2_cross_config_drift_report",
        "tolerance_source": "#108",
        "issue": ISSUE,
        "pr": PR,
        "created_at_utc": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "launch": _launch_metadata(rank_env),
        "runtime": _runtime_metadata(device, distributed, rank_env),
        "target": {
            "model": args.model,
            "global_num_query_heads": QWEN3_8B_HEADS,
            "global_num_kv_heads": QWEN3_8B_KV_HEADS,
            "head_dim": QWEN3_8B_HEAD_DIM,
            "dtype": args.dtype,
            "batch": args.batch,
            "seq_len": seq_len,
            "causal": True,
        },
        "te_context_parallel_merge": te_status,
        "dlogp": {
            "status": "requested" if args.include_dlogp else "not_requested",
            "reason": None if args.include_dlogp else "selected-logprob chain was not requested",
            "source": "synthetic_fp32_lm_head_projection",
        },
        "cases": cases,
    }

    try:
        with _thread_limit(args.num_threads):
            for tp_world_size in _parse_int_csv(args.tp_world_sizes, name="tp_world_sizes"):
                for cp_world_size in _parse_int_csv(args.cp_world_sizes, name="cp_world_sizes"):
                    for kv_chunk_size in kv_chunk_sizes:
                        cases.append(
                            _run_case(
                                args,
                                device=device,
                                seq_len=seq_len,
                                tp_world_size=tp_world_size,
                                cp_world_size=cp_world_size,
                                kv_chunk_size=kv_chunk_size,
                                te_adapter=te_adapter,
                                include_dlogp=args.include_dlogp,
                            )
                        )
    finally:
        if distributed["initialized"]:
            import torch.distributed as dist

            if sys.exc_info()[0] is None:
                dist.barrier()
            dist.destroy_process_group()
    return report


def write_report(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    rank = int(report["launch"]["rank"])
    if rank == 0:
        if args.output is not None:
            write_report(report, args.output)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _run_case(
    args: argparse.Namespace,
    *,
    device: torch.device,
    seq_len: int,
    tp_world_size: int,
    cp_world_size: int,
    kv_chunk_size: int | None,
    te_adapter: TEContextParallelMergeAdapter | None,
    include_dlogp: bool,
) -> dict[str, object]:
    _validate_topology(tp_world_size, cp_world_size)
    dtype = _dtype_from_name(args.dtype)
    local_hq = QWEN3_8B_HEADS // tp_world_size
    local_hkv = QWEN3_8B_KV_HEADS // tp_world_size
    case_seed = _case_seed(args.seed, tp_world_size, cp_world_size, kv_chunk_size)
    q, k, v, rope_report = _make_qkv(
        batch=args.batch,
        local_hq=local_hq,
        local_hkv=local_hkv,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
        seed=case_seed,
        compose_rope=args.compose_rope,
    )
    dout = _make_dout(
        batch=args.batch,
        local_hq=local_hq,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
        seed=case_seed + 17,
    )
    attention = DeterministicCPAttentionReferenceOp()
    reference_out, reference_lse = attention.forward_fp32_with_lse(
        q,
        k,
        v,
        causal=True,
        cp_world_size=1,
        kv_chunk_size=None,
    )
    candidate_fp32_out, candidate_fp32_lse = attention.forward_fp32_with_lse(
        q,
        k,
        v,
        causal=True,
        cp_world_size=cp_world_size,
        kv_chunk_size=kv_chunk_size,
    )
    candidate_dtype_out, candidate_dtype_lse = attention.forward_with_lse(
        q,
        k,
        v,
        causal=True,
        cp_world_size=cp_world_size,
        kv_chunk_size=kv_chunk_size,
        output_dtype=dtype,
    )

    split_kv_policy = "disabled" if kv_chunk_size is None else "fixed"
    attention_mode = "prefill" if kv_chunk_size is None else "chunked_prefill"
    q_bounds = _split_bounds(seq_len, cp_world_size)
    kv_bounds = _kv_block_bounds(seq_len, cp_world_size, kv_chunk_size)
    runtime_plan_set = build_reference_split_kv_runtime_plan_set(
        (seq_len,) * args.batch,
        tp_world_size=tp_world_size,
        cp_world_size=cp_world_size,
        kv_chunk_size=kv_chunk_size,
        backend="deterministic_cp_reference",
    )
    dlogp_report = _dlogp_report(
        candidate_dtype_out,
        reference_out.to(dtype),
        batch=args.batch,
        seq_len=seq_len,
        local_hidden=local_hq * QWEN3_8B_HEAD_DIM,
        seed=case_seed + 101,
        device=device,
        enabled=include_dlogp,
    )
    case: dict[str, object] = {
        "case_name": _case_name(tp_world_size, cp_world_size, kv_chunk_size, args.dtype),
        "attention_mode": attention_mode,
        "model": args.model,
        "topology": {
            "tp_world_size": tp_world_size,
            "cp_world_size": cp_world_size,
            "tp_rank": 0,
            "logical_cp_ranks": cp_world_size,
            "local_num_query_heads": local_hq,
            "local_num_kv_heads": local_hkv,
            "local_query_head_range": [0, local_hq],
            "local_kv_head_range": [0, local_hkv],
            "head_dim": QWEN3_8B_HEAD_DIM,
            "q_sequence_bounds": [list(item) for item in q_bounds],
            "kv_block_bounds": [list(item) for item in kv_bounds],
        },
        "provenance": {
            "backend": "deterministic_cp_reference",
            "reference_backend": "cp1_fp32_prefill",
            "candidate_backend": "cp_reference",
            "dtype": args.dtype,
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_domain": "attention",
            "merge_order": "global_block_index",
            "split_kv_policy": split_kv_policy,
            "requested_split_kv_policy": split_kv_policy,
            "requested_split_kv_size": kv_chunk_size,
            "actual_split_kv_plans": split_kv_execution_plan_provenance(
                seq_len,
                cp_world_size=cp_world_size,
                kv_chunk_size=kv_chunk_size,
                backend="deterministic_cp_reference",
            ),
            "actual_split_kv_plan_set": runtime_plan_set.to_dict(),
            "kv_chunk_size": kv_chunk_size,
            "block_metadata_hash": _block_metadata_hash(kv_bounds),
            "scale_placement": "scores_after_qk_matmul",
            "mask_application_order": ["scale", "causal_mask", "key_padding_mask"],
            "dropout_policy": "disabled",
            "deterministic_controls": {
                "reference": "strict_fp32_math_inside_cp_attention",
                "num_threads": args.num_threads,
            },
            "rope": rope_report["provenance"],
        },
        "drift": {
            "cp_merge_fp32": {
                "out": _drift_stats(candidate_fp32_out, reference_out),
                "lse": _drift_stats(candidate_fp32_lse, reference_lse),
                "source_class": "reduction_and_collective_drift",
            },
            "dtype_path_vs_fp32": {
                "out": _drift_stats(candidate_dtype_out, reference_out),
                "lse": _drift_stats(candidate_dtype_lse, reference_lse),
                "source_class": "arithmetic_schedule_drift",
            },
            "rope": rope_report["drift"],
        },
        "merge_order_probe": _merge_order_probe(
            attention,
            q,
            k,
            v,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
        ),
        "te_merge_oracle": _te_merge_oracle_probe(
            attention,
            q,
            k,
            v,
            cp_world_size=cp_world_size,
            kv_chunk_size=kv_chunk_size,
            te_adapter=te_adapter,
        ),
        "per_rank": _per_rank_forward_drifts(
            candidate_fp32_out,
            candidate_fp32_lse,
            reference_out,
            reference_lse,
            cp_world_size,
        ),
        "backward": {"status": "not_requested"},
        "dlogp": dlogp_report,
    }
    distributed_reference = _run_distributed_p2p_reference(
        q,
        k,
        v,
        reference_out,
        reference_lse,
        device=device,
        seq_len=seq_len,
        tp_world_size=tp_world_size,
        cp_world_size=cp_world_size,
        kv_chunk_size=kv_chunk_size,
    )
    case["distributed_p2p_reference"] = distributed_reference
    if args.include_backward:
        backward = compare_cp_attention_backward(
            q,
            k,
            v,
            dout,
            causal=True,
            candidate_cp_world_size=cp_world_size,
            candidate_kv_chunk_size=kv_chunk_size,
            output_dtype=dtype,
        )
        case["backward"] = {
            "status": "available",
            "report": backward.to_dict(),
        }
    return case


def _run_distributed_p2p_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reference_out: torch.Tensor,
    reference_lse: torch.Tensor,
    *,
    device: torch.device,
    seq_len: int,
    tp_world_size: int,
    cp_world_size: int,
    kv_chunk_size: int | None,
) -> dict[str, object]:
    """Exercise the actual P2P reference when launched as a matching NCCL job."""

    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return {"status": "not_requested", "reason": "process_group_not_initialized"}
    backend = str(dist.get_backend()).lower()
    world_size = int(dist.get_world_size())
    if device.type != "cuda":
        return {
            "status": "skipped",
            "reason": "P2P NCCL reference requires CUDA",
            "backend": backend,
        }
    if "nccl" not in backend:
        return {
            "status": "skipped",
            "reason": "P2P reference requires NCCL",
            "backend": backend,
        }
    if world_size != cp_world_size:
        return {
            "status": "skipped",
            "reason": "WORLD_SIZE must equal cp_world_size for the CP reference",
            "world_size": world_size,
            "cp_world_size": cp_world_size,
        }

    try:
        from rl_engine.backends.cuda.attention.cp_comm import (
            AttentionCPBlockMetadata,
            AttentionCPCommunicationPlan,
            AttentionCPMergedState,
            AttentionCPPartialState,
            AttentionParallelSpec,
            P2PNCCLAttentionCPCommunication,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "rl_engine.backends.cuda.attention.cp_comm":
            raise
        return {
            "status": "unavailable",
            "reason": "PR7 CP communication module is not present in this checkout",
            "required_dependency": "#279",
        }

    rank = int(dist.get_rank())
    owner_ranges = _split_bounds(seq_len, cp_world_size)
    block_bounds = _kv_block_bounds(seq_len, cp_world_size, kv_chunk_size)
    blocks: list[AttentionCPBlockMetadata] = []
    owner_block_counts = [0] * cp_world_size
    for block_index, (start, end) in enumerate(block_bounds):
        owner = next(
            owner_rank
            for owner_rank, (owner_start, owner_end) in enumerate(owner_ranges)
            if owner_start <= start < owner_end
        )
        blocks.append(
            AttentionCPBlockMetadata(
                global_block_index=block_index,
                kv_block_start=start,
                kv_block_end=end,
                owner_cp_rank=owner,
                owner_tp_rank=0,
            )
        )
        owner_block_counts[owner] += 1

    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=tp_world_size,
            tp_rank=0,
            cp_world_size=cp_world_size,
            cp_rank=rank,
        ),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=tuple(blocks),
        expected_kv_token_range=(0, seq_len),
        query_token_ranges=tuple(_split_bounds(q.size(2), cp_world_size)),
    )
    attention = DeterministicCPAttentionReferenceOp()
    local_states: list[AttentionCPPartialState] = []
    for block in blocks:
        if block.owner_cp_rank != rank:
            continue
        state = attention.local_partial_state(
            q,
            k[:, :, block.kv_block_start : block.kv_block_end, :],
            v[:, :, block.kv_block_start : block.kv_block_end, :],
            q_start=0,
            k_start=block.kv_block_start,
            total_kv_len=seq_len,
            total_query_len=q.size(2),
            causal=True,
        )
        local_states.append(
            AttentionCPPartialState(
                out=state.out,
                lse=state.lse,
                block=block,
            )
        )
    communication = P2PNCCLAttentionCPCommunication()
    gathered = communication.all_gather_partial_states(tuple(local_states), plan)
    merged = merge_attention_partial_states(
        [
            AttentionPartialState(
                out=state.out,
                lse=state.lse,
                block_start=state.block.kv_block_start,
                block_end=state.block.kv_block_end,
            )
            for state in gathered
        ]
    )
    local = communication.reduce_scatter_merged_state(
        AttentionCPMergedState(out=merged.out, lse=merged.lse),
        plan,
    )
    q_start, q_end = plan.query_token_ranges[rank]
    reference_local_out = reference_out[:, :, q_start:q_end, :]
    reference_local_lse = reference_lse[:, :, q_start:q_end]
    return {
        "status": "available",
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "transport": "p2p_nccl_reference",
        "manifest_block_count": len(blocks),
        "owner_block_counts": owner_block_counts,
        "gathered_block_indices": [state.block.global_block_index for state in gathered],
        "query_range": [q_start, q_end],
        "out": _drift_stats(local.out, reference_local_out),
        "lse": _drift_stats(local.lse, reference_local_lse),
    }


def _make_qkv(
    *,
    batch: int,
    local_hq: int,
    local_hkv: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    compose_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q_pre = torch.randn(batch, local_hq, seq_len, QWEN3_8B_HEAD_DIM, generator=gen)
    k_pre = torch.randn(batch, local_hkv, seq_len, QWEN3_8B_HEAD_DIM, generator=gen)
    v = torch.randn(batch, local_hkv, seq_len, QWEN3_8B_HEAD_DIM, generator=gen)
    q_pre = q_pre.to(device=device, dtype=dtype)
    k_pre = k_pre.to(device=device, dtype=dtype)
    v = v.to(device=device, dtype=dtype)

    positions = (
        torch.arange(seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(
            batch,
            -1,
        )
    )
    if not compose_rope:
        return q_pre, k_pre, v, _rope_report_disabled()

    rope = NativeRoPEOp()
    q_rope_dtype = rope.forward(q_pre, positions, theta=QWEN3_8B_ROPE_THETA)
    k_rope_dtype = rope.forward(k_pre, positions, theta=QWEN3_8B_ROPE_THETA)
    q_rope_fp32 = rope.forward_fp32(q_pre, positions, theta=QWEN3_8B_ROPE_THETA)
    k_rope_fp32 = rope.forward_fp32(k_pre, positions, theta=QWEN3_8B_ROPE_THETA)
    return (
        q_rope_dtype,
        k_rope_dtype,
        v,
        {
            "provenance": {
                "rope_state": "post_rope",
                "rope_theta": QWEN3_8B_ROPE_THETA,
                "rope_scaling": None,
                "rotary_dim": QWEN3_8B_HEAD_DIM,
                "position_ids": "arange(seq_len)",
                "cache_position": "same_as_position_ids",
                "query_position_offsets": [0 for _ in range(batch)],
                "key_position_offsets": [0 for _ in range(batch)],
                "k_cache_rope_state": "post_rope",
                "rope_cast_at": "rope_output",
                "rope_output_dtype": _dtype_name(dtype),
                "fusion_boundary": "unfused_rope_attention_reference",
            },
            "drift": {
                "status": "available",
                "q": _drift_stats(q_rope_dtype, q_rope_fp32),
                "k": _drift_stats(k_rope_dtype, k_rope_fp32),
            },
        },
    )


def _make_dout(
    *,
    batch: int,
    local_hq: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(
        batch,
        local_hq,
        seq_len,
        QWEN3_8B_HEAD_DIM,
        generator=gen,
        dtype=dtype,
    ).to(device=device)


def _rope_report_disabled() -> dict[str, object]:
    return {
        "provenance": {
            "rope_state": "not_composed",
            "fusion_boundary": "attention_only",
        },
        "drift": {
            "status": "not_composed",
            "q": None,
            "k": None,
        },
    }


def _merge_order_probe(
    attention: DeterministicCPAttentionReferenceOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cp_world_size: int,
    kv_chunk_size: int | None,
) -> dict[str, object]:
    reversed_out: list[torch.Tensor] = []
    reversed_lse: list[torch.Tensor] = []
    sorted_out: list[torch.Tensor] = []
    sorted_lse: list[torch.Tensor] = []
    kv_bounds = _kv_block_bounds(k.size(2), cp_world_size, kv_chunk_size)
    for q_start, q_end in _split_bounds(q.size(2), cp_world_size):
        if q_start == q_end:
            continue
        states = _partial_states_for_query_block(
            attention,
            q,
            k,
            v,
            q_start=q_start,
            q_end=q_end,
            kv_bounds=kv_bounds,
        )
        sorted_merge = merge_attention_partial_states(states)
        reversed_merge = merge_attention_partial_states(list(reversed(states)))
        sorted_out.append(sorted_merge.out)
        sorted_lse.append(sorted_merge.lse)
        reversed_out.append(reversed_merge.out)
        reversed_lse.append(reversed_merge.lse)
    if not sorted_out:
        return {
            "status": "empty_query",
            "arrival_order_policy": "ignored_then_sorted_by_global_block_index",
        }
    return {
        "status": "available",
        "arrival_order_policy": "ignored_then_sorted_by_global_block_index",
        "out": _drift_stats(torch.cat(reversed_out, dim=2), torch.cat(sorted_out, dim=2)),
        "lse": _drift_stats(torch.cat(reversed_lse, dim=2), torch.cat(sorted_lse, dim=2)),
    }


def _te_merge_oracle_probe(
    attention: DeterministicCPAttentionReferenceOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cp_world_size: int,
    kv_chunk_size: int | None,
    te_adapter: TEContextParallelMergeAdapter | None,
) -> dict[str, object]:
    if te_adapter is None:
        return {
            "status": "unavailable",
            "te_module": TE_CONTEXT_PARALLEL_MODULE,
            "fallback": "deterministic_cp_reference",
        }

    ours_out: list[torch.Tensor] = []
    ours_lse: list[torch.Tensor] = []
    te_out: list[torch.Tensor] = []
    te_lse: list[torch.Tensor] = []
    kv_bounds = _kv_block_bounds(k.size(2), cp_world_size, kv_chunk_size)
    for q_start, q_end in _split_bounds(q.size(2), cp_world_size):
        if q_start == q_end:
            continue
        states = _partial_states_for_query_block(
            attention,
            q,
            k,
            v,
            q_start=q_start,
            q_end=q_end,
            kv_bounds=kv_bounds,
        )
        ours = merge_attention_partial_states(states)
        te = te_adapter.merge(states)
        ours_out.append(ours.out)
        ours_lse.append(ours.lse)
        te_out.append(te.out)
        te_lse.append(te.lse)
    if not ours_out:
        return {"status": "empty_query", "te_version": te_adapter.version}
    return {
        "status": "available",
        "te_version": te_adapter.version,
        "te_module": TE_CONTEXT_PARALLEL_MODULE,
        "te_symbols": list(TE_SYMBOLS),
        "out": _drift_stats(torch.cat(te_out, dim=2), torch.cat(ours_out, dim=2)),
        "lse": _drift_stats(torch.cat(te_lse, dim=2), torch.cat(ours_lse, dim=2)),
    }


def _partial_states_for_query_block(
    attention: DeterministicCPAttentionReferenceOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_start: int,
    q_end: int,
    kv_bounds: Sequence[tuple[int, int]],
) -> list[AttentionPartialState]:
    return [
        attention.local_partial_state(
            q[:, :, q_start:q_end, :],
            k[:, :, k_start:k_end, :],
            v[:, :, k_start:k_end, :],
            q_start=q_start,
            k_start=k_start,
            total_kv_len=k.size(2),
            total_query_len=q.size(2),
            causal=True,
        )
        for k_start, k_end in kv_bounds
        if k_start != k_end
    ]


def _per_rank_forward_drifts(
    candidate_out: torch.Tensor,
    candidate_lse: torch.Tensor,
    reference_out: torch.Tensor,
    reference_lse: torch.Tensor,
    cp_world_size: int,
) -> list[dict[str, object]]:
    per_rank = []
    for rank, (q_start, q_end) in enumerate(_split_bounds(candidate_out.size(2), cp_world_size)):
        per_rank.append(
            {
                "rank": rank,
                "query_start": q_start,
                "query_end": q_end,
                "out": _drift_stats(
                    candidate_out[:, :, q_start:q_end, :],
                    reference_out[:, :, q_start:q_end, :],
                ),
                "lse": _drift_stats(
                    candidate_lse[:, :, q_start:q_end],
                    reference_lse[:, :, q_start:q_end],
                ),
            }
        )
    return per_rank


def _drift_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match "
            f"reference shape {tuple(reference.shape)}"
        )
    diff = (candidate.float() - reference.float()).abs().reshape(-1)
    active_count = int(diff.numel())
    if active_count == 0:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p95_abs": 0.0,
            "p99_abs": 0.0,
            "active_count": 0,
        }
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "p95_abs": float(torch.quantile(diff, 0.95).item()),
        "p99_abs": float(torch.quantile(diff, 0.99).item()),
        "active_count": active_count,
    }


def _dlogp_report(
    candidate_out: torch.Tensor,
    reference_out: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    local_hidden: int,
    seed: int,
    device: torch.device,
    enabled: bool,
) -> dict[str, object]:
    """Run the selected-token log-probability leg on the attention outputs.

    The benchmark intentionally uses a small deterministic synthetic projection,
    but performs both logits and log-softmax in FP32.  The reference is first cast
    to the candidate's final-write dtype, so this gate compares the same BF16 cast
    boundary and does not mislabel the expected FP32-to-BF16 write as CP drift.
    """

    if not enabled:
        return {
            "status": "not_requested",
            "reason": "use --include-dlogp to exercise the selected-logprob chain",
        }
    if candidate_out.shape != reference_out.shape:
        raise ValueError("candidate and reference attention outputs must have matching shapes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab_size = 17
    weight = torch.randn(
        vocab_size,
        local_hidden,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    target_ids = (
        torch.arange(batch * seq_len, device=device, dtype=torch.long).reshape(batch, seq_len)
        % vocab_size
    )
    active_mask = torch.ones((batch, seq_len), device=device, dtype=torch.bool)
    if seq_len > 1:
        active_mask[:, 0] = False

    def project(out: torch.Tensor) -> torch.Tensor:
        hidden = out.float().transpose(1, 2).reshape(batch, seq_len, local_hidden)
        logits = torch.matmul(hidden, weight.transpose(0, 1))
        return selected_logprobs_reference(
            logits,
            target_ids,
            mask=active_mask,
            output_dtype=torch.float32,
        )

    candidate_logp = project(candidate_out)
    reference_logp = project(reference_out)
    return {
        "status": "available",
        "projection": "synthetic_fp32_lm_head_projection",
        "reference_cast_dtype": str(reference_out.dtype).removeprefix("torch."),
        "vocab_size": vocab_size,
        "active_token_count": int(active_mask.sum().item()),
        "drift": _drift_stats(
            candidate_logp[active_mask],
            reference_logp[active_mask],
        ),
    }


def _rank_env() -> dict[str, int | bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "torchrun": "RANK" in os.environ or "WORLD_SIZE" in os.environ,
    }


def _resolve_device(device_arg: str, rank_env: dict[str, int | bool]) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        local_rank = int(rank_env["local_rank"])
        if torch.cuda.device_count() > 0:
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _maybe_init_process_group(
    args: argparse.Namespace,
    device: torch.device,
    rank_env: dict[str, int | bool],
) -> dict[str, object]:
    initialized = False
    backend = None
    if args.init_process_group and int(rank_env["world_size"]) > 1:
        import torch.distributed as dist

        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        initialized = True
    return {
        "initialized": initialized,
        "backend": backend,
        "transport": "torchrun_env_rank_aware",
    }


def _runtime_metadata(
    device: torch.device,
    distributed: dict[str, object],
    rank_env: dict[str, int | bool],
) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "distributed": distributed,
        "rank_env": rank_env,
    }


def _launch_metadata(rank_env: dict[str, int | bool]) -> dict[str, object]:
    return {
        "command": _shell_join(sys.argv),
        "rank": int(rank_env["rank"]),
        "world_size": int(rank_env["world_size"]),
        "local_rank": int(rank_env["local_rank"]),
        "torchrun": bool(rank_env["torchrun"]),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch < 1:
        raise ValueError("batch must be >= 1")
    if args.seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if args.num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    for tp_world_size in _parse_int_csv(args.tp_world_sizes, name="tp_world_sizes"):
        _validate_topology(tp_world_size, 1)
    for cp_world_size in _parse_int_csv(args.cp_world_sizes, name="cp_world_sizes"):
        _validate_topology(1, cp_world_size)
    _parse_kv_chunk_sizes(args.kv_chunk_sizes)


def _validate_topology(tp_world_size: int, cp_world_size: int) -> None:
    if tp_world_size < 1 or cp_world_size < 1:
        raise ValueError("tp_world_size and cp_world_size must be >= 1")
    if QWEN3_8B_HEADS % tp_world_size != 0:
        raise ValueError("Qwen3 query heads must be divisible by tp_world_size")
    if QWEN3_8B_KV_HEADS % tp_world_size != 0:
        raise ValueError("Qwen3 KV heads must be divisible by tp_world_size")


def _parse_int_csv(value: str, *, name: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not parsed:
        raise ValueError(f"{name} must contain at least one integer")
    return tuple(parsed)


def _parse_kv_chunk_sizes(value: str) -> tuple[int | None, ...]:
    parsed: list[int | None] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"none", "full", "no_split"}:
            parsed.append(None)
            continue
        try:
            size = int(item)
        except ValueError as exc:
            raise ValueError("kv_chunk_sizes must contain integers or 'none'") from exc
        if size < 1:
            raise ValueError("kv chunk sizes must be >= 1")
        parsed.append(size)
    if not parsed:
        raise ValueError("kv_chunk_sizes must contain at least one entry")
    return tuple(parsed)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _split_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    base, extra = divmod(length, parts)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        width = base + (1 if index < extra else 0)
        end = start + width
        bounds.append((start, end))
        start = end
    return bounds


def _kv_block_bounds(
    length: int,
    cp_world_size: int,
    kv_chunk_size: int | None,
) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    for start, end in _split_bounds(length, cp_world_size):
        if kv_chunk_size is None:
            bounds.append((start, end))
            continue
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + kv_chunk_size, end)
            bounds.append((cursor, chunk_end))
            cursor = chunk_end
    return bounds


def _block_metadata_hash(bounds: Sequence[tuple[int, int]]) -> str:
    payload = json.dumps([list(item) for item in bounds], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _case_seed(seed: int, tp_world_size: int, cp_world_size: int, kv_chunk_size: int | None) -> int:
    return seed + tp_world_size * 101 + cp_world_size * 17 + (kv_chunk_size or 0)


def _case_name(
    tp_world_size: int,
    cp_world_size: int,
    kv_chunk_size: int | None,
    dtype: str,
) -> str:
    mode = "prefill" if kv_chunk_size is None else f"chunk{kv_chunk_size}"
    return f"qwen3_8b_tp{tp_world_size}_cp{cp_world_size}_{mode}_{dtype}"


def _transformer_engine_version() -> str:
    for package in ("transformer-engine", "transformer_engine"):
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _shell_join(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return " ".join(argv)
    return shlex.join(argv)


@contextlib.contextmanager
def _thread_limit(num_threads: int) -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


if __name__ == "__main__":
    raise SystemExit(main())
