# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""PR7 strict paged-layout Attention validation entry point.

The default dry-run mode is CI/local friendly: it builds the paged-KV plan and
provenance without requiring a GPU vendor backend. On a GPU host, omit
``--dry-run`` to run the platform production core and its strict reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.backends.cuda.attention.cp_comm import (  # noqa: E402
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
)
from rl_engine.backends.cuda.attention.flash_attn import StrictFlashAttention4Core  # noqa: E402
from rl_engine.backends.cuda.attention.flashinfer_paged_attention import (  # noqa: E402
    FlashInferPagedAttentionConfig,
    FlashInferQwen3PagedAttentionOp,
    FlashInferRoPEFusionConfig,
    FlashInferSplitKVPolicy,
    FlashInferUnavailable,
    _apply_strict_rope,
    _materialize_strict_logical_kv,
    build_flashinfer_paged_kv_plan,
)
from rl_engine.contracts.operators.attention import (  # noqa: E402
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
)
from rl_engine.validation.common.attention_comparison import (  # noqa: E402
    AttentionPathResult,
    DecodeAttentionInputs,
    DecodeKVCacheMetadata,
    run_decode_full_prefill_reference,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    device = torch.device("cuda" if args.device == "cuda" else "cpu")
    inputs = _make_inputs(args, device)
    config = _make_config(args)
    config.validate(head_dim=args.head_dim, query_len=args.query_len)
    plan = build_flashinfer_paged_kv_plan(
        inputs.metadata,
        batch_size=args.batch_size,
        query_len=args.query_len,
        cache_capacity=inputs.k_cache.size(2),
        device=device,
    )
    report: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "executed",
        "pr": "PR7",
        "target": "Qwen3-8B TP-local paged Attention; CP transport validated separately",
        "mode": config.mode,
        "device": str(device),
        "shape": {
            "batch_size": args.batch_size,
            "query_len": args.query_len,
            "kv_seq_len": args.kv_seq_len,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "rope": config.rope.provenance(args.head_dim),
        "split_kv": {
            **config.split_kv.to_dict(),
            "provenance_status": "requested_only_dry_run",
            "actual_plan_required_for_strict_pass": config.require_batch_invariant,
            "requested_execution_plans": [
                config.split_kv.resolve(
                    int(seq_len),
                    backend="flashinfer_dry_run_requested_only",
                ).to_dict()
                for seq_len in plan.kv_seq_lens.tolist()
            ],
        },
        "communication": config.cp_comm_plan.provenance()
        | {"cp_comm_required": config.require_cp_comm},
        "paged_kv_plan": plan.provenance(),
        "tests_expected": [
            "platform production core vs direct same-core reference",
            "split-K disabled/fixed policy drift",
            "batch composition/position invariant sweep",
            "attention-domain LSE export drift",
            "strict platform core with separate multi-rank AG/RS forward/backward evidence",
        ],
        "thresholds": {
            "out_max_abs": args.out_atol,
            "lse_max_abs": args.lse_atol,
            "dlogp_max_abs": args.dlogp_atol,
        },
    }
    if args.dry_run:
        report["passed"] = False
        report["acceptance_eligible"] = False
        report["errors"] = ["dry-run does not execute the FlashInfer candidate"]
        _emit(report, json_output=args.json, output=args.output)
        return 0

    if device.type != "cuda":
        raise SystemExit("non-dry-run PR7 validation requires --device cuda")
    op = FlashInferQwen3PagedAttentionOp()
    try:
        candidate = op(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.metadata,
            config=config,
        )
    except (FlashInferUnavailable, RuntimeError) as exc:
        # Keep optional dependency failures machine-readable while preserving
        # a non-zero exit so aggregate acceptance cannot pass closed.
        report.update(
            {
                "status": "not_available",
                "passed": False,
                "acceptance_eligible": False,
                "errors": [f"Attention backend unavailable: {exc}"],
                "unavailable_reason": str(exc),
            }
        )
        _emit(report, json_output=args.json, output=args.output)
        return 1
    reference_inputs = replace(
        inputs,
        metadata=replace(
            inputs.metadata,
            q_rope_state="pre_rope",
            k_cache_rope_state="pre_rope",
        ),
    )
    pytorch_reference = run_decode_full_prefill_reference(reference_inputs)
    reference = (
        _run_strict_platform_reference(inputs, config, plan)
        if config.strict_mode
        else pytorch_reference
    )
    out_stats = _drift_stats(candidate.out, reference.out)
    lse_stats = _drift_stats(candidate.lse, reference.lse)
    dlogp_stats = _selected_logprob_drift(
        candidate.out,
        reference.out,
        seed=args.seed + 101,
        vocab_size=args.vocab_size,
    )
    report["candidate_provenance"] = candidate.provenance
    report["split_kv"].update(
        {
            "provenance_status": "runtime_verified",
            "actual_execution_plans": candidate.provenance.get(
                "actual_split_kv_plans",
                candidate.provenance.get("strict_core_row_plans"),
            ),
            "actual_plan_set": candidate.provenance.get("actual_split_kv_plan_set"),
        }
    )
    report["drift"] = {
        "out": out_stats,
        "lse": lse_stats,
        "dlogp": dlogp_stats,
    }
    report["reference_backend"] = "rlkernel.pytorch.full_logical_kv_reference"
    if config.strict_mode:
        report.update(_strict_execution_report_fields(candidate.provenance))
        report["diagnostic_drift_vs_pytorch"] = {
            "out": _drift_stats(candidate.out, pytorch_reference.out),
            "lse": _drift_stats(candidate.lse, pytorch_reference.lse),
            "dlogp": _selected_logprob_drift(
                candidate.out,
                pytorch_reference.out,
                seed=args.seed + 101,
                vocab_size=args.vocab_size,
            ),
        }
    if config.require_batch_invariant:
        report["batch_invariant_sweep"] = _run_batch_invariance_sweep(
            op,
            inputs,
            candidate,
            config,
        )
    report["page_layout_invariant_sweep"] = _run_page_layout_invariance_sweep(
        op,
        inputs,
        candidate,
        config,
    )
    errors = _acceptance_errors(report, args)
    report["errors"] = errors
    report["passed"] = not errors
    report["acceptance_eligible"] = not errors
    report["status"] = "passed" if not errors else "failed"
    _emit(report, json_output=args.json, output=args.output)
    return 0 if not errors else 1


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-len", type=int, default=1)
    parser.add_argument("--kv-seq-len", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=2357)
    parser.add_argument("--vocab-size", type=int, default=257)
    parser.add_argument("--out-atol", type=float, default=1.0e-2)
    parser.add_argument("--lse-atol", type=float, default=2.0e-3)
    parser.add_argument("--dlogp-atol", type=float, default=2.0e-3)
    parser.add_argument("--tp-world-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--cp-world-size", type=int, default=2)
    parser.add_argument("--cp-rank", type=int, default=0)
    parser.add_argument(
        "--cp-comm-backend",
        choices=["cuda_ag_rs", "local_debug"],
        default="cuda_ag_rs",
    )
    parser.add_argument("--require-cp-comm", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="use FlashInfer only for paged layout and execute the RL-Kernel shared core",
    )
    parser.add_argument("--fixed-split-size", type=int, default=None)
    parser.add_argument(
        "--split-kv-policy",
        choices=["disabled", "fixed", "auto"],
        default="disabled",
    )
    parser.add_argument(
        "--require-batch-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    for name in ("out_atol", "lse_atol", "dlogp_atol"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.vocab_size < 2:
        parser.error("--vocab-size must be >= 2")
    if 32 % args.tp_world_size != 0 or 8 % args.tp_world_size != 0:
        parser.error("--tp-world-size must divide Qwen3-8B's 32 query and 8 KV heads")
    expected_q_heads = 32 // args.tp_world_size
    expected_kv_heads = 8 // args.tp_world_size
    if args.q_heads != expected_q_heads or args.kv_heads != expected_kv_heads:
        parser.error(
            "--q-heads/--kv-heads must describe the TP-local Qwen3-8B shard: "
            f"expected {expected_q_heads}/{expected_kv_heads} for TP={args.tp_world_size}"
        )
    if args.head_dim != 128:
        parser.error("--head-dim must be 128 for the Qwen3-8B acceptance target")
    if args.strict and args.split_kv_policy != "disabled":
        parser.error("--strict requires --split-kv-policy disabled")
    return args


def _make_config(args: argparse.Namespace) -> FlashInferPagedAttentionConfig:
    if args.split_kv_policy == "disabled":
        split_kv = FlashInferSplitKVPolicy.disabled()
    elif args.split_kv_policy == "fixed":
        if args.fixed_split_size is None:
            raise SystemExit("--split-kv-policy fixed requires --fixed-split-size")
        split_kv = FlashInferSplitKVPolicy.fixed(args.fixed_split_size)
    else:
        split_kv = FlashInferSplitKVPolicy.auto()
    return FlashInferPagedAttentionConfig(
        mode=args.mode,
        workspace_size_bytes=128 * 1024 * 1024,
        require_batch_invariant=args.require_batch_invariant,
        rope=FlashInferRoPEFusionConfig(
            rope_theta=1_000_000.0,
            rope_scale=1.0,
            rotary_dim=args.head_dim,
            q_rope_state="pre_rope",
            k_cache_rope_state="pre_rope",
        ),
        split_kv=split_kv,
        cp_comm_plan=AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(
                tp_world_size=args.tp_world_size,
                tp_rank=args.tp_rank,
                cp_world_size=args.cp_world_size,
                cp_rank=args.cp_rank,
            ),
            backend=args.cp_comm_backend,
            status="interface_only",
        ),
        require_cp_comm=args.require_cp_comm,
        strict_mode=args.strict,
    )


def _make_inputs(args: argparse.Namespace, device: torch.device) -> DecodeAttentionInputs:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.kv_seq_len % args.page_size != 0:
        raise SystemExit("--kv-seq-len must be divisible by --page-size for this scaffold")
    if args.mode == "decode" and args.query_len != 1:
        raise SystemExit("--mode decode requires --query-len 1")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    q = torch.randn(
        args.batch_size,
        args.q_heads,
        args.query_len,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    k_cache = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.kv_seq_len,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn(
        args.batch_size,
        args.kv_heads,
        args.kv_seq_len,
        args.head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    page_count = args.kv_seq_len // args.page_size
    block_table = torch.arange(page_count, device=device, dtype=torch.long).repeat(
        args.batch_size,
        1,
    )
    positions = torch.arange(args.kv_seq_len, device=device, dtype=torch.long).repeat(
        args.batch_size,
        1,
    )
    query_start = args.kv_seq_len - args.query_len
    query_positions = torch.arange(
        query_start,
        args.kv_seq_len,
        device=device,
        dtype=torch.long,
    ).repeat(args.batch_size, 1)
    metadata = DecodeKVCacheMetadata(
        cache_position=query_positions.clone(),
        kv_seq_lens=torch.full(
            (args.batch_size,),
            args.kv_seq_len,
            device=device,
            dtype=torch.long,
        ),
        block_table=block_table,
        global_token_positions=positions,
        query_position_ids=query_positions.clone(),
        key_position_ids=positions.clone(),
        page_size=args.page_size,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )
    return DecodeAttentionInputs(q=q, k_cache=k_cache, v_cache=v_cache, metadata=metadata)


def _run_strict_platform_reference(
    inputs: DecodeAttentionInputs,
    config: FlashInferPagedAttentionConfig,
    paged_plan: Any,
) -> AttentionPathResult:
    """Call the platform production core directly on the logical KV sequence."""

    core = config.deterministic_core or _strict_attention_core(config.split_kv)
    rope = config.strict_rope_op or _strict_rope_op()
    logical_k, logical_v, key_positions = _materialize_strict_logical_kv(
        inputs.k_cache,
        inputs.v_cache,
        inputs.metadata,
        paged_plan,
    )
    query_positions = inputs.metadata.query_position_ids
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for batch_index, seq_len_value in enumerate(paged_plan.kv_seq_lens.tolist()):
        seq_len = int(seq_len_value)
        q_row = inputs.q[batch_index : batch_index + 1]
        k_row = logical_k[batch_index : batch_index + 1, :, :seq_len, :]
        v_row = logical_v[batch_index : batch_index + 1, :, :seq_len, :]
        q_pos = query_positions[batch_index : batch_index + 1]
        k_pos = key_positions[batch_index : batch_index + 1, :seq_len]
        result = core.forward_with_lse(
            _apply_strict_rope(rope, q_row, q_pos, config.rope.rope_theta),
            _apply_strict_rope(rope, k_row, k_pos, config.rope.rope_theta),
            v_row,
            causal=config.causal,
            scale=config.softmax_scale,
            query_position_ids=q_pos,
            key_position_ids=k_pos,
            output_dtype=inputs.q.dtype,
        )
        outputs.append(result.out)
        lses.append(result.lse)
    return AttentionPathResult(
        name="direct_platform_strict_attention",
        out=torch.cat(outputs, dim=0),
        lse=torch.cat(lses, dim=0),
        provenance={
            "strict_core_id": core.core_id,
            "strict_schedule": core.strict_schedule,
            "split_kv_policy": "disabled",
        },
    )


def _strict_execution_report_fields(provenance: dict[str, Any]) -> dict[str, Any]:
    """Describe the backend that actually executed, independent of the host running tests."""

    backend = provenance.get("actual_backend", "unknown")
    return {
        "target": (
            f"Qwen3-8B TP-local {backend} strict production core; "
            "CP transport validated separately"
        ),
        "reference_backend": backend,
        "rope": {
            "rope_backend": provenance.get("rope_backend"),
            "rope_fusion": provenance.get("rope_fusion"),
            "rope_fusion_boundary": provenance.get("rope_fusion_boundary"),
            "rope_theta": provenance.get("rope_theta"),
            "rotary_dim": provenance.get("rotary_dim"),
            "q_rope_state": provenance.get("q_rope_state"),
            "k_cache_rope_state": provenance.get("k_cache_rope_state"),
        },
    }


def _run_batch_invariance_sweep(
    op: FlashInferQwen3PagedAttentionOp,
    inputs: DecodeAttentionInputs,
    batch_result: Any,
    config: FlashInferPagedAttentionConfig,
) -> dict[str, Any]:
    rows = []
    max_out = 0.0
    max_lse = 0.0
    for batch_index in range(inputs.q.size(0)):
        single = _select_batch_row(inputs, batch_index)
        single_result = op(
            single.q,
            single.k_cache,
            single.v_cache,
            single.metadata,
            config=config,
        )
        out_diff = (
            single_result.out.float() - batch_result.out[batch_index : batch_index + 1].float()
        ).abs()
        lse_diff = (
            single_result.lse.float() - batch_result.lse[batch_index : batch_index + 1].float()
        ).abs()
        row_out = float(out_diff.max().item())
        row_lse = float(lse_diff.max().item())
        max_out = max(max_out, row_out)
        max_lse = max(max_lse, row_lse)
        rows.append({"batch_index": batch_index, "out_max_abs": row_out, "lse_max_abs": row_lse})
    return {
        "method": "single_row_vs_same_row_inside_batch",
        "row_count": len(rows),
        "out_max_abs": max_out,
        "lse_max_abs": max_lse,
        "rows": rows,
        "passed": max_out == 0.0 and max_lse == 0.0,
    }


def _run_page_layout_invariance_sweep(
    op: FlashInferQwen3PagedAttentionOp,
    inputs: DecodeAttentionInputs,
    base_result: Any,
    config: FlashInferPagedAttentionConfig,
) -> dict[str, Any]:
    page_size = inputs.metadata.page_size
    page_count = inputs.k_cache.size(2) // page_size
    if page_count < 2:
        return {
            "method": "logical_page_table_permutation",
            "status": "not_applicable",
            "passed": True,
            "reason": "fewer than two physical pages",
        }
    permutation = torch.arange(
        page_count - 1,
        -1,
        -1,
        device=inputs.k_cache.device,
        dtype=torch.long,
    )
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(page_count, device=permutation.device)

    def permute_cache(cache: torch.Tensor) -> torch.Tensor:
        pages = cache.reshape(
            cache.size(0),
            cache.size(1),
            page_count,
            page_size,
            cache.size(3),
        )
        return pages[:, :, permutation].reshape_as(cache).contiguous()

    block_table = inverse[inputs.metadata.block_table.long()]
    positions = inputs.metadata.global_token_positions.reshape(
        inputs.q.size(0), page_count, page_size
    )[:, permutation].reshape_as(inputs.metadata.global_token_positions)
    key_positions = inputs.metadata.key_position_ids.reshape(
        inputs.q.size(0), page_count, page_size
    )[:, permutation].reshape_as(inputs.metadata.key_position_ids)
    permuted = DecodeAttentionInputs(
        q=inputs.q,
        k_cache=permute_cache(inputs.k_cache),
        v_cache=permute_cache(inputs.v_cache),
        metadata=replace(
            inputs.metadata,
            block_table=block_table,
            global_token_positions=positions,
            key_position_ids=key_positions,
        ),
        scale=inputs.scale,
        output_dtype=inputs.output_dtype,
        rope_theta=inputs.rope_theta,
        rope_rotary_dim=inputs.rope_rotary_dim,
        rope_cast_at=inputs.rope_cast_at,
        q_rope_output_dtype=inputs.q_rope_output_dtype,
        k_cache_rope_output_dtype=inputs.k_cache_rope_output_dtype,
    )
    candidate = op(
        permuted.q,
        permuted.k_cache,
        permuted.v_cache,
        permuted.metadata,
        config=config,
    )
    out = _drift_stats(candidate.out, base_result.out)
    lse = _drift_stats(candidate.lse, base_result.lse)
    return {
        "method": "logical_page_table_permutation",
        "permutation": permutation.detach().cpu().tolist(),
        "out": out,
        "lse": lse,
        "passed": out["max_abs"] == 0.0 and lse["max_abs"] == 0.0,
    }


def _drift_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference tensors must have matching shapes")
    candidate_fp32 = candidate.float()
    reference_fp32 = reference.float()
    raw = (candidate_fp32 - reference_fp32).abs()
    diff = torch.where(candidate_fp32 == reference_fp32, torch.zeros_like(raw), raw).reshape(-1)
    if diff.numel() == 0:
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
        "active_count": int(diff.numel()),
    }


def _selected_logprob_drift(
    candidate_out: torch.Tensor,
    reference_out: torch.Tensor,
    *,
    seed: int,
    vocab_size: int,
) -> dict[str, Any]:
    batch, heads, seq_len, head_dim = candidate_out.shape
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(
        vocab_size,
        heads * head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(candidate_out.device)
    weight.mul_(1.0 / math.sqrt(heads * head_dim))
    target_ids = (
        torch.arange(
            batch * seq_len,
            device=candidate_out.device,
            dtype=torch.long,
        ).reshape(batch, seq_len)
        % vocab_size
    )

    def selected(out: torch.Tensor) -> torch.Tensor:
        hidden = out.float().transpose(1, 2).reshape(batch, seq_len, heads * head_dim)
        logits = torch.matmul(hidden, weight.transpose(0, 1))
        return torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    return _drift_stats(selected(candidate_out), selected(reference_out))


def _acceptance_errors(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = []
    drift = report["drift"]
    for name, threshold in (
        ("out", args.out_atol),
        ("lse", args.lse_atol),
        ("dlogp", args.dlogp_atol),
    ):
        value = drift.get(name, {}).get("max_abs")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            errors.append(f"{name} max_abs must be finite and non-negative")
        elif value > threshold:
            errors.append(f"{name} max_abs={value} exceeds {threshold}")
    provenance = report.get("candidate_provenance", {})
    if not str(report.get("device", "")).startswith("cuda"):
        errors.append("strict PR7 acceptance requires a CUDA execution")
    shape = report.get("shape", {})
    expected_shape = {
        "q_heads": 32 // args.tp_world_size,
        "kv_heads": 8 // args.tp_world_size,
        "head_dim": 128,
    }
    if not isinstance(shape, dict) or any(
        shape.get(key) != expected for key, expected in expected_shape.items()
    ):
        errors.append("runtime shape is not the Qwen3-8B TP-local head shard")
    if provenance.get("attention_mode") != args.mode:
        errors.append("runtime attention mode differs from the requested mode")
    if provenance.get("fallback") is not False:
        errors.append("FlashInfer execution used or omitted fallback provenance")
    if args.strict:
        platform = provenance.get("platform", "cuda")
        if platform not in {"cuda", "rocm"}:
            errors.append("strict runtime platform provenance is invalid")
        is_rocm = platform == "rocm"
        expected_core = (
            STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
            if is_rocm
            else STRICT_ATTENTION_PRODUCTION_CORE_ID
        )
        expected_schedule = (
            STRICT_ATTENTION_ROCM_SCHEDULE_ID if is_rocm else STRICT_ATTENTION_FA4_SCHEDULE_ID
        )
        expected_backend = "aiter.rocm.ck_dense_mha" if is_rocm else "flash_attention_4.cute"
        if provenance.get("strict_mode") is not True:
            errors.append("strict runtime did not execute the shared Attention core")
        if provenance.get("strict_core_id") != expected_core:
            errors.append("strict runtime core identity is invalid")
        if provenance.get("strict_schedule") != expected_schedule:
            errors.append("strict runtime arithmetic schedule is invalid")
        if provenance.get("actual_backend") != expected_backend:
            errors.append("strict runtime backend identity is invalid")
        if provenance.get("native_attention_arithmetic") is not True:
            errors.append("strict runtime did not execute the native production arithmetic")
        if provenance.get("num_splits") != 1:
            errors.append("strict runtime did not prove one reduction partition")
        if provenance.get("deterministic_backward") is not True:
            errors.append("strict runtime did not request deterministic backward")
        if provenance.get("reference_only") is not False:
            errors.append("strict runtime selected the reference core")
        if is_rocm:
            if provenance.get("split_kv_control") != "dense_non_split_api":
                errors.append("strict ROCm runtime did not use AITER dense non-Split-K MHA")
            if provenance.get("aiter_api_source") != "aiter.ops.mha":
                errors.append("strict ROCm runtime did not prove the AITER API source")
            if not provenance.get("aiter_source_sha256"):
                errors.append("strict ROCm runtime did not fingerprint AITER MHA")
        elif provenance.get("fa_api_source") != "flash_attn.cute.interface":
            errors.append("strict CUDA runtime did not prove the FA4 CuTe API source")
        strict_plans = provenance.get("strict_core_row_plans")
        if not isinstance(strict_plans, list) or not strict_plans:
            errors.append("strict no-Split-K execution plans are missing")
        elif any(plan.get("actual_split_kv_policy") != "disabled" for plan in strict_plans):
            errors.append("strict runtime did not keep Split-KV disabled")
        expected_rope_backends = (
            {"rlkernel.rocm.deterministic_rope"}
            if is_rocm
            else {
                "rlkernel.cuda.rope_sm90",
                "rlkernel.cuda.rope_sm90_op",
            }
        )
        if provenance.get("rope_backend") not in expected_rope_backends:
            errors.append("strict runtime did not use the RL-Kernel WS1 RoPE operator")
    elif provenance.get("pos_encoding_mode") != "ROPE_LLAMA":
        errors.append("FlashInfer runtime did not use ROPE_LLAMA")
    if provenance.get("rope_theta") != 1_000_000.0 or provenance.get("rotary_dim") != 128:
        errors.append("FlashInfer runtime RoPE identity does not match Qwen3-8B")
    if provenance.get("arithmetic_semantics_verified") is not True:
        errors.append("runtime arithmetic semantics were not verified")
    if not args.strict:
        if not provenance.get("actual_split_kv_plans"):
            errors.append("actual Split-KV runtime plans are missing")
        plan_set = provenance.get("actual_split_kv_plan_set")
        if not isinstance(plan_set, dict) or plan_set.get("coverage") != (
            "complete_batch_tp_cp_owner_cartesian_product"
        ):
            errors.append("complete batch/TP/CP/owner Split-KV plan set is missing")
    for sweep_name in ("batch_invariant_sweep", "page_layout_invariant_sweep"):
        if report.get(sweep_name, {}).get("passed") is not True:
            errors.append(f"{sweep_name} failed")
    return errors


def _strict_attention_core(split_kv):
    if torch.version.hip is not None:
        from rl_engine.backends.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore

        return StrictRocmAiterCKAttentionCore(split_kv=split_kv)
    return StrictFlashAttention4Core(split_kv=split_kv)


def _strict_rope_op():
    from rl_engine.backends.cuda.rope.rope import RocmDeterministicRoPEOp, RoPESM90Op

    return RocmDeterministicRoPEOp() if torch.version.hip is not None else RoPESM90Op()


def _select_batch_row(inputs: DecodeAttentionInputs, batch_index: int) -> DecodeAttentionInputs:
    metadata = inputs.metadata
    cp_block_owners = (
        None
        if metadata.cp_block_owners is None
        else metadata.cp_block_owners[batch_index : batch_index + 1]
    )
    selected_metadata = DecodeKVCacheMetadata(
        cache_position=metadata.cache_position[batch_index : batch_index + 1],
        kv_seq_lens=metadata.kv_seq_lens[batch_index : batch_index + 1],
        block_table=metadata.block_table[batch_index : batch_index + 1],
        global_token_positions=metadata.global_token_positions[batch_index : batch_index + 1],
        query_position_ids=metadata.query_position_ids[batch_index : batch_index + 1],
        key_position_ids=metadata.key_position_ids[batch_index : batch_index + 1],
        page_size=metadata.page_size,
        prefix_cache_key=metadata.prefix_cache_key,
        prefix_cache_enabled=metadata.prefix_cache_enabled,
        prefix_length=metadata.prefix_length,
        prefix_cache_fingerprint=metadata.prefix_cache_fingerprint,
        q_rope_state=metadata.q_rope_state,
        k_cache_rope_state=metadata.k_cache_rope_state,
        cp_block_owners=cp_block_owners,
    )
    return replace(
        inputs,
        q=inputs.q[batch_index : batch_index + 1],
        k_cache=inputs.k_cache[batch_index : batch_index + 1],
        v_cache=inputs.v_cache[batch_index : batch_index + 1],
        metadata=selected_metadata,
    )


def _emit(
    report: dict[str, Any],
    *,
    json_output: bool,
    output: Path | None,
) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"PR7 FlashInfer check: {report['status']}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
