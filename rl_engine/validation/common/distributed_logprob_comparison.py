# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Distributed WS2 comparison for the vocab-parallel logprob reference."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import shlex
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

if __package__ in (None, ""):
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

_import_output = (
    contextlib.redirect_stdout(sys.stderr)
    if __package__ in (None, "")
    else contextlib.nullcontext()
)
with _import_output:
    from rl_engine.contracts.numerical import load_contract as load_tolerance_contract
    from rl_engine.contracts.operators.logprob import (
        LogprobContract,
        LogprobDType,
        LogprobRole,
        MaskSpec,
        ReductionSpec,
        ShardingSpec,
    )
    from rl_engine.reference.logprob.vocab_parallel_logp import BACKEND_ID, DEFAULT_NUM_VOCAB_TILES
    from rl_engine.runtime.registry import KernelRegistry
    from rl_engine.validation.common.logprob_comparison import route_rl_kernel_logs_to_stderr
    from rl_engine.validation.common.logprob_drift import LogprobDriftStats, summarize_logprob_drift

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_TOLERANCE_DTYPES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}
_PROCESS_GROUP_TIMEOUT = datetime.timedelta(minutes=5)
_RELATIVE_ERROR_FLOOR = 1.0e-12


@dataclass(frozen=True)
class DistributedLogprobCase:
    tp_world_size: int
    cp_world_size: int
    dtype: str = "bf16"
    requested_backend: str = BACKEND_ID
    real_vocab_size: int = 151936
    padded_vocab_size: int = 151936
    num_vocab_tiles: int = DEFAULT_NUM_VOCAB_TILES
    batch_size: int = 2
    sequence_length: int = 16
    prompt_tokens: int = 8
    seed: int = 123
    ignore_index: int = -100

    def __post_init__(self) -> None:
        for name in (
            "tp_world_size",
            "cp_world_size",
            "real_vocab_size",
            "padded_vocab_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}")
        if not self.requested_backend or self.requested_backend.lower() == "auto":
            raise ValueError("distributed cases require an explicit non-auto backend")
        if self.padded_vocab_size < self.real_vocab_size:
            raise ValueError("padded_vocab_size must be at least real_vocab_size")
        if self.num_vocab_tiles < self.tp_world_size:
            raise ValueError("num_vocab_tiles must be at least tp_world_size")
        if self.padded_vocab_size % self.num_vocab_tiles != 0:
            raise ValueError("num_vocab_tiles must divide padded_vocab_size exactly")
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        if not 0 <= self.prompt_tokens <= self.sequence_length:
            raise ValueError("prompt_tokens must be in [0, sequence_length]")

    @property
    def world_size(self) -> int:
        return self.tp_world_size * self.cp_world_size

    @property
    def num_tokens(self) -> int:
        return self.batch_size * self.sequence_length

    @property
    def case_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class RankTopology:
    global_rank: int
    world_size: int
    tp_rank: int
    tp_world_size: int
    cp_rank: int
    cp_world_size: int
    tp_group_ranks: tuple[int, ...]


@dataclass(frozen=True)
class DriftDetail:
    stats: LogprobDriftStats
    max_rel: float
    worst_global_token: int | None
    worst_target_id: int | None
    worst_owner_rank: int | None
    candidate_value: float | None
    reference_value: float | None
    atol: float
    rtol: float
    passed: bool


@dataclass(frozen=True)
class RankLogprobReport:
    global_rank: int
    tp_rank: int
    tp_world_size: int
    cp_rank: int
    cp_world_size: int
    sp_world_size: int
    dp_world_size: int
    token_start: int
    token_end: int
    vocab_start: int
    vocab_end: int
    device: str
    requested_backend: str
    actual_backend: str
    fallback: bool
    contract_fingerprint: str
    contract: dict[str, Any]
    capability: dict[str, Any]
    tp_outputs_bitwise_replicated: bool
    lse: DriftDetail
    dlogp: DriftDetail
    passed: bool


@dataclass(frozen=True)
class DistributedLogprobReport:
    schema_version: int
    case_id: str
    case: dict[str, Any]
    launch_command: str
    environment: dict[str, Any]
    ranks: tuple[RankLogprobReport, ...]
    aggregate: dict[str, DriftDetail]
    bitwise_fingerprints: dict[str, Any]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RankPayload:
    report: RankLogprobReport
    candidate_logp: torch.Tensor
    candidate_lse: torch.Tensor
    reference_logp: torch.Tensor
    reference_lse: torch.Tensor
    active_mask: torch.Tensor
    target_ids: torch.Tensor
    global_positions: torch.Tensor


def _strict_report_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def plan_distributed_logprob_cases(
    *,
    tp_world_sizes: Sequence[int] = (1, 2, 4),
    cp_world_sizes: Sequence[int] = (1, 2),
    **overrides: Any,
) -> tuple[DistributedLogprobCase, ...]:
    """Build the scoped issue #241 topology product in deterministic order."""

    return tuple(
        DistributedLogprobCase(tp_world_size=tp, cp_world_size=cp, **overrides)
        for tp in tp_world_sizes
        for cp in cp_world_sizes
    )


def rank_topology(case: DistributedLogprobCase, global_rank: int) -> RankTopology:
    if not 0 <= global_rank < case.world_size:
        raise ValueError(f"global_rank must be in [0, {case.world_size})")
    cp_rank, tp_rank = divmod(global_rank, case.tp_world_size)
    group_start = cp_rank * case.tp_world_size
    return RankTopology(
        global_rank=global_rank,
        world_size=case.world_size,
        tp_rank=tp_rank,
        tp_world_size=case.tp_world_size,
        cp_rank=cp_rank,
        cp_world_size=case.cp_world_size,
        tp_group_ranks=tuple(range(group_start, group_start + case.tp_world_size)),
    )


def token_shard_bounds(num_tokens: int, cp_world_size: int) -> tuple[tuple[int, int], ...]:
    """Partition token rows contiguously, allowing a one-row imbalance."""

    if num_tokens < cp_world_size:
        raise ValueError("num_tokens must be at least cp_world_size")
    quotient, remainder = divmod(num_tokens, cp_world_size)
    bounds = []
    cursor = 0
    for cp_rank in range(cp_world_size):
        count = quotient + int(cp_rank < remainder)
        bounds.append((cursor, cursor + count))
        cursor += count
    return tuple(bounds)


def vocab_shard_bounds(case: DistributedLogprobCase) -> tuple[tuple[int, int], ...]:
    """Assign complete global vocab tiles to TP ranks."""

    tile_size = case.padded_vocab_size // case.num_vocab_tiles
    quotient, remainder = divmod(case.num_vocab_tiles, case.tp_world_size)
    bounds = []
    cursor_tiles = 0
    for tp_rank in range(case.tp_world_size):
        tile_count = quotient + int(tp_rank < remainder)
        start = cursor_tiles * tile_size
        cursor_tiles += tile_count
        bounds.append((start, cursor_tiles * tile_size))
    return tuple(bounds)


def format_launch_command(
    case: DistributedLogprobCase,
    *,
    output: str | pathlib.Path,
    device: str = "cuda",
    dist_backend: str | None = None,
) -> str:
    backend = dist_backend or ("nccl" if device == "cuda" else "gloo")
    arguments = [
        "torchrun",
        "--standalone",
        f"--nproc-per-node={case.world_size}",
        "rl_engine/validation/common/distributed_logprob_comparison.py",
        "--tp",
        str(case.tp_world_size),
        "--cp",
        str(case.cp_world_size),
        "--dtype",
        case.dtype,
        "--backend",
        case.requested_backend,
        "--real-vocab",
        str(case.real_vocab_size),
        "--padded-vocab",
        str(case.padded_vocab_size),
        "--num-vocab-tiles",
        str(case.num_vocab_tiles),
        "--batch",
        str(case.batch_size),
        "--seq",
        str(case.sequence_length),
        "--prompt-tokens",
        str(case.prompt_tokens),
        "--seed",
        str(case.seed),
        "--device",
        device,
        "--dist-backend",
        backend,
        "--output",
        str(output),
    ]
    return shlex.join(arguments)


def _canonical_inputs(
    case: DistributedLogprobCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    logits = torch.randn(
        case.num_tokens,
        case.padded_vocab_size,
        generator=generator,
        dtype=torch.float32,
    )
    targets = torch.randint(
        0,
        case.real_vocab_size,
        (case.num_tokens,),
        generator=generator,
        dtype=torch.long,
    )
    active = torch.ones((case.batch_size, case.sequence_length), dtype=torch.bool)
    active[:, : case.prompt_tokens] = False
    active = active.reshape(-1)
    targets = targets.masked_fill(~active, case.ignore_index)
    return logits, targets, active


def _make_contract(
    case: DistributedLogprobCase,
    topology: RankTopology,
    active_mask: torch.Tensor,
) -> LogprobContract:
    return LogprobContract(
        role=LogprobRole.TRAIN,
        dtype=LogprobDType(case.dtype),
        mask=MaskSpec(
            num_tokens=int(active_mask.numel()),
            active_mask=tuple(bool(value) for value in active_mask.tolist()),
            ignore_index=case.ignore_index,
        ),
        sharding=ShardingSpec(
            tp_rank=topology.tp_rank,
            tp_world_size=case.tp_world_size,
            vocab_shard_bounds=vocab_shard_bounds(case),
            real_vocab_size=case.real_vocab_size,
            padded_vocab_size=case.padded_vocab_size,
            cp_rank=topology.cp_rank,
            cp_world_size=case.cp_world_size,
        ),
        reduction=ReductionSpec(),
    )


def _fp32_oracle(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    active_mask: torch.Tensor,
    real_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    real_logits = logits[:, :real_vocab_size].float()
    lse = torch.logsumexp(real_logits, dim=-1)
    safe_targets = target_ids.masked_fill(~active_mask, 0)
    selected = real_logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    logp = torch.where(active_mask, selected - lse, torch.zeros_like(lse))
    return logp, lse


def _resolve_tolerance(dtype: str) -> tuple[float, float]:
    entry = load_tolerance_contract()["accuracy"]["default"]["logprob"]
    tolerance = entry[_TOLERANCE_DTYPES[dtype]]
    return float(tolerance["atol"]), float(tolerance["rtol"])


def _drift_detail(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    target_ids: torch.Tensor,
    global_positions: torch.Tensor,
    sharding: ShardingSpec,
    atol: float,
    rtol: float,
    mask: torch.Tensor | None = None,
) -> DriftDetail:
    stats = summarize_logprob_drift(candidate, reference, mask=mask)
    diff = (candidate.float() - reference.float()).abs()
    selected = torch.ones_like(diff, dtype=torch.bool) if mask is None else mask.to(diff.device)
    if not bool(selected.any().item()):
        return DriftDetail(stats, 0.0, None, None, None, None, None, atol, rtol, True)

    selected_diff = diff[selected]
    selected_ref = reference.float()[selected]
    relative = selected_diff.double() / selected_ref.double().abs().clamp_min(_RELATIVE_ERROR_FLOOR)
    selected_indices = torch.arange(diff.numel(), device=diff.device)[selected]
    worst_selected = int(selected_diff.argmax().item())
    worst_local = int(selected_indices[worst_selected].item())
    target_id = int(target_ids[worst_local].item())
    close = selected_diff <= atol + rtol * selected_ref.abs()
    return DriftDetail(
        stats=stats,
        max_rel=float(relative.max().item()),
        worst_global_token=int(global_positions[worst_local].item()),
        worst_target_id=target_id,
        worst_owner_rank=sharding.owner_rank(target_id) if target_id >= 0 else None,
        candidate_value=float(candidate[worst_local].float().item()),
        reference_value=float(reference[worst_local].float().item()),
        atol=atol,
        rtol=rtol,
        passed=bool(close.all().item()),
    )


def _tp_outputs_replicated(
    logp: torch.Tensor,
    lse: torch.Tensor,
    *,
    tp_group: Any,
    tp_world_size: int,
) -> bool:
    if tp_world_size == 1:
        return True
    import torch.distributed as dist

    gathered_logp = [torch.empty_like(logp) for _ in range(tp_world_size)]
    gathered_lse = [torch.empty_like(lse) for _ in range(tp_world_size)]
    dist.all_gather(gathered_logp, logp.contiguous(), group=tp_group)
    dist.all_gather(gathered_lse, lse.contiguous(), group=tp_group)
    return all(torch.equal(logp, value) for value in gathered_logp) and all(
        torch.equal(lse, value) for value in gathered_lse
    )


def _execute_rank(
    case: DistributedLogprobCase,
    topology: RankTopology,
    *,
    device: torch.device,
    tp_group: Any,
) -> _RankPayload:
    full_logits, full_targets, full_active = _canonical_inputs(case)
    token_start, token_end = token_shard_bounds(case.num_tokens, case.cp_world_size)[
        topology.cp_rank
    ]
    vocab_start, vocab_end = vocab_shard_bounds(case)[topology.tp_rank]
    token_slice = slice(token_start, token_end)
    local_active = full_active[token_slice].to(device=device)
    local_targets = full_targets[token_slice].to(device=device)
    local_fp32 = full_logits[token_slice].to(device=device)
    local_logits = local_fp32[:, vocab_start:vocab_end].to(_DTYPES[case.dtype]).contiguous()
    positions = torch.arange(token_start, token_end, device=device, dtype=torch.long)

    contract = _make_contract(case, topology, local_active.cpu())
    dispatch = KernelRegistry().get_logprob_op(
        contract,
        requested_backend=case.requested_backend,
    )
    fallback = bool(dispatch.provenance["fallback"])
    if fallback:
        raise RuntimeError("distributed logprob dispatch materialized through a fallback")
    requested_policy = case.requested_backend.lower()
    if requested_policy not in {"reference", "production"} and (
        case.requested_backend != dispatch.capability.backend_id
    ):
        raise RuntimeError(
            f"requested backend {case.requested_backend!r} materialized as "
            f"{dispatch.capability.backend_id!r}"
        )

    candidate_logp, candidate_lse = dispatch.op(
        local_logits,
        local_targets,
        contract=contract,
        tp_group=tp_group,
        num_vocab_tiles=case.num_vocab_tiles,
        validate=True,
    )
    reference_logp, reference_lse = _fp32_oracle(
        local_fp32,
        local_targets,
        local_active,
        case.real_vocab_size,
    )
    replicated = _tp_outputs_replicated(
        candidate_logp,
        candidate_lse,
        tp_group=tp_group,
        tp_world_size=case.tp_world_size,
    )
    atol, rtol = _resolve_tolerance(case.dtype)
    lse_drift = _drift_detail(
        candidate_lse,
        reference_lse,
        target_ids=local_targets,
        global_positions=positions,
        sharding=contract.sharding,
        atol=atol,
        rtol=rtol,
    )
    dlogp_drift = _drift_detail(
        candidate_logp,
        reference_logp,
        target_ids=local_targets,
        global_positions=positions,
        sharding=contract.sharding,
        atol=atol,
        rtol=rtol,
        mask=local_active,
    )
    rank_report = RankLogprobReport(
        global_rank=topology.global_rank,
        tp_rank=topology.tp_rank,
        tp_world_size=topology.tp_world_size,
        cp_rank=topology.cp_rank,
        cp_world_size=topology.cp_world_size,
        sp_world_size=1,
        dp_world_size=1,
        token_start=token_start,
        token_end=token_end,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        device=str(device),
        requested_backend=case.requested_backend,
        actual_backend=dispatch.capability.backend_id,
        fallback=fallback,
        contract_fingerprint=contract.cross_rank_fingerprint(),
        contract=contract.to_dict(),
        capability=dispatch.capability.to_dict(),
        tp_outputs_bitwise_replicated=replicated,
        lse=lse_drift,
        dlogp=dlogp_drift,
        passed=replicated and lse_drift.passed and dlogp_drift.passed,
    )
    return _RankPayload(
        report=rank_report,
        candidate_logp=candidate_logp.detach().cpu(),
        candidate_lse=candidate_lse.detach().cpu(),
        reference_logp=reference_logp.detach().cpu(),
        reference_lse=reference_lse.detach().cpu(),
        active_mask=local_active.cpu(),
        target_ids=local_targets.cpu(),
        global_positions=positions.cpu(),
    )


def _aggregate_payloads(
    case: DistributedLogprobCase,
    payloads: Sequence[_RankPayload],
) -> dict[str, DriftDetail]:
    representatives = sorted(
        (payload for payload in payloads if payload.report.tp_rank == 0),
        key=lambda payload: payload.report.cp_rank,
    )
    if len(representatives) != case.cp_world_size:
        raise RuntimeError("missing one or more CP representatives in rank reports")
    candidate_logp = torch.cat([payload.candidate_logp for payload in representatives])
    candidate_lse = torch.cat([payload.candidate_lse for payload in representatives])
    reference_logp = torch.cat([payload.reference_logp for payload in representatives])
    reference_lse = torch.cat([payload.reference_lse for payload in representatives])
    active_mask = torch.cat([payload.active_mask for payload in representatives])
    target_ids = torch.cat([payload.target_ids for payload in representatives])
    positions = torch.cat([payload.global_positions for payload in representatives])
    sharding = _make_contract(
        case,
        rank_topology(case, 0),
        active_mask,
    ).sharding
    atol, rtol = _resolve_tolerance(case.dtype)
    return {
        "lse": _drift_detail(
            candidate_lse,
            reference_lse,
            target_ids=target_ids,
            global_positions=positions,
            sharding=sharding,
            atol=atol,
            rtol=rtol,
        ),
        "dlogp": _drift_detail(
            candidate_logp,
            reference_logp,
            target_ids=target_ids,
            global_positions=positions,
            sharding=sharding,
            atol=atol,
            rtol=rtol,
            mask=active_mask,
        ),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash the exact CPU tensor bytes for cross-topology comparisons."""

    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _aggregate_bitwise_fingerprints(
    case: DistributedLogprobCase,
    payloads: Sequence[_RankPayload],
) -> dict[str, Any]:
    representatives = sorted(
        (payload for payload in payloads if payload.report.tp_rank == 0),
        key=lambda payload: payload.report.cp_rank,
    )
    if len(representatives) != case.cp_world_size:
        raise RuntimeError("missing one or more CP representatives in rank reports")
    candidate_logp = torch.cat([payload.candidate_logp for payload in representatives])
    candidate_lse = torch.cat([payload.candidate_lse for payload in representatives])
    return {
        "candidate_logp_sha256": _tensor_sha256(candidate_logp),
        "candidate_lse_sha256": _tensor_sha256(candidate_lse),
        "dtype": str(candidate_logp.dtype).replace("torch.", ""),
        "shape": list(candidate_logp.shape),
    }


def _create_tp_group(case: DistributedLogprobCase, topology: RankTopology) -> Any:
    if case.world_size == 1:
        return None
    import torch.distributed as dist

    selected = None
    for cp_rank in range(case.cp_world_size):
        start = cp_rank * case.tp_world_size
        ranks = list(range(start, start + case.tp_world_size))
        group = dist.new_group(ranks=ranks)
        if topology.global_rank in ranks:
            selected = group
    return selected


def run_distributed_logprob_case(
    case: DistributedLogprobCase,
    *,
    device_name: str = "cuda",
    dist_backend: str | None = None,
    output: str | pathlib.Path,
) -> DistributedLogprobReport | None:
    """Run one materialized topology; only global rank zero returns the report."""

    import torch.distributed as dist

    backend = dist_backend or ("nccl" if device_name == "cuda" else "gloo")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != case.world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} does not match TP*CP={case.world_size}; "
            "launch exactly the topology declared by the case"
        )
    owns_process_group = world_size > 1 and not dist.is_initialized()
    try:
        if device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        elif device_name == "cpu":
            device = torch.device("cpu")
        else:
            raise ValueError("device must be cuda or cpu")

        if owns_process_group:
            dist.init_process_group(backend=backend, timeout=_PROCESS_GROUP_TIMEOUT)
        if dist.is_initialized():
            if dist.get_world_size() != world_size or dist.get_rank() != rank:
                raise RuntimeError("initialized process group does not match RANK/WORLD_SIZE")

        topology = rank_topology(case, rank)
        tp_group = _create_tp_group(case, topology)
        payload = _execute_rank(case, topology, device=device, tp_group=tp_group)
        if world_size == 1:
            payloads = [payload]
        else:
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, payload)
            payloads = gathered

        report = None
        if rank == 0:
            aggregate = _aggregate_payloads(case, payloads)
            bitwise_fingerprints = _aggregate_bitwise_fingerprints(case, payloads)
            rank_reports = tuple(
                payload.report
                for payload in sorted(payloads, key=lambda item: item.report.global_rank)
            )
            actual_backends = sorted({rank_report.actual_backend for rank_report in rank_reports})
            reduction_specs = {
                json.dumps(rank_report.contract["reduction"], sort_keys=True)
                for rank_report in rank_reports
            }
            materialization_consistent = len(actual_backends) == 1 and len(reduction_specs) == 1
            launch_command = format_launch_command(
                case,
                output=output,
                device=device_name,
                dist_backend=backend,
            )
            report = DistributedLogprobReport(
                schema_version=1,
                case_id=case.case_id,
                case=asdict(case),
                launch_command=launch_command,
                environment={
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "dist_backend": backend,
                    "world_size": world_size,
                    "sp_world_size": 1,
                    "dp_world_size": 1,
                    "materialization": {
                        "actual_backends": actual_backends,
                        "consistent": materialization_consistent,
                    },
                    "communication": {
                        "logprob_merge_axis": "tp_vocab",
                        "cp_is_merge_axis": False,
                        "report_collection": ("all_gather_object" if world_size > 1 else "none"),
                    },
                },
                ranks=rank_reports,
                aggregate=aggregate,
                bitwise_fingerprints=bitwise_fingerprints,
                passed=(
                    materialization_consistent
                    and all(rank_report.passed for rank_report in rank_reports)
                    and all(detail.passed for detail in aggregate.values())
                ),
            )
            output_path = pathlib.Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                _strict_report_json(report.to_dict()) + "\n",
                encoding="utf-8",
            )
        if world_size > 1:
            dist.barrier()
        return report
    finally:
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


def _case_from_args(args: argparse.Namespace) -> DistributedLogprobCase:
    return DistributedLogprobCase(
        tp_world_size=args.tp,
        cp_world_size=args.cp,
        dtype=args.dtype,
        requested_backend=args.backend,
        real_vocab_size=args.real_vocab,
        padded_vocab_size=args.padded_vocab,
        num_vocab_tiles=args.num_vocab_tiles,
        batch_size=args.batch,
        sequence_length=args.seq,
        prompt_tokens=args.prompt_tokens,
        seed=args.seed,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the WS2 distributed logprob drift report.")
    parser.add_argument("--plan", action="store_true", help="Print the six scoped launch commands.")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--backend", default=BACKEND_ID)
    parser.add_argument("--real-vocab", type=int, default=151936)
    parser.add_argument("--padded-vocab", type=int, default=151936)
    parser.add_argument("--num-vocab-tiles", type=int, default=DEFAULT_NUM_VOCAB_TILES)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dist-backend", choices=("nccl", "gloo"), default=None)
    parser.add_argument("--output", default="artifacts/ws2-logprob/report.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    route_rl_kernel_logs_to_stderr()
    args = _parse_args(argv)
    if args.plan:
        cases = plan_distributed_logprob_cases(
            dtype=args.dtype,
            requested_backend=args.backend,
            real_vocab_size=args.real_vocab,
            padded_vocab_size=args.padded_vocab,
            num_vocab_tiles=args.num_vocab_tiles,
            batch_size=args.batch,
            sequence_length=args.seq,
            prompt_tokens=args.prompt_tokens,
            seed=args.seed,
        )
        commands = [
            format_launch_command(
                case,
                output=pathlib.Path(args.output).parent
                / f"tp{case.tp_world_size}-cp{case.cp_world_size}.json",
                device=args.device,
                dist_backend=args.dist_backend,
            )
            for case in cases
        ]
        print(json.dumps({"commands": commands}, indent=2))
        return

    case = _case_from_args(args)
    report = run_distributed_logprob_case(
        case,
        device_name=args.device,
        dist_backend=args.dist_backend,
        output=args.output,
    )
    if report is not None:
        print(_strict_report_json(report.to_dict()))
        if not report.passed:
            raise SystemExit(1)


__all__ = [
    "DistributedLogprobCase",
    "DistributedLogprobReport",
    "DriftDetail",
    "RankLogprobReport",
    "RankTopology",
    "format_launch_command",
    "plan_distributed_logprob_cases",
    "rank_topology",
    "run_distributed_logprob_case",
    "token_shard_bounds",
    "vocab_shard_bounds",
]


if __name__ == "__main__":
    main()
