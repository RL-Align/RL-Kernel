# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P3 (WS2) on real GPUs: per-rank readback through the strict TP route.

Every rank scores its rows and LM-head shard once through the real strict
route (``RecordedTrainingProvider``: VIME request types, ``MegatronLogpOperator``,
``LinearLogpWrapper`` on an NCCL TP group), validates its own call with the T05
gate, and exchanges the per-rank readback over a CPU gloo group. Rank 0 checks
the rank set and writes the report plus per-token FP32 bytes, so runs on
different physical TP sizes can be compared byte for byte.

Launch, for example (set ``RL_KERNEL_STRICT_CANONICAL_TP`` to share a boundary):

    torchrun --standalone --nproc_per_node=4 tests/megatron_training_ranks_tp.py \
        --tp 4 --out runs/<id>/tp4.json
    python tests/megatron_training_ranks_tp.py --compare runs/<id>/tp1.json runs/<id>/tp4.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from p7_doubles import t02_sampling  # noqa: E402
from p7_doubles.fixtures import make_fixture  # noqa: E402

from rl_engine.alignment.cross_config.adapters import megatron_training_gate as gate  # noqa: E402
from rl_engine.alignment.cross_config.adapters import megatron_training_ranks as ranks  # noqa: E402
from rl_engine.alignment.cross_config.adapters.megatron_training import (  # noqa: E402
    PreUpdateWeightLock,
    ShardDigest,
    T05ContractError,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_provider import (  # noqa: E402
    M10ProviderInput,
    RecordedTrainingProvider,
)

SAMPLE_LENGTHS = (64, 32)
REAL_VOCAB, PADDED_VOCAB, WIDTH = 500, 512, 128


def _fixture(tp: int, cp: int):
    return make_fixture(
        f"ws2_tp{tp}_cp{cp}",
        sample_lengths=SAMPLE_LENGTHS,
        width=WIDTH,
        real_vocab=REAL_VOCAB,
        padded_vocab=PADDED_VOCAB,
        tp_world=tp,
        cp_world=cp,
    )


def _groups(tp: int, cp: int) -> tuple[Any, Any, Any]:
    """Megatron order (TP fastest): NCCL TP and CP groups plus a CPU gloo group."""

    world, rank = dist.get_world_size(), dist.get_rank()
    tp_group = cp_group = None
    for cp_rank in range(cp):
        members = [cp_rank * tp + tp_rank for tp_rank in range(tp)]
        group = dist.new_group(members)
        if rank in members:
            tp_group = group
    for tp_rank in range(tp):
        members = [cp_rank * tp + tp_rank for cp_rank in range(cp)]
        group = dist.new_group(members)
        if rank in members:
            cp_group = group
    gloo = dist.new_group(list(range(world)), backend="gloo")
    return tp_group, cp_group, gloo


def _score(fixture, *, tp_rank: int, cp_rank: int, tp_group: Any) -> gate.TrainingM10Evidence:
    device = torch.device("cuda", torch.cuda.current_device())
    rows = list(fixture.cp_local_rows(cp_rank))
    start, end = fixture.vocab_shards[tp_rank]
    weight = fixture.lm_head_weight
    shards = tuple(
        ShardDigest.from_tensor(weight[s:e], tp_rank=r, vocab_start=s, vocab_end=e)
        for r, (s, e) in enumerate(fixture.vocab_shards)
    )
    m10 = M10ProviderInput(
        final_decoder_rows=fixture.hidden[rows].to(device),
        lm_head_weight_local=weight[start:end].to(device),
        target_token_ids=fixture.target_ids[rows].to(device),
        active_token_mask=fixture.active_mask[rows].clone(),
        vocab_start=start,
        real_vocab_size=REAL_VOCAB,
        padded_vocab_size=PADDED_VOCAB,
        token_layout_world_size=fixture.cp_world,
        token_layout_rank=cp_rank,
        token_layout="single" if fixture.cp_world == 1 else "zigzag",
        row_layout=gate.RowLayout(
            tuple(f"s{index}" for index in range(len(SAMPLE_LENGTHS))), SAMPLE_LENGTHS
        ),
        checkpoint_id="ckpt-0",
        weight_id="w-0",
        weight_lock=PreUpdateWeightLock("ckpt-0", "w-0", shards),
        lm_head_lora=False,
        sampling_record=t02_sampling.record(t02_sampling.IDENTITY_REQUEST),
        tensor_parallel_group=tp_group,
        metadata={"tp_rank": tp_rank, "tp_world_size": fixture.tp_world},
    )
    return RecordedTrainingProvider().score(m10)


def _per_token(evidence: gate.TrainingM10Evidence) -> dict[str, int]:
    raw = evidence.selected_logprob.contiguous().view(torch.int32)
    mask = evidence.active_token_mask
    return {
        f"{key[0]}:{key[1]}": int(raw[index])
        for index, key in enumerate(evidence.row_keys)
        if key is not None and bool(mask[index])
    }


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("VIME_RL_KERNEL_STRICT", "1")
    os.environ.setdefault("RL_KERNEL_LOGP_CASE", "R/R")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world, rank = dist.get_world_size(), dist.get_rank()
    tp, cp = args.tp, world // args.tp
    if tp * cp != world:
        raise SystemExit(f"world {world} is not a multiple of TP {tp}")
    if args.no_tp_group and world != 1:
        raise SystemExit("--no-tp-group is a TP1 single-process record")
    tp_group, cp_group, gloo = _groups(tp, cp)
    tp_rank, cp_rank = rank % tp, rank // tp
    fixture = _fixture(tp, cp)

    # VIME always passes Megatron's TP group (size 1 at TP1), which selects the
    # canonical ``_tp`` entry; ``--no-tp-group`` records the ungrouped single-GPU entry.
    score_group = None if args.no_tp_group else tp_group
    evidence = _score(fixture, tp_rank=tp_rank, cp_rank=cp_rank, tp_group=score_group)
    coords = ranks.observe_rank_coordinates(
        tp_group=score_group, cp_group=cp_group if cp > 1 else None
    )
    readbacks = ranks.gather_rank_readbacks(
        ranks.RankReadback.from_evidence(evidence, coords), gloo
    )
    tokens: list[Any] = [None] * world
    dist.all_gather_object(tokens, _per_token(evidence), group=gloo)

    topology = ranks.ExpectedRankTopology(tp=tp, cp=cp)
    expected_rows = {(f"s{sample}", position) for sample, position in fixture.row_identity()}
    result: dict[str, Any] = {"tp": tp, "cp": cp, "world": world}
    if rank == 0:
        report = ranks.validate_rank_set(readbacks, topology, expected_row_keys=expected_rows)
        result["report"] = report.to_dict()
        result["report_fingerprint"] = report.fingerprint
        merged: dict[str, int] = {}
        for item in tokens:
            for key, value in item.items():
                if merged.setdefault(key, value) != value:
                    raise SystemExit(f"TP peers disagree on token {key}")
        result["tokens"] = dict(sorted(merged.items()))
        negatives: dict[str, str] = {}
        if world > 1:
            dropped = [item for item in readbacks if item.coordinates["global_rank"] != world - 1]
            try:
                ranks.validate_rank_set(dropped, topology)
                negatives["drop_last_rank"] = "PASSED (expected MISSING_RANK)"
            except T05ContractError as error:
                negatives["drop_last_rank"] = error.status
            duplicated = readbacks + [readbacks[0]]
            try:
                ranks.validate_rank_set(duplicated, topology)
                negatives["duplicate_rank"] = "PASSED (expected IDENTITY_DRIFT)"
            except T05ContractError as error:
                negatives["duplicate_rank"] = error.status
        result["negatives"] = negatives
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True))
        print(
            f"tp={tp} cp={cp} complete={report.complete} certified={report.certified} "
            f"tokens={len(merged)} negatives={negatives}"
        )
        ok = (
            negatives.get("drop_last_rank", "MISSING_RANK") == "MISSING_RANK"
            and negatives.get("duplicate_rank", "IDENTITY_DRIFT") == "IDENTITY_DRIFT"
        )
    else:
        ok = True
    dist.barrier(group=gloo)
    dist.destroy_process_group()
    return 0 if ok else 1


def compare(paths: list[str]) -> int:
    loaded = [json.loads(Path(path).read_text()) for path in paths]
    reference = loaded[0]["tokens"]
    status = 0
    for path, item in zip(paths[1:], loaded[1:], strict=True):
        tokens = item["tokens"]
        differ = sorted(key for key in reference if tokens.get(key) != reference[key])
        extra = sorted(set(tokens) - set(reference))
        print(
            f"{paths[0]} vs {path}: {len(reference)} tokens, {len(differ)} differ, "
            f"{len(extra)} extra"
        )
        if differ or extra:
            status = 1
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--no-tp-group", action="store_true", help="TP1 only: call without a TP group"
    )
    parser.add_argument("--out", default="ws2.json")
    parser.add_argument("--compare", nargs="+")
    args = parser.parse_args()
    if args.compare:
        return compare(args.compare)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
