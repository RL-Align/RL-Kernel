# SPDX-License-Identifier: Apache-2.0
"""Isolated-process test helper. Gloo is a test transport, not a production CP backend."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.backends.rocm.attention.indexer_icv import AiterIndexerICVBackend
from rl_engine.contracts.operators.indexer_icv import IndexerICVContract
from rl_engine.integrations.common.indexer_icv import bind_icv_queries, bind_global_icv_keys
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV
from tests.models.deepseek_v4.icv_helpers import byte_equal, ids, payload


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def replay(output):
    backend = AiterIndexerICVBackend()
    hashes = {}
    for precision in ("mxfp4", "e4m3fn", "e4m3fnuz"):
        c = IndexerICVContract(
            num_heads=64,
            head_dim=128,
            query_tile_rows=3,
            key_tile_rows=31,
            precision="mxfp4" if precision == "mxfp4" else "fp8",
            fp8_encoding=None if precision == "mxfp4" else precision,
        )
        op = IndexerICV(c, backend=backend)
        qv, qs = payload(c, 9, q=True, seed=161601, device="cuda")
        kv, ks = payload(c, 67, seed=161602, device="cuda")
        queries = bind_icv_queries(
            op, qv, qs, ids(7, 9, device="cuda"), layout=c.layout, version="epoch1"
        )
        keys = bind_global_icv_keys(
            op,
            kv,
            ks,
            ids(7, 67, device="cuda"),
            layout=c.layout,
            version="epoch1",
            request_key_counts={7: 67},
        )
        result = op.materialize(queries, keys)[0].values
        hashes[precision] = {
            "q_bytes": digest(qv),
            "q_scale_bytes": digest(qs),
            "k_bytes": digest(kv),
            "k_scale_bytes": digest(ks),
            "icv_bytes": digest(result),
            "implementation": op.implementation_fingerprint,
        }
    output.write_text(json.dumps(hashes, indent=2) + "\n")


def gloo_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        c = IndexerICVContract(num_heads=3, head_dim=128, query_tile_rows=3, key_tile_rows=7)
        # Each rank owns an uneven subset of 17 logical compressed rows.
        full_values, full_scales = payload(c, 17, seed=161603)
        selected = torch.arange(17)[torch.arange(17) % 2 == rank]
        local_n = torch.tensor([len(selected)], dtype=torch.int64)
        lengths = [torch.empty_like(local_n) for _ in range(2)]
        dist.all_gather(lengths, local_n)
        max_rows = max(int(x.item()) for x in lengths)
        values = torch.zeros((max_rows, 64), dtype=torch.uint8)
        scales = torch.full((max_rows, 4), 127, dtype=torch.uint8)
        logical = torch.full((max_rows, 2), -1, dtype=torch.int64)
        values[: len(selected)] = full_values[selected]
        scales[: len(selected)] = full_scales[selected]
        logical[: len(selected)] = ids(7, 17)[selected]
        gathered = []
        for tensor in (values, scales, logical):
            buffers = [torch.empty_like(tensor) for _ in range(2)]
            dist.all_gather(buffers, tensor)
            gathered.append(buffers)
        if rank == 0:
            # Rank 1 arrives first in this materialization to catch rank/local
            # offsets being accidentally used as global compressed-row indices.
            packed_values = torch.cat(tuple(reversed(gathered[0]))).cuda()
            packed_scales = torch.cat(tuple(reversed(gathered[1]))).cuda()
            packed_ids = torch.cat(tuple(reversed(gathered[2]))).cuda()
            owners = torch.cat((torch.ones(max_rows), torch.zeros(max_rows))).long().cuda()
            op = IndexerICV(c)
            qv, qs = payload(c, 9, q=True, seed=161604, device="cuda")
            queries = bind_icv_queries(
                op, qv, qs, ids(7, 9, device="cuda"), layout=c.layout, version="epoch1"
            )
            actual_keys = bind_global_icv_keys(
                op,
                packed_values,
                packed_scales,
                packed_ids,
                layout=c.layout,
                version="epoch1",
                request_key_counts={7: 17},
                key_owners=owners,
                cp_size=2,
            )
            expected_keys = bind_global_icv_keys(
                op,
                full_values.cuda(),
                full_scales.cuda(),
                ids(7, 17, device="cuda"),
                layout=c.layout,
                version="epoch1",
                request_key_counts={7: 17},
            )
            actual = op.materialize(queries, actual_keys)[0].values
            expected = op.materialize(queries, expected_keys)[0].values
            byte_equal(actual, expected)
            Path(output).write_text(
                json.dumps(
                    {
                        "backend": "gloo_cpu_transport_then_one_gpu_icv",
                        "world_size": 2,
                        "byte_equal": True,
                        "rank_lengths": [int(x.item()) for x in lengths],
                        "output_sha256": digest(actual),
                    },
                    indent=2,
                )
                + "\n"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "gloo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "replay":
        replay(args.output)
    else:
        store = args.output.parent / "gloo-file-store"
        mp.spawn(
            gloo_worker, args=(store.resolve().as_uri(), str(args.output)), nprocs=2, join=True
        )


if __name__ == "__main__":
    main()
