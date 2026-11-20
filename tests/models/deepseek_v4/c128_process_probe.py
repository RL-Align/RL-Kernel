# SPDX-License-Identifier: Apache-2.0
"""Fresh-process numerical evidence and actual uneven CPU Gloo transport."""

import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.contracts.operators.kv_compressor_c128 import C128Contract
from rl_engine.models.deepseek_v4.kv_compressor_c128 import C128Compressor
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from tests.models.deepseek_v4.c128_helpers import fixture


def digest(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def single(path):
    op = C128Compressor(C128Contract())
    x, ids, a, n, c, s = fixture(384, device="cuda", requires_grad=True)
    with canonical_backward_session() as session:
        out = op(op.bind(x, ids, version="v1"), a, n, c, s, version="v1", parameter_id="probe")
        out.values.sum().backward()
        session.validate_complete()
    Path(path).write_text(
        json.dumps(
            {
                "output": digest(out.values),
                "dx": digest(x.grad),
                "dape": digest(a.grad),
                "dnorm": digest(n.grad),
                "implementation": op.implementation_fingerprint,
            },
            indent=2,
        )
    )


def cp_rank(rank, path):
    dist.init_process_group(
        "gloo", init_method=Path(path + ".init").resolve().as_uri(), rank=rank, world_size=2
    )
    x, ids, a, n, c, s = fixture(384)
    starts, counts = (0, 150), (150, 234)
    local = torch.zeros((234, 1024), dtype=torch.bfloat16)
    local[: counts[rank]].copy_(x[starts[rank] : starts[rank] + counts[rank]])
    gathered = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(gathered, local)
    global_x = torch.cat([buf[:count] for buf, count in zip(gathered, counts, strict=True)]).cuda()
    owners = torch.cat((torch.zeros(150), torch.ones(234))).long().cuda()
    op = C128Compressor(C128Contract())
    bound = op.bind(global_x, ids.cuda(), version="v1", cp_size=2, rank=rank, owners=owners)
    output = op(bound, a.cuda(), n.cuda(), c.cuda(), s.cuda(), version="v1")
    torch.save(
        {"ids": output.group_ids.cpu(), "values": output.values.cpu()}, path + f".rank{rank}.pt"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mode, path = sys.argv[1:]
    if mode == "single":
        single(path)
    elif mode == "cp":
        mp.spawn(cp_rank, args=(path,), nprocs=2, join=True)
    else:
        raise ValueError(mode)
