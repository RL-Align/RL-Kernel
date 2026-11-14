# SPDX-License-Identifier: Apache-2.0
"""Compare captured original sparse operands at matching prefill positions."""
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
ap, rp = root / "actor-trace-v0-A", root / "generation-trace"
a = json.loads((ap / "manifest.json").read_text())
r = [json.loads(s) for s in (rp / "manifest.jsonl").read_text().splitlines()]


def load(events, name, path):
    e = next(
        e for e in events if e["name"] == name and e["layer"] == 0 and e.get("forward", 0) == 0
    )
    return torch.load(path / e["file"], weights_only=True, map_location="cpu")


def compare(name, x, y):
    assert x.dtype == y.dtype and x.shape == y.shape, (name, x.shape, y.shape, x.dtype, y.dtype)
    x, y = x.contiguous(), y.contiguous()
    bad = (x.view(torch.uint8) != y.view(torch.uint8)).reshape(x.numel(), x.element_size()).any(-1)
    record = dict(stage=name, dtype=str(x.dtype), elements=x.numel(), mismatch_count=int(bad.sum()))
    if bad.any():
        i = int(bad.nonzero()[0])
        xx = x.reshape(-1)[i : i + 1]
        yy = y.reshape(-1)[i : i + 1]
        record.update(
            index=i,
            rollout=xx.item(),
            actor=yy.item(),
            rollout_bits=bytes(xx.view(torch.uint8).tolist())[::-1].hex(),
            actor_bits=bytes(yy.view(torch.uint8).tolist())[::-1].hex(),
        )
    print(json.dumps(record), flush=True)


rinputs = load(r, "layer.0.attn.sparse_prefill.input", rp)
aq, ak, sink, indices, scale = load(a, "layer.0.attn.sparse.input", ap)
n = rinputs["q"].shape[0]
compare("Q", rinputs["q"], aq[:n, 0])
ri = rinputs["indices"]
print("GEOMETRY", ri.shape, indices.shape, rinputs["kv"].shape, ak.shape, flush=True)
print("INDICES", ri[:3, :8].tolist(), indices[0, :3, :8].tolist(), flush=True)
ni = indices[0, :n].contiguous()
width = max(ri.shape[1], ni.shape[1])
compare(
    "selected indices",
    torch.nn.functional.pad(ri, (0, width - ri.shape[1]), value=-1),
    torch.nn.functional.pad(ni, (0, width - ni.shape[1]), value=-1),
)
compare("gathered KV", rinputs["kv"][:n, 0], ak[:n, 0])
compare("attention sink", rinputs["attn_sink"], sink)
ro = load(r, "layer.0.attn.sparse_prefill.output", rp)
ao = load(a, "layer.0.attn.sparse.output", ap)
compare("attention output before inverse RoPE", ro, ao[:n, 0].reshape_as(ro))
