# SPDX-License-Identifier: Apache-2.0
"""Compare original same-prefix GPU tensors in execution order, with raw bytes."""

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    root = args.run
    actor, rollout = root / "actor-trace-v0-A", root / "generation-trace"
    am = json.loads((actor / "manifest.json").read_text())
    rm = [json.loads(x) for x in (rollout / "manifest.jsonl").read_text().splitlines()]

    def get(events, name, directory, role):
        event = next(
            e
            for e in events
            if e["name"] == name and e["layer"] == 0 and (role == "actor" or e["forward"] == 0)
        )
        return torch.load(directory / event["file"], weights_only=True, map_location="cpu")

    def a(name):
        return get(am, name, actor, "actor")

    def r(name):
        return get(rm, name, rollout, "rollout")

    def canon(x):
        while isinstance(x, (tuple, list)):
            x = x[0]
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        return x[:129].contiguous()

    qkv = canon(r("layer.0.attn.method._fused_wqa_wkv_gemm"))
    qr, rs, kv = r("layer.0.attn.method._split_qkv_and_norm")
    aq, ass = a("layer.0.attn.linear_q_up_proj.normalized_quantized_input")
    pairs = [
        ("mhc post_mix", canon(r("rlk.post_mix")), canon(a("rlk.post_mix"))),
        ("initial norm", canon(r("rlk.normalized")), canon(a("rlk.normalized"))),
        ("wq_a", qkv[:, :1024].contiguous(), canon(a("layer.0.attn.linear_q_down_proj.output"))),
        ("wkv", qkv[:, 1024:].contiguous(), canon(a("layer.0.attn.linear_kv_proj.output"))),
        ("QNorm FP8 codes", qr, canon(aq)),
        ("QNorm FP32 scales", rs, canon(ass)),
        (
            "wq_b",
            canon(r("layer.0.attn.method._wq_b_proj")),
            canon(a("layer.0.attn.linear_q_up_proj.output")),
        ),
        ("KV norm", kv, canon(a("layer.0.attn.kv_layernorm.output"))),
        ("attention output", canon(r("layer.0.attn.output")), canon(a("layer.0.attn.output"))),
    ]
    results = []
    for label, x, y in pairs:
        assert x.shape == y.shape and x.dtype == y.dtype, (
            label,
            x.shape,
            y.shape,
            x.dtype,
            y.dtype,
        )
        xx = x.contiguous().view(torch.uint8).reshape(x.numel(), x.element_size())
        yy = y.contiguous().view(torch.uint8).reshape(y.numel(), y.element_size())
        bad = (xx != yy).any(-1)
        record = dict(
            stage=label,
            shape=list(x.shape),
            dtype=str(x.dtype),
            elements=x.numel(),
            mismatch_count=int(bad.sum()),
        )
        if bad.any():
            i = int(bad.nonzero()[0])
            record.update(
                first_index=i,
                rollout=x.reshape(-1)[i].item(),
                actor=y.reshape(-1)[i].item(),
                rollout_bits=bytes(xx[i].tolist())[::-1].hex(),
                actor_bits=bytes(yy[i].tolist())[::-1].hex(),
            )
        results.append(record)
        print(json.dumps(record), flush=True)
    (root / "first-difference.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
