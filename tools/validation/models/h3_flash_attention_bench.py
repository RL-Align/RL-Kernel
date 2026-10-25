# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Accuracy and timing of h3-flash-attention-v1 vs #553 (H3FullAttentionOp), SDPA and FP64."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from rl_engine.backends.shared.triton.attention.h3_flash import H3FlashAttentionOp

H, D = 56, 128


def timed(fn, repeats=5):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out) * 1e3


def fp64(q, k, v, grad):
    q, k, v = (x.detach().double().requires_grad_() for x in (q, k, v))
    out = torch.einsum(
        "bhst,bthd->bshd", (torch.einsum("bshd,bthd->bhst", q, k) / D**0.5).softmax(-1), v
    )
    out.backward(grad.double())
    return out, q.grad, k.grad, v.grad


def run(op, q, k, v, grad):
    q, k, v = (x.detach().clone().requires_grad_() for x in (q, k, v))
    out = op(q, k, v)
    out.backward(grad)
    return out, q.grad, k.grad, v.grad


def sdpa(q, k, v):
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    return torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[12, 925, 4096, 8349])
    ap.add_argument("--out")
    args = ap.parse_args()
    ops = {"flash_v1": H3FlashAttentionOp(), "sdpa_provider": sdpa}
    try:
        from rl_engine.backends.cuda.attention.h3_full_attention import H3FullAttentionOp

        H3FullAttentionOp()(
            *(torch.zeros(1, 8, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        )
        ops["h3_full_553"] = H3FullAttentionOp()
    except Exception as exc:  # noqa: BLE001
        print("553 unavailable:", exc)
    rows = []
    for idx, s in enumerate(args.seqs):
        g = torch.Generator(device="cuda").manual_seed(1)
        q, k, v = (
            torch.randn(1, s, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
            for _ in range(3)
        )
        grad = torch.randn(1, s, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
        ref = fp64(q, k, v, grad) if s <= 4096 else None
        names = list(ops)
        shift = idx % len(names)  # rotate the measurement order per sequence length
        order = names[shift:] + names[:shift]
        for name in order:
            op = ops[name]
            if name == "h3_full_553" and s > 8349:
                continue
            got = run(op, q, k, v, grad)
            row = {
                "seq": s,
                "impl": name,
                "order": order,
                "fwd_ms": timed(lambda: op(q, k, v)),
                "fwd_bwd_ms": timed(lambda: run(op, q, k, v, grad), 3),
            }
            if ref is not None:
                for n, a, r in zip(("out", "dq", "dk", "dv"), got, ref):
                    row[f"rel_{n}"] = ((a.double() - r).norm() / r.norm()).item()
            rows.append(row)
            print(json.dumps(row))
            del got
        del ref
    if args.out:
        json.dump(
            {"device": torch.cuda.get_device_name(), "rows": rows}, open(args.out, "w"), indent=1
        )


if __name__ == "__main__":
    main()
