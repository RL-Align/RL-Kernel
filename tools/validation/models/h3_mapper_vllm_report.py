# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measurements of the RFC #420 ``h3_checkpoint_mapper`` and ``vllm_runtime_adapter_h3`` rows on
the real checkpoint: JSON report plus a PNG.

    python tools/validation/models/h3_mapper_vllm_report.py --checkpoint <dir with transformer/> \
        --out report.json --png figure.png
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from rl_engine.backends.shared.triton.gemm import h3_projection as g
from rl_engine.integrations.engines.rollout.vllm.h3_omni import install_h3_rollout
from rl_engine.models.minimax_h3 import checkpoint_mapper as cm

GROUP = re.compile(r"^((?:token_refiner\.refiner_blocks|transformer_blocks)\.\d+)\.")


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def mapping_report(root):
    index = {}
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                index[k] = f
    groups = defaultdict(list)
    for name in index:
        m = GROUP.match(name)
        groups[m.group(1) if m else "model-level"].append(name)
    tensors = native_count = 0
    bitwise, t_fwd, t_back = True, 0.0, 0.0
    native_names = set()
    for names in groups.values():
        state = {}
        for n in names:
            with safe_open(index[n], "pt") as h:
                state[n] = h.get_tensor(n)
        t0 = time.perf_counter()
        native = cm.diffusers_to_native(state)
        t1 = time.perf_counter()
        back = cm.native_to_diffusers(native)
        t_back += time.perf_counter() - t1
        t_fwd += t1 - t0
        tensors += len(state)
        native_count += len(native)
        native_names |= set(native)
        bitwise &= set(back) == set(state) and all(
            torch.equal(back[n], state[n]) and back[n].dtype == state[n].dtype for n in state
        )
    return (
        dict(
            checkpoint_tensors=tensors,
            native_tensors=native_count,
            unique_native_names=len(native_names),
            groups=len(groups),
            roundtrip_bitwise=bool(bitwise),
            map_seconds=t_fwd,
            inverse_seconds=t_back,
        ),
        index,
    )


def load(index, name):
    with safe_open(index[name], "pt") as h:
        return h.get_tensor(name).cuda()


def timeit(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


class Lin(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight), None


def layer_rows(index, tokens=(70, 925)):
    rows = []
    for blk in (0, 25, 49):
        p = f"transformer_blocks.{blk}."
        q, k, v = (load(index, p + f"attn.to_{s}.weight") for s in "qkv")
        o = load(index, p + "attn.to_out.0.weight")
        f1 = load(index, p + "ff.net.0.proj.weight")
        f2 = load(index, p + "ff.net.2.weight")
        native = cm.diffusers_to_native(
            {
                p + "attn.to_q.weight": q,
                p + "attn.to_k.weight": k,
                p + "attn.to_v.weight": v,
                p + "attn.to_out.0.weight": o,
                p + "ff.net.0.proj.weight": f1,
                p + "ff.net.2.weight": f2,
            }
        )
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Module()])
        b = model.blocks[0]
        b.attn, b.mlp = torch.nn.Module(), torch.nn.Module()
        b.attn.qkv_proj = Lin(native["blocks.%d.attn.qkv_proj.weight" % blk])
        b.attn.out_proj = Lin(native["blocks.%d.attn.out_proj.weight" % blk])
        b.mlp.fc1 = Lin(native["blocks.%d.mlp.fc1.weight" % blk])
        b.mlp.fc2 = Lin(native["blocks.%d.mlp.fc2.weight" % blk])
        for t in tokens:
            gen = torch.Generator().manual_seed(blk * 1000 + t)
            hidden = q.shape[1]
            x = torch.randn(1, t, hidden, generator=gen).bfloat16().cuda()
            xo = torch.randn(1, t, o.shape[1], generator=gen).bfloat16().cuda()
            x2 = torch.randn(1, t, f2.shape[1], generator=gen).bfloat16().cuda()
            ref = {
                "qkv_proj": (x, b.attn.qkv_proj),
                "out_proj": (xo, b.attn.out_proj),
                "fc1": (x, b.mlp.fc1),
                "fc2": (x2, b.mlp.fc2),
            }
            torch_out = {n: m(xi)[0] for n, (xi, m) in ref.items()}
            torch_ms = {n: timeit(lambda m=m, xi=xi: m(xi)) for n, (xi, m) in ref.items()}
            with install_h3_rollout(model):
                out = {n: m(xi)[0] for n, (xi, m) in ref.items()}
                ms = {n: timeit(lambda m=m, xi=xi: m(xi)) for n, (xi, m) in ref.items()}
                again = {n: m(xi)[0] for n, (xi, m) in ref.items()}
            flat = x.reshape(-1, hidden)
            sep = torch.cat([g.gemm(flat, w, b_t=True) for w in (q, k, v)], dim=1)
            half = f1.shape[0] // 2
            sep_fc1 = g.gemm(flat, f1, b_t=True)
            fc1_native = out["fc1"].reshape(t, -1)
            swapped = torch.cat([sep_fc1[:, half:], sep_fc1[:, :half]], dim=1)
            for n, (xi, m) in ref.items():
                exact = xi.reshape(-1, xi.shape[-1]).double() @ m.weight.double().t()
                rows.append(
                    dict(
                        block=blk,
                        tokens=t,
                        linear=n,
                        repeatable=bool(torch.equal(out[n], again[n])),
                        rel_err_vs_fp64=rel(out[n].reshape(t, -1), exact),
                        torch_rel_err_vs_fp64=rel(torch_out[n].reshape(t, -1), exact),
                        fused_equals_separate=(
                            bool(torch.equal(out[n].reshape(t, -1), sep))
                            if n == "qkv_proj"
                            else None
                        ),
                        fc1_equals_diffusers_swapped=(
                            bool(torch.equal(fc1_native, swapped)) if n == "fc1" else None
                        ),
                        adapter_ms=ms[n],
                        torch_ms=torch_ms[n],
                        slowdown=ms[n] / torch_ms[n],
                    )
                )
    return rows


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["layers"]
    names = ("qkv_proj", "out_proj", "fc1", "fc2")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    m = report["mapping"]
    ax[0].bar(
        ["checkpoint\ntensors", "native\ntensors", "round trip\nbitwise equal"],
        [
            m["checkpoint_tensors"],
            m["native_tensors"],
            m["checkpoint_tensors"] if m["roundtrip_bitwise"] else 0,
        ],
    )
    ax[0].set_title("Checkpoint mapper (real 33B checkpoint)")
    w = 0.38
    for j, key in enumerate(("rel_err_vs_fp64", "torch_rel_err_vs_fp64")):
        vals = [max(r[key] for r in rows if r["linear"] == n) for n in names]
        ax[1].bar(
            [i + (j - 0.5) * w for i in range(4)],
            vals,
            w,
            label=("adapter" if j == 0 else "torch F.linear"),
        )
    ax[1].set_xticks(range(4), names)
    ax[1].set_yscale("log")
    ax[1].set_title("Relative error vs FP64 (worst of 3 blocks x 2 sizes)")
    ax[1].legend()
    for t, c in ((70, "C0"), (925, "C1")):
        vals = [
            sum(r["slowdown"] for r in rows if r["linear"] == n and r["tokens"] == t) / 3
            for n in names
        ]
        ax[2].plot(names, vals, "o-", color=c, label=f"{t} tokens")
    ax[2].axhline(1, color="gray", lw=1)
    ax[2].set_title("Adapter time / torch F.linear time (mean over blocks)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--png", type=Path)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    mapping, index = mapping_report(args.checkpoint / "transformer")
    report = dict(
        device=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        mapping=mapping,
        layers=layer_rows(index),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.png:
        plot(report, args.png)
    print(json.dumps(mapping, indent=1))
    for r in report["layers"]:
        print(r)


if __name__ == "__main__":
    main()
