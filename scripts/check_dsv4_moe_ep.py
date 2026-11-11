# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Check that the DSv4 MoE block output does not depend on the expert-parallel size.

Every rank builds the same full weights (fixed seed) and keeps its expert slice; the
same global batch is split over the ranks with uneven sizes (including empty ranks
and decode-like single tokens). The EP = R outputs and input gradients, gathered on
rank 0, must equal byte for byte the EP = 1 block run on the whole batch with all
experts. Scored and
hash-routed layers; two runs each (run to run).

With ``--lora`` both blocks carry the same LoRA adapters (routed fc1 / fc2, shared
expert fc1 / fc2). Outputs and input gradients must still match byte for byte. The
routed adapters' gradients (summed over the EP group inside the backward) must match
the EP = 1 gradients to rounding: partial sums per rank reorder the token reduction.

Usage (SM90, a ``KERNEL_ALIGN_MOE_SM90=1`` build, DeepGEMM):
    torchrun --nproc-per-node 8 scripts/check_dsv4_moe_ep.py [--hidden 1024 --ffn 256]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

from rl_engine.models.deepseek_v4.moe import Dsv4MoE
from rl_engine.models.deepseek_v4.router import Dsv4Router
from rl_engine.moe.mx_format import mx_quantize

E, SCALE, VOCAB = 256, 1.5, 4096


def _weights(hidden, ffn, shared_ffn, device):
    g = torch.Generator(device=device).manual_seed(1234)
    gate = (torch.randn(E, hidden, generator=g, device=device) / hidden**0.5).bfloat16()
    bias = torch.randn(E, generator=g, device=device) * 0.05
    tid2eid = torch.randint(0, E, (VOCAB, 6), generator=g, device=device, dtype=torch.int32)
    w1 = mx_quantize(
        torch.randn(E, 2 * ffn, hidden, generator=g, device=device) / hidden**0.5, "e2m1"
    )
    w2 = mx_quantize(torch.randn(E, hidden, ffn, generator=g, device=device) / ffn**0.5, "e2m1")
    s1 = (torch.randn(2 * shared_ffn, hidden, generator=g, device=device) / hidden**0.5).bfloat16()
    s2 = (torch.randn(hidden, shared_ffn, generator=g, device=device) / shared_ffn**0.5).bfloat16()
    return gate, bias, tid2eid, w1, w2, s1, s2


def _slice(t, lo, hi):
    from rl_engine.moe.mx_format import MXTensor

    return MXTensor(
        t.codes[lo:hi].contiguous(),
        t.scales[lo:hi].contiguous(),
        t.elem_format,
        (hi - lo, *t.shape[1:]),
    )


def _cases(world):
    g = torch.Generator().manual_seed(world)
    yield "uneven", [int(v) for v in torch.randint(0, 97, (world,), generator=g)]
    yield "decode", [1] * world
    yield "one-rank", [0] * (world - 1) + [300]
    yield "empty-first", [0] + [64] * (world - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--shared-ffn", type=int, default=256)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--lora", action="store_true")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    gate, bias, tid2eid, w1, w2, s1, s2 = _weights(args.hidden, args.ffn, args.shared_ffn, dev)
    local = E // world
    lo, hi = rank * local, (rank + 1) * local
    group = dist.new_group(list(range(world)))

    def make_lora():
        if not args.lora:
            return None
        from rl_engine.models.deepseek_v4.lora import LoraAdapter, MoeLora

        h, f, fs = args.hidden, args.ffn, args.shared_ffn
        lora = MoeLora(
            fc1=LoraAdapter.init(h, 2 * f, 16, 32, device=dev, seed=11, zero_b=False),
            fc2=LoraAdapter.init(f, h, 16, 32, device=dev, seed=12, zero_b=False),
            shared_fc1=LoraAdapter.init(h, 2 * fs, 16, 32, device=dev, seed=13, zero_b=False),
            shared_fc2=LoraAdapter.init(fs, h, 16, 32, device=dev, seed=14, zero_b=False),
        )
        for _, ad in lora.adapters():
            ad.a.requires_grad_(True)
            ad.b.requires_grad_(True)
        return lora

    def block(hash_layer, ep):
        router = Dsv4Router(
            gate,
            bias=None if hash_layer else bias,
            tid2eid=tid2eid if hash_layer else None,
            routed_scaling_factor=SCALE,
        )
        if ep:
            return Dsv4MoE(
                router,
                _slice(w1, lo, hi),
                _slice(w2, lo, hi),
                s1,
                s2,
                ep_group=group,
                lora=make_lora(),
            )
        return Dsv4MoE(router, w1, w2, s1, s2, lora=make_lora())

    ok = True
    backward = not args.forward_only
    for hash_layer in (False, True):
        ep_moe = block(hash_layer, True)
        ref_moe = block(hash_layer, False) if rank == 0 else None
        for name, sizes in _cases(world):
            n = sum(sizes)
            g = torch.Generator(device=dev).manual_seed(n + 7)
            x = torch.randn(n, args.hidden, generator=g, device=dev).bfloat16()
            res = torch.randn(n, args.hidden, generator=g, device=dev).bfloat16()
            ids = torch.randint(0, VOCAB, (n,), generator=g, device=dev)
            a = sum(sizes[:rank])
            b = a + sizes[rank]
            dout = torch.randn(n, args.hidden, generator=g, device=dev).bfloat16()

            def gather(t):
                pad = torch.zeros(max(sizes), args.hidden, dtype=t.dtype, device=dev)
                pad[: t.shape[0]] = t
                parts = [torch.empty_like(pad) for _ in range(world)]
                dist.all_gather(parts, pad, group=group)
                return torch.cat([parts[r][: sizes[r]] for r in range(world)])

            def routed_grads(moe):
                if moe.lora is None:
                    return []
                out = []
                for name in ("fc1", "fc2"):
                    ad = getattr(moe.lora, name)
                    out += [ad.a.grad.float().clone(), ad.b.grad.float().clone()]
                    ad.a.grad = ad.b.grad = None
                for name in ("shared_fc1", "shared_fc2"):
                    ad = getattr(moe.lora, name)
                    ad.a.grad = ad.b.grad = None
                return out

            outs, grads, lgrads = [], [], []
            for _ in range(2):
                xr = x[a:b].clone().requires_grad_(True)
                y = ep_moe.forward(xr, res[a:b].contiguous(), ids[a:b].contiguous())
                if backward:
                    y.backward(dout[a:b].contiguous())
                    grads.append(gather(xr.grad))
                    lgrads.append(routed_grads(ep_moe))
                outs.append(gather(y.detach()))
            if rank == 0:
                xf = x.clone().requires_grad_(True)
                ref = ref_moe.forward(xf, res, ids)
                same = torch.equal(outs[0].view(torch.int16), ref.view(torch.int16))
                rerun = torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
                gsame = grerun = True
                lclose = True
                lerr = 0.0
                if backward:
                    ref.backward(dout)
                    gsame = torch.equal(grads[0].view(torch.int16), xf.grad.view(torch.int16))
                    grerun = torch.equal(grads[0].view(torch.int16), grads[1].view(torch.int16))
                    for got, want in zip(lgrads[0], routed_grads(ref_moe)):
                        err = float((got - want).norm() / want.norm().clamp_min(1e-30))
                        lerr = max(lerr, err)
                    lclose = lerr < 1e-4
                good = same and rerun and gsame and grerun and lclose
                ok &= good
                diff = (outs[0].float() - ref.float()).abs().max().item() if n else 0.0
                kind = "hash" if hash_layer else "scored"
                bwd = f", dx {'same' if gsame and grerun else 'DIFFERS'}" if backward else ""
                if backward and args.lora:
                    bwd += f", LoRA grads rel {lerr:.1e}"
                print(
                    f"EP={world} {kind:6} {name:11} sizes={sizes} "
                    f"{'PASS' if good else 'FAIL'} (max |diff| {diff:.3g}, "
                    f"run-to-run {'same' if rerun else 'DIFFERS'}{bwd})",
                    flush=True,
                )
    flag = torch.tensor([1 if ok else 0], device=dev)
    dist.broadcast(flag, 0)
    dist.destroy_process_group()
    if rank == 0:
        print("ALL EP CHECKS PASS" if flag.item() else "EP CHECKS FAILED", flush=True)
    return 0 if flag.item() else 1


if __name__ == "__main__":
    sys.exit(main())
