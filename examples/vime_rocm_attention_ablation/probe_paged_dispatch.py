"""Isolate CK dispatch, physical KV addressing and HIP replay on identical data.

This is a diagnostic, not an end-to-end bitwise/performance acceptance test.
The inflated max-Q variant identifies dispatch effects; it is not a proposed
production optimization because it also launches unused query blocks.
"""

from __future__ import annotations

import argparse
import json
from functools import partial

import torch

from rl_engine.kernels.ops.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore


def report(label, actual, expected, **context):
    actual = actual.contiguous()
    expected = expected.contiguous()
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
    mismatch = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
    print(
        json.dumps(
            dict(
                check=label,
                finite=finite,
                bitwise=mismatch == 0,
                mismatch=mismatch,
                elements=actual.numel(),
                max_abs=float((actual.float() - expected.float()).abs().max()),
                **context,
            )
        ),
        flush=True,
    )


def invoke(label, fn, profile):
    print(json.dumps(dict(begin=label)), flush=True)
    result = fn()
    torch.cuda.synchronize()
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            result = fn()
            torch.cuda.synchronize()
        kernels = sorted(
            {
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA and "fmha" in e.name.lower()
            }
        )
        print(json.dumps(dict(profile=label, kernels=kernels)), flush=True)
    return result


def cache_for(k, v, layout, guard_pages):
    batch, heads, length, dim = k.shape
    page_size = 16
    pages = (length + page_size - 1) // page_size
    count = batch * pages + 1
    if layout == "large":
        count = max(count, (2**31 // (page_size * heads * 2 * dim * 2)) + batch * pages + 1)
    if layout == "separate":
        kc = torch.empty((count, page_size, heads, dim), device=k.device, dtype=k.dtype)
        vc = torch.empty_like(kc)
    else:
        cache = torch.empty((count, page_size, heads, 2 * dim), device=k.device, dtype=k.dtype)
        kc, vc = cache.split(dim, dim=-1)
    # Reverse high physical page IDs; all referenced pages and tail bytes are initialized.
    ids = torch.arange(
        count - 1, count - 1 - batch * pages, -1, device=k.device, dtype=torch.int64
    ).reshape(batch, pages)
    kp = torch.zeros((batch, pages * page_size, heads, dim), device=k.device, dtype=k.dtype)
    vp = torch.zeros_like(kp)
    kp[:, :length].copy_(k.transpose(1, 2))
    vp[:, :length].copy_(v.transpose(1, 2))
    kc.index_copy_(0, ids.flatten(), kp.reshape(-1, page_size, heads, dim))
    vc.index_copy_(0, ids.flatten(), vp.reshape(-1, page_size, heads, dim))
    kc[0].zero_()
    vc[0].zero_()
    table = ids.to(torch.int32).contiguous()
    assert bool(((table >= 0) & (table < count)).all())
    assert torch.equal(kc[ids].reshape_as(kp)[:, :length], k.transpose(1, 2))
    assert torch.equal(vc[ids].reshape_as(vp)[:, :length], v.transpose(1, 2))
    if guard_pages:
        # CK reads physical page IDs for a whole 128-token KV tile before
        # applying the logical sequence mask. Fill out that tile with page 0.
        padded = torch.zeros((batch, ((pages + 7) // 8) * 8), dtype=torch.int32, device=k.device)
        padded[:, :pages].copy_(table)
        table = padded
    return kc, vc, table


def packed_forward(
    core, packed, kc, vc, table, seqs, cuq, indptr, maxq, length, causal=True, lse=True
):
    return core.forward_paged_varlen_with_lse(
        packed,
        kc,
        vc,
        page_table=table,
        seqused_k=seqs,
        cu_seqlens_q=cuq,
        kv_indptr=indptr,
        max_seqlen_q=maxq,
        max_seqlen_k=length,
        causal=causal,
        scale=128**-0.5,
        return_lse=lse,
    ).out


@torch.inference_mode()
def run_case(args, seed, length, batch):
    torch.manual_seed(seed)
    core = StrictRocmAiterCKAttentionCore()
    if args.fixed_tile:
        from rl_engine.kernels.ops.rocm.attention.fixed_paged_ck import fixed_paged_prefill

        core._mha_batch_prefill = partial(fixed_paged_prefill, tile_m=args.fixed_tile)
    q = torch.randn((batch, 8, length, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, 2, length, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    positions = torch.arange(length, device="cuda").expand(batch, -1).contiguous()
    # The training core executes one logical sequence per invocation, even
    # when rollout packs several requests into a single launch.
    ref = (
        torch.cat(
            [
                invoke(
                    f"train/{row}",
                    lambda row=row: (
                        core.forward_with_lse(
                            q[row : row + 1],
                            k[row : row + 1],
                            v[row : row + 1],
                            causal=True,
                            scale=128**-0.5,
                            query_position_ids=positions[row : row + 1],
                            key_position_ids=positions[row : row + 1],
                        ).out
                    ),
                    args.profile,
                )
                for row in range(batch)
            ]
        )
        .transpose(1, 2)
        .contiguous()
    )
    for layout in args.layouts:
        kc, vc, table = cache_for(k, v, layout, args.guard_pages)
        context = dict(
            seed=seed,
            length=length,
            batch=batch,
            layout=layout,
            pool_bytes=kc.shape[0] * kc.stride(0) * kc.element_size(),
        )
        torch.cuda.synchronize()
        print(
            json.dumps(
                dict(
                    cache_validated=True,
                    min_page=int(table.min()),
                    max_page=int(table.max()),
                    **context,
                )
            ),
            flush=True,
        )
        seqs = torch.full((batch,), length, dtype=torch.int32, device="cuda")
        indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * table.shape[1]
        for qlen in dict.fromkeys((length, min(length, 128), min(length, 17), 1)):
            packed = q[:, :, -qlen:].transpose(1, 2).contiguous().reshape(batch * qlen, 8, 128)
            cuq = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * qlen
            expected = ref[:, -qlen:].reshape_as(packed)
            for maxq in dict.fromkeys((qlen, max(4096, length))):
                call = partial(
                    packed_forward, core, packed, kc, vc, table, seqs, cuq, indptr, maxq, length
                )
                label = f"{layout}/q{qlen}/maxq{maxq}"
                actual = invoke(label, call, args.profile)
                report("full_vs_suffix", actual, expected, qlen=qlen, maxq=maxq, **context)
                if qlen == 1:
                    decode = invoke(
                        label + "/decode", lambda call=call: call(False, False), args.profile
                    )
                    report("causal_lse_vs_decode", decode, actual, qlen=qlen, maxq=maxq, **context)
                    if args.graph and maxq == 1:
                        # Retain graph addresses while lengths and active requests change.
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                call(False, False)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            graph_out = call(False, False)
                        original_table = table.clone()
                        for active in range(batch, 0, -1):
                            for new_length in (length, max(1, length - 1), max(1, length - 16)):
                                table.copy_(original_table)
                                table[active:].zero_()
                                seqs.fill_(new_length)
                                seqs[active:].fill_(1)
                                eager = call(False, False).clone()
                                graph.replay()
                                torch.cuda.synchronize()
                                report(
                                    "graph_vs_eager",
                                    graph_out,
                                    eager,
                                    active=active,
                                    new_length=new_length,
                                    **context,
                                )
                        table.copy_(original_table)
                        seqs.fill_(length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 3])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[513, 4096, 4171, 7168])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["separate", "interleaved", "large"],
        default=["separate", "interleaved", "large"],
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--guard-pages", action="store_true")
    parser.add_argument("--fixed-tile", type=int, choices=[64, 128])
    args = parser.parse_args()
    print(
        json.dumps(dict(gpu=str(torch.cuda.get_device_properties(0)), args=vars(args))), flush=True
    )
    for seed in args.seeds:
        for length in args.seq_lens:
            for batch in args.batches:
                run_case(args, seed, length, batch)


if __name__ == "__main__":
    main()
