# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""vLLM rollout with the RL-Kernel DSv4 plugin against ``Dsv4Model``, layer by layer, in bytes.

Phase 1 runs vLLM (``rl_engine/integrations/dsv4/vllm_model.py``, batch-invariant mode,
chunked prefill under a small token budget so steps mix decode tokens with prefill chunks)
and samples at ``--temperature`` (``processed_logprobs``, verl's mode); the plugin dumps
every step's per-layer streams, sub-layer intermediates and final hidden state. Phase 2
(a fresh process) loads the same checkpoint into ``Dsv4Model`` (the actor), runs one packed
prefill of every generated sequence, and compares every token vLLM processed at every
layer, then the log-prob vLLM reported for every sampled token against
``Dsv4Model.selected_logprob`` at the same temperature (FP32 bits). Exit 0 when all are
byte-identical.

    PYTHONPATH=. python scripts/check_dsv4_vllm_gate.py --ckpt ~/dsv4-work/dsv4-toy256
    PYTHONPATH=. python scripts/check_dsv4_vllm_gate.py --ckpt DIR --lora 16,32,7

``--dp N`` runs vLLM with N data-parallel attention ranks and the MoE expert-parallel over
them (``enable_expert_parallel``; one process per rank, one GPU each). Rank r gets its own
prompts (rotated lengths, its own seed), so the ranks finish at different steps and an idle
rank runs vLLM's dummy forwards, which take part in the expert exchange. Every rank's
tokens are compared with the same single-GPU ``Dsv4Model`` (EP 1).

    PYTHONPATH=. python scripts/check_dsv4_vllm_gate.py --ckpt DIR --dp 2 --lora 16,32,7

``--graph`` runs vLLM with its CUDA graphs (the plugin's dumps then cover only the eager
steps; every sampled log-prob is still compared). ``--lora-sync`` installs the adapters the
way verl does: reserved at load (``RL_KERNEL_DSV4_LORA``), copied in after the engine starts.

``--non-bi`` runs the plugin's non-invariant ablation (``RL_KERNEL_DSV4_NON_BI=1``,
``VLLM_BATCH_INVARIANT=0``, paged FP8 decode): the bytes then differ by design, and the
report shows by how much (with ``--lora``, a small difference also shows the adapters ran).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import torch


def rank_inputs(args, r: int):
    """Data-parallel rank r's prompt lengths, generation lengths and seed."""
    prompts_n = [int(v) for v in args.prompts.split(",")]
    generate = [int(v) for v in args.generate.split(",")]
    k = r % len(prompts_n)
    return prompts_n[k:] + prompts_n[:k], generate[k:] + generate[:k], args.seed + 1000 * r


def _install_lora(worker, rank: int, alpha: float, seed: int) -> None:
    """On a vLLM worker: the adapters ``attach_lora(..., seed, zero_b=False)`` gives the actor."""
    from rl_engine.models.deepseek_v4.model import build_lora_adapters

    m = worker.model_runner.get_model().model
    layers = [layer.ffn.layer for layer in m.rlk_layers()]
    dev = m.hc_head_fn.device
    adapters = build_lora_adapters(
        m.rlk_config, layers, rank, alpha, device=dev, seed=seed, zero_b=False
    )
    m.rlk_set_lora(adapters)


def run_vllm(args, dump: pathlib.Path) -> None:
    os.environ["RL_KERNEL_DSV4_ROLLOUT"] = "1"
    if args.non_bi:
        os.environ["RL_KERNEL_DSV4_NON_BI"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "0"
        os.environ["VLLM_DS4_DECODE_KERNEL"] = "paged"
    else:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # register() reaches the engine
    if args.lora and args.lora_sync:
        # verl's order: reserve zero adapters at load (before any graph capture), install the
        # real ones once the engine runs (rlk_set_lora copies them in place).
        os.environ["RL_KERNEL_DSV4_LORA"] = ",".join(args.lora.split(",")[:2])
    elif args.lora:
        os.environ["RL_KERNEL_DSV4_TEST_LORA"] = args.lora
    # Before vLLM: the plugin must load RL-Kernel's FlashMLA ahead of vLLM's attention ops.
    # isort: off
    from rl_engine.integrations.dsv4.vllm_model import register
    from vllm import LLM, SamplingParams

    # isort: on

    register()
    # Set before the engine starts (its worker processes inherit it, as with --dp); the
    # plugin writes only once the directory exists, which is after the warmup forwards.
    root = dump
    if args.dp == 1:  # with --dp, run_dp clears it before any rank starts
        shutil.rmtree(root, ignore_errors=True)
    os.environ["RL_KERNEL_DSV4_DUMP"] = str(root)
    dp = {}
    if args.dp > 1:
        # The data-parallel size and rank come from VLLM_DP_* (run_dp); LLM() refuses
        # data_parallel_size itself (that is the in-process launcher).
        dp = {"enable_expert_parallel": True}
    # --graph: vLLM's breakable CUDA graphs (the fork's default for DSv4) for uniform decode
    # batches; capture sizes up to --max-seqs keep the startup short.
    graph_kw = {}
    if args.graph:
        sizes = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256) if n <= args.max_seqs]
        graph_kw = {"compilation_config": {"cudagraph_capture_sizes": sizes}}
    llm = LLM(
        **dp,
        model=args.ckpt,
        tensor_parallel_size=1,
        max_model_len=args.max_len,
        max_num_batched_tokens=args.budget,
        max_num_seqs=args.max_seqs,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=not args.graph,
        **graph_kw,
        kv_cache_dtype="fp8",
        logprobs_mode="processed_logprobs",  # verl's: log softmax(z / T)
        generation_config="vllm",  # no checkpoint sampling defaults (top-k / top-p)
        skip_tokenizer_init=True,
        seed=args.seed,
    )
    if args.lora and args.lora_sync:
        import functools

        rank, alpha, lseed = args.lora.split(",")
        llm.collective_rpc(
            functools.partial(_install_lora, rank=int(rank), alpha=float(alpha), seed=int(lseed))
        )
    prompts_n, generate, seed = rank_inputs(args, args.dp_rank)
    if args.dp > 1:
        dump = dump / f"dp{args.dp_rank}"  # the plugin writes under <root>/dp<r>/
    shutil.rmtree(dump, ignore_errors=True)  # a rank still warming up may have written
    root.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed + 1)
    vocab = llm.llm_engine.model_config.hf_config.vocab_size
    prompts = [torch.randint(vocab, (n,), generator=g).tolist() for n in prompts_n]
    params = [
        SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=n,
            ignore_eos=True,
            detokenize=False,
            logprobs=0,
            seed=seed + i,
        )
        for i, n in enumerate(generate)
    ]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    seqs = [p + list(o.outputs[0].token_ids) for p, o in zip(prompts, outs)]
    logps = [
        [lp[t].logprob for t, lp in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]
        for o in outs
    ]
    dump.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": prompts, "seqs": seqs, "logprobs": logps}, dump / "seqs.pt")
    print(f"vLLM done: {len(seqs)} requests, {sum(map(len, seqs))} tokens")


def compare(args, dump: pathlib.Path) -> dict:
    from rl_engine.models.deepseek_v4.attention import Dsv4Batch
    from rl_engine.models.deepseek_v4.loader import load_model
    from rl_engine.models.deepseek_v4.model import attach_lora

    data = torch.load(dump / "seqs.pt")
    # vLLM never feeds the last generated token.
    fed = [s[:-1] for s in data["seqs"]]
    model = load_model(args.ckpt, max_model_len=args.max_len)
    if args.lora:
        rank, alpha, seed = args.lora.split(",")
        attach_lora(model, int(rank), float(alpha), seed=int(seed), zero_b=False)
    lens = [len(s) for s in fed]
    ids = torch.tensor([t for s in fed for t in s], device="cuda")
    batch = Dsv4Batch.from_lens(lens, lens, model.config.sliding_window, "cuda")
    caches = model.new_caches(len(fed), args.max_len)
    hidden, streams = model.forward_batch(ids, batch, caches, return_streams=True)
    # Per-layer intermediates (attention in / out, MoE in / out) of the same packed prefill.
    from rl_engine.kernels.ops.cuda.mhc import mhc_post_cuda, mhc_pre_cuda

    detail = []
    inner = []

    class Rec:  # wraps a callable weight and records its output
        def __init__(self, fn, rec, name, arg=None):
            self.fn, self.rec, self.name, self.arg = fn, rec, name, arg

        def __getattr__(self, k):
            return getattr(self.fn, k)

        def __call__(self, *a, **kw):
            y = self.fn(*a, **kw)
            self.rec[self.name] = y.clone()
            if self.arg:
                self.rec[self.arg] = a[0].clone()
            return y

    for layer in model.layers:
        rec = {}
        inner.append(rec)
        w = layer.attn.w
        w.fused_wqa_wkv = Rec(w.fused_wqa_wkv, rec, "qr_kv")
        w.wq_b = Rec(w.wq_b, rec, "q", arg="qr")
        w.o_proj = Rec(w.o_proj, rec, "y", arg="o")
    with torch.no_grad():
        caches2 = model.new_caches(len(fed), args.max_len)
        st = model.embed_streams(ids)
        for layer, cache in zip(model.layers, caches2):
            p = layer.hc_attn
            h, post, comb = mhc_pre_cuda(st, p.fn, p.scale, p.base, p.norm)[:3]
            a = layer.attn.forward_batch(h, batch, cache)
            st = mhc_post_cuda(a, st, post, comb)
            p = layer.hc_ffn
            h2, post, comb = mhc_pre_cuda(st, p.fn, p.scale, p.base, p.norm)[:3]
            m = layer.moe.forward(h2, None, ids if layer.hash_routing else None)
            st = mhc_post_cuda(m, st, post, comb)
            detail.append((h, a, h2, m))
    # The actor records wq_b's base output; add its LoRA delta to compare with vLLM's.
    from rl_engine.models.deepseek_v4.lora import lora_apply

    for layer, rec in zip(model.layers, inner):
        ad = layer.attn.w.lora.get("wq_b")
        if ad is not None:
            rec["q"] = lora_apply(rec["q"], rec["qr"], ad)
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)

    layers = len(streams)
    sites = ("attn_in", "attn_out", "moe_in", "moe_out")
    inner_sites = ("qr_kv", "q", "o", "y")
    sub_differ = [dict.fromkeys(sites + inner_sites, 0) for _ in range(layers)]
    differ = [0] * (layers + 1)
    maxdiff = [0.0] * (layers + 1)
    compared = 0
    o_bad = [[] for _ in range(layers)]  # (request, position) of differing attention outputs

    def bad_rows(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.cuda().reshape(y.shape)
        return (x.view(torch.int16) != y.view(torch.int16)).flatten(1).any(1)

    steps = sorted((dump / "rank0").glob("step*.pt"))
    for path in steps:
        d = torch.load(path)
        qsl, tok, pos = d["query_start_loc"].tolist(), d["input_ids"], d["positions"]
        for a, b in zip(qsl[:-1], qsl[1:]):
            if b == a:
                continue
            s, e = int(pos[a]), int(pos[b - 1]) + 1
            seg = tok[a:b].tolist()
            cand = [r for r, f in enumerate(fed) if len(f) >= e and f[s:e] == seg]
            if len(cand) != 1:
                raise RuntimeError(f"{path.name}: segment {a}:{b} (pos {s}..{e}) matches {cand}")
            r = cand[0]
            ref = slice(offs[r] + s, offs[r] + e)
            pairs = [(d["streams"][i], streams[i]) for i in range(layers)]
            pairs.append((d["hidden"], hidden))
            for i, (got, want) in enumerate(pairs):
                x, y = got[a:b].cuda(), want[ref]
                differ[i] += int(bad_rows(x, y).sum())
                maxdiff[i] = max(maxdiff[i], float((x.float() - y.float()).abs().max()))
            for li in range(layers):
                for k, site in enumerate(sites):
                    got = d["detail"][li][k][a:b]
                    sub_differ[li][site] += int(bad_rows(got, detail[li][k][ref]).sum())
                for site in inner_sites:
                    got = d["inner"][li][site][a:b]
                    bad = bad_rows(got, inner[li][site][ref])
                    sub_differ[li][site] += int(bad.sum())
                    if site == "o":
                        o_bad[li] += [(r, s + j) for j in bad.nonzero().flatten().tolist()]
            compared += b - a
    # Sampled-token log-probs: generated token j of request r is predicted at position P + j - 1.
    temp = args.temperature if args.temperature > 1e-5 else 1.0
    lp_total = lp_differ = 0
    lp_max = 0.0
    for r, (prompt, got) in enumerate(zip(data["prompts"], data["logprobs"])):
        n = len(got)
        first = offs[r] + len(prompt) - 1
        rows = torch.arange(first, first + n, device="cuda")
        targets = torch.tensor(data["seqs"][r][len(prompt) :], device="cuda")
        want = model.selected_logprob(hidden[rows], targets, temp)[0]
        got = torch.tensor(got, dtype=torch.float32, device="cuda")
        lp_total += n
        lp_differ += int((got.view(torch.int32) != want.view(torch.int32)).sum())
        lp_max = max(lp_max, float((got - want).abs().max()))
    expected = sum(lens)
    rows = [
        {
            "layer": i if i < layers else "final",
            "tokens_differ": differ[i],
            "max_abs_diff": maxdiff[i],
        }
        for i in range(layers + 1)
    ]
    # Graph replays skip the plugin's Python, so --graph dumps only the eager (prefill) steps:
    # every dumped token and every sampled log-prob must still match.
    covered = compared <= expected if args.graph else compared == expected
    passed = covered and not any(differ) and lp_total and not lp_differ
    return {
        "sub_layers": [{"layer": li, **sub_differ[li]} for li in range(layers)],
        "attention_out_differ": [sorted(o) for o in o_bad],
        "passed": bool(passed),
        "logprobs_compared": lp_total,
        "logprobs_differ": lp_differ,
        "logprob_max_abs_diff": lp_max,
        "tokens_compared": compared,
        "tokens_expected": expected,
        "steps": len(steps),
        "layers": rows,
    }


def run_dp(args, base: list[str]) -> None:
    """One vLLM process per data-parallel rank (vLLM's offline data-parallel launch)."""
    import socket

    shutil.rmtree(pathlib.Path(args.dump).resolve(), ignore_errors=True)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Cap each rank's CPU thread pools, as torchrun does: unset, OpenBLAS / OpenMP start one
    # thread per CPU in every rank, and 8 ranks pass a per-user thread limit (ulimit -u 2048
    # on some Slurm sites), failing with "Resource temporarily unavailable".
    threads = {k: os.environ.get(k, "4") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    procs = []
    for r in range(args.dp):
        env = dict(
            os.environ,
            **threads,
            VLLM_DP_RANK=str(r),
            VLLM_DP_RANK_LOCAL=str(r),
            VLLM_DP_SIZE=str(args.dp),
            VLLM_DP_MASTER_IP="127.0.0.1",
            VLLM_DP_MASTER_PORT=str(port),
        )
        cmd = base + ["--phase", "vllm", "--dp-rank", str(r)]
        procs.append(subprocess.Popen(cmd, env=env))
    # A rank that dies leaves the others blocked in collectives: stop them on the first failure.
    while any(p.poll() is None for p in procs):
        if any(p.poll() for p in procs):
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            break
        time.sleep(1)
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f"vLLM data-parallel ranks exited with {codes}")


def merge_reports(reports: list[dict]) -> dict:
    """Sum the data-parallel ranks' counts; per-layer rows add up, maxima take the max."""
    out = dict(reports[0])
    out["passed"] = all(r["passed"] for r in reports)
    for k in ("logprobs_compared", "logprobs_differ", "tokens_compared", "tokens_expected"):
        out[k] = sum(r[k] for r in reports)
    out["steps"] = sum(r["steps"] for r in reports)
    out["logprob_max_abs_diff"] = max(r["logprob_max_abs_diff"] for r in reports)
    out["layers"] = [
        {
            **rows[0],
            "tokens_differ": sum(x["tokens_differ"] for x in rows),
            "max_abs_diff": max(x["max_abs_diff"] for x in rows),
        }
        for rows in zip(*(r["layers"] for r in reports))
    ]
    out["sub_layers"] = [
        {k: (v if k == "layer" else sum(x[k] for x in rows)) for k, v in rows[0].items()}
        for rows in zip(*(r["sub_layers"] for r in reports))
    ]
    out["per_rank"] = [{k: r[k] for k in ("passed", "logprobs_differ")} for r in reports]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="300,140,517,64,1000,33")
    ap.add_argument("--generate", default="6,9,4,12,3,20")
    ap.add_argument("--budget", type=int, default=256, help="max_num_batched_tokens")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--lora", default="", help="rank,alpha,seed of random adapters; '' = none")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.4)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="dsv4_vllm_gate_dump")
    ap.add_argument("--phase", choices=["all", "vllm", "compare"], default="all")
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--non-bi", action="store_true", help="the plugin's non-invariant ablation")
    ap.add_argument("--dp", type=int, default=1, help="data-parallel ranks (EP over them)")
    ap.add_argument("--graph", action="store_true", help="vLLM CUDA graphs (not eager)")
    ap.add_argument(
        "--lora-sync",
        action="store_true",
        help="with --lora: reserve adapters at load, install them after the engine starts",
    )
    ap.add_argument("--dp-rank", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    dump = pathlib.Path(args.dump).resolve()

    if args.phase == "vllm":
        run_vllm(args, dump)
        return 0
    if args.phase == "all":
        # Each phase in its own process: vLLM's engine and its memory are gone before phase 2.
        base = [sys.executable, __file__] + [a for a in sys.argv[1:] if not a.startswith("--phase")]
        if args.dp == 1:
            subprocess.run(base + ["--phase", "vllm"], check=True)
        else:
            run_dp(args, base)
    if args.dp == 1:
        report = compare(args, dump)
    else:
        reports = [compare(args, dump / f"dp{r}") for r in range(args.dp)]
        report = merge_reports(reports)
    report["config"] = {k: v for k, v in vars(args).items() if k not in ("phase", "out")}
    verdict = "PASS" if report["passed"] else "FAIL"
    if args.non_bi:
        verdict = f"non-invariant ablation, byte-identical: {report['passed']}"
    print(f"rollout (vLLM + RL-Kernel) vs actor (Dsv4Model): {verdict}")
    print(
        f"  tokens compared {report['tokens_compared']} / {report['tokens_expected']}, "
        f"{report['steps']} steps"
    )
    print(
        f"  sampled-token log-probs differ {report['logprobs_differ']} / "
        f"{report['logprobs_compared']}, max |diff| {report['logprob_max_abs_diff']:.3e}"
    )
    for row in report["layers"]:
        print(
            f"  layer {row['layer']!s:>5}  tokens differ {row['tokens_differ']:6d}  "
            f"max |diff| {row['max_abs_diff']:.3e}"
        )
    print("  tokens differing per site (attention / MoE in, out; inside attention qr_kv, q, o, y):")
    for row in report["sub_layers"]:
        print("   ", row)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["passed"] or args.non_bi else 1


if __name__ == "__main__":
    raise SystemExit(main())
