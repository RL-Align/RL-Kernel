# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""vLLM main with the RL-Kernel DSv4.1 plugin against ``Dsv41Model``, layer by layer, in bytes.

Phase 1 runs vLLM (``rl_engine/integrations/dsv41/vllm_model.py``, eager, chunked prefill
under a small token budget so steps mix decode tokens with prefill chunks) and samples at
``--temperature`` (``processed_logprobs``, verl's mode); the plugin dumps every step's
per-layer streams and final hidden state. Phase 2 (a fresh process) loads the checkpoint
into ``Dsv41Model`` (the actor), prefills every generated sequence, and compares every token
vLLM processed at every layer, then the log-prob vLLM reported for every sampled token
against ``Dsv41Model.selected_logprob`` at the same temperature (FP32 bits). Exit 0 when all
are byte-identical.

    PYTHONPATH=. python scripts/check_dsv41_vllm_gate.py --ckpt TOY_DIR --lora 16,32,7
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import torch


def run_vllm(args, dump: pathlib.Path) -> None:
    os.environ["RL_KERNEL_DSV41_ROLLOUT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # register() reaches the engine
    if args.lora:
        os.environ["RL_KERNEL_DSV41_TEST_LORA"] = args.lora
    from vllm import LLM, SamplingParams

    from rl_engine.integrations.dsv41.vllm_model import register

    register()
    shutil.rmtree(dump, ignore_errors=True)
    os.environ["RL_KERNEL_DSV41_DUMP"] = str(dump)  # written once the directory exists
    llm = LLM(
        model=args.ckpt,
        tensor_parallel_size=1,
        max_model_len=args.max_len,
        max_num_batched_tokens=args.budget,
        max_num_seqs=args.max_seqs,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_util,
        enforce_eager=True,
        logprobs_mode="processed_logprobs",  # verl's: log softmax(z / T)
        generation_config="vllm",  # no checkpoint sampling defaults (top-k / top-p)
        skip_tokenizer_init=True,
        seed=args.seed,
    )
    dump.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed + 1)
    hf = llm.llm_engine.model_config.hf_config
    vocab = getattr(hf, "text_config", hf).vocab_size
    prompts_n = [int(v) for v in args.prompts.split(",")]
    generate = [int(v) for v in args.generate.split(",")]
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
            seed=args.seed + i,
        )
        for i, n in enumerate(generate)
    ]
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params)
    seqs = [p + list(o.outputs[0].token_ids) for p, o in zip(prompts, outs)]
    logps = [
        [lp[t].logprob for t, lp in zip(o.outputs[0].token_ids, o.outputs[0].logprobs)]
        for o in outs
    ]
    torch.save({"prompts": prompts, "seqs": seqs, "logprobs": logps}, dump / "seqs.pt")
    print(f"vLLM done: {len(seqs)} requests, {sum(map(len, seqs))} tokens")


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per row: all elements bitwise equal."""
    ai = a.contiguous().view(torch.int16 if a.element_size() == 2 else torch.int32)
    bi = b.contiguous().view(torch.int16 if b.element_size() == 2 else torch.int32)
    return (ai == bi).flatten(1).all(1)


def compare(args, dump: pathlib.Path) -> dict:
    from rl_engine.models.deepseek_v41 import Dsv41Config
    from rl_engine.models.deepseek_v41.attention import Dsv41Batch
    from rl_engine.models.deepseek_v41.engram import build_compressed_token_map, ngram_windows
    from rl_engine.models.deepseek_v41.loader import load_model

    data = torch.load(dump / "seqs.pt")
    ckpt = pathlib.Path(args.ckpt)
    cfg = Dsv41Config.from_pretrained(ckpt)
    token_map = None
    if cfg.engram_layer_ids:
        token_map, _ = build_compressed_token_map(str(ckpt / "tokenizer.json"))
    model = load_model(ckpt, cfg, max_model_len=args.max_len, token_map=token_map,
                       engram_table_device="mmap")  # fmt: skip
    if args.lora:
        from rl_engine.models.deepseek_v4.model import attach_lora

        rank, alpha, seed = args.lora.split(",")
        torch.manual_seed(int(seed))
        attach_lora(model, int(rank), float(alpha), seed=int(seed), zero_b=False)
    # The actor: one prefill per sequence (its bytes do not depend on the batch).
    ref_streams, ref_hidden = [], []
    with torch.no_grad():
        for seq in data["seqs"]:
            ids = torch.tensor(seq, device="cuda")
            n = len(seq)
            batch = Dsv41Batch.from_lens([n], [n], cfg.sliding_window)
            win = None
            if model.hasher is not None:
                pos = torch.arange(n, device="cuda")
                win = ngram_windows([ids], [pos], cfg.engram_max_ngram_size)
            caches = model.new_caches(1, args.max_len)
            h, per = model.forward_batch(ids, batch, caches, engram_windows=win,
                                         return_streams=True)  # fmt: skip
            ref_hidden.append(h)
            ref_streams.append(per)

    layers = len(model.layers)
    differ = [0] * (layers + 1)  # per layer, then the final hidden
    compared = steps = unmatched = 0
    seqs = [torch.tensor(s) for s in data["seqs"]]
    for f in sorted(dump.glob("rank0/step*.pt")):
        d = torch.load(f)
        steps += 1
        qsl = d["query_start_loc"].tolist()
        for r in range(len(qsl) - 1):
            lo, hi = qsl[r], qsl[r + 1]
            if hi == lo:
                continue
            ids = d["input_ids"][lo:hi]
            p0 = int(d["positions"][lo])
            match = [
                i for i, s in enumerate(seqs)
                if len(s) >= p0 + hi - lo and torch.equal(s[p0 : p0 + hi - lo], ids)
            ]  # fmt: skip
            if len(match) != 1:
                unmatched += hi - lo
                continue
            q = match[0]
            for i in range(layers):
                got = d["streams"][i][lo:hi].cuda()
                differ[i] += int((~_bits_equal(got, ref_streams[q][i][p0 : p0 + hi - lo])).sum())
            got = d["hidden"][lo:hi].cuda()
            differ[layers] += int((~_bits_equal(got, ref_hidden[q][p0 : p0 + hi - lo])).sum())
            compared += hi - lo

    # Sampled-token log-probs: the token at p was sampled from the logits of position p - 1.
    lp_differ = lp_n = 0
    max_abs = 0.0
    with torch.no_grad():
        for q, (prompt, got) in enumerate(zip(data["prompts"], data["logprobs"])):
            if not got:
                continue
            seq = data["seqs"][q]
            p, n = len(prompt), len(got)
            rows = ref_hidden[q][p - 1 : p - 1 + n]
            tgt = torch.tensor(seq[p : p + n], device="cuda")
            t = args.temperature if args.temperature > 0 else 1.0
            ref = model.selected_logprob(rows, tgt, t)[0]
            vl = torch.tensor(got, dtype=torch.float32, device="cuda")
            neq = ref.view(torch.int32) != vl.view(torch.int32)
            lp_differ += int(neq.sum())
            lp_n += n
            max_abs = max(max_abs, float((ref - vl).abs().max()))
    expected = sum(len(s) for s in data["seqs"]) - sum(1 for _ in data["seqs"])
    report = {
        "steps": steps,
        "tokens_compared": compared,
        "tokens_unmatched": unmatched,
        "tokens_expected_about": expected,
        "layers": [{"layer": i, "tokens_differ": v} for i, v in enumerate(differ[:-1])]
        + [{"layer": "final", "tokens_differ": differ[-1]}],
        "logprobs_compared": lp_n,
        "logprobs_differ": lp_differ,
        "logprob_max_abs_diff": max_abs,
    }
    report["passed"] = bool(
        steps and compared and not unmatched and not any(differ) and lp_n and not lp_differ
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", default="300,140,517,64,1000,33")
    ap.add_argument("--generate", default="6,9,4,12,3,20")
    ap.add_argument("--budget", type=int, default=256, help="max_num_batched_tokens")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--lora", default="", help="rank,alpha,seed of random adapters; '' = none")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.4)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="dsv41_vllm_gate_dump")
    ap.add_argument("--phase", choices=["all", "vllm", "compare"], default="all")
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args()
    dump = pathlib.Path(args.dump).resolve()
    if args.phase == "vllm":
        run_vllm(args, dump)
        return 0
    if args.phase == "all":
        # Each phase in its own process: vLLM's engine and its memory are gone before phase 2.
        base = [sys.executable, __file__] + [a for a in sys.argv[1:] if not a.startswith("--phase")]
        subprocess.run(base + ["--phase", "vllm"], check=True)
    report = compare(args, dump)
    report["config"] = {k: v for k, v in vars(args).items() if k not in ("phase", "out")}
    print(f"rollout (vLLM main + RL-Kernel) vs actor (Dsv41Model): "
          f"{'PASS' if report['passed'] else 'FAIL'}")  # fmt: skip
    print(f"  tokens compared {report['tokens_compared']} (unmatched "
          f"{report['tokens_unmatched']}), {report['steps']} steps")  # fmt: skip
    lp = report["logprob_max_abs_diff"]
    print(f"  sampled-token log-probs differ {report['logprobs_differ']} / "
          f"{report['logprobs_compared']}, max |diff| {lp:.3e}")  # fmt: skip
    for row in report["layers"]:
        print(f"  layer {row['layer']!s:>5}  tokens differ {row['tokens_differ']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
