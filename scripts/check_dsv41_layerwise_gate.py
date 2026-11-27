# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run the DSv4.1 layerwise exact-zero gate (``rl_engine/models/deepseek_v41/layerwise_gate.py``).

Rollout (staggered arrivals, chunked prefill under a token budget, decode next to prefill
chunks) against the actor recompute (one packed prefill, reversed request order) and the
training forward, compared at every layer. Exit code 0 when every layer, the final hidden
state and the selected-token log-probs are byte-identical.

    # random checkpoint, the toy config (7 layers, every role, an Engram layer)
    PYTHONPATH=. python scripts/check_dsv41_layerwise_gate.py --out report.json
    # the first N layers of the release (Engram tables stay in host memory)
    PYTHONPATH=. python scripts/check_dsv41_layerwise_gate.py --ckpt DIR --layers 4
    # more layers: routed experts over N GPUs (every rank runs the same gate; EP is bitwise
    # to EP 1), Engram tables mapped from the shards
    PYTHONPATH=. torchrun --nproc-per-node 4 scripts/check_dsv41_layerwise_gate.py \
        --ckpt DIR --layers 22 --engram-tables mmap
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", help="checkpoint directory (default: a random toy checkpoint)")
    ap.add_argument("--layers", type=int, default=None, help="keep only the first layers")
    ap.add_argument("--prompts", default="300,140,517,64,1000,33")
    ap.add_argument("--generate", default="6,9,4,12,3,20")
    ap.add_argument("--budget", type=int, default=256, help="rollout tokens per step")
    ap.add_argument("--lora-rank", type=int, default=16, help="0 = no adapters")
    ap.add_argument("--no-train", action="store_true", help="skip the training forward")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--engram-tables", default="cpu", help="device of the Engram tables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args()

    from rl_engine.models.deepseek_v4.model import attach_lora
    from rl_engine.models.deepseek_v41 import Dsv41Config, toy_config
    from rl_engine.models.deepseek_v41.engram import build_compressed_token_map
    from rl_engine.models.deepseek_v41.layerwise_gate import layerwise_gate
    from rl_engine.models.deepseek_v41.loader import load_model, random_checkpoint

    ep_group, rank = None, 0
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist

        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
        ep_group = dist.group.WORLD

    t0 = time.time()
    if args.ckpt:
        cfg = Dsv41Config.from_pretrained(args.ckpt)
        src = args.ckpt
        token_map, _ = build_compressed_token_map(str(pathlib.Path(args.ckpt) / "tokenizer.json"))
        tables = args.engram_tables
    else:
        cfg = Dsv41Config.from_dict({**toy_config(vocab_size=2048), "moe_intermediate_size": 256})
        src = random_checkpoint(cfg, seed=args.seed)
        token_map, tables = list(range(cfg.vocab_size)), None
    model = load_model(
        src,
        cfg,
        max_model_len=args.max_len,
        num_layers=args.layers,
        token_map=token_map,
        engram_table_device=tables,
        ep_group=ep_group,
    )
    if rank:
        globals()["print"] = lambda *a, **k: None
    print(f"loaded {len(model.layers)} layers in {time.time() - t0:.0f} s", flush=True)
    torch.manual_seed(args.seed)  # the same adapters on every rank
    if args.lora_rank:
        attach_lora(model, rank=args.lora_rank, alpha=2.0 * args.lora_rank, zero_b=False)

    prompts = [int(v) for v in args.prompts.split(",")]
    generate = [int(v) for v in args.generate.split(",")]
    g = torch.Generator().manual_seed(args.seed + 1)
    ids = [
        torch.randint(cfg.vocab_size, (p + n,), generator=g).cuda()
        for p, n in zip(prompts, generate)
    ]
    report = layerwise_gate(model, ids, prompts, args.budget, train=not args.no_train)
    report["config"] = {
        "ckpt": args.ckpt or "random toy",
        "layers": len(model.layers),
        "roles": [
            f"{cfg.layer_role(i).mode} r{cfg.layer_role(i).ratio}" for i in range(len(model.layers))
        ],
        "prompts": prompts,
        "generate": generate,
        "budget": args.budget,
        "lora_rank": args.lora_rank,
        "ep": 1 if ep_group is None else torch.distributed.get_world_size(),
        "device": torch.cuda.get_device_name(),
    }
    for name in ("actor", "train"):
        if name not in report:
            continue
        r = report[name]
        print(f"rollout vs {name}: {'PASS' if r['passed'] else 'FAIL'}")
        for row in r["layers"]:
            print(
                f"  layer {row['layer']:3d} ({report['config']['roles'][row['layer']]:9s})  "
                f"tokens differ {row['tokens_differ']:6d}  max |diff| {row['max_abs_diff']:.3e}"
            )
        print(f"  final hidden same: {r['hidden_same']}  logprob same: {r['logp_same']}")
    print("GATE", "PASS" if report["passed"] else "FAIL", f"({report['tokens']} tokens)")
    if args.out and rank == 0:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
