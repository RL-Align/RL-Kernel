# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Run the DSv4 layerwise exact-zero gate (``rl_engine/models/deepseek_v4/layerwise_gate.py``).

Rollout (continuous batching: staggered arrivals, chunked prefill under a token budget,
decode next to prefill chunks) against the actor recompute (one packed prefill, reversed
request order) and the training forward, compared at every layer. Exit code 0 when
every layer, the final hidden state and the selected-token log-probs are byte-identical.

    # random checkpoint at the real dimensions, layer types window / window / C4 / C128
    PYTHONPATH=. python scripts/check_dsv4_layerwise_gate.py --out report.json
    # the first N layers of a real checkpoint (advance N one layer at a time)
    PYTHONPATH=. python scripts/check_dsv4_layerwise_gate.py --ckpt DIR --layers 3
"""

from __future__ import annotations

import argparse
import json

import torch

ROPE = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "type": "yarn",
}
QUANT = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", help="checkpoint directory (default: a random checkpoint)")
    ap.add_argument("--layers", type=int, default=None, help="keep only the first layers")
    ap.add_argument("--prompts", default="300,140,517,64,1000,33")
    ap.add_argument("--generate", default="6,9,4,12,3,20")
    ap.add_argument("--budget", type=int, default=256, help="rollout tokens per step")
    ap.add_argument("--lora-rank", type=int, default=16, help="0 = no adapters")
    ap.add_argument("--no-train", action="store_true", help="skip the training forward")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args()

    from rl_engine.models.deepseek_v4.config import Dsv4Config
    from rl_engine.models.deepseek_v4.layerwise_gate import layerwise_gate
    from rl_engine.models.deepseek_v4.loader import load_model, random_checkpoint
    from rl_engine.models.deepseek_v4.model import attach_lora

    if args.ckpt:
        cfg = Dsv4Config.from_pretrained(args.ckpt)
        src = args.ckpt
    else:
        cfg = Dsv4Config.from_dict(
            dict(
                vocab_size=2048,
                num_hidden_layers=args.layers or 4,
                compress_ratios=([0, 0, 4, 128] * 16)[: args.layers or 4],
                num_hash_layers=2,
                rope_scaling=ROPE,
                quantization_config=QUANT,
            )
        )
        src = random_checkpoint(cfg, seed=args.seed)
    model = load_model(src, cfg, max_model_len=args.max_len, num_layers=args.layers)
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
        "ckpt": args.ckpt or "random",
        "layers": len(model.layers),
        "compress_ratios": [layer.attn.c.compress_ratio for layer in model.layers],
        "prompts": prompts,
        "generate": generate,
        "budget": args.budget,
        "lora_rank": args.lora_rank,
        "device": torch.cuda.get_device_name(),
    }
    for name in ("actor", "train"):
        if name not in report:
            continue
        r = report[name]
        print(f"rollout vs {name}: {'PASS' if r['passed'] else 'FAIL'}")
        for row in r["layers"]:
            print(
                f"  layer {row['layer']:3d}  tokens differ {row['tokens_differ']:6d}  "
                f"max |diff| {row['max_abs_diff']:.3e}"
            )
        print(f"  final hidden same: {r['hidden_same']}  logprob same: {r['logp_same']}")
    print("GATE", "PASS" if report["passed"] else "FAIL", f"({report['tokens']} tokens)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
