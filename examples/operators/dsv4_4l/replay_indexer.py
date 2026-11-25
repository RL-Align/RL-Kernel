# SPDX-License-Identifier: Apache-2.0
"""Replay actual long-case indexer operands in the production FP8 domain."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig

from examples.operators.dsv4_4l.replay_q_up import compare
from rl_engine.backends.rocm.attention.qkv_preprocess import DSV4QKVPreprocess
from rl_engine.integrations.common.device_identity import hip_device_identity
from rl_engine.integrations.engines.train.megatron.native_indexer import production_indexer_select


def load(root, name):
    entries = json.loads((root / "manifest.json").read_text())
    e = next(e for e in entries if e["name"] == name and e["layer"] == 2)
    return torch.load(root / e["file"], map_location="cuda", weights_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-run", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    actor = next((args.trace_run / "common-fixed-actor/traces").glob("actor-rank-0-*-single-2177"))
    rollout = args.trace_run / "N1/fixed-rollout/policy-sync/traces/rollout-rank-0-single-2177"
    stem = "layer.2.attn.core_attention.indexer"
    raw_q = load(actor, stem + ".linear_wq_b.output")[0]
    t, batch, _ = raw_q.shape
    raw_q = raw_q.reshape(t, batch, 64, 128)
    kv = load(actor, stem + ".compressor.linear_wkv.output")[0]
    score = load(actor, stem + ".compressor.linear_wgate.output")[0]
    weights = load(actor, stem + ".linear_weights_proj.output")[0]
    cfg = AutoConfig.from_pretrained(args.model)
    cache = DSV4QKVPreprocess(cfg, 4).cos_sin_cache
    files = list(args.model.glob("*.safetensors"))
    with safe_open(str(files[0]), framework="pt", device="cuda") as source:
        ape = source.get_tensor("layers.2.attn.indexer.compressor.ape")
        norm = source.get_tensor("layers.2.attn.indexer.compressor.norm.weight")
    _, indices, operands = production_indexer_select(
        raw_q, kv, score, ape, norm, weights, cache, cfg.rms_norm_eps
    )
    original = load(rollout, "layer.2.attn.indexer.indexer_op.output")[:t]
    native_q, _, native_weights = load(rollout, "layer.2.attn.indexer.output")
    valid = torch.arange(1, t + 1, device=indices.device)[:, None] // 4

    def selection(value):
        value = value.reshape(t, 512)
        return torch.where((value >= 0) & (value < valid), value, -1).sort(-1).values

    checks = dict(
        q_codes=compare(operands["q"][0], native_q),
        folded_head_weights=compare(operands["weights"][0], native_weights),
        selected_key_sets=compare(selection(indices[0]), selection(original)),
    )
    result = dict(
        status="PASS" if all(v["mismatch_count"] == 0 for v in checks.values()) else "FAIL",
        rank=0,
        identity=hip_device_identity(),
        checks=checks,
        scope="actual FP8 indexer replay only; not full-model or GRPO acceptance",
        trace_run=str(args.trace_run),
        profile="native inference FP8 Q/K RoPE/C4/MQA/Top512; unchanged original BF16/F32 weights",
    )
    (args.output / "rank-0.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
