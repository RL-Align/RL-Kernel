# SPDX-License-Identifier: Apache-2.0
"""Replay original real Q-up operands; operator diagnosis, never model acceptance."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from rl_engine.backends.rocm.fp8 import fp8_dtype
from rl_engine.backends.rocm.norm.qnorm_fp8 import _FusedQNorm
from rl_engine.integrations.common.device_identity import hip_device_identity
from rl_engine.integrations.engines.rollout.vllm.q_up_projection import fixed_q_up_fp8


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError(
            f"replay tensor shape/dtype mismatch: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        )
    x = a.contiguous().reshape(-1).view(torch.uint8).reshape(a.numel(), a.element_size())
    y = b.contiguous().reshape(-1).view(torch.uint8).reshape(b.numel(), b.element_size())
    bad = (x != y).any(-1)
    return dict(elements=a.numel(), mismatch_count=int(bad.sum()))


def event(root, name):
    rows = json.loads((root / "manifest.json").read_text())
    e = next(e for e in rows if e["name"] == name and e["layer"] == 0)
    return torch.load(root / e["file"], map_location="cuda", weights_only=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-run", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.cuda.set_device(0)
    a = next((args.trace_run / "common-fixed-actor/traces").glob("actor-rank-0-*-single-2047"))
    b = next(
        (args.trace_run / "N1/fixed-rollout/policy-sync/traces").glob("rollout-rank-0-single-2047")
    )
    raw = event(a, "layer.0.attn.linear_q_down_proj.output")[0]
    actor = event(a, "layer.0.attn.linear_q_up_proj.output")[0].reshape(raw.shape[0], -1)
    original = event(b, "layer.0.attn.method._wq_b_proj.output")
    packed = event(b, "layer.0.attn.method._wq_b_proj.input")["args"]
    q, scale = packed
    files = list(args.model.glob("*.safetensors"))
    if len(files) != 1:
        raise ValueError("replay requires the original single-file seven-layer source")
    with safe_open(str(files[0]), framework="pt", device="cuda") as source:
        weight = source.get_tensor("layers.0.attn.wq_b.weight")
        ws = source.get_tensor("layers.0.attn.wq_b.scale").float()
        norm = source.get_tensor("layers.0.attn.q_norm.weight")
    eps = json.loads((args.model / "config.json").read_text())["rms_norm_eps"]
    _, actor_q, actor_scale = _FusedQNorm.apply(raw, norm, eps)
    if fp8_dtype(q.device) == torch.float8_e4m3fnuz:
        codes = weight.view(torch.uint8)
        weight = torch.where(codes == 128, 0, codes).to(torch.uint8).view(torch.float8_e4m3fnuz)
        ws = ws * 2.0
    replay = fixed_q_up_fp8(q, weight, scale, ws)
    checks = dict(
        qnorm_codes=compare(actor_q, q),
        qnorm_scales=compare(actor_scale, scale),
        fixed_ck_vs_actor=compare(actor, replay),
        original_native_vs_actor=compare(actor, original),
    )
    passed = all(
        checks[k]["mismatch_count"] == 0
        for k in ["qnorm_codes", "qnorm_scales", "fixed_ck_vs_actor"]
    )
    passed = passed and checks["original_native_vs_actor"]["mismatch_count"] > 0
    result = dict(
        status="PASS" if passed else "FAIL",
        rank=0,
        identity=hip_device_identity(),
        checks=checks,
        scope="saved real Q-up arithmetic replay only; no full-model, GRPO or EP acceptance",
        trace_run=str(args.trace_run),
    )
    (args.output / "rank-0.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
