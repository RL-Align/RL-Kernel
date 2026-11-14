# SPDX-License-Identifier: Apache-2.0
"""CPU replay of captured QNorm operands; references never count as GPU parity."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def mismatch(x, y):
    assert x.dtype == y.dtype and x.shape == y.shape
    a = x.contiguous().view(torch.uint8).reshape(x.numel(), x.element_size())
    b = y.contiguous().view(torch.uint8).reshape(y.numel(), y.element_size())
    changed = (a != b).any(-1)
    out = {"elements": x.numel(), "mismatch_count": int(changed.sum())}
    if changed.any():
        i = int(changed.nonzero()[0])
        out["first"] = dict(
            index=i,
            left=x.flatten()[i].item(),
            right=y.flatten()[i].item(),
            left_bits=bytes(a[i].tolist())[::-1].hex(),
            right_bits=bytes(b[i].tolist())[::-1].hex(),
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actor = args.trace / "actor-v0-trace"
    rollout = args.trace / "baseline/generation-trace"
    aa = json.loads((actor / "manifest.json").read_text())
    rr = [json.loads(s) for s in (rollout / "manifest.jsonl").read_text().splitlines()]

    def load(events, name, root, forward=None):
        e = next(
            e for e in events if e["name"] == name and (forward is None or e["forward"] == forward)
        )
        return torch.load(root / e["file"], weights_only=True, map_location="cpu")

    x = load(aa, "layer.0.attn.q_layernorm.input", actor)[0][:129].reshape(129, 1024)
    fused_input = load(rr, "layer.0.attn.method._fused_wqa_wkv_gemm", rollout, 0)
    if isinstance(fused_input, (tuple, list)):
        fused_input = fused_input[0]
    fused_input = fused_input[:129, :1024].contiguous()
    native = load(aa, "layer.0.attn.q_layernorm.output", actor)[:129].reshape(129, 1024)
    codes, scales, _ = load(rr, "layer.0.attn.method._split_qkv_and_norm", rollout, 0)
    files = list(args.checkpoint.glob("*.safetensors"))
    assert len(files) == 1
    with safe_open(str(files[0]), framework="pt", device="cpu") as weights:
        weight = weights.get_tensor("layers.0.attn.q_norm.weight")
    config = json.loads((args.checkpoint / "config.json").read_text())
    eps = config.get("rms_norm_eps", 1e-6)
    xf = x.float()
    square_sum = (xf * xf).sum(-1, keepdim=True)
    variance = square_sum / x.shape[-1]
    inv = (variance + eps).rsqrt()
    before_cast = (xf * inv) * weight.float()
    after_cast = before_cast.to(torch.bfloat16)
    records = {
        "scope": "captured GPU operands; CPU reference stages are not actual GPU intermediates",
        "actual_norm_input": mismatch(x, fused_input),
        "cpu_round_to_bf16_vs_actual_native_norm": mismatch(after_cast, native),
        "reference_epsilon": eps,
    }
    stages = dict(
        input=x,
        weight=weight,
        cpu_square_sum=square_sum,
        cpu_mean=variance,
        cpu_rsqrt=inv,
        cpu_weighted_fp32=before_cast,
        cpu_first_bf16=after_cast,
        actual_native_bf16=native,
        actual_rollout_codes=codes,
        actual_rollout_scales=scales,
    )
    for name, value in (("fp32_before_store", before_cast), ("bf16_after_store", native.float())):
        grouped = value.reshape(129, 8, 128)
        scale = grouped.abs().amax(-1).clamp_min(1e-10) * torch.tensor(1 / 240, dtype=torch.float32)
        quant = (
            (grouped * scale.reciprocal()[..., None]).reshape(129, 1024).to(torch.float8_e4m3fnuz)
        )
        records[name] = {"scale": mismatch(scale, scales), "codes": mismatch(quant, codes)}
        stages[name + "_scale"] = scale
        stages[name + "_codes"] = quant
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(stages, args.output / "stages.pt")
    (args.output / "diagnostic.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
