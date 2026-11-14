# SPDX-License-Identifier: Apache-2.0
"""Compare actual four-layer boundaries at identical prefill/decode positions."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--actor-case", default="v0-A")
    parser.add_argument("--rollout-forward", type=int, default=0)
    parser.add_argument("--decode-offset", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    actor = args.run / f"actor-trace-{args.actor_case}"
    rollout = args.run / "generation-trace"
    am = json.loads((actor / "manifest.json").read_text())
    rm = [json.loads(s) for s in (rollout / "manifest.jsonl").read_text().splitlines()]
    rm = [e for e in rm if e["forward"] == args.rollout_forward]
    cases = json.loads((args.run / "cases-v0.json").read_text())
    prompt = len(next(c for c in cases if c["id"] == "A")["prompt_token_ids"])

    def load(events, directory, name, layer=None, occurrence=0):
        found = [e for e in events if e["name"] == name and (layer is None or e["layer"] == layer)]
        event = found[occurrence]
        value = torch.load(directory / event["file"], weights_only=True, map_location="cpu")
        while isinstance(value, (tuple, list)):
            value = value[0]
        return value

    def canonical(value, role):
        if args.decode_offset is None:
            value = value[:prompt]
        elif role == "actor":
            start = prompt + args.decode_offset - 1
            value = value[start : start + 1]
        return value.reshape(value.shape[0], -1)

    records = []

    def compare(label, layer, x, y):
        x, y = canonical(x, "rollout"), canonical(y, "actor")
        assert x.dtype == y.dtype and x.shape == y.shape, (
            label,
            x.dtype,
            y.dtype,
            x.shape,
            y.shape,
        )
        word = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]
        xb, yb = x.reshape(-1).view(word), y.reshape(-1).view(word)
        bad = xb != yb
        record = dict(
            layer=layer,
            stage=label,
            dtype=str(x.dtype),
            elements=x.numel(),
            mismatch_count=int(bad.sum()),
        )
        if bad.any():
            i = int(bad.nonzero()[0])
            mask = (1 << (8 * x.element_size())) - 1
            record.update(
                index=i,
                rollout=x.reshape(-1)[i].item(),
                actor=y.reshape(-1)[i].item(),
                rollout_bits=f"{int(xb[i]) & mask:0{x.element_size() * 2}x}",
                actor_bits=f"{int(yb[i]) & mask:0{y.element_size() * 2}x}",
            )
        records.append(record)
        print(json.dumps(record), flush=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        return bool(bad.any())

    for layer in range(4):
        for label, rname, aname, n in (
            ("mHC residual", "rlk.residual", "rlk.residual", 0),
            ("mHC post_mix", "rlk.post_mix", "rlk.post_mix", 0),
            ("attention norm", "rlk.normalized", "rlk.normalized", 0),
            (
                "main Q",
                "layer.X.attn.method.forward_mqa.input",
                "layer.X.attn.method.get_query_key_value_tensors",
                0,
            ),
            (
                "wo_a input",
                "layer.X.attn.wo_a.original_fp8.input",
                "layer.X.attn.group_output_projection.input",
                0,
            ),
            (
                "wo_a output",
                "layer.X.attn.wo_a.original_fp8.output",
                "layer.X.attn.group_output_projection.output",
                0,
            ),
            ("attention output", "layer.X.attn.output", "layer.X.attn.output", 0),
            ("FFN norm", "rlk.normalized", "rlk.normalized", 1),
            (
                "shared expert",
                "layer.X.ffn.shared_experts.output",
                "layer.X.ffn.shared_experts.output",
                0,
            ),
            ("FFN output", "layer.X.ffn.output", "layer.X.ffn.output", 0),
        ):
            x = load(rm, rollout, rname.replace("X", str(layer)), layer, n)
            y = load(am, actor, aname.replace("X", str(layer)), layer, n)
            if compare(label, layer, x, y):
                return
    for label, name in (("final mHC head", "rlk.head"), ("final norm", "rlk.normalized")):
        if compare(
            label,
            None,
            load(rm, rollout, name, occurrence=-1),
            load(am, actor, name, occurrence=-1),
        ):
            return


if __name__ == "__main__":
    main()
