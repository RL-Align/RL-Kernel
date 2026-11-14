# SPDX-License-Identifier: Apache-2.0
"""Byte-compare two actual actor traces and report the first differing boundary."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def leaves(value, path=""):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from leaves(v, f"{path}/{i}")
    elif isinstance(value, dict):
        for key, v in value.items():
            yield from leaves(v, f"{path}/{key}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    lm = json.loads((args.left / "manifest.json").read_text())
    rm = json.loads((args.right / "manifest.json").read_text())
    right = defaultdict(list)
    used = defaultdict(int)
    for e in rm:
        right[(e["layer"], e["name"])].append(e)
    records = []
    for event in lm:
        key = (event["layer"], event["name"])
        occurrences = right.get(key, [])
        index = used[key]
        used[key] += 1
        if index >= len(occurrences):
            continue
        other = occurrences[index]
        x = torch.load(args.left / event["file"], weights_only=True, map_location="cpu")
        y = torch.load(args.right / other["file"], weights_only=True, map_location="cpu")
        xl, yl = dict(leaves(x)), dict(leaves(y))
        for path in xl.keys() & yl.keys():
            a, b = xl[path], yl[path]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError((key, path, a.dtype, b.dtype, a.shape, b.shape))
            aa = a.contiguous().reshape(-1).view(torch.uint8).reshape(a.numel(), a.element_size())
            bb = b.contiguous().reshape(-1).view(torch.uint8).reshape(b.numel(), b.element_size())
            bad = (aa != bb).any(-1)
            if bad.any():
                i = int(bad.nonzero()[0])
                record = dict(
                    layer=key[0],
                    stage=key[1],
                    leaf=path,
                    dtype=str(a.dtype),
                    elements=a.numel(),
                    mismatch_count=int(bad.sum()),
                    first_index=i,
                    left=a.reshape(-1)[i].item(),
                    right=b.reshape(-1)[i].item(),
                    left_bits=bytes(aa[i].tolist())[::-1].hex(),
                    right_bits=bytes(bb[i].tolist())[::-1].hex(),
                )
                records.append(record)
                if len(records) == 1:
                    print("FIRST_ACTOR_SELF_REPEAT_DIFFERENCE", json.dumps(record), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "actor self repeat, not actor-rollout parity",
                "first_difference": records[0] if records else None,
                "differences": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
