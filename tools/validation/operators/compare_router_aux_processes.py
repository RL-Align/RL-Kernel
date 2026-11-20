# SPDX-License-Identifier: Apache-2.0
"""Compare actual fresh-process auxiliary loss/count/mass/gradient payloads."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "raw-a", "raw-b", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    first, second = [json.loads(path.read_text()) for path in (args.first, args.second)]
    assert (
        first["passed"]
        and second["passed"]
        and first["normal_routing"]
        and second["normal_routing"]
    )
    assert not first["fixed_replay"] and not second["fixed_replay"]
    assert first["cases"] == second["cases"] and len(first["cases"]) > 0
    a, b = [
        torch.load(path, map_location="cpu", weights_only=True) for path in (args.raw_a, args.raw_b)
    ]
    assert a.keys() == b.keys()
    size = 0
    for key in a:
        x, y = a[key].contiguous(), b[key].contiguous()
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), key
        size += x.numel() * x.element_size()
    report = {
        "schema": "rlkernel.router_aux_loss.fresh_process_bytes.v1",
        "passed": True,
        "processes": 2,
        "normal_cases": len(first["cases"]),
        "tensors_compared": len(a),
        "raw_payload_bytes_compared": size,
        "actual_uint8_payloads_equal": True,
        "includes_signed_zero": True,
        "fields": [
            "logits",
            "logical_ids",
            "bias",
            "loss",
            "dlogits",
            "experts",
            "counts",
            "masses",
            "probabilities",
        ],
        "fixed_route_replay": False,
        "raw_artifacts": "ignored artifacts/task34/raw-{a,b}.pt",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
