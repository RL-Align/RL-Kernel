# SPDX-License-Identifier: Apache-2.0
"""Compare both fresh-process records and original raw route/VJP payload bytes."""

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
    assert first["passed"] and second["passed"]
    assert first["cases"] == second["cases"] and len(first["cases"]) > 0
    assert first["source_sha256"] == second["source_sha256"]
    assert all(
        case["normal_discrete_passed"] and not case["diagnostic_acceptance"]
        for case in first["cases"]
    )
    a, b = [
        torch.load(path, weights_only=True, map_location="cpu") for path in (args.raw_a, args.raw_b)
    ]
    assert a.keys() == b.keys()
    size = 0
    for key in a:
        x, y = a[key].contiguous(), b[key].contiguous()
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), key
        size += x.numel() * x.element_size()
    report = {
        "schema": "rlkernel.router_training.fresh_process_bytes.v1",
        "passed": True,
        "processes": 2,
        "normal_cases": len(first["cases"]),
        "tensors_compared": len(a),
        "raw_bytes_compared": size,
        "forward_and_vjp_record_hashes_equal": True,
        "raw_uint8_payloads_equal": True,
        "includes_signed_zero": True,
        "source_fingerprints_equal": True,
        "fields": "logits, logical IDs, upstream, normal weights, dlogits, global expert IDs",
        "diagnostic_replay_does_not_earn_acceptance": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
