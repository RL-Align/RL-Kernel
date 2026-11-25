# SPDX-License-Identifier: Apache-2.0
"""Independent replay of saved original FP32 response scores, without torch.

This audits numerical evidence only. Worker placement, real execution, optimizer
continuity, and synchronization require their own receipts; this file cannot
establish those facts from equal score files.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def compare_fp32_bytes(left, right, expected_tokens):
    """Return exact bit evidence, including signed zero and the first difference."""
    if type(expected_tokens) is not int or expected_tokens <= 0:
        raise ValueError("expected_tokens must be a positive integer")
    if len(left) != len(right) or len(left) != 4 * expected_tokens:
        raise ValueError("FP32 bytes do not cover exactly all expected response tokens")
    mismatches = 0
    total = maximum = 0.0
    first = None
    for index, ((a,), (b,)) in enumerate(
        zip(struct.iter_unpack("<f", left), struct.iter_unpack("<f", right), strict=True)
    ):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"nonfinite response logprob at logical token {index}")
        ab = left[index * 4 : index * 4 + 4]
        bb = right[index * 4 : index * 4 + 4]
        delta = abs(a - b)
        total += delta
        maximum = max(maximum, delta)
        if ab != bb:
            mismatches += 1
            if first is None:
                first = dict(
                    token_index=index,
                    left=a,
                    right=b,
                    left_bits=f"0x{int.from_bytes(ab, 'little'):08x}",
                    right_bits=f"0x{int.from_bytes(bb, 'little'):08x}",
                )
    return dict(
        status="FAIL" if mismatches else "PASS",
        expected_tokens=expected_tokens,
        compared_tokens=expected_tokens,
        mismatch_count=mismatches,
        mean_abs_diff=total / expected_tokens,
        max_abs_diff=maximum,
        finite=True,
        first_difference=first,
    )


def _read_manifest(path, suite_sha256, snapshot_sha256, required):
    manifest = json.loads(path.read_text())
    if manifest["input_suite_sha256"] != suite_sha256:
        raise ValueError("logical input suite differs")
    if manifest["snapshot_sha256"] != snapshot_sha256:
        raise ValueError("fixed-weight comparison requires the same snapshot")
    if manifest["dtype"] != "float32" or manifest["encoding"] != "little-endian-ieee754":
        raise ValueError("original FP32 storage is required")
    if manifest["temperature"] != 1 or manifest["top_k"] != -1 or manifest["top_p"] != 1:
        raise ValueError("fixed replay requires untruncated temperature-one raw scores")
    rows = {r["id"]: r for r in manifest["rows"]}
    if len(rows) != len(manifest["rows"]) or set(rows) != set(required):
        raise ValueError(
            "missing, duplicate, or unexpected logical sample; ranks must not duplicate tokens"
        )
    if not manifest.get("producer", {}).get("engine"):
        raise ValueError("score producer identity is required")
    return manifest, rows


def verify_saved_pair(suite_path, left_path, right_path, snapshot_sha256):
    suite_path, left_path, right_path = map(Path, (suite_path, left_path, right_path))
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    required = {r["id"]: r for r in suite["samples"]}
    if len(required) != len(suite["samples"]) or not required:
        raise ValueError("logical suite must contain unique nonempty sample observations")
    suite_hash = hashlib.sha256(suite_bytes).hexdigest()
    _, left_rows = _read_manifest(left_path, suite_hash, snapshot_sha256, required)
    _, right_rows = _read_manifest(right_path, suite_hash, snapshot_sha256, required)
    checked = []
    for identity, case in required.items():
        count = len(case["response_ids"])
        if not count or len(case["position_ids"]) != len(case["prompt_ids"]) + count:
            raise ValueError("invalid logical token/position coverage")
        if case["mask"] != "full-causal-unpadded-all-response-tokens":
            raise ValueError("the fixed suite mask must cover every response token")
        blobs, paths = [], []
        for root, rows in ((left_path.parent, left_rows), (right_path.parent, right_rows)):
            row = rows[identity]
            if row["response_ids"] != case["response_ids"] or row["expected_tokens"] != count:
                raise ValueError("saved scores changed response IDs or token coverage")
            path = (root / row["file"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("score path escapes its evidence directory")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("raw score file hash mismatch")
            blobs.append(raw)
            paths.append(path)
        if paths[0].samefile(paths[1]):
            raise ValueError("independent actor/rollout evidence cannot reuse the same score file")
        checked.append(dict(id=identity, **compare_fp32_bytes(*blobs, count)))
    expected = sum(row["expected_tokens"] for row in checked)
    mismatch = sum(row["mismatch_count"] for row in checked)
    return dict(
        scope="saved raw FP32 numerical evidence only; not execution or GRPO acceptance",
        status="FAIL" if mismatch else "PASS",
        snapshot_sha256=snapshot_sha256,
        input_suite_sha256=suite_hash,
        expected_tokens=expected,
        compared_tokens=sum(row["compared_tokens"] for row in checked),
        mismatch_count=mismatch,
        mean_abs_diff=sum(row["mean_abs_diff"] * row["expected_tokens"] for row in checked)
        / expected,
        max_abs_diff=max(row["max_abs_diff"] for row in checked),
        rows=checked,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--snapshot-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    try:
        result = verify_saved_pair(args.suite, args.left, args.right, args.snapshot_sha256)
    except Exception as e:
        result = dict(status="FAIL", error=f"{type(e).__name__}: {e}")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
