# SPDX-License-Identifier: Apache-2.0
"""INK-70: validate actual paired image-text run evidence, not launch a model."""

import argparse
import json
import re
from pathlib import Path

IDENTITY = ("platform", "hardware", "gpu_count", "checkpoint_sha256", "dataset_sha256",
            "processor_sha256", "recipe_sha256", "miles_revision", "sglang_revision",
            "steps", "precision", "vision_policy")


def compare_multimodal(native, strict):
    if any(native.get(k) is None or native[k] != strict.get(k) for k in IDENTITY):
        raise ValueError("incomparable multimodal runs")
    if native["platform"] not in ("cuda", "rocm") or native["precision"] != "bf16-dequant":
        raise ValueError("expected CUDA/ROCm BF16")
    if native["vision_policy"] != "frozen":
        raise ValueError("trainable vision requires separate qualification")
    for key in IDENTITY:
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", native[key]):
            raise ValueError("invalid evidence digest")
    steps = native["steps"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("invalid step count")
    metrics = {}
    for mode, report in (("native", native), ("strict", strict)):
        if report.get("mode") != mode or report.get("capture_mode") != "real-weight":
            raise ValueError("missing real-weight run evidence")
        rows = report["step_metrics"]
        if len(rows) != steps:
            raise ValueError("incomplete multimodal steps")
        counts = dict(images=0, image_tokens=0, selected_tokens=0,
                      feature_row_mismatches=0, bit_mismatches=0)
        for i, row in enumerate(rows, 1):
            if row["step"] != i:
                raise ValueError("wrong step order")
            if not re.fullmatch(r"[0-9a-f]{64}", row["image_input_sha256"]):
                raise ValueError("missing image input provenance")
            for key in counts:
                if type(row[key]) is not int or row[key] < 0:
                    raise ValueError("invalid count")
                counts[key] += row[key]
            if min(row["images"], row["image_tokens"], row["selected_tokens"]) <= 0:
                raise ValueError("text-only or empty sample is not multimodal evidence")
            if row["bit_mismatches"] > row["selected_tokens"] or (
                row["feature_row_mismatches"] > row["image_tokens"]
            ):
                raise ValueError("invalid mismatch counts")
            if mode == "strict" and (
                row.get("logical_visits") != 48 or row.get("fallback_free") is not True
            ):
                raise ValueError("strict runtime evidence incomplete")
        metrics[mode] = counts
    b = metrics["strict"]
    return {"metrics": metrics, "strict_zero_mismatch": not (
        b["feature_row_mismatches"] or b["bit_mismatches"]
    ), "platform": native["platform"], "independently_certified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    p.add_argument("strict", type=Path)
    args = p.parse_args()
    result = compare_multimodal(json.loads(args.native.read_text()),
                                json.loads(args.strict.read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["strict_zero_mismatch"] else 1)


if __name__ == "__main__":
    main()
