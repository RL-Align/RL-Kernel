# SPDX-License-Identifier: Apache-2.0
"""INK-72: compare matched measured run metrics without fabricating timings."""

import argparse
import json
import math
from pathlib import Path

MATCH = ("platform", "hardware", "gpu_count", "checkpoint_sha256", "recipe_sha256",
         "dataset_sha256", "miles_revision", "sglang_revision", "steps", "precision",
         "ep", "rollout_tp", "warmup_steps")


def compare_performance(native, strict):
    if any(native.get(k) is None or native[k] != strict.get(k) for k in MATCH):
        raise ValueError("unmatched hardware, recipe or measurement scope")
    if native["platform"] not in ("cuda", "rocm") or native["precision"] != "bf16-dequant":
        raise ValueError("invalid GPU BF16 comparison")
    results = {}
    for mode, report in (("native", native), ("strict", strict)):
        if report.get("mode") != mode or report.get("capture_mode") != "real-weight":
            raise ValueError("missing recorded run evidence")
        samples = report["samples"]
        if len(samples) != report["steps"] - report["warmup_steps"] or not samples:
            raise ValueError("incomplete measured steps")
        count = 0
        elapsed = 0.0
        for row in samples:
            if type(row["generated_tokens"]) is not int or row["generated_tokens"] <= 0:
                raise ValueError("invalid generated token count")
            for key in ("step_seconds", "prefill_ms", "decode_ms", "peak_memory_bytes"):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f"invalid measurement {key}")
            count += row["generated_tokens"]
            elapsed += row["step_seconds"]
        results[mode] = {
            "generated_tokens": count,
            "tokens_per_second": count / elapsed,
            "steps_per_second": len(samples) / elapsed,
            "mean_step_seconds": elapsed / len(samples),
            "mean_prefill_ms": sum(s["prefill_ms"] for s in samples) / len(samples),
            "mean_decode_ms": sum(s["decode_ms"] for s in samples) / len(samples),
            "peak_memory_bytes": max(s["peak_memory_bytes"] for s in samples),
        }
    a, b = results["native"], results["strict"]
    ratios = {k: b[k] / a[k] for k in ("tokens_per_second", "steps_per_second",
                                      "mean_step_seconds", "mean_prefill_ms",
                                      "mean_decode_ms", "peak_memory_bytes")}
    return {"platform": native["platform"], "metrics": results, "strict_over_native": ratios,
            "throughput_target_met": ratios["tokens_per_second"] >= .9,
            "bitwise_validated": False, "independently_certified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    p.add_argument("strict", type=Path)
    args = p.parse_args()
    print(json.dumps(compare_performance(json.loads(args.native.read_text()),
                                         json.loads(args.strict.read_text())), indent=2))


if __name__ == "__main__":
    main()
