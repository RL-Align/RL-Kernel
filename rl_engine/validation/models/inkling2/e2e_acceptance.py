# SPDX-License-Identifier: Apache-2.0
"""Validate paired recorded GRPO evidence; never infer qualification from a plan."""

import math
import re

DIGEST = re.compile(r"[0-9a-f]{64}")
IDENTITY = ("platform", "hardware", "gpu_count", "checkpoint_sha256", "dataset_sha256",
            "recipe_sha256", "miles_revision", "sglang_revision", "precision",
            "steps", "ep", "rollout_tp")


def validate_pair(native: dict, strict: dict, *, platform: str, steps: int) -> dict:
    if platform not in ("cuda", "rocm") or steps not in (1, 10, 100, 200):
        raise ValueError("unsupported platform or stage")
    if any(native.get(k) is None or native[k] != strict.get(k) for k in IDENTITY):
        raise ValueError("incompatible paired run identity")
    if native["platform"] != platform or native["steps"] != steps:
        raise ValueError("wrong requested stage")
    if native["precision"] != "bf16-dequant" or native["rollout_tp"] != 1:
        raise ValueError("expected BF16-dequant TP1 rollout")
    if native["gpu_count"] not in (2, 4, 8) or native["ep"] != native["gpu_count"]:
        raise ValueError("unexpected initial EP topology")
    if platform == "cuda" and (native["hardware"] != "B200" or native["gpu_count"] != 8):
        raise ValueError("CUDA reference requires eight B200 GPUs")
    if platform == "rocm" and native["hardware"] not in ("MI350X", "MI355X"):
        raise ValueError("initial ROCm reference requires MI350X/MI355X")
    for key in ("checkpoint_sha256", "dataset_sha256", "recipe_sha256"):
        if not DIGEST.fullmatch(str(native[key])):
            raise ValueError("invalid run identity digest")
    counts = {}
    for mode, report in (("native", native), ("strict", strict)):
        if report.get("mode") != mode or report.get("capture_mode") != "real-weight":
            raise ValueError("missing real-weight mode evidence")
        rows = report["step_metrics"]
        if len(rows) != steps:
            raise ValueError("incomplete step evidence")
        count = mismatches = 0
        for step, row in enumerate(rows, start=1):
            if row["step"] != step or not DIGEST.fullmatch(str(row["replay_sha256"])):
                raise ValueError("invalid step or replay identity")
            n, m = row["selected_tokens"], row["bit_mismatches"]
            if type(n) is not int or type(m) is not int or not 0 <= m <= n or n == 0:
                raise ValueError("invalid mismatch counts")
            if any(not math.isfinite(row[k]) for k in ("loss", "reward", "kl")):
                raise ValueError("nonfinite training metric")
            if mode == "strict" and (
                row.get("fallback_free") is not True or row.get("logical_visits") != 48
            ):
                raise ValueError("missing strict visit/fallback evidence")
            count += n
            mismatches += m
        counts[mode] = {"selected_tokens": count, "bit_mismatches": mismatches}
    return {"platform": platform, "steps": steps, "counts": counts,
            "strict_zero_mismatch": counts["strict"]["bit_mismatches"] == 0,
            "acceptance_from_reports_only": True, "independently_certified": False}
