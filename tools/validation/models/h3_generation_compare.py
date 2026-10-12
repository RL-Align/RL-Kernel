# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare fixed full-generation runs, preserving failed quality gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from tools.validation.models.h3_full_model import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provider-repeat", action="store_true")
    a = p.parse_args()
    contract = json.loads(a.contract.read_text())
    rows = []
    if a.output.exists():
        raise ValueError("never overwrite a prior comparison")
    for seed in contract["seeds"]:
        left, right = a.reference / str(seed), a.candidate / str(seed)
        reports = [json.loads((root / "report.json").read_text()) for root in (left, right)]
        if any(
            r["preflight"] or r["seed"] != seed or r["contract_sha256"] != digest(a.contract)
            for r in reports
        ):
            raise ValueError("wrong generation evidence/contract")
        if reports[0]["candidate"] or reports[1]["candidate"] == a.provider_repeat:
            raise ValueError("reference/candidate path mismatch")
        initial = [
            torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
            for root in (left, right)
        ]
        if initial[0].keys() != initial[1].keys() or not initial[0]:
            raise ValueError("missing packed initial noise evidence")
        if any(not torch.equal(value, initial[1][name]) for name, value in initial[0].items()):
            raise ValueError("different initial noise/conditioning")
        x = np.load(left / "video.npy", mmap_mode="r")
        y = np.load(right / "video.npy", mmap_mode="r")
        if x.shape != y.shape:
            raise ValueError("video geometry differs")
        mse = 0.0
        ssim = 0.0
        temporal = 0.0
        previous = None
        for frame in range(len(x)):
            delta = x[frame].astype("float64") - y[frame]
            mse += float(np.mean(delta**2))
            ssim += float(
                structural_similarity(x[frame], y[frame], channel_axis=-1, data_range=1.0)
            )
            if previous is not None:
                temporal += float(np.mean((delta - previous) ** 2))
            previous = delta
        audio = np.load(left / "audio.npy")
        other = np.load(right / "audio.npy")
        if audio.shape != other.shape:
            raise ValueError("audio dimensions differ")
        audio_error = float(
            np.linalg.norm((audio - other).astype("float64"))
            / max(np.linalg.norm(audio.astype("float64")), 1e-12)
        )
        row = dict(
            seed=seed,
            psnr_db=float(-10 * np.log10(max(mse / len(x), 1e-30))),
            ssim=ssim / len(x),
            temporal_rmse=float(np.sqrt(temporal / (len(x) - 1))),
            audio_relative_l2=audio_error,
            audio_max_abs=float(np.max(np.abs(audio - other))),
        )
        predictions = []
        if reports[0]["steps"] != reports[1]["steps"]:
            raise ValueError("different scheduler step counts")
        for step in range(reports[0]["steps"]):
            v, w = [
                torch.load(root / f"step-{step:03}.pt", map_location="cpu", weights_only=True)
                for root in (left, right)
            ]
            predictions.append(
                [
                    dict(
                        max_abs=(b.float() - c.float()).abs().max().item(),
                        relative_l2=(
                            (b.float() - c.float()).double().norm()
                            / b.double().norm().clamp_min(1e-12)
                        ).item(),
                    )
                    for b, c in zip(v, w)
                ]
            )
        bound = contract["quality_bounds"]
        row["passed"] = (
            row["psnr_db"] >= bound["minimum_psnr_db"]
            and row["ssim"] >= bound["minimum_ssim"]
            and row["temporal_rmse"] <= bound["maximum_temporal_rmse"]
            and audio_error <= bound["maximum_audio_relative_l2"]
        )
        row["step_errors"] = predictions
        rows.append(row)
    passed = all(r["passed"] for r in rows)
    a.output.write_text(
        json.dumps(
            dict(
                status="passed" if passed else "failed",
                contract_sha256=digest(a.contract),
                seeds=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
