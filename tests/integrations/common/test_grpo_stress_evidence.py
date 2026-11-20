# SPDX-License-Identifier: Apache-2.0
"""Fail-closed regressions using actual full-verl stress evidence, never mock scores."""
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from examples.operators.dsv4_4l.verify_grpo import verify_stress


@pytest.fixture
def recorded(tmp_path):
    source = os.environ.get("RLK_GRPO_STRESS_CAPTURE")
    if not source:
        pytest.skip("requires a real completed full-verl v3 stress evidence directory")
    source = Path(source)
    for name in (
        "stress-v3.json",
        "post-update-v3.json",
        "fixed-sequences.json",
        "sync-v3-raw.json",
        "stress-v3-actor-readback.json",
        "stress-v3-rollout-readback.json",
    ):
        shutil.copy2(source / name, tmp_path / name)
    # Copy only small scored-sequence evidence; no checkpoint/policy tensor archives.
    for group in ("fixed", "fresh"):
        for i, name in enumerate("ABC"):
            file = f"audit-v3-{group}-{i:02d}-{name}-{name}.pt"
            shutil.copy2(source / file, tmp_path / file)
    shutil.copytree(source / "stress-v3", tmp_path / "stress-v3")
    summary = json.loads((tmp_path / "stress-v3.json").read_text())
    pids = {int(k): v for k, v in summary["pids"].items()}
    return tmp_path, summary["repeats"], pids


def test_real_completed_stress_evidence(recorded):
    root, repeats, pids = recorded
    assert verify_stress(root, repeats, pids)["compared_tokens"] > 0


@pytest.mark.parametrize("corruption", ("one_ulp", "both_scores", "nan", "stale_version"))
def test_corrupted_raw_stress_cannot_pass(recorded, corruption):
    root, repeats, pids = recorded
    path = root / "stress-v3/00000.pt"
    raw = torch.load(path, weights_only=True)
    if corruption == "one_ulp":
        raw["rollout"][0].view(torch.int32)[0] ^= 1
    elif corruption == "both_scores":
        # A fabricated zero-difference pair must still disagree with the actual actor baseline.
        raw["rollout"][0].view(torch.int32)[0] ^= 1
        raw["actor"][0].view(torch.int32)[0] ^= 1
    elif corruption == "nan":
        raw["rollout"][0][0] = float("nan")
    else:
        raw["weight_version"] = 2
    torch.save(raw, path)
    with pytest.raises((AssertionError, RuntimeError)):
        verify_stress(root, repeats, pids)


@pytest.mark.parametrize("corruption", ("missing_iteration", "duplicate_iteration", "count"))
def test_incomplete_or_incorrect_stress_matrix_cannot_pass(recorded, corruption):
    root, repeats, pids = recorded
    path = root / "stress-v3/comparisons.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if corruption == "missing_iteration":
        rows.pop()
    elif corruption == "duplicate_iteration":
        rows[-1]["iteration"] = 0
    else:
        path = root / "stress-v3.json"
        summary = json.loads(path.read_text())
        summary["expected_tokens"] -= 1
        path.write_text(json.dumps(summary))
    if corruption != "count":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises((AssertionError, RuntimeError)):
        verify_stress(root, repeats, pids)
