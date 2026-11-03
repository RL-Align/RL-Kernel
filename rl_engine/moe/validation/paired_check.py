# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Torch paired check for formal goldens.

The raw Torch reference must run a paired check on every
formal golden and record diagnostics; its differences must never override
a strict verdict, and missing evidence maps to MISSING_PROVENANCE.

Rules implemented:
- Every formal golden (a fixture manifest entry) must carry
  a paired-check record: the raw Torch reference executed on the same case
  inputs, with diagnostics (max/mean abs diff vs the golden's oracle).
- The paired diff is DIAGNOSTIC ONLY: it can never flip a strict verdict
  (byte-exact gates). It exists to catch "golden itself drifted from Torch"
  early and to keep evidence for audits.
- Shape/dtype mismatches and non-finite diagnostics are invalid evidence,
  because they do not describe a meaningful elementwise comparison.
- A golden without a paired record is not acceptable evidence:
  MISSING_PROVENANCE (67).
- Until the start kit is published (anchor_pending), the manifest lookup
  returns empty and every check reports golden-missing — tests then skip.
  Nothing here guesses or fabricates a manifest path.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import torch

from rl_engine.moe.t01_verdicts import RouterVerdict
from rl_engine.moe.validation.runner import ValidationReport, make_fail, make_pass

#: Repository-level fixture manifest location, published with the start kit.
#: Its absence is the documented anchor_pending state:
#: we look it up, we never guess.
GOLDEN_MANIFEST_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "fixtures" / "router" / "manifest.json"
)


@dataclass(frozen=True)
class GoldenEntry:
    """One formal golden from the fixture manifest.

    ``name`` links the entry to a paired record. ``fixture_path`` and
    ``checksum`` identify the fixture. ``kind`` describes the fixture family,
    while ``router_mode`` distinguishes hash and learned routing.
    """

    name: str  # e.g. "learned_dropless_basic"
    fixture_path: str
    checksum: str
    kind: str  # random | near_tie | exact_tie | padding
    router_mode: str  # hash | learned


@dataclass(frozen=True)
class PairedRecord:
    """Diagnostic evidence that raw Torch ran on this golden's case.

    The record proves execution and stores diagnostics; it does not prove
    strict correctness. L3a/L3b own the strict comparison verdict.
    """

    golden_name: str
    torch_max_abs_diff: float
    torch_mean_abs_diff: float
    n_tensors_compared: int
    diagnostic_note: str = ""


def _paired_record_problem(expected_name: str, record: PairedRecord) -> str | None:
    """Return why paired evidence is invalid, or ``None`` if acceptable."""
    if record.golden_name != expected_name:
        return f"record belongs to {record.golden_name!r}, expected {expected_name!r}"
    if record.n_tensors_compared <= 0:
        return "record compared zero tensors"
    if not math.isfinite(record.torch_max_abs_diff) or not math.isfinite(
        record.torch_mean_abs_diff
    ):
        return "record contains non-finite diagnostics"
    if record.torch_max_abs_diff < 0.0 or record.torch_mean_abs_diff < 0.0:
        return "record contains negative absolute-difference diagnostics"
    if record.torch_mean_abs_diff > record.torch_max_abs_diff:
        return "record mean absolute difference exceeds its maximum"
    return None


def load_golden_manifest(path: pathlib.Path | None = None) -> list[GoldenEntry]:
    """Read the fixture manifest; [] means anchor_pending (no guessing).

    Callers may supply a test path. The default path is fixed by contract;
    missing or malformed data never triggers a fallback-path search.
    """
    manifest = path if path is not None else GOLDEN_MANIFEST_PATH
    if not manifest.is_file():
        return []
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = raw.get("goldens", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []
    out: list[GoldenEntry] = []
    for item in entries or []:
        try:
            out.append(
                GoldenEntry(
                    name=str(item["name"]),
                    fixture_path=str(item["fixture_path"]),
                    checksum=str(item["checksum"]),
                    kind=str(item.get("kind", "random")),
                    router_mode=str(item.get("router_mode", "learned")),
                )
            )
        except (KeyError, TypeError):
            continue  # malformed entry: skip, do not fabricate
    return out


def run_paired_check(
    case_id: str,
    golden: GoldenEntry,
    golden_tensors: dict[str, torch.Tensor],
    torch_reference: Callable[[dict[str, Any]], dict[str, torch.Tensor]] | None,
    case_inputs: dict[str, Any],
    *,
    existing_record: PairedRecord | None = None,
) -> ValidationReport:
    """Run raw Torch on the golden's case and record diagnostics.

    The diff NEVER flips strict verdicts; missing execution evidence is
    MISSING_PROVENANCE. If the fixture ships a paired record,
    ``existing_record`` short-circuits the local Torch run (still evidence).
    """
    if existing_record is None:
        if not callable(torch_reference):
            return make_fail(
                "paired",
                case_id,
                RouterVerdict.MISSING_PROVENANCE,
                "no Torch reference executed on this golden",
            )
        try:
            ref = torch_reference(case_inputs)
        except Exception as exc:  # noqa: BLE001 — evidence failure, not a crash mask
            return make_fail(
                "paired",
                case_id,
                RouterVerdict.MISSING_PROVENANCE,
                f"Torch reference failed to run: {type(exc).__name__}: {exc}",
            )
        if not isinstance(ref, dict):
            return make_fail(
                "paired",
                case_id,
                RouterVerdict.MISSING_PROVENANCE,
                f"Torch reference returned {type(ref).__name__}, expected a tensor mapping",
            )

        # Both aggregates report the worst tensor: the largest elementwise
        # error and the largest per-tensor mean error, respectively.
        max_abs = 0.0
        mean_abs = 0.0
        n = 0
        missing = sorted(set(golden_tensors) - set(ref))
        if missing:
            return make_fail(
                "paired",
                case_id,
                RouterVerdict.MISSING_PROVENANCE,
                f"Torch reference omitted required golden tensors: {missing}",
                missing=missing,
            )
        if not golden_tensors:
            return make_fail(
                "paired",
                case_id,
                RouterVerdict.MISSING_PROVENANCE,
                "golden contains no tensors to compare",
            )
        for key, gold_t in golden_tensors.items():
            r = ref[key]
            if not isinstance(gold_t, torch.Tensor) or not isinstance(r, torch.Tensor):
                return make_fail(
                    "paired",
                    case_id,
                    RouterVerdict.MISSING_PROVENANCE,
                    f"Torch/golden entry {key!r} is not a tensor on both sides",
                    tensor=key,
                )
            if gold_t.numel() == 0:
                return make_fail(
                    "paired",
                    case_id,
                    RouterVerdict.MISSING_PROVENANCE,
                    f"Torch/golden tensor {key!r} has no elements to compare",
                    tensor=key,
                )
            # An incompatible tensor is missing comparison evidence, not an
            # infinite numeric difference.
            if r.shape != gold_t.shape or r.dtype != gold_t.dtype:
                return make_fail(
                    "paired",
                    case_id,
                    RouterVerdict.MISSING_PROVENANCE,
                    f"Torch/golden tensor {key!r} is not comparable: "
                    f"Torch shape={tuple(r.shape)} dtype={r.dtype}; "
                    f"golden shape={tuple(gold_t.shape)} dtype={gold_t.dtype}",
                    tensor=key,
                    torch_shape=list(r.shape),
                    golden_shape=list(gold_t.shape),
                    torch_dtype=str(r.dtype),
                    golden_dtype=str(gold_t.dtype),
                )
            d = float((r.double() - gold_t.double()).abs().max().item())
            mean_d = float((r.double() - gold_t.double()).abs().mean().item())
            max_abs = max(max_abs, d)
            mean_abs = max(mean_abs, mean_d)
            n += 1
        record = PairedRecord(
            golden_name=golden.name,
            torch_max_abs_diff=max_abs,
            torch_mean_abs_diff=mean_abs,
            n_tensors_compared=n,
        )
    else:
        record = existing_record

    # Pre-shipped records pass through the same validation as fresh records.
    problem = _paired_record_problem(golden.name, record)
    if problem is not None:
        return make_fail(
            "paired",
            case_id,
            RouterVerdict.MISSING_PROVENANCE,
            f"paired {problem}",
        )

    diagnostic_exact = record.torch_max_abs_diff == 0.0 and record.torch_mean_abs_diff == 0.0
    diagnostic_warning = (
        None
        if diagnostic_exact
        else ("Torch/golden diagnostics differ; strict L3a/L3b verdict unchanged")
    )
    return make_pass(
        "paired",
        case_id,
        f"paired evidence complete; golden={golden.name} "
        f"torch_max_abs={record.torch_max_abs_diff:.3e} over "
        f"{record.n_tensors_compared} tensors; strict verdict unchanged",
        paired_record=dataclasses.asdict(record),
        diagnostic_exact=diagnostic_exact,
        diagnostic_warning=diagnostic_warning,
    )


def paired_gate_for_goldens(
    case_id: str,
    goldens: list[GoldenEntry],
    records: dict[str, PairedRecord],
) -> ValidationReport:
    """Gate: every formal golden must have paired evidence.

    This audit does not rerun Torch. It requires a matching record with at
    least one compared tensor and finite diagnostics for every manifest entry.
    """
    if not goldens:
        return make_fail(
            "paired-gate",
            case_id,
            RouterVerdict.MISSING_PROVENANCE,
            "formal golden manifest is absent or empty",
        )
    missing = [
        g.name
        for g in goldens
        if g.name not in records or _paired_record_problem(g.name, records[g.name]) is not None
    ]
    if missing:
        return make_fail(
            "paired-gate",
            case_id,
            RouterVerdict.MISSING_PROVENANCE,
            f"goldens without Torch paired evidence: {missing}",
            missing=missing,
        )
    return make_pass("paired-gate", case_id, f"{len(goldens)} goldens carry paired diagnostics")
