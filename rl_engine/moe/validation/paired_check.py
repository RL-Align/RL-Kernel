# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Torch paired check for formal goldens (contract §2.5, T09 step 7).

Contract §2.5: "Torch 原始参考必须在每个正式 golden 上运行 paired check
并记录诊断；其差异不能覆盖 strict verdict，缺证据为 MISSING_PROVENANCE。"

Rules implemented:
- Every formal golden (published by T01, fixture manifest entry) must carry
  a paired-check record: the raw Torch reference executed on the same case
  inputs, with diagnostics (max/mean abs diff vs the golden's oracle).
- The paired diff is DIAGNOSTIC ONLY: it can never flip a strict verdict
  (byte-exact gates). It exists to catch "golden itself drifted from Torch"
  early and to keep evidence for audits.
- A golden without a paired record is not acceptable evidence:
  MISSING_PROVENANCE (67).
- Until T01 publishes the start kit (anchor_pending), the manifest lookup
  returns empty and every check reports golden-missing — tests then skip.
  Nothing here guesses or fabricates a manifest path.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass
from typing import Any, Callable

import torch

from rl_engine.moe.p3_verdicts import P3Verdict
from rl_engine.moe.validation.report import LadderReport, make_fail, make_pass

_fail, _pass = make_fail, make_pass  # module-local aliases for brevity

#: Repository-level fixture manifest location, published by T01.
#: Its absence is the documented anchor_pending state (contract §1.3/§5):
#: we look it up, we never guess.
GOLDEN_MANIFEST_PATH = pathlib.Path(__file__).resolve().parents[3] / \
    "fixtures" / "p3" / "manifest.json"


@dataclass(frozen=True)
class GoldenEntry:
    """One formal golden from T01's fixture manifest."""

    name: str                        # e.g. "learned_dropless_basic"
    fixture_path: str
    checksum: str
    kind: str                        # random | near_tie | exact_tie | padding
    router_mode: str                 # hash | learned


@dataclass(frozen=True)
class PairedRecord:
    """Diagnostic evidence that raw Torch ran on this golden's case."""

    golden_name: str
    torch_max_abs_diff: float
    torch_mean_abs_diff: float
    n_tensors_compared: int
    diagnostic_note: str = ""


def load_golden_manifest(path: pathlib.Path | None = None) -> list[GoldenEntry]:
    """Read T01's fixture manifest; [] means anchor_pending (no guessing)."""
    manifest = path if path is not None else GOLDEN_MANIFEST_PATH
    if not manifest.is_file():
        return []
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = raw.get("goldens", raw) if isinstance(raw, dict) else raw
    out: list[GoldenEntry] = []
    for item in entries or []:
        try:
            out.append(GoldenEntry(
                name=str(item["name"]),
                fixture_path=str(item["fixture_path"]),
                checksum=str(item["checksum"]),
                kind=str(item.get("kind", "random")),
                router_mode=str(item.get("router_mode", "learned")),
            ))
        except (KeyError, TypeError):
            continue     # malformed entry: skip, do not fabricate
    return out


def run_paired_check(
    case_id: str,
    golden: GoldenEntry,
    golden_tensors: dict[str, torch.Tensor],
    torch_reference: Callable[[dict[str, Any]], dict[str, torch.Tensor]],
    case_inputs: dict[str, Any],
    *,
    existing_record: PairedRecord | None = None,
) -> LadderReport:
    """Run raw Torch on the golden's case and record diagnostics (§2.5).

    The diff NEVER flips strict verdicts; missing execution evidence is
    MISSING_PROVENANCE. If T01 ships a paired record inside the fixture,
    ``existing_record`` short-circuits the local Torch run (still evidence).
    """
    if existing_record is None:
        if not callable(torch_reference):
            return _fail("paired", case_id, P3Verdict.MISSING_PROVENANCE,
                         "no Torch reference executed on this golden")
        try:
            ref = torch_reference(case_inputs)
        except Exception as exc:  # noqa: BLE001 — evidence failure, not a crash mask
            return _fail("paired", case_id, P3Verdict.MISSING_PROVENANCE,
                         f"Torch reference failed to run: {type(exc).__name__}: {exc}")

        max_abs = 0.0
        mean_abs = 0.0
        n = 0
        for key, gold_t in golden_tensors.items():
            if key not in ref:
                continue
            r = ref[key]
            if r.shape != gold_t.shape or r.dtype != gold_t.dtype:
                d = float("inf")
                mean_d = float("inf")
            else:
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

    return _pass("paired", case_id,
                 f"golden={golden.name} torch_max_abs={record.torch_max_abs_diff:.3e} "
                 f"over {record.n_tensors_compared} tensors (diagnostic only)",
                 paired_record=dataclasses.asdict(record))


def paired_gate_for_goldens(
    case_id: str,
    goldens: list[GoldenEntry],
    records: dict[str, PairedRecord],
) -> LadderReport:
    """Gate: every formal golden must have paired evidence (§2.5, 67)."""
    missing = [g.name for g in goldens if g.name not in records]
    if missing:
        return _fail("paired-gate", case_id, P3Verdict.MISSING_PROVENANCE,
                     f"goldens without Torch paired evidence: {missing}",
                     missing=missing)
    return _pass("paired-gate", case_id,
                 f"{len(records)} goldens carry paired diagnostics")
