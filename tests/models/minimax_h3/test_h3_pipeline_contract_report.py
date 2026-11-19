# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The pipeline-contract evidence tool on a host where Diffusers and SGLang are missing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOL = (
    Path(__file__).resolve().parents[3] / "tools/validation/models/h3_pipeline_contract_report.py"
)


def _tool():
    spec = importlib.util.spec_from_file_location("h3_pipeline_contract_report", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(reference: dict) -> dict:
    """What ``main`` records when ``reference()`` / ``timing()`` find no provider."""

    return {
        "row": "fl2va_pipeline_contract",
        "environment": {"gpus": [], "commit": "0" * 40},
        "repeatability": {"case": {"runs": 5, "identical": True, "sequence_length": 10}},
        "reference": reference,
        "timing": {"rl_kernel": {"median_ms": 1.0, "min_ms": 0.9, "runs": 15, "label": "toy"}},
        "findings": {},
    }


SKIPPED = {
    "diffusers_static": {"skipped": "ModuleNotFoundError: No module named 'diffusers'"},
    "sglang_prefix": {"skipped": "ModuleNotFoundError: No module named 'sglang'"},
}


def test_nothing_compared_is_not_a_pass():
    tool = _tool()
    assert tool.verdict(_report(SKIPPED)) is False
    compared = {**SKIPPED, "diffusers_static": {"cases": 3, "bitwise_equal": 3}}
    assert tool.verdict(_report(compared)) is True
    compared["diffusers_static"]["bitwise_equal"] = 2
    assert tool.verdict(_report(compared)) is False


def test_charts_without_providers_draw_what_ran(tmp_path):
    pytest.importorskip("matplotlib")
    tool = _tool()
    # Neither provider: no equality chart, a timing chart of ours alone.
    written = tool.charts(_report(SKIPPED), str(tmp_path / "none"))
    assert written == ["fl2va-timing.png", "fl2va-repeatability.png"]
    # A compared reference whose timing is missing (the provider failed in timing()).
    compared = {**SKIPPED, "diffusers_static": {"cases": 3, "bitwise_equal": 3}}
    written = tool.charts(_report(compared), str(tmp_path / "one"))
    assert written == ["fl2va-equality.png", "fl2va-timing.png", "fl2va-repeatability.png"]
