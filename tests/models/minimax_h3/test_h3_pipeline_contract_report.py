# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The pipeline-contract evidence tool on a host where Diffusers and SGLang are missing."""

from __future__ import annotations

import importlib.util
import sys
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
        "environment": {
            "gpus": [],
            "commit": "0" * 40,
            "provider_source_sha256": _tool().PROVIDER_SOURCE_SHA256,
        },
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


def test_an_empty_comparison_is_not_a_pass():
    tool = _tool()
    # A reference that imported but ran no case compared nothing.
    empty = {**SKIPPED, "diffusers_static": {"cases": 0, "bitwise_equal": 0}}
    assert tool.verdict(_report(empty)) is False
    # Next to a real comparison it still means the harness broke.
    mixed = {**empty, "sglang_prefix": {"cases": 3, "bitwise_equal": 3}}
    assert tool.verdict(_report(mixed)) is False
    # No repeatability case is no H-C7 evidence.
    report = _report({**SKIPPED, "diffusers_static": {"cases": 3, "bitwise_equal": 3}})
    assert tool.verdict(report) is True
    report["repeatability"] = {}
    assert tool.verdict(report) is False


def test_a_diffusers_comparison_counts_only_against_the_pinned_source(tmp_path, monkeypatch):
    tool = _tool()
    compared = {**SKIPPED, "diffusers_static": {"cases": 3, "bitwise_equal": 3}}
    report = _report(compared)
    assert tool.verdict(report) is True
    for other in (None, "0" * 64):  # no Diffusers source found, or another one
        report["environment"]["provider_source_sha256"] = other
        assert tool.verdict(report) is False
    # SGLang is prior art without a pin: its comparison alone does not need one.
    sglang_only = _report({**SKIPPED, "sglang_prefix": {"cases": 3, "bitwise_equal": 3}})
    sglang_only["environment"]["provider_source_sha256"] = None
    assert tool.verdict(sglang_only) is True

    # A Diffusers whose files differ from the pinned ones (a stand-in package here) does
    # not hash to the pin.
    package = tmp_path / "diffusers"
    (package / "modular_pipelines" / "minimax_h3").mkdir(parents=True)
    for name in ("__init__.py", "modular_pipelines/__init__.py", "image_processor.py"):
        (package / name).write_text("")
    (package / "modular_pipelines" / "minimax_h3" / "__init__.py").write_text("")

    def ours(name):
        return name == "diffusers" or name.startswith("diffusers.")

    # monkeypatch would restore the real modules but keep the stand-in ones imported below, so
    # every diffusers entry is saved and put back by hand, whether Diffusers was imported or not.
    saved = {m: sys.modules.pop(m) for m in [m for m in sys.modules if ours(m)]}
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        source = tool.provider_source_sha256()
        assert source is not None and source != tool.PROVIDER_SOURCE_SHA256
    finally:
        for name in [m for m in sys.modules if ours(m)]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_rows_offer_only_the_row_modules_present(monkeypatch, capsys):
    """Each row's module ships with its own PR: a row whose module is absent is refused by
    ``--row`` (argparse, exit 2) instead of failing on import."""

    tool = _tool()
    present = {r for r, m in tool.ROW_MODULES.items() if (TOOL.parent / f"{m}.py").is_file()}
    assert set(tool.ROWS) == present and present
    for row in sorted(set(tool.ROW_MODULES) - present):
        monkeypatch.setattr(sys, "argv", ["tool", "--row", row, "--out", "unused.json"])
        with pytest.raises(SystemExit) as exit_info:
            tool.main()
        assert exit_info.value.code == 2
        assert f"invalid choice: '{row}'" in capsys.readouterr().err
