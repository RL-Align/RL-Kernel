# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The ``visual_vae_contract`` evidence and chart tools (CPU only, no checkpoint)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.models.minimax_h3 import visual_vae as vv
from tools.validation.models import h3_visual_vae_evidence as ev


def _backend_flags() -> tuple[bool, bool, bool, bool]:
    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    return cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32


@pytest.fixture
def foreign_flags():
    """Ambient flags as another library might leave them: TF32 flipped both ways."""

    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    saved = _backend_flags()
    cudnn.deterministic, cudnn.benchmark = False, True
    cudnn.allow_tf32, matmul.allow_tf32 = False, True
    try:
        yield _backend_flags()
    finally:
        cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32 = saved


def test_pinned_reference_runs_under_the_contract_policy(foreign_flags) -> None:
    p = vv.RELEASE_POLICY
    with ev._flags(True):
        assert _backend_flags() == (
            p.cudnn_deterministic,
            p.cudnn_benchmark,
            p.cudnn_allow_tf32,
            p.matmul_allow_tf32,
        )
    with ev._flags(False):
        # PyTorch's defaults: nondeterministic cuDNN, no autotuning, TF32 for cuDNN only.
        assert _backend_flags() == (False, False, True, False)
    assert _backend_flags() == foreign_flags


def test_cmp_records_a_max_on_shape_mismatch() -> None:
    out = ev._cmp(torch.zeros(1, 2), torch.zeros(1, 3))
    assert out["bitwise_equal"] is False
    assert out["max_abs"] is None and out["shape_a"] == [1, 2] and out["shape_b"] == [1, 3]
    assert ev._cmp(torch.zeros(2), torch.tensor([0.0, 0.5]))["max_abs"] == 0.5
    assert ev._cmp(torch.zeros(2), torch.tensor([0.0, math.nan]))["max_abs"] is None


def test_cmp_compares_bytes_and_dtypes() -> None:
    wide = ev._cmp(torch.tensor([0.5, 1.0]), torch.tensor([0.5, 1.0], dtype=torch.float16))
    assert wide["bitwise_equal"] is False and wide["same_dtype"] is False
    signed_zero = ev._cmp(torch.tensor([0.0, 1.0]), torch.tensor([-0.0, 1.0]))
    assert signed_zero["bitwise_equal"] is False and signed_zero["mismatches"] == 1
    assert signed_zero["max_abs"] == 0.0
    nan = torch.tensor([math.nan, 1.0])
    same_nan = ev._cmp(nan, nan.clone())
    assert same_nan["bitwise_equal"] is True and same_nan["mismatches"] == 0
    assert same_nan["max_abs"] == 0.0
    equal = ev._cmp(torch.arange(6.0).view(2, 3), torch.arange(6.0).view(2, 3))
    assert equal == {
        "bitwise_equal": True,
        "same_dtype": True,
        "max_abs": 0.0,
        "mismatches": 0,
        "numel": 6,
    }


def _strict_loads(text: str):
    def reject(name: str):
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(text, parse_constant=reject)


def test_report_is_strict_json(tmp_path) -> None:
    def mismatch_above_batch_1(x: torch.Tensor) -> torch.Tensor:
        return x if x.shape[0] == 1 else x[..., :1]

    report = {
        "cmp": ev._cmp(torch.zeros(1, 2), torch.zeros(1, 3)),
        "sweep": ev._sweep(mismatch_above_batch_1, torch.zeros(3, 2)),
    }
    out = tmp_path / "evidence.json"
    ev.write_report(out, report)
    loaded = _strict_loads(out.read_text(encoding="utf-8"))
    assert loaded["cmp"]["max_abs"] is None
    assert loaded["sweep"]["max_abs"] is None and loaded["sweep"]["differing_cases"] == 5
    with pytest.raises(ValueError):
        ev.write_report(tmp_path / "bad.json", {"max_abs": math.nan})


# --- charts -------------------------------------------------------------------------------


@pytest.fixture
def plot():
    pytest.importorskip("matplotlib")
    from tools.validation.models import plot_h3_visual_vae

    return plot_h3_visual_vae


OPS = ("encode_keyframe", "encode_video", "decode_video")
BATCH = {"encode_keyframe": 3, "encode_video": 2, "decode_video": 2}


def _sweep(op: str, max_abs: float = 0.0) -> dict:
    cases = [
        {"batch": b, "row": r, "bitwise_equal": max_abs == 0.0, "max_abs": max_abs}
        for b in range(2, BATCH[op] + 1)
        for r in range(b)
    ]
    return {"cases": cases}


def _report(device_name: str, repeats: int = 3, max_abs: float | None = 0.0) -> dict:
    """A report with every field the charts read (no ``contract_cases``, as recorded)."""

    run = {k: {"digests": ["d"] * repeats, "bitwise_equal": True} for k in OPS}
    parity = {k: {"max_abs": max_abs} for k in OPS}
    timing = {k: {"median_ms": 1.0, "repeats": 5} for k in OPS}
    return {
        "rl_kernel_commit": "0" * 40,
        "environment": {"device_name": device_name},
        "inputs": {"video": [2, 3, 22, 64, 96]},
        "checks": {
            "run_to_run": run,
            "diffusers_parity": {"diffusers_pinned": parity, "diffusers_default_flags": parity},
            "batch_invariance": {
                "contract_per_row": {k: True for k in OPS},
                "diffusers_batched_default_flags": {k: _sweep(k, 1e-3) for k in OPS},
                "diffusers_batched_pinned_flags": {k: _sweep(k) for k in OPS},
            },
            "tiling": {
                "tiled_vs_untiled_encode_keyframe": {"max_abs": 0.1},
                "tiled_vs_untiled_decode_video": {"max_abs": 0.1},
            },
            "preprocessing": {
                "div255_cpu_vs_cuda_mismatches": 126,
                "lut_equals_cuda_release": True,
            },
        },
        "timing": {
            "contract": timing,
            "diffusers_default_flags": timing,
            "diffusers_pinned_flags": timing,
            "contract_untiled": timing,
        },
    }


def test_gpu_tag_keeps_the_chip_name(plot) -> None:
    assert plot.gpu_tag(_report("NVIDIA B200")) == "b200"
    assert plot.gpu_tag(_report("NVIDIA H100")) == "h100"
    assert plot.gpu_tag(_report("NVIDIA H100 80GB HBM3")) == "h100-80gb-hbm3"
    # Names sharing their last word get different output files.
    assert plot.gpu_tag(_report("NVIDIA H200 141GB HBM3")) != plot.gpu_tag(
        _report("NVIDIA H100 80GB HBM3")
    )


def test_accuracy_rows_state_the_recorded_repeat_count(plot) -> None:
    labels = [label for label, _ in plot.accuracy_rows(_report("NVIDIA B200", repeats=2))]
    assert "run to run (2 repeats), keyframe encode" in labels
    assert not any("3 repeats" in label for label in labels)


@pytest.mark.parametrize("missing", [None, math.nan])  # strict JSON null; older reports NaN
def test_shape_mismatch_charts_as_differs(plot, missing) -> None:
    rows = plot.accuracy_rows(_report("NVIDIA B200", max_abs=missing))
    assert plot.fmt_delta(rows[3][1]) == plot.DIFFERS


def test_equal_values_with_other_bytes_chart_as_differs(plot) -> None:
    assert plot.max_abs({"max_abs": 0.0, "bitwise_equal": False}) != 0.0
    assert plot.max_abs({"max_abs": 0.0, "bitwise_equal": True}) == 0.0
    assert plot.max_abs({"max_abs": 0.0}) == 0.0  # parity entries without the flag
    assert plot.fmt_delta(plot.max_abs({"max_abs": 0.0, "bitwise_equal": False})) == plot.DIFFERS


def test_contract_batch_cases_are_charted_per_case(plot) -> None:
    report = _report("NVIDIA B200")
    bi = report["checks"]["batch_invariance"]
    # Reports without per-case contract results: the per-step verdict, as recorded.
    assert {row[1] for row in plot.batch_cases(report)} == {0.0}
    bi["contract_per_row"]["encode_keyframe"] = False
    bi["contract_cases"] = {k: _sweep(k) for k in OPS}
    bi["contract_cases"]["encode_keyframe"]["cases"][3]["max_abs"] = 0.5  # batch 3, row 1
    rows = {row[0]: row[1] for row in plot.batch_cases(report)}
    assert rows["keyframe encode, batch 3 row 1"] == 0.5
    assert rows.pop("keyframe encode, batch 3 row 1") and set(rows.values()) == {0.0}


def _write_reports(tmp_path: Path, *device_names: str) -> list[str]:
    paths = []
    for i, name in enumerate(device_names):
        path = tmp_path / f"evidence-{i}.json"
        path.write_text(json.dumps(_report(name)), encoding="utf-8")
        paths.append(str(path))
    return paths


def test_reports_from_the_same_gpu_are_rejected(plot, tmp_path, monkeypatch) -> None:
    reports = _write_reports(tmp_path, "NVIDIA B200", "NVIDIA B200")
    out = tmp_path / "charts"
    monkeypatch.setattr(sys, "argv", ["plot", *reports, "--out-dir", str(out)])
    with pytest.raises(SystemExit):
        plot.main()
    assert not out.exists() or not any(out.iterdir())


def test_tables_are_written_as_utf8_under_any_locale(plot, tmp_path) -> None:
    """The tables hold non-ASCII text ("Δ", "—"); an ASCII locale must not break them."""

    reports = _write_reports(tmp_path, "NVIDIA B200", "NVIDIA H100")
    out = tmp_path / "charts"
    repo = Path(__file__).resolve().parents[3]
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    script = repo / "tools" / "validation" / "models" / "plot_h3_visual_vae.py"
    subprocess.run(
        [sys.executable, str(script), *reports, "--out-dir", str(out)],
        check=True,
        cwd=repo,
        env=env,
    )
    table = (out / "h3-visual-vae-accuracy-b200.md").read_text(encoding="utf-8")
    assert "max \\|Δ\\|" in table
    assert (out / "h3-visual-vae-batch-invariance-b200-vs-h100.md").exists()


def test_a_single_repeat_is_rejected(monkeypatch, tmp_path) -> None:
    """One run gives one digest per step, and run-to-run determinism would pass unchecked."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    out = tmp_path / "evidence.json"
    argv = ["ev", "--vae-dir", str(tmp_path), "--out", str(out), "--repeats", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        ev.main()
    assert exc.value.code == 2
    assert not out.exists()
