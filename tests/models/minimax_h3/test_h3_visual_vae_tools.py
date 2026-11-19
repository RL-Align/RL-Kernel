# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The ``visual_vae_contract`` evidence and chart tools (CPU only, no checkpoint)."""

from __future__ import annotations

import math

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
    assert math.isnan(out["max_abs"])
    assert ev._cmp(torch.zeros(2), torch.tensor([0.0, 0.5]))["max_abs"] == 0.5


# --- charts -------------------------------------------------------------------------------


@pytest.fixture
def plot():
    pytest.importorskip("matplotlib")
    from tools.validation.models import plot_h3_visual_vae

    return plot_h3_visual_vae


def _report(device_name: str, repeats: int = 3, max_abs: float = 0.0) -> dict:
    ops = ("encode_keyframe", "encode_video", "decode_video")
    run = {k: {"digests": ["d"] * repeats, "bitwise_equal": True} for k in ops}
    parity = {k: {"max_abs": max_abs} for k in ops}
    return {
        "environment": {"device_name": device_name},
        "checks": {
            "run_to_run": run,
            "diffusers_parity": {"diffusers_pinned": parity, "diffusers_default_flags": parity},
            "batch_invariance": {"contract_per_row": {k: True for k in ops}},
            "tiling": {
                "tiled_vs_untiled_encode_keyframe": {"max_abs": 0.1},
                "tiled_vs_untiled_decode_video": {"max_abs": 0.1},
            },
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


def test_shape_mismatch_charts_as_differs(plot) -> None:
    rows = plot.accuracy_rows(_report("NVIDIA B200", max_abs=math.nan))
    assert plot.fmt_delta(rows[3][1]) == plot.DIFFERS
