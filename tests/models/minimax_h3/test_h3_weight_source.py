# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The H3 MLP evidence says which weights it was measured with and refuses unpinned runs."""

from __future__ import annotations

import pytest
import torch

from rl_engine.validation.models.h3_report import h3_weights_source
from rl_engine.validation.models.h3_weights import WEIGHTS_ENV, h3_weights_dir
from tools.validation.models.h3_evidence import (
    check_weight_source,
    measure_case,
    require_pinned_weights,
)


def test_without_the_pinned_checkpoint_the_source_is_random(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    assert h3_weights_source() == "random"


def test_with_the_pinned_checkpoint_the_source_is_pinned():
    if h3_weights_dir() is None:
        pytest.skip(f"{WEIGHTS_ENV} not set; run tools/weights/prepare_h3_weights.py")
    assert h3_weights_source() == "pinned"


def test_mlp_evidence_is_refused_without_pinned_weights():
    random_run = {"accuracy": {"weights_source": "random"}}
    pinned_run = {"accuracy": {"weights_source": "pinned"}}
    with pytest.raises(SystemExit, match="pinned time_embedder weights"):
        check_weight_source("timestep_mlp_fp32", random_run, allow_random=False)
    check_weight_source("timestep_mlp_fp32", random_run, allow_random=True)
    check_weight_source("timestep_mlp_fp32", pinned_run, allow_random=False)
    # the sinusoid op has no weights, so it records none
    check_weight_source("timestep_sinusoid_h3", {"accuracy": {}}, allow_random=False)


def test_mlp_evidence_is_refused_before_measuring_without_pinned_weights(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    with pytest.raises(SystemExit, match="pinned time_embedder weights"):
        require_pinned_weights("timestep_mlp_fp32", allow_random=False)
    require_pinned_weights("timestep_mlp_fp32", allow_random=True)
    require_pinned_weights("timestep_sinusoid_h3", allow_random=False)


def test_the_figure_title_names_the_recorded_weight_source():
    pytest.importorskip("matplotlib")
    from tools.validation.models.plot_h3_evidence import mlp_title

    assert "pinned time_embedder weights" in mlp_title({"accuracy": {"weights_source": "pinned"}})
    assert "random" in mlp_title({"accuracy": {"weights_source": "random"}})
    assert "not recorded" in mlp_title({"accuracy": {}})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_perf_row_keeps_the_weight_source_of_its_case():
    a = torch.randn(128, 128, device="cuda")
    case = {
        "op": "matmul",
        "case": "T=1",
        "backend": "torch",
        "bytes": 1 << 16,
        "weights": "random",
        "candidate": lambda: a @ a,
    }
    row = measure_case(case)
    assert row["weights"] == "random"
    assert row["candidate_us"] > 0
    assert "weights" not in measure_case({k: v for k, v in case.items() if k != "weights"})
