# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The H3 MLP evidence says which weights it was measured with and refuses unpinned runs."""

from __future__ import annotations

import pytest
import torch

from rl_engine.validation.models.h3_report import MLP_NAMES, MLP_SHAPES, h3_params
from rl_engine.validation.models.h3_weights import WEIGHTS_ENV, h3_weights_dir
from tools.validation.models.h3_evidence import check_weight_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_without_the_pinned_checkpoint_the_params_are_marked_random(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    params, source = h3_params(MLP_NAMES, MLP_SHAPES)
    assert source == "random"
    assert [tuple(p.shape) for p in params] == MLP_SHAPES
    assert all(p.is_cuda for p in params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_with_the_pinned_checkpoint_the_params_are_marked_pinned():
    if h3_weights_dir() is None:
        pytest.skip(f"{WEIGHTS_ENV} not set; run tools/weights/prepare_h3_weights.py")
    params, source = h3_params(MLP_NAMES, MLP_SHAPES)
    assert source == "pinned"
    assert [tuple(p.shape) for p in params] == MLP_SHAPES


def test_mlp_evidence_is_refused_without_pinned_weights():
    random_run = {"accuracy": {"weights_source": "random"}}
    pinned_run = {"accuracy": {"weights_source": "pinned"}}
    with pytest.raises(SystemExit, match="pinned time_embedder weights"):
        check_weight_source("timestep_mlp_fp32", random_run, allow_random=False)
    check_weight_source("timestep_mlp_fp32", random_run, allow_random=True)
    check_weight_source("timestep_mlp_fp32", pinned_run, allow_random=False)
    # the sinusoid op has no weights, so it records none
    check_weight_source("timestep_sinusoid_h3", {"accuracy": {}}, allow_random=False)


def test_the_figure_title_names_the_recorded_weight_source():
    pytest.importorskip("matplotlib")
    from tools.validation.models.plot_h3_evidence import mlp_title

    assert "pinned time_embedder weights" in mlp_title({"accuracy": {"weights_source": "pinned"}})
    assert "random" in mlp_title({"accuracy": {"weights_source": "random"}})
    assert "not recorded" in mlp_title({"accuracy": {}})
