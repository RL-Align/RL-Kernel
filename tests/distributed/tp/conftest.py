# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fixtures for the MiniMax-H3 tensor-parallel tests (RFC #420).

``h3_weights_cpu`` is the same fixture as in ``tests/models/minimax_h3/conftest.py``: the pinned
tensors come from ``$RL_KERNEL_H3_WEIGHTS`` (written by ``tools/weights/prepare_h3_weights.py``)
and the test skips when they are unavailable.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.validation.models.h3_weights import (
    WEIGHTS_ENV,
    h3_weights_dir,
    load_h3_conditioning_weights,
)


@pytest.fixture(scope="session")
def h3_weights_cpu() -> dict[str, torch.Tensor]:
    """Load pinned CPU weights once per session, skipping unavailable artifacts."""

    if h3_weights_dir() is None:
        pytest.skip(f"{WEIGHTS_ENV} not set; run tools/weights/prepare_h3_weights.py")
    try:
        return load_h3_conditioning_weights("cpu")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
