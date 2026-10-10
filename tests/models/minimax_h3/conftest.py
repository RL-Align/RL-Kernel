# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared fixtures for the MiniMax-H3 (RFC #420) operator tests.

Real-weight tests read the pinned tensors from ``$RL_KERNEL_H3_WEIGHTS``
(written by ``tools/weights/prepare_h3_weights.py``) and skip when it is unset.
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
