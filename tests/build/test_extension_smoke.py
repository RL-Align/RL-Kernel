# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Regression guard for issue #191.

On a GPU host the compiled extension ``rl_engine._C`` MUST be present and
launchable; a missing or arch-mismatched build must fail loudly here instead of
silently degrading to the pure-PyTorch fallbacks. On a CPU host (no CUDA) the
test skips, so the CPU CI job - which legitimately has no ``_C`` - stays green.

Enforcement is opt-in via ``RL_KERNEL_REQUIRE_EXT=1``, which the GPU CI
orchestrator sets right before pytest (after it has built the extension). That
way the check does not fail in environments not expected to have ``_C`` compiled -
a plain ``pytest`` run, or CI running an older orchestrator that has not built it
yet (the compiled kernels are always enforced separately by tools/checks/ci_smoke.py).
"""
import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.skipif(
    os.environ.get("RL_KERNEL_REQUIRE_EXT") != "1",
    reason="extension enforcement is CI-only (orchestrator sets RL_KERNEL_REQUIRE_EXT=1)",
)
def test_compiled_extension_present_and_launches():
    # Import directly from the package, NOT from rl_engine.backends.extension, which
    # deliberately swallows the ImportError and falls back to _C = None.
    try:
        from rl_engine import _C
    except ImportError as exc:  # Bug A: the extension was never built
        pytest.fail(
            "rl_engine._C is missing on a GPU host - the CUDA extension was not "
            "built. Install with `pip install --no-build-isolation -e .`. "
            f"Underlying error: {exc}"
        )

    logits = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    token_ids = torch.randint(0, 32, (4,), device="cuda", dtype=torch.long)
    out = _C.fused_logp(logits, token_ids)
    torch.cuda.synchronize()  # Bug B: an arch mismatch surfaces on synchronize
    assert tuple(out.shape) == (4,)
