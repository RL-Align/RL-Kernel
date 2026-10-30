# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Schema validation for the AITER entry points the strict ROCm core calls.

The CUDA core validates the FA4 CuTe API by parameter name before it runs.
AITER hides its signature behind a JIT wrapper (``inspect.signature`` reports
``(*args, **kwargs)``), so the equivalent check reads the registered Torch
schema. The strict calls are positional, which makes argument *order* part of
the contract too: an upstream insertion would silently reinterpret everything
after it while every call still type-checks.
"""

from __future__ import annotations

import pytest

from rl_engine.backends.rocm.attention.flash_attn import (
    _AITER_BATCH_PREFILL_POSITIONAL_CONTRACT,
    _AITER_BATCH_PREFILL_REQUIRED_KEYWORDS,
    _AITER_BWD_POSITIONAL_CONTRACT,
    _AITER_BWD_REQUIRED_KEYWORDS,
    _AITER_FWD_POSITIONAL_CONTRACT,
    _AITER_FWD_REQUIRED_KEYWORDS,
    StrictRocmAttentionUnavailable,
    _validate_aiter_schema,
)


def _aiter_available() -> bool:
    try:
        import aiter.ops.mha  # noqa: F401
    except Exception:
        return False
    return True


requires_aiter = pytest.mark.skipif(not _aiter_available(), reason="AITER is not installed")


def test_positional_contract_matches_the_strict_call_sites() -> None:
    """The tuples must stay in step with what the autograd Function passes.

    ``_AiterCKAttentionFn`` calls both ops positionally. If someone edits a
    call site without editing the contract, the schema check would still pass
    while the call means something else.
    """

    assert _AITER_FWD_POSITIONAL_CONTRACT[:6] == (
        "q",
        "k",
        "v",
        "dropout_p",
        "softmax_scale",
        "is_causal",
    )
    # The forward passes (-1, -1, 0, True, False) after is_causal.
    assert _AITER_FWD_POSITIONAL_CONTRACT[6:] == (
        "window_size_left",
        "window_size_right",
        "sink_size",
        "return_softmax_lse",
        "return_dropout_randval",
    )
    assert "out" in _AITER_FWD_REQUIRED_KEYWORDS
    # The backward pins determinism positionally, so its slot must not move.
    assert _AITER_BWD_POSITIONAL_CONTRACT[-1] == "deterministic"
    assert _AITER_BWD_POSITIONAL_CONTRACT.index("softmax_lse") == 5
    assert "rng_state" in _AITER_BWD_REQUIRED_KEYWORDS


@requires_aiter
def test_installed_aiter_satisfies_the_strict_contract() -> None:
    _validate_aiter_schema(
        "mha_fwd",
        _AITER_FWD_POSITIONAL_CONTRACT,
        required_keywords=_AITER_FWD_REQUIRED_KEYWORDS,
    )
    _validate_aiter_schema(
        "mha_bwd",
        _AITER_BWD_POSITIONAL_CONTRACT,
        required_keywords=_AITER_BWD_REQUIRED_KEYWORDS,
    )
    _validate_aiter_schema(
        "mha_batch_prefill",
        _AITER_BATCH_PREFILL_POSITIONAL_CONTRACT,
        required_keywords=_AITER_BATCH_PREFILL_REQUIRED_KEYWORDS,
    )


@requires_aiter
def test_reordered_positional_contract_fails_closed() -> None:
    """A swap the installed schema does not have must be rejected."""

    swapped = ("q", "k", "v", "softmax_scale", "dropout_p")
    with pytest.raises(StrictRocmAttentionUnavailable, match="positional contract changed"):
        _validate_aiter_schema("mha_fwd", swapped)


@requires_aiter
def test_missing_keyword_control_fails_closed() -> None:
    with pytest.raises(StrictRocmAttentionUnavailable, match="missing strict controls"):
        _validate_aiter_schema(
            "mha_fwd",
            _AITER_FWD_POSITIONAL_CONTRACT,
            required_keywords=frozenset({"num_splits"}),
        )


def test_unregistered_operator_fails_closed() -> None:
    with pytest.raises(StrictRocmAttentionUnavailable, match="cannot read the Torch schema"):
        _validate_aiter_schema("mha_fwd_that_does_not_exist", ("q",))
