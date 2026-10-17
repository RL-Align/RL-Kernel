# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Canonical registration and fallback for the shared softcapped logprob op."""

import pytest
import torch

from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)
from rl_engine.runtime.provenance.identity import stable_operator_path
from rl_engine.runtime.registry import KernelRegistry, OpBackend


def test_new_operator_uses_canonical_identity():
    """This new operator has no previously released legacy module identity."""
    assert OpBackend.PYTORCH_NATIVE_SOFTCAPPED_SELECTED_LOGPROB.value == (
        "rl_engine.reference.logprob.softcapped_selected_logprob."
        "NativeSoftcappedSelectedLogprobOp"
    )
    assert OpBackend.TRITON_SOFTCAPPED_SELECTED_LOGPROB.value == (
        "rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob."
        "TritonSoftcappedSelectedLogprobOp"
    )
    assert stable_operator_path(NativeSoftcappedSelectedLogprobOp()) == (
        OpBackend.PYTORCH_NATIVE_SOFTCAPPED_SELECTED_LOGPROB.value
    )


@pytest.mark.parametrize("platform", ("cpu", "musa", "npu"))
def test_registry_native_dispatch(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    op = registry.get_op("softcapped_selected_logprob")
    assert isinstance(op, NativeSoftcappedSelectedLogprobOp)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    output = op(logits, torch.tensor([1]))
    expected = torch.log_softmax(30.0 * torch.tanh(logits.double() / 30.0), dim=-1)[:, 1]
    assert output.shape == (1,) and output.dtype == torch.float32
    assert output.device == logits.device
    torch.testing.assert_close(output.double(), expected)


@pytest.mark.parametrize("platform", ("cuda", "rocm"))
def test_registry_fallback_when_triton_cannot_load(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    load_backend = registry._load_backend
    attempted = []

    def without_triton(backend):
        attempted.append(backend)
        if backend == OpBackend.TRITON_SOFTCAPPED_SELECTED_LOGPROB:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", without_triton)
    assert isinstance(
        registry.get_op("softcapped_selected_logprob"), NativeSoftcappedSelectedLogprobOp
    )
    assert attempted == [
        OpBackend.TRITON_SOFTCAPPED_SELECTED_LOGPROB,
        OpBackend.PYTORCH_NATIVE_SOFTCAPPED_SELECTED_LOGPROB,
    ]
