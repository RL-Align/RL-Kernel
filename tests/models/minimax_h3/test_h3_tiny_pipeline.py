# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``capture_steps`` puts back every step it wrapped, also when its setup fails."""

from __future__ import annotations

import sys
import types

import pytest


def test_capture_steps_restores_wrapped_steps_when_a_step_is_missing(monkeypatch):
    from tests.models.minimax_h3 import _tiny_pipeline

    class Step:
        def __call__(self, components, state):
            return components, state

    original = Step.__call__
    # Stand-in for the pinned step modules: ``Step`` exists, ``RenamedStep`` does not.
    steps = types.ModuleType("diffusers.modular_pipelines.minimax_h3")
    steps.before_denoise = types.SimpleNamespace(Step=Step)
    steps.before_encoder = types.SimpleNamespace()
    steps.encoders = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", steps)
    monkeypatch.setattr(_tiny_pipeline, "CAPTURE", {"Step": ("x",), "RenamedStep": ("y",)})

    # ``next()`` finds no ``RenamedStep``; the generator re-raises its StopIteration.
    with pytest.raises(RuntimeError):
        with _tiny_pipeline.capture_steps():
            pass
    assert Step.__call__ is original
