# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The tiny-pipeline helpers: `capture_steps` restores what it wrapped; the loader tries the
cache only first and never caches a partly loaded pipeline."""

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


def _fake_diffusers(calls, missing):
    """``ModularPipeline`` stand-in: each load attempt leaves the next ``missing`` names None,
    as the pinned ``load_components`` does when it cannot load a component."""

    class Pipe:
        def __init__(self):
            self.null_component_names = []

        @classmethod
        def from_pretrained(cls, repo, **hub):
            calls.append(("from_pretrained", hub))
            return cls()

        def load_components(self, torch_dtype=None, **hub):
            calls.append(("load_components", hub))
            self.null_component_names = list(missing.pop(0))

        def register_to_config(self, **config):
            self.config = config

    return types.SimpleNamespace(ModularPipeline=Pipe)


def test_tiny_pipeline_loads_cache_only_first_and_never_caches_a_partial_pipeline(monkeypatch):
    from tests.models.minimax_h3 import _tiny_pipeline

    # The cached copy lacks the processor; the Hub retry completes the pipeline.
    calls: list = []
    fake = _fake_diffusers(calls, [["processor"], []])
    monkeypatch.setattr(_tiny_pipeline, "require_tiny_pipeline", lambda: fake)
    monkeypatch.setattr(_tiny_pipeline, "_PIPELINES", {})
    pipe = _tiny_pipeline.load_tiny_pipeline()
    assert calls == [
        ("from_pretrained", {"local_files_only": True}),
        ("load_components", {"local_files_only": True}),
        ("from_pretrained", {}),
        ("load_components", {}),
    ]
    assert pipe.null_component_names == [] and _tiny_pipeline._PIPELINES["pipe"] is pipe

    # Still incomplete after the retry (offline, partial cache): skip, cache nothing.
    fake = _fake_diffusers([], [["processor"], ["processor"]])
    monkeypatch.setattr(_tiny_pipeline, "require_tiny_pipeline", lambda: fake)
    monkeypatch.setattr(_tiny_pipeline, "_PIPELINES", {})
    with pytest.raises(pytest.skip.Exception, match="processor"):
        _tiny_pipeline.load_tiny_pipeline()
    assert _tiny_pipeline._PIPELINES == {}
