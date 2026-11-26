# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``capture_steps`` restores every step class as it was, also when one inherits another's call."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from tests.models.minimax_h3 import _tiny_pipeline as tp


class _State(dict):
    pass


def _fake_steps(monkeypatch):
    class Base:
        def __call__(self, components, state):
            state["x"] = 1
            return components, state

    class Sub(Base):  # inherits ``__call__``
        pass

    package = ModuleType("diffusers.modular_pipelines.minimax_h3")
    for name in ("before_denoise", "before_encoder", "encoders"):
        setattr(package, name, ModuleType(name))
    package.before_denoise.Base, package.before_denoise.Sub = Base, Sub
    for name, module in (
        ("diffusers", ModuleType("diffusers")),
        ("diffusers.modular_pipelines", ModuleType("diffusers.modular_pipelines")),
        ("diffusers.modular_pipelines.minimax_h3", package),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(tp, "CAPTURE", {"Base": ("x",), "Sub": ("x",)})
    return Base, Sub


def test_an_inherited_call_is_wrapped_once_and_restored(monkeypatch):
    Base, Sub = _fake_steps(monkeypatch)
    base_call = Base.__dict__["__call__"]
    with tp.capture_steps() as captured:
        Sub()(None, _State())
        assert captured == {"Sub": {"x": 1}}  # the real call ran once, under the subclass name
        Base()(None, _State())
        assert captured["Base"] == {"x": 1}
    assert Base.__dict__["__call__"] is base_call
    assert "__call__" not in Sub.__dict__  # inherits again, as before


def test_a_missing_step_restores_the_ones_already_wrapped(monkeypatch):
    Base, _ = _fake_steps(monkeypatch)
    base_call = Base.__dict__["__call__"]
    monkeypatch.setattr(tp, "CAPTURE", {"Base": ("x",), "Missing": ("x",)})
    with pytest.raises(RuntimeError):  # StopIteration inside the generator (PEP 479)
        with tp.capture_steps():
            pass
    assert Base.__dict__["__call__"] is base_call
