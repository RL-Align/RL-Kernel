# SPDX-License-Identifier: Apache-2.0
"""Evidence failures must not look like completed four-layer acceptance."""

import asyncio
from types import SimpleNamespace

import pytest

from examples.operators.dsv4_4l import audit_dependencies, cycle, native_trace


def test_audit_rejects_wrong_upstream_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(audit_dependencies.subprocess, "check_output", lambda *a, **k: "wrong")
    with pytest.raises(RuntimeError, match="verl revision mismatch"):
        audit_dependencies.audit(tmp_path)


def test_partial_json_sentinel_is_not_treated_as_completed(monkeypatch, tmp_path):
    path = tmp_path / "actor-ready.json"
    path.write_text("{")
    monkeypatch.setattr(cycle, "ROOT", tmp_path)

    async def publish():
        await asyncio.sleep(0.01)
        path.write_text('{"version": 1}')

    async def check():
        writer = asyncio.create_task(publish())
        result = await cycle.wait_file(path, SimpleNamespace(poll=lambda: None), timeout=2)
        await writer
        return result

    assert asyncio.run(check()) == {"version": 1}


def test_failed_trace_install_restores_existing_observer(tmp_path):
    def observer(*args):
        pass

    model = SimpleNamespace(_rlk_p1=SimpleNamespace(observer=observer))
    with pytest.raises(AttributeError):
        native_trace.install(model, tmp_path)
    assert model._rlk_p1.observer is observer
