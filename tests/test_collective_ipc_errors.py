# SPDX-License-Identifier: Apache-2.0
"""IPC error reporting must preserve the original error on Python 3.10 too."""

from types import SimpleNamespace

import pytest

from rl_engine.distributed.collectives import DeterministicCollective

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("supports_notes", [False, True])
def test_ipc_allocation_preserves_error_with_optional_notes(monkeypatch, supports_notes):
    error = RuntimeError("synthetic IPC allocation failure")
    notes = []
    monkeypatch.setattr(error, "add_note", notes.append if supports_notes else None, raising=False)

    def fail_ipc_meta(staging):
        raise error

    collective = SimpleNamespace(
        rank=0,
        max_size_bytes=16,
        device="cpu",
        _extension=SimpleNamespace(deterministic_collective_ipc_meta=fail_ipc_meta),
    )
    with pytest.raises(RuntimeError) as caught:
        DeterministicCollective._create_cuda_ipc_state(collective)
    assert caught.value is error
    if supports_notes:
        assert len(notes) == 1
        assert "IPC allocation: rank=0, capacity=16" in notes[0]
    else:
        assert notes == []
