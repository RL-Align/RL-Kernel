# SPDX-License-Identifier: Apache-2.0
"""Exact task-33 dispatch and source/API identity failures."""

from types import SimpleNamespace

import pytest

from rl_engine.backends.rocm.moe import router_training as backend
from rl_engine.contracts.operators.router_training import CONTRACT_ID, REPLAY_CONTRACT_ID
from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(device="rocm", dtype="float32", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    resolution = session.resolve(
        semantic_op="router_training",
        requested_backend=CONTRACT_ID,
        target=target,
        requirements=OperatorRequirements(
            device=device,
            dtype=dtype,
            topology={"tensor_parallel_size": tp},
            alignment_properties={"graph": graph},
        ),
    )
    return session, resolution


@pytest.mark.parametrize("target", ["training", "rollout"])
def test_exact_registration(target):
    session, result = resolve(target=target)
    assert type(session.instantiate(result)).__name__ == "RouterTrainingOp"
    assert result.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [{"device": "cpu"}, {"device": "cuda"}, {"dtype": "bfloat16"}, {"tp": 2}, {"graph": True}],
)
def test_unsupported_dispatch_has_no_fallback(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)


def test_missing_api_fails(monkeypatch):
    def missing(name):
        raise ImportError("missing")

    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires the pinned"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_changed_api_contract_fails(monkeypatch):
    backend.implementation.cache_clear()
    monkeypatch.setattr(
        backend.importlib,
        "import_module",
        lambda name: SimpleNamespace(ROUTER_REPLAY_CONTRACT_ID="wrong"),
    )
    with pytest.raises(RuntimeError, match="API contract"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_changed_source_fails(monkeypatch, tmp_path):
    path = tmp_path / "api.py"
    path.write_text("# unrelated implementation\n")
    backend.implementation.cache_clear()
    monkeypatch.setattr(
        backend.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            ROUTER_REPLAY_CONTRACT_ID=REPLAY_CONTRACT_ID, __file__=str(path)
        ),
    )
    with pytest.raises(RuntimeError, match="source fingerprint"):
        backend.implementation()
    backend.implementation.cache_clear()
