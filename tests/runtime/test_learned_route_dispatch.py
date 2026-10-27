# SPDX-License-Identifier: Apache-2.0
"""Exact learned-router registration and dependency failure behavior."""

from types import SimpleNamespace

import pytest

from rl_engine.backends.rocm.moe import learned_route as backend
from rl_engine.contracts.operators.learned_route import AITER_CONTRACT_ID, LAUNCH
from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(device="rocm", dtype="float32", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    resolution = session.resolve(
        semantic_op="learned_route",
        requested_backend="rlkernel.learned_route.fp32.bias_selection.capacity_mask.v1",
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
    assert type(session.instantiate(result)).__name__ == "LearnedRouteOp"
    assert result.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [{"device": "cpu"}, {"device": "cuda"}, {"dtype": "bfloat16"}, {"tp": 2}, {"graph": True}],
)
def test_no_fallback_for_unsupported_requests(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)


def test_missing_aiter_companion_fails(monkeypatch):
    def missing(name):
        raise ImportError("missing task32 API")

    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires pinned"):
        backend.implementation()
    backend.implementation.cache_clear()


@pytest.mark.parametrize("contract,launch", [("wrong", LAUNCH), (AITER_CONTRACT_ID, {})])
def test_changed_contract_or_launch_rejects(monkeypatch, contract, launch):
    api = SimpleNamespace(LEARNED_ROUTE_CONTRACT_ID=contract, LAUNCH=launch)
    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match="arithmetic/launch"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_changed_source_fingerprint_rejects(monkeypatch, tmp_path):
    source = tmp_path / "wrong.py"
    source.write_text("# unrelated implementation\n")
    api = SimpleNamespace(
        LEARNED_ROUTE_CONTRACT_ID=AITER_CONTRACT_ID, LAUNCH=LAUNCH, __file__=str(source)
    )
    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match="source fingerprint"):
        backend.implementation()
    backend.implementation.cache_clear()
