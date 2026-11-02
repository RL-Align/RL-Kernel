# SPDX-License-Identifier: Apache-2.0
"""Exact Hash backend dispatch and no-silent-fallback behavior."""

import pytest

from rl_engine.backends.rocm.moe import hash_route as backend
from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(device="rocm", dtype="float32", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    resolution = session.resolve(
        semantic_op="hash_route",
        requested_backend="rlkernel.hash_route.fp32.slot_order.clamp_min.v1",
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
def test_exact_hash_registration(target):
    session, resolution = resolve(target=target)
    assert type(session.instantiate(resolution)).__name__ == "HashRouteOp"
    assert resolution.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [{"device": "cpu"}, {"device": "cuda"}, {"dtype": "bfloat16"}, {"tp": 2}, {"graph": True}],
)
def test_unsupported_hash_dispatch_rejects(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)


def test_missing_hash_patch_does_not_fall_back(monkeypatch):
    def missing(name):
        raise ImportError("missing task30 API")

    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires the pinned AITER"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_wrong_hash_contract_rejected(monkeypatch):
    from types import SimpleNamespace

    backend.implementation.cache_clear()
    monkeypatch.setattr(
        backend.importlib,
        "import_module",
        lambda name: SimpleNamespace(HASH_ROUTE_CONTRACT_ID="wrong"),
    )
    with pytest.raises(RuntimeError, match="arithmetic contract mismatch"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_wrong_hash_source_rejected(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from rl_engine.contracts.operators.hash_route import AITER_CONTRACT_ID

    path = tmp_path / "wrong.py"
    path.write_text("# unrelated implementation\n")
    api = SimpleNamespace(HASH_ROUTE_CONTRACT_ID=AITER_CONTRACT_ID, __file__=str(path))
    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match="source fingerprint mismatch"):
        backend.implementation()
    backend.implementation.cache_clear()
