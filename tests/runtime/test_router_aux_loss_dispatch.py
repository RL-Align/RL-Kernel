# SPDX-License-Identifier: Apache-2.0
"""Task-34 optional-training registration and exact dependency identity guards."""

from types import SimpleNamespace

import pytest

from rl_engine.backends.rocm.moe import router_aux_loss as backend
from rl_engine.contracts.operators.router_aux_loss import AITER_CONTRACT_ID, CONTRACT_ID, LAUNCH
from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(device="rocm", dtype="float32", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    result = session.resolve(
        semantic_op="router_aux_loss",
        requested_backend=CONTRACT_ID,
        target=target,
        requirements=OperatorRequirements(
            device=device,
            dtype=dtype,
            topology={"tensor_parallel_size": tp},
            alignment_properties={"graph": graph},
        ),
    )
    return session, result


def test_exact_optional_training_registration():
    session, result = resolve()
    assert type(session.instantiate(result)).__name__ == "RouterAuxLossOp"
    assert result.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "cpu"},
        {"device": "cuda"},
        {"dtype": "bfloat16"},
        {"tp": 2},
        {"graph": True},
        {"target": "rollout"},
    ],
)
def test_unsupported_semantic_request_does_not_fallback(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)


def test_missing_aiter_companion_rejected(monkeypatch):
    def missing(name):
        raise ImportError("missing task34")

    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires the pinned"):
        backend.implementation()
    backend.implementation.cache_clear()


@pytest.mark.parametrize("contract,launch", [("wrong", LAUNCH), (AITER_CONTRACT_ID, {})])
def test_changed_arithmetic_or_launch_rejected(monkeypatch, contract, launch):
    backend.implementation.cache_clear()
    monkeypatch.setattr(
        backend.importlib,
        "import_module",
        lambda name: SimpleNamespace(AUX_LOSS_CONTRACT_ID=contract, LAUNCH=launch),
    )
    with pytest.raises(RuntimeError, match="arithmetic/launch"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_changed_source_rejected(monkeypatch, tmp_path):
    path = tmp_path / "api.py"
    path.write_text("# unrelated\n")
    backend.implementation.cache_clear()
    monkeypatch.setattr(
        backend.importlib,
        "import_module",
        lambda name: SimpleNamespace(
            AUX_LOSS_CONTRACT_ID=AITER_CONTRACT_ID, LAUNCH=LAUNCH, __file__=str(path)
        ),
    )
    with pytest.raises(RuntimeError, match="source fingerprint"):
        backend.implementation()
    backend.implementation.cache_clear()
