# SPDX-License-Identifier: Apache-2.0
"""Stable Top-6 exact dispatch and source/contract guards."""

from types import SimpleNamespace

import pytest

from rl_engine.backends.rocm.moe import stable_topk6 as backend
from rl_engine.contracts.operators.stable_topk6 import AITER_CONTRACT_ID, LAUNCH
from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(device="rocm", dtype="float32", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    resolution = session.resolve(
        semantic_op="stable_topk6",
        requested_backend="rlkernel.stable_topk6.fp32.global_id.v1",
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
def test_exact_stable_topk6_registration(target):
    session, resolution = resolve(target=target)
    assert type(session.instantiate(resolution)).__name__ == "StableTopK6Op"
    assert resolution.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [{"device": "cpu"}, {"device": "cuda"}, {"dtype": "bfloat16"}, {"tp": 2}, {"graph": True}],
)
def test_unsupported_dispatch_rejects_without_a_fallback(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)


def test_missing_aiter_patch_is_an_explicit_error(monkeypatch):
    def missing(name):
        raise ImportError("missing task31 API")

    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="requires the pinned AITER"):
        backend.implementation()
    backend.implementation.cache_clear()


@pytest.mark.parametrize("contract,launch", [("wrong", LAUNCH), (AITER_CONTRACT_ID, {})])
def test_changed_contract_or_launch_rejected(monkeypatch, contract, launch):
    backend.implementation.cache_clear()
    api = SimpleNamespace(STABLE_TOPK6_CONTRACT_ID=contract, LAUNCH=launch)
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match="arithmetic/launch contract mismatch"):
        backend.implementation()
    backend.implementation.cache_clear()


def test_changed_kernel_source_rejected(monkeypatch, tmp_path):
    path = tmp_path / "wrong.py"
    path.write_text("# unrelated implementation\n")
    api = SimpleNamespace(
        STABLE_TOPK6_CONTRACT_ID=AITER_CONTRACT_ID, LAUNCH=LAUNCH, __file__=str(path)
    )
    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match="source fingerprint mismatch"):
        backend.implementation()
    backend.implementation.cache_clear()
