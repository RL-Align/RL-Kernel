# SPDX-License-Identifier: Apache-2.0
"""Exact task-29 dispatch eligibility and companion-source verification."""

from types import SimpleNamespace

import pytest

from rl_engine.backends.rocm.moe import router_sqrt_softplus as backend
from rl_engine.contracts.operators.router_sqrt_softplus import AITER_CONTRACT_ID, BLOCK_SIZE, LAUNCH
from rl_engine.runtime.registry import _default_semantic_descriptors
from rl_engine.runtime.semantic_registry import (
    OperatorRequirements,
    OperatorResolutionError,
    SemanticOperatorCatalog,
)


def resolve(device="rocm", dtype="float32", tp=1, graph=False, target="training"):
    session = SemanticOperatorCatalog(_default_semantic_descriptors()).session()
    result = session.resolve(
        semantic_op="router_sqrt_softplus",
        requested_backend="rlkernel.router_sqrt_softplus.fp32.stable_tail.v1",
        target=target,
        requirements=OperatorRequirements(
            device=device,
            dtype=dtype,
            topology={"tensor_parallel_size": tp},
            alignment_properties={"graph": graph},
        ),
    )
    return session, result


@pytest.mark.parametrize("target", ["training", "rollout"])
def test_exact_registration(target):
    session, result = resolve(target=target)
    assert type(session.instantiate(result)).__name__ == "RouterSqrtSoftplusOp"
    assert result.trace.fallback_attempts == ()


@pytest.mark.parametrize(
    "kwargs",
    [{"device": "cpu"}, {"device": "cuda"}, {"dtype": "bfloat16"}, {"tp": 2}, {"graph": True}],
)
def test_unsupported_dispatch_never_falls_back(kwargs):
    with pytest.raises(OperatorResolutionError):
        resolve(**kwargs)


def test_missing_companion_patch_is_an_error(monkeypatch):
    def unavailable(name):
        raise ImportError("not installed")

    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="requires the pinned AITER companion patch"):
        backend.implementation()
    backend.implementation.cache_clear()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"ROUTER_SQRT_SOFTPLUS_CONTRACT_ID": "unknown"}, "arithmetic contract"),
        ({"BLOCK": 128}, "launch configuration"),
        ({"LAUNCH": {"num_warps": 8}}, "launch configuration"),
    ],
)
def test_incompatible_companion_api_rejected(monkeypatch, change, match):
    api = SimpleNamespace(
        ROUTER_SQRT_SOFTPLUS_CONTRACT_ID=AITER_CONTRACT_ID, BLOCK=BLOCK_SIZE, LAUNCH=dict(LAUNCH)
    )
    for name, value in change.items():
        setattr(api, name, value)
    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match=match):
        backend.implementation()
    backend.implementation.cache_clear()


def test_source_fingerprint_mismatch_rejected(monkeypatch, tmp_path):
    source = tmp_path / "wrong.py"
    source.write_text("# unrelated source\n")
    api = SimpleNamespace(
        ROUTER_SQRT_SOFTPLUS_CONTRACT_ID=AITER_CONTRACT_ID,
        BLOCK=BLOCK_SIZE,
        LAUNCH=dict(LAUNCH),
        __file__=str(source),
    )
    backend.implementation.cache_clear()
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: api)
    with pytest.raises(RuntimeError, match="sources do not match"):
        backend.implementation()
    backend.implementation.cache_clear()
