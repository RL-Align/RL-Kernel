# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded P1 fixtures and the golden-hash manifest (start-kit acceptance data).

Fixtures are regenerated deterministically from seeds; the committed manifest
``tests/fixtures/p1/golden_hashes.json`` anchors the golden bytes in CI. If a
torch upgrade ever changes RNG or libm behavior, the manifest test fails
loudly instead of the goldens drifting silently.

Fixture geometry is a scaled-down layer (``hidden=128`` -> ``K=512``) so the
serial oracle stays CPU-cheap; ``hc_mult``, ``controller_n``,
``sinkhorn_iters`` and both epsilons are the real production constants.
``LayerContract.assert_production()`` pins the full DSv4 geometry separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from rl_engine.mhc import oracle
from rl_engine.mhc.contract import (
    ORACLE_PROFILE,
    SCHEMA_VERSION,
    ControllerParams,
    GradBoundary,
    LayerContract,
    NormParams,
    ResidualBatch,
    tensor_sha256,
)
from rl_engine.mhc.trace import MHCTrace

FIXTURE_HIDDEN = 128
BASE_SEED = 2026

DEFAULT_MANIFEST_PATH = Path("tests/fixtures/p1/golden_hashes.json")

BLOCK_CASES: dict[str, dict[str, Any]] = {
    "one_row": {"tokens": 1, "geometry": "one-row"},
    "packed_t16": {"tokens": 16},
    "packed_t7_odd": {"tokens": 7},
    "fused_pre_norm": {"tokens": 16, "fusion_mode": "fused-pre-norm"},
    "mixer_frozen": {"tokens": 16, "trainability": "mixer-frozen"},
}


def _seed_for(name: str) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return BASE_SEED + int.from_bytes(digest[:4], "little")


def _gen(name: str) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(_seed_for(name))
    return g


def _randn(g: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float32) * scale


def make_contract(name: str) -> LayerContract:
    spec = BLOCK_CASES[name]
    return LayerContract(
        hidden=FIXTURE_HIDDEN,
        layer_index=spec.get("layer_index", 3),
        fusion_mode=spec.get("fusion_mode", "unfused"),
        trainability=spec.get("trainability", "full"),
    )


def make_batch(name: str) -> ResidualBatch:
    spec = BLOCK_CASES[name]
    contract = make_contract(name)
    g = _gen(name)
    t, d, n, k = spec["tokens"], contract.hidden, contract.controller_n, contract.flat_k
    batch = ResidualBatch(
        r_old=_randn(g, t, contract.hc_mult, d).to(torch.bfloat16),
        y_sublayer=_randn(g, t, d).to(torch.bfloat16),
        controller=ControllerParams(
            weight=_randn(g, n, k, scale=1.0 / float(k) ** 0.5),
            alpha_pre=_randn(g, 1, scale=0.5),
            alpha_post=_randn(g, 1, scale=0.5),
            alpha_res=_randn(g, 1, scale=0.5),
            bias=_randn(g, n, scale=0.1),
        ),
        norm=NormParams(gamma=(1.0 + _randn(g, d, scale=0.05)).to(torch.bfloat16)),
        token_id=torch.arange(t, dtype=torch.int64) + 1000,
        contract=contract,
        row_geometry=spec.get("geometry", "packed"),
    ).sealed()
    batch.validate()
    return batch


def make_grads(name: str, batch: ResidualBatch) -> GradBoundary:
    g = _gen(name + ".grad")
    t, d, s = batch.tokens, batch.hidden, batch.contract.hc_mult
    grads = GradBoundary(
        d_r_new=_randn(g, t, s, d).to(torch.bfloat16),
        d_normalized=_randn(g, t, d).to(torch.bfloat16),
        d_residual=_randn(g, t, d, scale=0.25).to(torch.bfloat16),
    )
    grads.validate(batch)
    return grads


def make_sinkhorn_edge_inputs() -> torch.Tensor:
    """Edge inputs for P1-D1: saturating sigmoids, tied logits, degenerate rows.

    Row 0 pushes both sigmoid legs to the flat ends; row 1 makes every COMB
    logit identical (a uniform Sinkhorn fixed point); row 2 is all zeros; row 3
    gives one row of the 4x4 a huge value so the ``sum + eps`` guard is the
    only thing keeping the normalize finite -- the case where swapping in a
    ``clamp`` would change bytes.
    """
    rows = [
        torch.tensor([-30.0, -8.0, 8.0, 30.0] * 6, dtype=torch.float32),
        torch.cat([torch.zeros(8), torch.full((16,), 0.75)]),
        torch.zeros(24, dtype=torch.float32),
        torch.cat(
            [
                torch.tensor([0.5, -0.5, 2.0, -2.0, 1.0, -1.0, 3.0, -3.0]),
                torch.tensor([40.0, -40.0, 0.0, 0.0] + [0.0] * 12),
            ]
        ),
    ]
    return torch.stack(rows)


def make_rms_edge_inputs() -> torch.Tensor:
    """Edge inputs for P1-D5: a zero row (eps is the only guard), tiny and large
    magnitudes, and an exact-power-of-two row where ``rsqrt`` and ``1/sqrt``
    are most likely to agree by accident."""
    rows = [
        torch.zeros(FIXTURE_HIDDEN),
        torch.full((FIXTURE_HIDDEN,), 2.0**-12),
        torch.full((FIXTURE_HIDDEN,), 4.0),
        torch.linspace(-8.0, 8.0, FIXTURE_HIDDEN),
    ]
    return torch.stack(rows).to(torch.bfloat16)


def _run_case(name: str) -> dict[str, str]:
    batch = make_batch(name)
    trace = MHCTrace(numeric_profile=ORACLE_PROFILE)
    r_new, saved = oracle.mhc_block_forward(batch, trace)
    grads = make_grads(name, batch)
    out = oracle.mhc_block_backward(batch, saved, grads, trace)
    hashes = trace.hashes()
    for key, grad in out.items():
        if grad is not None:
            hashes[f"grad.{key}"] = tensor_sha256(grad)
    hashes["r_new"] = tensor_sha256(r_new)
    return hashes


def golden_manifest() -> dict[str, Any]:
    """Recompute every golden hash from seeds with the FP32 oracle."""
    cases: dict[str, dict[str, str]] = {name: _run_case(name) for name in BLOCK_CASES}

    contract = LayerContract(hidden=FIXTURE_HIDDEN)
    h = make_sinkhorn_edge_inputs()
    pre, post, c, saved = oracle.hc_split_sinkhorn_fwd(h, contract)
    g = _gen("sinkhorn_edges.grad")
    dh = oracle.hc_split_sinkhorn_bwd(_randn(g, 4, 4), _randn(g, 4, 4), _randn(g, 4, 4, 4), saved)
    cases["sinkhorn_edges"] = {
        "pre": tensor_sha256(pre),
        "post": tensor_sha256(post),
        "c": tensor_sha256(c),
        "grad.dh": tensor_sha256(dh),
    }

    x = make_rms_edge_inputs()
    gamma = (1.0 + _randn(_gen("rms_edges.gamma"), FIXTURE_HIDDEN, scale=0.05)).to(torch.bfloat16)
    y, residual, rsaved = oracle.rmsnorm_residual_fwd(x, gamma, contract.rmsnorm_eps)
    ge = _gen("rms_edges.grad")
    dx, dgamma = oracle.rmsnorm_residual_bwd(
        _randn(ge, *x.shape).to(torch.bfloat16),
        _randn(ge, *x.shape, scale=0.25).to(torch.bfloat16),
        x,
        gamma,
        rsaved,
    )
    cases["rms_edges"] = {
        "y": tensor_sha256(y),
        "residual": tensor_sha256(residual),
        "grad.dx": tensor_sha256(dx),
        "grad.dgamma": tensor_sha256(dgamma),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "numeric_profile": ORACLE_PROFILE,
        "fixture_hidden": FIXTURE_HIDDEN,
        "cases": cases,
    }


def write_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden_manifest(), indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def slice_batch(batch: ResidualBatch, start: int, stop: int) -> ResidualBatch:
    """A token sub-range of a batch, used by the batch-invariance tests."""
    return replace(
        batch,
        r_old=batch.r_old[start:stop],
        y_sublayer=batch.y_sublayer[start:stop],
        token_id=batch.token_id[start:stop],
        row_geometry="one-row" if stop - start == 1 else "packed",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 golden-hash manifest tool")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    if args.write_manifest:
        print(f"wrote {write_manifest(args.path)}")
    else:
        print(json.dumps(golden_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
