# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Seeded P5 fixtures and the golden-hash manifest (start-kit acceptance data).

Fixtures are regenerated deterministically from seeds; the committed manifest
``tests/fixtures/p5/golden_hashes.json`` anchors the golden bytes in CI. If a
torch upgrade ever changes RNG or libm behavior, the manifest test fails
loudly instead of the goldens drifting silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rl_engine.moe import oracle
from rl_engine.moe.contract import (
    ORACLE_PROFILE,
    SCHEMA_VERSION,
    ExpertBatch,
    LoRAParams,
    SharedBatch,
    tensor_sha256,
)
from rl_engine.moe.mx_format import MXTensor, mx_quantize
from rl_engine.moe.trace import ExpertTrace

FIXTURE_HIDDEN = 128
FIXTURE_FFN = 64
FIXTURE_RANK = 8
BASE_SEED = 2026

DEFAULT_MANIFEST_PATH = Path("tests/fixtures/p5/golden_hashes.json")

E2E_CASES: dict[str, dict[str, Any]] = {
    "base_only_one_row": {"rows": 1, "offsets": [0, 1, 1], "lora": False, "geometry": "one-row"},
    "base_only_packed": {"rows": 24, "offsets": [0, 6, 12, 18, 24], "lora": False},
    "lora_only": {"rows": 24, "offsets": [0, 6, 12, 18, 24], "lora": True, "base_zero": True},
    "base_plus_lora": {"rows": 24, "offsets": [0, 6, 12, 18, 24], "lora": True},
    "uneven_experts": {"rows": 24, "offsets": [0, 0, 17, 17, 24], "lora": True},
}

SHARED_CASES: dict[str, dict[str, Any]] = {
    "shared_t1": {"tokens": 1},
    "shared_t16": {"tokens": 16},
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


def _make_base_weights(
    g: torch.Generator, n_experts: int, zero: bool = False
) -> tuple[MXTensor, MXTensor]:
    h, f = FIXTURE_HIDDEN, FIXTURE_FFN
    scale = 1.0 / float(h) ** 0.5
    w1 = _randn(g, n_experts, 2 * f, h, scale=scale)
    w2 = _randn(g, n_experts, h, f, scale=1.0 / float(f) ** 0.5)
    if zero:
        w1 = torch.zeros_like(w1)
        w2 = torch.zeros_like(w2)
    return mx_quantize(w1, "e2m1"), mx_quantize(w2, "e2m1")


def _make_lora(g: torch.Generator) -> LoRAParams:
    h, f, r = FIXTURE_HIDDEN, FIXTURE_FFN, FIXTURE_RANK
    return LoRAParams(
        a1=_randn(g, r, h, scale=0.1).to(torch.bfloat16),
        b1=_randn(g, 2 * f, r, scale=0.1).to(torch.bfloat16),
        a2=_randn(g, r, f, scale=0.1).to(torch.bfloat16),
        b2=_randn(g, h, r, scale=0.1).to(torch.bfloat16),
        alpha=0.5,
    )


def make_expert_batch(name: str) -> ExpertBatch:
    spec = E2E_CASES[name]
    g = _gen(name)
    rows = spec["rows"]
    offsets = torch.tensor(spec["offsets"], dtype=torch.int32)
    n_experts = offsets.numel() - 1
    w1, w2 = _make_base_weights(g, n_experts, zero=spec.get("base_zero", False))
    lora = _make_lora(g) if spec.get("lora") else None
    batch = ExpertBatch(
        x=_randn(g, rows, FIXTURE_HIDDEN).to(torch.bfloat16),
        expert_offsets=offsets,
        p_s=torch.rand(rows, generator=g, dtype=torch.float32),
        w1=w1,
        w2=w2,
        lora=lora,
        output_slot=torch.arange(rows, dtype=torch.int32),
        row_geometry=spec.get("geometry", "packed"),
    )
    batch = ExpertBatch(
        **{**batch.__dict__, "weight_fingerprint": batch.compute_weight_fingerprint()}
    )
    batch.validate()
    return batch


def make_shared_batch(name: str) -> SharedBatch:
    spec = SHARED_CASES[name]
    g = _gen(name)
    h, f = FIXTURE_HIDDEN, FIXTURE_FFN
    batch = SharedBatch(
        x=_randn(g, spec["tokens"], h).to(torch.bfloat16),
        w_fc1=_randn(g, 2 * f, h, scale=1.0 / float(h) ** 0.5).to(torch.bfloat16),
        w_fc2=_randn(g, h, f, scale=1.0 / float(f) ** 0.5).to(torch.bfloat16),
    )
    batch.validate()
    return batch


def make_grad_output(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    g = _gen(name + ".grad")
    return _randn(g, *shape).to(torch.bfloat16)


def make_act_quant_edge_inputs() -> torch.Tensor:
    """Edge inputs for P5-1 (#60): powers of two, ties, zero rows, subnormal scales."""
    rows = []
    rows.append(torch.tensor([2.0**k for k in range(-16, 16)], dtype=torch.float32))
    rows.append(torch.tensor([17.0, 18.0, 19.0, 20.0] * 8, dtype=torch.float32))
    rows.append(torch.zeros(32, dtype=torch.float32))
    rows.append(torch.linspace(-6.0, 6.0, 32, dtype=torch.float32))
    x = torch.stack(rows)
    return x.to(torch.bfloat16)


def make_swiglu_boundary_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edge inputs for P5-2 (#63): gate/up exactly at, inside, and beyond the clamps."""
    gate_vals = [-12.0, -10.0, -1.0, 0.0, 1.0, 9.5, 10.0, 10.5]
    up_vals = [-10.5, -10.0, -9.5, 0.0, 0.5, 9.5, 10.0, 10.5]
    gate = torch.tensor([gate_vals * 4] * 3, dtype=torch.float32)
    up = torch.tensor([up_vals * 4] * 3, dtype=torch.float32)
    p_s = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    return gate, up, p_s


def _mx_hashes(prefix: str, t: MXTensor) -> dict[str, str]:
    return {f"{prefix}.codes": tensor_sha256(t.codes), f"{prefix}.scales": tensor_sha256(t.scales)}


def golden_manifest() -> dict[str, Any]:
    """Recompute every golden hash from seeds with the FP32 oracle."""
    cases: dict[str, dict[str, str]] = {}
    for name in E2E_CASES:
        batch = make_expert_batch(name)
        trace = ExpertTrace(numeric_profile=ORACLE_PROFILE)
        y, saved = oracle.routed_expert_forward(batch, trace)
        dy = make_grad_output(name, tuple(y.shape))
        grads = oracle.routed_expert_backward(batch, saved, dy, trace)
        hashes = trace.hashes()
        for key, grad in grads.items():
            if grad is not None:
                hashes[f"grad.{key}"] = tensor_sha256(grad)
        cases[name] = hashes
    for name in SHARED_CASES:
        shared = make_shared_batch(name)
        y, saved = oracle.shared_expert_mlp_fwd(shared)
        dy = make_grad_output(name, tuple(y.shape))
        dx = oracle.shared_expert_mlp_bwd(dy, shared, saved)
        cases[name] = {"shared_out": tensor_sha256(y), "grad.dx": tensor_sha256(dx)}
    q_edge = oracle.mxfp8_act_quant_fwd(make_act_quant_edge_inputs())
    cases["act_quant_edges"] = _mx_hashes("act_quant", q_edge)
    gate, up, p_s = make_swiglu_boundary_inputs()
    h, sw_saved = oracle.clamp_swiglu_weighted_fwd(gate, up, p_s)
    dh = make_grad_output("swiglu_boundary", tuple(h.shape))
    dgate, dup, dp_s = oracle.clamp_swiglu_weighted_bwd(dh, sw_saved)
    assert dp_s is not None
    cases["swiglu_boundary"] = {
        "h": tensor_sha256(h),
        "grad.dgate": tensor_sha256(dgate),
        "grad.dup": tensor_sha256(dup),
        "grad.dp_s": tensor_sha256(dp_s),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "numeric_profile": ORACLE_PROFILE,
        "cases": cases,
    }


def write_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden_manifest(), indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 golden-hash manifest tool")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--path", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    if args.write_manifest:
        out = write_manifest(args.path)
        print(f"wrote {out}")
    else:
        print(json.dumps(golden_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
