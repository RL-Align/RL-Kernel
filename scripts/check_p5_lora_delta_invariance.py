# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariance probe for P5-3 shared_grouped_lora_delta (#62).

A row's output must be bit-identical whether it is computed alone or inside a
larger batch. torch.matmul does not guarantee this: cuBLAS picks tile shapes and
split-k strategies from the matrix dimensions, so the accumulation order changes
with M.

A naive probe -- one seed, one shape, only the final output -- is misleading on
three counts, and this script addresses each:

  * The intermediate BF16 cast absorbs most FP32 drift, so a non-deterministic
    GEMM can still produce an identical `y`. Every boundary is checked, not just
    the output, and the drift counts are printed so a real pass can be told
    apart from one that got lucky.
  * A violation only surfaces when a value lands near a BF16 rounding boundary,
    so several seeds are swept.
  * Small K hides the problem entirely: cuBLAS keeps one strategy across all M
    until the matrix is large enough to be worth splitting. Production geometry
    (K=4096, N=2048) is probed alongside the fixture geometry -- swapping a
    deterministic backend for torch.matmul goes undetected at 128x64.

Run with no arguments to compare all three backends in one pass.
"""

from __future__ import annotations

import argparse
import importlib

import torch

_MODULE = "rl_engine.moe.shared_grouped_lora_delta_provider"
DEFAULT_PROVIDERS = (
    f"{_MODULE}:LoRADeltaProvider",
    f"{_MODULE}:LoRADeltaCudaProvider",
    f"{_MODULE}:LoRADeltaTritonProvider",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2, 42, 2026)
SUB_BATCHES = (1, 2, 3, 8, 17)

# (label, M, K, N, r)
SHAPES = (
    ("fixture", 24, 128, 64, 8),
    ("fixture_wide", 24, 128, 128, 8),
    ("mid", 32, 256, 128, 16),
    ("production", 64, 4096, 2048, 8),
)
ALPHA = 0.5

# Boundaries that are per-row and therefore sliceable. dA and dB reduce over
# every row in the batch, so a sub-batch result is a different quantity, not a
# slice of the full one -- they are covered by the rerun check instead.
BOUNDARIES = ("u", "y", "dx")
STAGE = {"u": "fwd GEMM 1", "y": "fwd GEMM 2", "dx": "bwd dX"}


def _drift(sub: torch.Tensor, ref: torch.Tensor) -> tuple[int, float]:
    """How many elements differ, and by how much at worst."""
    diff = (sub.float() - ref.float()).abs()
    return int((diff > 0).sum().item()), float(diff.max().item())


def _operands(m: int, k: int, n: int, r: int, seed: int):
    torch.manual_seed(seed)
    return (
        torch.randn(m, k, dtype=torch.bfloat16, device=DEVICE),
        torch.randn(r, k, dtype=torch.bfloat16, device=DEVICE),
        torch.randn(n, r, dtype=torch.bfloat16, device=DEVICE),
        torch.randn(m, n, dtype=torch.bfloat16, device=DEVICE),
    )


def probe_shape(provider, m_full: int, k: int, n: int, r: int, seed: int) -> list[dict]:
    """Compare every sub-batch against the same rows inside the full batch."""
    x, a, b, dy = _operands(m_full, k, n, r, seed)

    y_full, u_full = provider.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)
    dx_full, _, _ = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, ALPHA, u_full)

    rows = []
    for m in SUB_BATCHES:
        if m > m_full:
            continue
        y_sub, u_sub = provider.shared_grouped_lora_delta_fwd(x[:m], a, b, ALPHA)
        dx_sub, _, _ = provider.shared_grouped_lora_delta_bwd(
            dy[:m], x[:m], a, b, ALPHA, u_sub
        )
        rows.append(
            {
                "m": m,
                "u": torch.equal(u_sub, u_full[:m]),
                "y": torch.equal(y_sub, y_full[:m]),
                "dx": torch.equal(dx_sub, dx_full[:m]),
                "u_drift": _drift(u_sub, u_full[:m]),
                "y_drift": _drift(y_sub, y_full[:m]),
            }
        )
    return rows


def probe_rerun(provider, repeats: int = 3) -> dict[str, bool]:
    """Run-to-run determinism, which also covers dA and dB.

    Batch invariance says nothing about repeating the same call, and dA/dB are
    not sliceable, so this is the only check those two gradients get.
    """
    x, a, b, dy = _operands(64, 4096, 2048, 8, seed=7)
    y0, u0 = provider.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)
    dx0, da0, db0 = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, ALPHA, u0)

    stable = dict.fromkeys(("y", "u", "dX", "dA", "dB"), True)
    for _ in range(repeats):
        y, u = provider.shared_grouped_lora_delta_fwd(x, a, b, ALPHA)
        dx, da, db = provider.shared_grouped_lora_delta_bwd(dy, x, a, b, ALPHA, u)
        for key, got, want in (
            ("y", y, y0), ("u", u, u0), ("dX", dx, dx0), ("dA", da, da0), ("dB", db, db0)
        ):
            stable[key] &= torch.equal(got, want)
    return stable


def resolve(spec: str):
    """Instantiate a provider from 'module.path:ClassName'."""
    module_name, class_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)()


def run_one(spec: str, verbose: bool) -> dict:
    provider = resolve(spec)
    backend = provider.provenance()["actual_backend"]
    print(f"\n{'#' * 78}")
    print(f"# {provider.name}   backend={backend}   device={DEVICE}")
    print(f"{'#' * 78}")

    total = 0
    per_boundary = dict.fromkeys(BOUNDARIES, 0)
    per_shape: dict[str, tuple[int, int]] = {}

    for label, m_full, k, n, r in SHAPES:
        shape_total = shape_failed = 0
        if verbose:
            print(f"\n=== {label}: M={m_full} K={k} N={n} r={r} ===")
            print(
                f"{'seed':>6} {'M':>4} {'u':>7} {'y':>7} {'dX':>7}"
                f"   {'u drift':>16}   {'y drift':>16}"
            )
        for seed in SEEDS:
            for row in probe_shape(provider, m_full, k, n, r, seed):
                total += 1
                shape_total += 1
                if not all(row[key] for key in BOUNDARIES):
                    shape_failed += 1
                for key in BOUNDARIES:
                    if not row[key]:
                        per_boundary[key] += 1
                if verbose:
                    u_cnt, u_mx = row["u_drift"]
                    y_cnt, y_mx = row["y_drift"]
                    print(
                        f"{seed:>6} {row['m']:>4} "
                        f"{str(row['u']):>7} {str(row['y']):>7} {str(row['dx']):>7}   "
                        f"{u_cnt:>5} / {u_mx:.2e}   {y_cnt:>5} / {y_mx:.2e}"
                    )
        per_shape[label] = (shape_failed, shape_total)

    failed = sum(f for f, _ in per_shape.values())
    rerun = probe_rerun(provider)

    print(f"\n--- {provider.name}: {total - failed}/{total} probes invariant ---")
    print("  per shape:")
    for label, (f, t) in per_shape.items():
        print(f"    {label:<14} {t - f:>3}/{t:<3} {'ok' if f == 0 else 'FAIL'}")
    print("  per boundary:")
    for key in BOUNDARIES:
        cnt = per_boundary[key]
        mark = "ok" if cnt == 0 else "FAIL"
        print(f"    {key:>2} ({STAGE[key]:<11}) {total - cnt:>3}/{total} {mark}")
    print("  run-to-run (the only check dA/dB get):")
    bad = [k for k, ok in rerun.items() if not ok]
    print(f"    {'all stable' if not bad else 'UNSTABLE: ' + ', '.join(bad)}")

    return {
        "name": provider.name,
        "total": total,
        "failed": failed,
        "per_boundary": per_boundary,
        "rerun_ok": not bad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        action="append",
        metavar="module.path:ClassName",
        help="repeatable; defaults to all three P5-3 backends",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="summary only")
    args = parser.parse_args()

    specs = args.provider or list(DEFAULT_PROVIDERS)
    results = [run_one(spec, verbose=not args.quiet) for spec in specs]

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"  {'backend':<26} {'invariant':>12}  {'u':>5} {'y':>5} {'dX':>5}  {'rerun':>7}")
    for res in results:
        pb = res["per_boundary"]
        print(
            f"  {res['name']:<26} "
            f"{res['total'] - res['failed']:>5}/{res['total']:<6} "
            f"{res['total'] - pb['u']:>5} {res['total'] - pb['y']:>5} "
            f"{res['total'] - pb['dx']:>5}  "
            f"{'ok' if res['rerun_ok'] else 'FAIL':>7}"
        )

    broken = [r["name"] for r in results if r["failed"] or not r["rerun_ok"]]
    print()
    if broken:
        print(f"RESULT: NOT batch-invariant -- {', '.join(broken)}")
        print(
            "  torch.matmul selects its tile and split-k strategy from M, so a\n"
            "  row's result depends on the batch it was computed in. Routing the\n"
            "  GEMMs through a fixed-K primitive fixes it."
        )
        return 1
    print("RESULT: all probed backends are batch-invariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
