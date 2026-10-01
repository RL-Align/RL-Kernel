#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measure how closely the GDN decode-step goldens track vLLM's providers (RFC #428 C6).

``tests/check_gdn_recurrent_golden.py`` asserts loose regression bounds; this runner
prints the measured values behind them, so the numbers quoted in
``docs/design/rfc428-c6-gdn-recurrent-replay.md`` can be reproduced. Inputs and seeds
are the check file's own helpers, imported from it rather than copied.

Three measurements, emitted as one JSON document on stdout:

* ``recurrent``: provider vs golden for the packed recurrent decode, per
  (batch, state dtype): max|d out|, max|d state| and bitwise mismatch counts.
* ``conv``: provider vs golden for ``causal_conv1d_update``, per (batch, cache dtype):
  output mismatch count and max|diff|, and whether the rolled state is bitwise equal.
* ``conv`` again with Triton FP fusion disabled, plus ``conv_kernels``: per compiled
  variant of the provider's conv-update kernel, its ``enable_fp_fusion`` option and
  instruction counts from the PTX and from the SASS.

The fusion-off arm runs in a child process with ``TRITON_DEFAULT_FP_FUSION=0`` and a
private ``TRITON_CACHE_DIR``. Flipping ``triton.knobs.language.default_fp_fusion``
in-process does not work: Triton's in-memory kernel cache is keyed on the launch
kwargs, and the knob is read only after a cache miss, so the fused variant is reused.

Count FMAs in the SASS, not only the PTX. With fusion on, Triton emits plain
``mul.f32``/``add.f32`` and leaves ptxas free to contract them into ``FFMA``; with
fusion off it passes ``--fmad=false`` to ptxas. A PTX with no ``fma.rn.f32`` can still
run FMAs. ``fusion_check`` reports whether each arm compiled what it claims to.

Requires CUDA and vLLM 0.30.0. Run it from a clean checkout so ``git_dirty`` is false,
and write the result outside the checkout: an untracked file inside it would itself
make the next run report ``git_dirty: true``.

    python scripts/ws1_gdn_provider_agreement.py > "${TMPDIR:-/tmp}/gdn_provider_agreement.json"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CHECK_FILE = REPO_ROOT / "tests" / "check_gdn_recurrent_golden.py"
_STATE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
_INT_VIEW = {2: torch.int16, 4: torch.int32}
# FLA_USE_FAST_OPS swaps the kernel's exp/log for fast_expf/fast_logf;
# TRITON_DEFAULT_FP_FUSION decides whether ptxas may contract mul/add.
_PROVENANCE_ENV = ("FLA_USE_FAST_OPS", "TRITON_DEFAULT_FP_FUSION")
# Opcode patterns. PTX: a rounding-qualified op (".rn") may not be contracted by ptxas;
# the plain form may. SASS: count opcode tokens, including modifiers such as FFMA.FTZ.
_PTX_OPS = {
    "fma_rn_f32": r"\bfma\.rn\.f32\b",
    "mul_f32": r"\bmul\.f32\b",
    "add_f32": r"\badd\.f32\b",
    "mul_rn_f32": r"\bmul\.rn\.f32\b",
    "add_rn_f32": r"\badd\.rn\.f32\b",
}
_SASS_OPS = {"FFMA": r"\bFFMA[\w.]*", "FMUL": r"\bFMUL[\w.]*", "FADD": r"\bFADD[\w.]*"}


def _load_check_module():
    """Import the check file by path; ``tests/`` is not a package."""
    spec = importlib.util.spec_from_file_location("check_gdn_recurrent_golden", _CHECK_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bit_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Elements whose bit patterns differ (NaN-safe, unlike ``a != b``)."""
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    view = _INT_VIEW[a.element_size()]
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum())


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _git(*args: str) -> str | None:
    """Stripped stdout, or ``None`` if git is missing or the command fails."""
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance() -> dict[str, Any]:
    import triton
    import vllm

    import rl_engine

    status = _git("status", "--porcelain")
    return {
        # None means "unknown" (no git, or not a checkout), never "clean".
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        # Which tree was imported, and the env knobs that change the provider's code.
        "rl_engine_file": rl_engine.__file__,
        "env": {key: os.environ.get(key) for key in _PROVENANCE_ENV},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "vllm": vllm.__version__,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "default_fp_fusion": bool(triton.knobs.language.default_fp_fusion),
    }


def _measure_recurrent(check, batches: list[int]) -> list[dict[str, Any]]:
    bounds = {name: check._RECURRENT_BOUNDS[dtype] for name, dtype in _STATE_DTYPES.items()}
    rows = []
    for batch in batches:
        for name, state_dtype in _STATE_DTYPES.items():
            # Same call as test_golden_matches_packed_decode_provider.
            inp = check._inputs(batch, batch + 2, state_dtype, torch.bfloat16, seed=batch)
            out_ref, state_ref = check._run_provider(inp)
            out_got, state_got = check._run_golden(inp)
            rows.append(
                {
                    "batch": batch,
                    "state_dtype": name,
                    "max_abs_diff_out": _max_abs_diff(out_got, out_ref),
                    "max_abs_diff_state": _max_abs_diff(state_got, state_ref),
                    "out_mismatch_elements": _bit_mismatches(out_got, out_ref),
                    "out_elements": out_ref.numel(),
                    "state_mismatch_elements": _bit_mismatches(state_got, state_ref),
                    "state_elements": state_ref.numel(),
                    "asserted_out_atol": bounds[name][0],
                    "asserted_state_atol": bounds[name][1],
                }
            )
    return rows


def _measure_conv(check, batches: list[int]) -> list[dict[str, Any]]:
    import triton

    fp_fusion = bool(triton.knobs.language.default_fp_fusion)  # this process's setting
    rows = []
    for batch in batches:
        for name, cache_dtype in _STATE_DTYPES.items():
            # Same call as the test_conv_* provider comparisons.
            inp = check._conv_inputs(batch, cache_dtype, seed=batch)
            (out_ref, state_ref), (out_got, state_got) = check._run_conv_pair(inp)
            rows.append(
                {
                    "batch": batch,
                    "cache_dtype": name,
                    "fp_fusion": fp_fusion,
                    "out_mismatch_elements": _bit_mismatches(out_got, out_ref),
                    "out_elements": out_ref.numel(),
                    "max_abs_diff_out": _max_abs_diff(out_got, out_ref),
                    "state_bitwise_equal": torch.equal(state_got, state_ref),
                }
            )
    return rows


def _count(patterns: dict[str, str], text: str) -> dict[str, int]:
    return {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}


def _conv_kernel_variants() -> list[dict[str, Any]] | dict[str, str]:
    """Options and instruction counts per compiled variant of the conv-update kernel."""
    try:
        from vllm.model_executor.layers.mamba.ops import causal_conv1d as conv_module

        kernel = conv_module._causal_conv1d_update_kernel
        while not hasattr(kernel, "device_caches") and hasattr(kernel, "fn"):
            kernel = kernel.fn
        variants = []
        for device, cache in kernel.device_caches.items():
            for compiled in cache[0].values():
                row: dict[str, Any] = {
                    "device": str(device),
                    "enable_fp_fusion": getattr(compiled.metadata, "enable_fp_fusion", None),
                    "ptx": _count(_PTX_OPS, compiled.asm["ptx"]),
                }
                try:
                    row["sass"] = _count(_SASS_OPS, compiled.asm["sass"])
                except Exception as exc:  # needs cuobjdump; report, do not fail
                    row["sass"] = {"unavailable": f"{type(exc).__name__}: {exc}"}
                variants.append(row)
        return variants
    except Exception as exc:  # introspection of Triton internals; report, do not fail
        return {"unavailable": f"{type(exc).__name__}: {exc}"}


def _conv_report(check, batches: list[int]) -> dict[str, Any]:
    return {"conv": _measure_conv(check, batches), "conv_kernels": _conv_kernel_variants()}


def _conv_report_without_fusion(batches: str) -> dict[str, Any]:
    """Rerun the conv arm in a child process that compiles with fusion off."""
    with tempfile.TemporaryDirectory(prefix="triton-nofusion-") as cache_dir:
        env = dict(os.environ, TRITON_DEFAULT_FP_FUSION="0", TRITON_CACHE_DIR=cache_dir)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--conv-only", "--batches", batches],
            env=env,
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
    return json.loads(child.stdout)


def _fusion_check(on: Any, off: Any) -> dict[str, Any]:
    """Whether each arm's compiled variants carry the fusion setting it claims."""

    def flags(variants: Any) -> list[Any] | None:
        if not isinstance(variants, list):
            return None
        return [v["enable_fp_fusion"] for v in variants]

    on_flags, off_flags = flags(on), flags(off)
    return {
        "fusion_on_variants": on_flags,
        "fusion_off_variants": off_flags,
        "fusion_off_effective": bool(off_flags) and all(f is False for f in off_flags),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batches",
        default="1,4,17,64",
        help="comma-separated batch sizes (default: the check file's 1,4,17,64)",
    )
    parser.add_argument(
        "--skip-fusion-off",
        action="store_true",
        help="do not rerun the conv comparison with Triton FP fusion disabled",
    )
    # Internal: the fusion-off child process prints only the conv arm.
    parser.add_argument("--conv-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required", file=sys.stderr)
        return 2
    try:
        import vllm  # noqa: F401
    except ImportError:
        print("vLLM is required", file=sys.stderr)
        return 2

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    check = _load_check_module()
    if args.conv_only:
        print(json.dumps(_conv_report(check, batches), sort_keys=True))
        return 0

    report: dict[str, Any] = {
        "runner": "scripts/ws1_gdn_provider_agreement.py",
        "provenance": _provenance(),
        "inputs": {
            "source": "tests/check_gdn_recurrent_golden.py (_inputs, _conv_inputs)",
            "seed": "seed=batch for every case",
            "batches": batches,
            "recurrent": "bf16 I/O, use_qk_l2norm_in_kernel=True, num_blocks=batch+2",
            "conv": "bias=True, activation=silu, dim_first=True, W=4, dim=8192",
        },
        "recurrent": _measure_recurrent(check, batches),
    }
    fused = _conv_report(check, batches)
    report["conv"] = fused["conv"]
    report["conv_kernels"] = {"fusion_on": fused["conv_kernels"]}
    if not args.skip_fusion_off:
        unfused = _conv_report_without_fusion(args.batches)
        report["conv"] += unfused["conv"]
        report["conv_kernels"]["fusion_off"] = unfused["conv_kernels"]
        report["fusion_check"] = _fusion_check(fused["conv_kernels"], unfused["conv_kernels"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
