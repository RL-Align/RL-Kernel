#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measure how closely the GDN decode-step goldens track vLLM's providers (RFC #428 C6).

``tests/check_gdn_recurrent_golden.py`` asserts loose regression bounds; this runner
prints the measured values behind them, so the numbers quoted in
``docs/design/rfc428-c6-gdn-recurrent-replay.md`` can be reproduced. Inputs and seeds
are the check file's own helpers, imported from it rather than copied.

Measurements, emitted as one JSON document on stdout:

* ``recurrent``: provider vs golden for the packed recurrent decode, per
  (batch, state dtype): max|d out|, max|d state| and bitwise mismatch counts.
* ``conv``: provider vs golden for ``causal_conv1d_update``, per (batch, cache dtype):
  output mismatch count and max|diff|, and whether the rolled state is bitwise equal.
* ``conv`` again with Triton FP fusion disabled, plus ``conv_kernels``: per compiled
  variant of the provider's conv-update kernel, its ``enable_fp_fusion`` option and
  instruction counts from the PTX and from the SASS.
* ``conv_silu`` localises the fp32-cache conv mismatches, on the same inputs widened to
  fp32 with an fp32 output: the pre-activation comparison (activation off), the SiLU
  comparison with its ULP histogram, and four Triton SiLU variants applied to the
  golden's pre-activation values, each compared bitwise with both sides.
* ``conv_noact_bf16``: the no-activation, bf16-output specialization against the golden,
  whether it equals the fp32-output run rounded to bf16, each mismatch's size, and the
  compiled variants of both specializations; with fusion on and off.

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
import math
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
    # Packed pairs (sm_100), approximate exp and division.
    "fma_rn_f32x2": r"\bfma\.rn\.f32x2\b",
    "mul_f32x2": r"\bmul(\.rn)?\.f32x2\b",
    "add_f32x2": r"\badd(\.rn)?\.f32x2\b",
    "ex2_approx": r"\bex2\.approx",
    "div_full_f32": r"\bdiv\.full\.f32\b",
}
_SASS_OPS = {"FFMA": r"\bFFMA[\w.]*", "FMUL": r"\bFMUL[\w.]*", "FADD": r"\bFADD[\w.]*"}
# Triton SiLU variants for conv_silu, by MODE of _triton_silu's kernel.
_SILU_VARIANTS = ("div_exp", "divrn_exp", "div_libexp", "divrn_libexp")


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


def _compiled_conv_variants() -> dict[tuple[str, Any], Any]:
    """Every compiled variant of vLLM's conv-update kernel in this process, by cache key."""
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as conv_module

    kernel = conv_module._causal_conv1d_update_kernel
    while not hasattr(kernel, "device_caches") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    return {
        (str(device), key): compiled
        for device, cache in kernel.device_caches.items()
        for key, compiled in cache[0].items()
    }


def _conv_kernel_variants(keys: Any = None) -> list[dict[str, Any]] | dict[str, str]:
    """Options and instruction counts per compiled variant (only ``keys``, if given)."""
    try:
        variants = []
        for (device, key), compiled in _compiled_conv_variants().items():
            if keys is not None and (device, key) not in keys:
                continue
            row: dict[str, Any] = {
                "device": device,
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


def _new_variant_keys(before: set[Any]) -> set[Any] | None:
    try:
        return set(_compiled_conv_variants()) - before
    except Exception:  # introspection of Triton internals; report, do not fail
        return None


def _bf16_ulps(diff: float, ref: float) -> float | None:
    """``diff`` in units of one bf16 ULP at the magnitude of ``ref`` (None at zero)."""
    if ref == 0.0:
        return None if diff else 0.0
    return diff / 2.0 ** (math.floor(math.log2(abs(ref))) - 7)


def _conv_noact_bf16(check, batches: list[int]) -> dict[str, Any]:
    """The no-activation conv with a bf16 output, against its fp32-output twin.

    The provider casts x to the fp32 cache dtype before launching, so the two runs
    compute the same thing and differ only in the output dtype, i.e. in which Triton
    specialization runs. Each specialization runs in its own loop so that the compiled
    variants it adds can be told apart.
    """
    import triton

    fp_fusion = bool(triton.knobs.language.default_fp_fusion)
    try:
        before = set(_compiled_conv_variants())
    except Exception:  # introspection of Triton internals; report, do not fail
        before = set()
    bf16_out = {}
    for batch in batches:
        inp = check._conv_inputs(batch, torch.float32, seed=batch)
        bf16_out[batch] = check._run_conv_pair(inp, activation=None)
    bf16_keys = _new_variant_keys(before)
    fp32_out = {}
    for batch in batches:
        inp = check._conv_inputs(batch, torch.float32, seed=batch)
        fp32_out[batch] = check._run_conv_pair(dict(inp, x=inp["x"].float()), activation=None)
    fp32_keys = _new_variant_keys(before | (bf16_keys or set()))

    rows = []
    for batch in batches:
        (b_ref, _), (b_got, _) = bf16_out[batch]
        (d_ref, _), (d_got, _) = fp32_out[batch]
        differ = torch.nonzero(b_ref.view(torch.int16) != b_got.view(torch.int16)).tolist()
        mismatches = []
        for r, ch in differ:
            provider, golden = b_ref[r, ch].item(), b_got[r, ch].item()
            mismatches.append(
                {
                    "abs_out": abs(golden),
                    "abs_diff": abs(provider - golden),
                    "bf16_ulps": _bf16_ulps(abs(provider - golden), golden),
                }
            )
        rows.append(
            {
                "batch": batch,
                "fp_fusion": fp_fusion,
                "out_mismatch_elements": len(differ),
                "out_elements": b_ref.numel(),
                "provider_eq_rne_of_fp32_out": torch.equal(b_ref, d_ref.to(torch.bfloat16)),
                "fp32_out_mismatch_elements": _bit_mismatches(d_got, d_ref),
                "mismatches": mismatches,
            }
        )
    return {
        "batches": rows,
        "bf16_out_kernels": _conv_kernel_variants(bf16_keys) if bf16_keys is not None else None,
        "fp32_out_kernels": _conv_kernel_variants(fp32_keys) if fp32_keys is not None else None,
    }


def _triton_silu():
    """``apply(x, mode)``: one of the four SiLU formulations, evaluated by Triton."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def silu(x_ptr, y_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if MODE == 0:  # the provider's source: x / (1 + tl.exp(-x))
            y = x / (1 + tl.exp(-x))
        elif MODE == 1:
            y = tl.math.div_rn(x, 1 + tl.exp(-x))
        elif MODE == 2:
            y = x / (1 + libdevice.exp(-x))
        else:
            y = tl.math.div_rn(x, 1 + libdevice.exp(-x))
        tl.store(y_ptr + offs, y, mask=mask)

    def apply(x: torch.Tensor, mode: int) -> torch.Tensor:
        flat = x.contiguous().view(-1)
        y = torch.empty_like(flat)
        silu[(triton.cdiv(flat.numel(), 1024),)](flat, y, flat.numel(), MODE=mode, BLOCK=1024)
        return y.view_as(x)

    return apply


def _ulp_histogram(a: torch.Tensor, b: torch.Tensor) -> dict[str, int]:
    """Differing fp32 elements by bit-pattern distance."""
    d = (a.contiguous().view(torch.int32).long() - b.contiguous().view(torch.int32).long()).abs()
    d = d[d != 0]
    return {
        "1": int((d == 1).sum()),
        "2": int((d == 2).sum()),
        "3-4": int(((d == 3) | (d == 4)).sum()),
        ">4": int((d > 4).sum()),
    }


def _conv_silu(check, batches: list[int]) -> list[dict[str, Any]]:
    """Where the fp32-cache conv mismatches come from, on fp32 inputs and outputs."""
    silu = _triton_silu()
    rows = []
    for batch in batches:
        inp = check._conv_inputs(batch, torch.float32, seed=batch)
        inp32 = dict(inp, x=inp["x"].float())  # the same values, kept in fp32 throughout
        (pre_ref, _), (pre_got, _) = check._run_conv_pair(inp32, activation=None)
        (out_ref, _), (out_got, _) = check._run_conv_pair(inp32)
        variants = {name: silu(pre_got, mode) for mode, name in enumerate(_SILU_VARIANTS)}
        rows.append(
            {
                "batch": batch,
                "elements": out_ref.numel(),
                "preactivation_mismatch_elements": _bit_mismatches(pre_got, pre_ref),
                "silu_mismatch_elements": _bit_mismatches(out_got, out_ref),
                "silu_ulp_histogram": _ulp_histogram(out_got, out_ref),
                # Applied to the golden's pre-activation values.
                "triton_silu_vs_provider": {
                    k: _bit_mismatches(v, out_ref) for k, v in variants.items()
                },
                "triton_silu_vs_golden": {
                    k: _bit_mismatches(v, out_got) for k, v in variants.items()
                },
            }
        )
    return rows


def _conv_report(check, batches: list[int]) -> dict[str, Any]:
    conv = _measure_conv(check, batches)
    kernels = _conv_kernel_variants()  # before the no-activation runs add their own
    return {
        "conv": conv,
        "conv_kernels": kernels,
        "conv_noact_bf16": _conv_noact_bf16(check, batches),
    }


def _arm_variants(arm: dict[str, Any]) -> list[Any] | None:
    """Every compiled variant an arm reports, or None if any listing is unavailable."""
    noact = arm["conv_noact_bf16"]
    lists = [arm["conv_kernels"], noact["bf16_out_kernels"], noact["fp32_out_kernels"]]
    if not all(isinstance(x, list) for x in lists):
        return None
    return [v for x in lists for v in x]


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
            "conv_silu": "the conv inputs with x widened to fp32; fp32 cache and output",
            "conv_noact_bf16": "the conv inputs with activation=None; fp32 cache",
        },
        "recurrent": _measure_recurrent(check, batches),
    }
    fused = _conv_report(check, batches)
    report["conv"] = fused["conv"]
    report["conv_kernels"] = {"fusion_on": fused["conv_kernels"]}
    report["conv_noact_bf16"] = {"fusion_on": fused["conv_noact_bf16"]}
    report["conv_silu"] = _conv_silu(check, batches)
    if not args.skip_fusion_off:
        unfused = _conv_report_without_fusion(args.batches)
        report["conv"] += unfused["conv"]
        report["conv_kernels"]["fusion_off"] = unfused["conv_kernels"]
        report["conv_noact_bf16"]["fusion_off"] = unfused["conv_noact_bf16"]
        report["fusion_check"] = _fusion_check(_arm_variants(fused), _arm_variants(unfused))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
