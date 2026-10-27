#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Direct bitwise comparison of one RFC #420 operator with its reuse candidates.

``h3_prior_art.py`` reports each implementation's batch invariance and its error against FP64.
Matching error summaries do not show that two implementations produce the same bits, so before
a candidate is adopted this runner compares outputs directly. Every candidate that
``h3_prior_art.candidates()`` defines runs on the same inputs (each global batch-invariant mode
in its own process, as in ``h3_prior_art.py``), and every forward output and leaf gradient is
compared byte for byte with diffusers' path and with the rl-kernel op:

    python tools/validation/models/h3_bitwise_reuse.py --op norm_modulate \\
        --out reports/experiments/h3-bitwise-reuse-b200/norm_modulate.json \\
        [--megatron-src /path/to/Megatron-LM] [--fast-math-variant]
    python tools/validation/models/plot_h3_bitwise_reuse.py \\
        reports/experiments/h3-bitwise-reuse-b200/*.json --out figure.png

For ``timestep_sinusoid`` the inputs are three timestep sets: 2^20 uniform draws, every 997th
FP32 bit pattern in [0, 1] (a dense sweep, mostly tiny values), and a 100001-point grid.
``--fast-math-variant`` also compiles ``csrc/cuda/h3/timestep_sinusoid.cu`` with
``--use_fast_math`` and compares it, to show what the fast-math build guard prevents. Other
operators use each candidate's own input generator at four sizes.

Intermediate tensors are written next to the report (not to /tmp, which cluster job epilogs
can clear) and removed afterwards. Run on a clean tree; the report records the commit, the tree
state and every library version.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h3_prior_art as PA
import torch

SEED = 7
SIZES = {
    "timestep_mlp": (1, 4, 64, 2048),
    "adaln_projection": (1, 4, 64, 2048),
    "adaln_row_gather": (1, 257, 4097, 32768),
    "norm_modulate": (1, 257, 4097, 32768),
    "gate_residual": (1, 257, 4097, 32768),
    "final_adaln_out": (1, 257, 4097, 32768),
}
SINUSOID_SETS = ("random_2^20", "dense_fp32_sweep", "grid_100001")
_FAST_MATH_BINDING = """
#include <torch/extension.h>
torch::Tensor h3_timestep_sinusoid_forward(torch::Tensor, int64_t, double, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &h3_timestep_sinusoid_forward); }
"""


def _sinusoid_inputs(name: str) -> dict[str, torch.Tensor]:
    if name == "random_2^20":
        t = torch.rand(1 << 20, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    elif name == "dense_fp32_sweep":
        bits = torch.arange(0, 0x3F800001, 997, dtype=torch.int32, device="cuda")
        t = bits.view(torch.float32)
    else:
        t = torch.linspace(0, 1, 100001, device="cuda")
    return {"t": t}


def _fast_math_sinusoid(build_dir: Path):
    from torch.utils.cpp_extension import load

    build_dir.mkdir(parents=True, exist_ok=True)
    binding = build_dir / "binding.cpp"
    binding.write_text(_FAST_MATH_BINDING)
    root = Path(__file__).resolve().parents[3]
    mod = load(
        name="h3_timestep_sinusoid_fast_math",
        sources=[str(root / "csrc/cuda/h3/timestep_sinusoid.cu"), str(binding)],
        build_directory=str(build_dir),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    return {
        "source": "rl-kernel timestep_sinusoid.cu compiled with --use_fast_math",
        "grads": [],
        "fn": lambda t: mod.forward(t, PA.F1, 10000.0, False),
    }


def _run(cand: dict[str, Any], inputs: dict[str, torch.Tensor]):
    args = {}
    for k, v in inputs.items():
        args[k] = v.detach().clone().requires_grad_(True) if k in cand["grads"] else v
    out = cand["fn"](**args)
    outs = list(out) if isinstance(out, (tuple, list)) else [out]
    live = [o for o in outs if o.requires_grad]
    if live:
        gen = torch.Generator("cuda").manual_seed(SEED)
        ups = [torch.randn(o.shape, device="cuda", generator=gen).to(o.dtype) for o in live]
        torch.autograd.backward(live, ups)
    grads = {k: args[k].grad.cpu() if args[k].grad is not None else None for k in cand["grads"]}
    return [o.detach().cpu() for o in outs], grads


def dump(op: str, mode: str, parts: Path, megatron_src: str | None, fast_math: bool) -> None:
    """Child process: run every candidate of one mode and save its outputs."""

    try:
        PA.enable_mode(mode, megatron_src)
    except Exception as exc:  # noqa: BLE001
        (parts / f"{mode}.unavailable").write_text(f"{type(exc).__name__}: {exc}"[:300])
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cands = PA.candidates(op, mode)
    if fast_math and op == "timestep_sinusoid" and mode == "plain":
        try:
            cands.append({"name": "rl_kernel_fast_math", **_fast_math_sinusoid(parts / "build")})
        except Exception as exc:  # noqa: BLE001
            (parts / "rl_kernel_fast_math.unavailable").write_text(
                f"{type(exc).__name__}: {exc}"[:300]
            )
    for cand in cands:
        name = cand.pop("name")
        if "unavailable" in cand:
            (parts / f"{name}.unavailable").write_text(cand["unavailable"][:300])
            continue
        cases = SINUSOID_SETS if op == "timestep_sinusoid" else SIZES[op]
        for case in cases:
            inputs = (
                _sinusoid_inputs(case) if op == "timestep_sinusoid" else cand["make"](SEED, case)
            )
            outs, grads = _run(cand, inputs)
            torch.save(
                {"source": cand["source"], "out": outs, "grads": grads},
                parts / f"{name}__{case}.pt",
            )
        print(f"  [{mode}] {name}: done", flush=True)
        torch.cuda.empty_cache()


def _same(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any] | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"equal": False, "note": f"{a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"}
    width = a.element_size()
    differ = (
        a.contiguous().view(torch.uint8).view(-1, width)
        != b.contiguous().view(torch.uint8).view(-1, width)
    ).any(1)
    return {
        "equal": not bool(differ.any()),
        "differ": int(differ.sum()),
        "numel": a.numel(),
        "max_abs": float((a.double() - b.double()).abs().max()),
    }


def _compare_case(run: dict[str, Any], ref_run: dict[str, Any] | None) -> dict[str, Any]:
    """Compare one case with the reference's run of it.

    A case the reference never saved is unavailable.

    A reference child can die after saving only some of its cases (an out-of-memory at the largest
    size, say); the comparisons that did complete must still reach the report.
    """

    if ref_run is None:
        return {"unavailable": "the reference run has no saved result for this case"}
    if len(run["out"]) != len(ref_run["out"]):
        # Outputs that have no counterpart are a difference, not something to drop: a zip would
        # stop at the shorter list and could report the case as bitwise equal.
        outs: list[dict[str, Any] | None] = [
            {"equal": False, "note": f"{len(run['out'])} vs {len(ref_run['out'])} outputs"}
        ]
    else:
        outs = [_same(a, b) for a, b in zip(run["out"], ref_run["out"], strict=True)]
    return {
        "out": outs,
        "grads": {g: _same(run["grads"].get(g), ref_run["grads"].get(g)) for g in run["grads"]},
    }


def compare(parts: Path) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for f in sorted(parts.glob("*.pt")):
        name, case = f.stem.rsplit("__", 1)
        runs.setdefault(name, {})[case] = torch.load(f)
    refs = [r for r in ("diffusers[plain]", "diffusers", "rl_kernel") if r in runs]
    out: dict[str, Any] = {}
    for name, by_case in runs.items():
        entry: dict[str, Any] = {"source": next(iter(by_case.values()))["source"]}
        for ref in refs:
            if ref == name:
                continue
            entry[f"vs {ref}"] = {
                case: _compare_case(run, runs[ref].get(case)) for case, run in by_case.items()
            }
        out[name] = entry
    for f in parts.glob("*.unavailable"):
        out[f.stem] = {"unavailable": f.read_text()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--op", required=True, choices=sorted(PA.OPS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--megatron-src", default=None, help="Megatron-LM source tree (optional)")
    parser.add_argument(
        "--fast-math-variant",
        action="store_true",
        help="timestep_sinusoid only: also compare a --use_fast_math build of the kernel",
    )
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if importlib.util.find_spec(PA.OPS[args.op]) is None:
        raise SystemExit(f"{args.op}: {PA.OPS[args.op]} is not on this branch")
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    parts = args.out.with_name(f".{args.out.stem}.parts")

    if args.mode is not None:  # child process: one global mode
        dump(args.op, args.mode, parts, args.megatron_src, args.fast_math_variant)
        return

    from rl_engine.validation.models.h3_chain import environment, git_state

    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True)
    modes = list(PA.MODES) if args.op in PA.GEMM_OPS else ["plain"]
    for mode in modes:
        print(f"[{args.op}] mode {mode}", flush=True)
        cmd = [sys.executable, __file__, "--op", args.op, "--out", str(args.out), "--mode", mode]
        if args.megatron_src:
            cmd += ["--megatron-src", args.megatron_src]
        if args.fast_math_variant:
            cmd.append("--fast-math-variant")
        proc = subprocess.run(cmd, env={**os.environ, **PA.MODES[mode]})
        if proc.returncode != 0:
            (parts / f"{mode}.unavailable").write_text(f"subprocess exited with {proc.returncode}")
    results = compare(parts)
    shutil.rmtree(parts, ignore_errors=True)
    report = {
        "kind": "h3_bitwise_reuse",
        "rfc": "RL-Align/RL-Kernel#420",
        "op": args.op,
        **git_state(),
        "environment": {
            **environment(),
            "libraries": {lib: PA._version(lib) for lib in PA.LIBS},
            "mode_environment": {m: PA.MODES[m] for m in modes},
        },
        "cases": list(SINUSOID_SETS if args.op == "timestep_sinusoid" else SIZES[args.op]),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out} (commit {report['rl_kernel_commit'][:7]}, "
        f"dirty={report['tracked_tree_dirty']})"
    )


if __name__ == "__main__":
    main()
