#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Accuracy, row invariance and latency of the Qwen3-Next norms vs existing implementations.

Writes one JSON report (RFC #428, reuse rule of #420): every candidate is compared with
an FP64 golden, checked for row invariance (a row computed alone vs inside a batch,
bitwise), and timed. Optional providers (transformers, vLLM, FlashInfer) are skipped
when they are not installed; the report says which ran.

    python tools/validation/models/qwen3_next_norm_evidence.py --out report.json
    python tools/validation/models/plot_qwen3_next_norm_evidence.py report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.backends.cuda.norm.rmsnorm import (  # noqa: E402
    Qwen3NextRMSNormCudaOp,
    Qwen3NextRMSNormGatedCudaOp,
)
from rl_engine.reference.norm.qwen3_next_rms_norm import (  # noqa: E402
    Qwen3NextRMSNormGatedOp,
    Qwen3NextRMSNormOp,
)

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Ops. Each candidate is fn(*row_inputs, weight) -> y, plus whether it has a
# backward. Row inputs are sliced together for the row-invariance check.
# --------------------------------------------------------------------------- #


def zero_centred_candidates(hidden: int) -> dict[str, dict[str, Any]]:
    cands: dict[str, dict[str, Any]] = {
        "rl-kernel CUDA": {"fn": Qwen3NextRMSNormCudaOp(), "backward": True},
        "rl-kernel PyTorch reference": {"fn": Qwen3NextRMSNormOp(), "backward": True},
    }
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

        module = Qwen3NextRMSNorm(hidden, eps=EPS).cuda()

        def hf(x, w, module=module):
            return torch.func.functional_call(module, {"weight": w}, (x,))

        cands["transformers Qwen3NextRMSNorm"] = {"fn": hf, "backward": True}
    except ImportError:
        pass
    try:
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            from vllm.model_executor.layers.layernorm import GemmaRMSNorm

            gemma = GemmaRMSNorm(hidden, eps=EPS).cuda()

        def vllm_fwd(x, w, gemma=gemma):
            gemma.weight.data = w.detach()
            return gemma.forward_cuda(x)

        cands["vLLM GemmaRMSNorm (forward only)"] = {"fn": vllm_fwd, "backward": False}
    except ImportError:
        pass
    try:
        import flashinfer

        cands["FlashInfer gemma_rmsnorm (forward only)"] = {
            "fn": lambda x, w: flashinfer.norm.gemma_rmsnorm(x, w, EPS),
            "backward": False,
        }
    except ImportError:
        pass
    return cands


def zero_centred_golden(x, w):
    x64 = x.double()
    return x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + EPS) * (1.0 + w.double())


def gated_candidates(hidden: int) -> dict[str, dict[str, Any]]:
    cuda_op, ref_op = Qwen3NextRMSNormGatedCudaOp(), Qwen3NextRMSNormGatedOp()
    cands: dict[str, dict[str, Any]] = {
        "rl-kernel CUDA": {"fn": lambda x, g, w: cuda_op(x, w, g, eps=EPS), "backward": True},
        "rl-kernel PyTorch reference": {
            "fn": lambda x, g, w: ref_op(x, w, g, eps=EPS),
            "backward": True,
        },
    }
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

        module = Qwen3NextRMSNormGated(hidden, eps=EPS).cuda()

        def hf(x, g, w, module=module):
            return torch.func.functional_call(module, {"weight": w}, (x, g))

        cands["transformers Qwen3NextRMSNormGated (cast-first)"] = {"fn": hf, "backward": True}
    except ImportError:
        pass
    try:
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            from vllm.model_executor.layers.layernorm import RMSNormGated

            gated = RMSNormGated(hidden, eps=EPS, norm_before_gate=True).cuda()

        def vllm_fwd(x, g, w, gated=gated):
            gated.weight.data = w.detach()
            return gated.forward_cuda(x, g)

        cands["vLLM RMSNormGated (forward only)"] = {"fn": vllm_fwd, "backward": False}
    except ImportError:
        pass
    return cands


def gated_golden(x, g, w):
    """vLLM's convention (the one #468 implements) in FP64: x * rstd * w * silu(gate)."""

    x64 = x.double()
    return (x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + EPS) * w.double()) * F.silu(
        g.double()
    )


OPS: dict[str, dict[str, Any]] = {
    "zero_centred_rmsnorm": {
        "hidden": 2048,  # decoder and final norms
        "row_inputs": 1,
        "weight": lambda h, gen: torch.randn(h, device="cuda", generator=gen) * 0.1,
        "candidates": zero_centred_candidates,
        "golden": zero_centred_golden,
        "timed_rows": (1024, 4096, 16384, 65536),
    },
    "gated_rmsnorm": {
        "hidden": 128,  # GDN value head dim; rows are tokens x heads
        "row_inputs": 2,
        "weight": lambda h, gen: 1.0 + torch.randn(h, device="cuda", generator=gen) * 0.1,
        "candidates": gated_candidates,
        "golden": gated_golden,
        "timed_rows": (4096, 16384, 65536, 262144),
    },
}


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


def _inputs(spec, rows: int, seed: int, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = spec["hidden"]
    row_inputs = [(torch.randn(rows, h, device="cuda", generator=g) * 2).to(dtype)]
    for _ in range(spec["row_inputs"] - 1):
        row_inputs.append(torch.randn(rows, h, device="cuda", generator=g).to(dtype))
    w = spec["weight"](h, g).to(dtype)
    up = torch.randn(rows, h, device="cuda", generator=g).to(dtype)
    return row_inputs, w, up


def _grads(fn, row_inputs, w, up, dtype=None):
    def leaf(t):
        return (t if dtype is None else t.to(dtype)).detach().clone().requires_grad_(True)

    rl, wl = [leaf(t) for t in row_inputs], leaf(w)
    out = fn(*rl, wl)
    out.backward(up if dtype is None else up.to(dtype))
    return out.detach(), [t.grad for t in rl], wl.grad


def accuracy(spec, cands, rows: int, seed: int) -> dict[str, Any]:
    row_inputs, w, up = _inputs(spec, rows, seed)
    ref_out, ref_drows, ref_dw = _grads(spec["golden"], row_inputs, w, up, torch.float64)
    result = {}
    for name, c in cands.items():
        entry: dict[str, Any] = {}
        if c["backward"]:
            out, drows, dw = _grads(c["fn"], row_inputs, w, up)
            pairs = [("dx", drows[0], ref_drows[0]), ("dweight", dw, ref_dw)]
            if len(drows) > 1:
                pairs.append(("dgate", drows[1], ref_drows[1]))
            for key, got, ref in pairs:
                err = (got.double() - ref).abs().max().item()
                entry[f"{key}_max_abs_over_absmax"] = err / ref.abs().max().item()
        else:
            with torch.no_grad():
                out = c["fn"](*row_inputs, w)
        err = (out.double() - ref_out).abs()
        entry["forward_max_abs"] = err.max().item()
        entry["forward_correctly_rounded_fraction"] = (
            (out == ref_out.to(out.dtype)).float().mean().item()
        )
        result[name] = entry
    return result


def row_invariance(spec, cands, seeds=(3, 4, 5), rows: int = 4096, step: int = 16):
    """256 rows computed alone vs the same rows inside a batch, bitwise."""

    result = {}
    for name, c in cands.items():
        fwd_bad = dx_bad = checked = 0
        for seed in seeds:
            row_inputs, w, up = _inputs(spec, rows, seed)
            if c["backward"]:
                full_out, full_drows, _ = _grads(c["fn"], row_inputs, w, up)
            else:
                with torch.no_grad():
                    full_out = c["fn"](*row_inputs, w)
            for i in range(0, rows, step):
                part = [t[i : i + 1] for t in row_inputs]
                if c["backward"]:
                    out, drows, _ = _grads(c["fn"], part, w, up[i : i + 1])
                    dx_bad += not all(torch.equal(d[0], fd[i]) for d, fd in zip(drows, full_drows))
                else:
                    with torch.no_grad():
                        out = c["fn"](*part, w)
                fwd_bad += not torch.equal(out[0], full_out[i])
                checked += 1
        result[name] = {
            "rows_checked": checked,
            "forward_rows_differing": fwd_bad,
            "dx_rows_differing": dx_bad if c["backward"] else None,
            "batch_rows": rows,
            "seeds": list(seeds),
        }
    return result


def _time_us(fn: Callable[[], Any], warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def latency(spec, cands) -> dict[str, Any]:
    result: dict[str, Any] = {name: {} for name in cands}
    for rows in spec["timed_rows"]:
        row_inputs, w, up = _inputs(spec, rows, seed=11)
        for name, c in cands.items():
            row: dict[str, float] = {}
            with torch.no_grad():
                row["forward_us"] = _time_us(lambda f=c["fn"]: f(*row_inputs, w))
            if c["backward"]:
                leaves = [t.detach().clone().requires_grad_(True) for t in [*row_inputs, w]]
                out = c["fn"](*leaves)
                row["backward_us"] = _time_us(
                    lambda o=out, lv=leaves: torch.autograd.grad(o, lv, up, retain_graph=True)
                )
            result[name][str(rows)] = row
    return result


# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment() -> dict[str, Any]:
    env = {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    for mod in ("transformers", "vllm", "flashinfer"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = None
    return env


def run_op(name: str, spec) -> dict[str, Any]:
    cands = spec["candidates"](spec["hidden"])
    print(f"[{name}] candidates: {', '.join(cands)}", flush=True)
    report = {
        "hidden": spec["hidden"],
        "accuracy": {str(r): accuracy(spec, cands, r, seed=r) for r in (257, 4096)},
        "row_invariance": row_invariance(spec, cands),
        "latency": latency(spec, cands),
    }
    for cand, entry in report["row_invariance"].items():
        print(f"  BI {cand}: {entry}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ops", default=",".join(OPS), help="comma list of ops")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "kind": "qwen3_next_norm_evidence",
        "rfc": "RL-Align/RL-Kernel#428",
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "environment": environment(),
        "eps": EPS,
        "dtype": "bfloat16",
        "ops": {name: run_op(name, OPS[name]) for name in args.ops.split(",")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} (commit {report['git_commit'][:7]}, dirty={report['git_dirty']})")


if __name__ == "__main__":
    main()
