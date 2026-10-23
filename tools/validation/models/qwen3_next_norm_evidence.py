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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.backends.cuda.norm.rmsnorm import Qwen3NextRMSNormCudaOp  # noqa: E402
from rl_engine.reference.norm.qwen3_next_rms_norm import Qwen3NextRMSNormOp  # noqa: E402

EPS = 1e-6
HIDDEN = 2048  # Qwen3-Next hidden size (decoder and final norms)
TIMED_ROWS = (1024, 4096, 16384, 65536)


# --------------------------------------------------------------------------- #
# Candidates: name -> (forward fn(x, w) -> y, has_backward)
# --------------------------------------------------------------------------- #


def zero_centred_candidates() -> dict[str, dict[str, Any]]:
    cands: dict[str, dict[str, Any]] = {
        "rl-kernel CUDA": {"fn": Qwen3NextRMSNormCudaOp(), "backward": True},
        "rl-kernel PyTorch reference": {"fn": Qwen3NextRMSNormOp(), "backward": True},
    }
    try:
        from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

        module = Qwen3NextRMSNorm(HIDDEN, eps=EPS).cuda()

        def hf(x, w, module=module):
            return torch.func.functional_call(module, {"weight": w}, (x,))

        cands["transformers Qwen3NextRMSNorm"] = {"fn": hf, "backward": True}
    except ImportError:
        pass
    try:
        from vllm.config import VllmConfig, set_current_vllm_config

        with set_current_vllm_config(VllmConfig()):
            from vllm.model_executor.layers.layernorm import GemmaRMSNorm

            gemma = GemmaRMSNorm(HIDDEN, eps=EPS).cuda()

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


def zero_centred_golden(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x64 = x.double()
    return x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + EPS) * (1.0 + w.double())


# --------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------- #


def _inputs(rows: int, seed: int, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, device="cuda", generator=g) * 2).to(dtype)
    w = (torch.randn(HIDDEN, device="cuda", generator=g) * 0.1).to(dtype)
    up = torch.randn(rows, HIDDEN, device="cuda", generator=g).to(dtype)
    return x, w, up


def _grads(fn, x, w, up, dtype=None):
    xl = (x if dtype is None else x.to(dtype)).detach().clone().requires_grad_(True)
    wl = (w if dtype is None else w.to(dtype)).detach().clone().requires_grad_(True)
    out = fn(xl, wl)
    out.backward(up if dtype is None else up.to(dtype))
    return out.detach(), xl.grad, wl.grad


def accuracy(cands, golden, rows: int, seed: int) -> dict[str, Any]:
    x, w, up = _inputs(rows, seed)
    ref_out, ref_dx, ref_dw = _grads(golden, x, w, up, torch.float64)
    result = {}
    for name, c in cands.items():
        entry: dict[str, Any] = {}
        if c["backward"]:
            out, dx, dw = _grads(c["fn"], x, w, up)
            for key, got, ref in (("dx", dx, ref_dx), ("dweight", dw, ref_dw)):
                err = (got.double() - ref).abs().max().item()
                entry[f"{key}_max_abs_over_absmax"] = err / ref.abs().max().item()
        else:
            with torch.no_grad():
                out = c["fn"](x, w)
        err = (out.double() - ref_out).abs()
        entry["forward_max_abs"] = err.max().item()
        entry["forward_correctly_rounded_fraction"] = (
            (out == ref_out.to(out.dtype)).float().mean().item()
        )
        result[name] = entry
    return result


def row_invariance(cands, seeds=(3, 4, 5), rows: int = 4096, step: int = 16) -> dict[str, Any]:
    """256 rows computed alone vs the same rows inside a batch, bitwise."""

    result = {}
    for name, c in cands.items():
        fwd_bad = dx_bad = checked = 0
        for seed in seeds:
            x, w, up = _inputs(rows, seed)
            if c["backward"]:
                full_out, full_dx, _ = _grads(c["fn"], x, w, up)
            else:
                with torch.no_grad():
                    full_out = c["fn"](x, w)
            for i in range(0, rows, step):
                sl = slice(i, i + 1)
                if c["backward"]:
                    out, dx, _ = _grads(c["fn"], x[sl], w, up[sl])
                    dx_bad += not torch.equal(dx[0], full_dx[i])
                else:
                    with torch.no_grad():
                        out = c["fn"](x[sl], w)
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


def latency(cands, rows_list=TIMED_ROWS) -> dict[str, Any]:
    result: dict[str, Any] = {name: {} for name in cands}
    for rows in rows_list:
        x, w, up = _inputs(rows, seed=11)
        for name, c in cands.items():
            row: dict[str, float] = {}
            with torch.no_grad():
                row["forward_us"] = _time_us(lambda f=c["fn"]: f(x, w))
            if c["backward"]:
                xl = x.detach().clone().requires_grad_(True)
                wl = w.detach().clone().requires_grad_(True)
                out = c["fn"](xl, wl)
                row["backward_us"] = _time_us(
                    lambda o=out: torch.autograd.grad(o, (xl, wl), up, retain_graph=True)
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


def run_op(name: str, cands, golden) -> dict[str, Any]:
    print(f"[{name}] candidates: {', '.join(cands)}", flush=True)
    report = {
        "accuracy": {str(r): accuracy(cands, golden, r, seed=r) for r in (257, 4096)},
        "row_invariance": row_invariance(cands),
        "latency": latency(cands),
    }
    for cand, entry in report["row_invariance"].items():
        print(f"  BI {cand}: {entry}", flush=True)
    return report


def build_report(ops: dict[str, tuple[Callable, Callable]]) -> dict[str, Any]:
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "kind": "qwen3_next_norm_evidence",
        "rfc": "RL-Align/RL-Kernel#428",
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "environment": environment(),
        "hidden": HIDDEN,
        "eps": EPS,
        "dtype": "bfloat16",
        "ops": {},
    }
    for name, (make_cands, golden) in ops.items():
        report["ops"][name] = run_op(name, make_cands(), golden)
    return report


OPS = {"zero_centred_rmsnorm": (zero_centred_candidates, zero_centred_golden)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    report = build_report(OPS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} (commit {report['git_commit'][:7]}, dirty={report['git_dirty']})")


if __name__ == "__main__":
    main()
