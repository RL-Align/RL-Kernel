#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Evidence for the generic fused-logp backward (#174): time, memory, accuracy, invariance.

Four backward paths for selected-token logprobs at Qwen vocab size (151936), BF16:

* ``torch log_softmax + gather``: plain autograd, the existing implementation;
* ``previous VJP``: this op's VJP before #174 (FP32 softmax over the full ``[N, V]``);
* ``chunked fallback``: the bounded row-chunked FP32 VJP, kept for ``_C`` builds
  without the fused kernel;
* ``fused kernel``: ``_C.fused_logp_backward``.

The last three share ``_C.fused_logp`` for the forward. Writes one JSON report:

    python benchmarks/fused_logp_backward_evidence.py --out report.json
    python benchmarks/plot_fused_logp_backward_evidence.py report.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import rl_engine.kernels.ops.cuda.loss.logp as logp_mod  # noqa: E402
from rl_engine.kernels.ops.base import _C  # noqa: E402

VOCAB = 151936
TIMED_ROWS = (8192, 16384, 32768)
CHUNKED = logp_mod.fused_logp_backward_chunked


class _ForwardOnly:
    """A ``_C`` stand-in without ``fused_logp_backward``: selects the Python VJP."""

    fused_logp = staticmethod(_C.fused_logp)


def _previous_vjp(logits, labels, grad_output, out_dtype, **_):
    """The VJP before #174: FP32 softmax over the whole ``[N, V]`` at once."""

    probs = torch.softmax(logits.float(), dim=-1)
    rows = torch.arange(logits.size(0), device=logits.device)
    probs[rows, labels] -= 1.0
    return (-grad_output.reshape(-1, 1).float() * probs).to(out_dtype)


@contextlib.contextmanager
def _vjp(vjp):
    """The autograd VJP reads the module attribute at backward time, so patch it
    for the whole forward + backward."""

    logp_mod.fused_logp_backward_chunked = vjp
    try:
        yield
    finally:
        logp_mod.fused_logp_backward_chunked = CHUNKED


def _autograd_fn(backend):
    return lambda logits, labels: logp_mod._FusedLogpAutograd.apply(logits, labels, backend)


def _torch_logp(logits, labels):
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels[:, None]).squeeze(-1)


#: name -> (forward fn, VJP the autograd node uses when the backend has no fused kernel)
CANDIDATES: dict[str, tuple[Callable, Callable]] = {
    "torch log_softmax + gather": (_torch_logp, CHUNKED),
    "previous VJP (full FP32 softmax)": (_autograd_fn(_ForwardOnly()), _previous_vjp),
    "chunked fallback": (_autograd_fn(_ForwardOnly()), CHUNKED),
    "fused kernel": (_autograd_fn(_C), CHUNKED),
}


def _inputs(rows: int, seed: int, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = (torch.randn(rows, VOCAB, device="cuda", generator=g) * 2).to(dtype)
    labels = torch.randint(0, VOCAB, (rows,), device="cuda", generator=g)
    grad = torch.randn(rows, device="cuda", generator=g)
    return logits, labels, grad


def _step(cand, logits, labels, grad):
    fn, vjp = cand
    x = logits.detach().requires_grad_(True)
    with _vjp(vjp):
        out = fn(x, labels)
        out.backward(grad.to(out.dtype))
    return out.detach(), x.grad


def accuracy(rows: int = 257, seed: int = 1) -> dict[str, Any]:
    logits, labels, grad = _inputs(rows, seed)
    x64 = logits.double()
    probs = torch.softmax(x64, dim=-1)
    ref_out = torch.log_softmax(x64, dim=-1).gather(-1, labels[:, None]).squeeze(-1)
    one_hot = torch.zeros_like(probs).scatter_(1, labels[:, None], 1.0)
    result = {}
    for name, fn in CANDIDATES.items():
        out, g = _step(fn, logits, labels, grad)
        # The upstream gradient reaches the VJP in the forward output's dtype.
        ref_grad = grad.to(out.dtype).double()[:, None] * (one_hot - probs)
        err = (g.double() - ref_grad).abs()
        result[name] = {
            "logp_max_abs": (out.double() - ref_out).abs().max().item(),
            "grad_max_abs": err.max().item(),
            "grad_max_abs_over_absmax": err.max().item() / ref_grad.abs().max().item(),
            "grad_correctly_rounded_fraction": (g == ref_grad.to(g.dtype)).float().mean().item(),
        }
    return result


def row_invariance(rows: int = 4096, step: int = 16, seeds=(3, 4, 5)) -> dict[str, Any]:
    """256 rows computed alone vs the same rows inside a batch, bitwise."""

    result = {}
    for name, fn in CANDIDATES.items():
        out_bad = grad_bad = checked = 0
        for seed in seeds:
            logits, labels, grad = _inputs(rows, seed)
            full_out, full_grad = _step(fn, logits, labels, grad)
            for i in range(0, rows, step):
                sl = slice(i, i + 1)
                out, g = _step(fn, logits[sl], labels[sl], grad[sl])
                out_bad += not torch.equal(out[0], full_out[i])
                grad_bad += not torch.equal(g[0], full_grad[i])
                checked += 1
            del logits, full_grad
        result[name] = {
            "rows_checked": checked,
            "logp_rows_differing": out_bad,
            "grad_rows_differing": grad_bad,
            "batch_rows": rows,
            "seeds": list(seeds),
        }
    return result


def _time_ms(fn: Callable[[], Any], warmup: int = 2, iters: int = 5) -> float:
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
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def performance() -> dict[str, Any]:
    result: dict[str, Any] = {name: {} for name in CANDIDATES}
    for rows in TIMED_ROWS:
        logits, labels, grad = _inputs(rows, seed=11)
        for name, fn in CANDIDATES.items():

            def step(fn=fn, logits=logits, labels=labels, grad=grad):
                _, g = _step(fn, logits, labels, grad)
                return g

            row = {"step_ms": _time_ms(step)}
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            g = step()
            torch.cuda.synchronize()
            row["peak_gib"] = (torch.cuda.max_memory_allocated() - base) / 2**30
            del g
            x = logits.detach().requires_grad_(True)
            with _vjp(fn[1]):
                out = fn[0](x, labels)
                go = grad.to(out.dtype)
                row["backward_ms"] = _time_ms(
                    lambda o=out, x=x, go=go: torch.autograd.grad(o, x, go, retain_graph=True)
                )
            del x, out
            torch.cuda.empty_cache()
            result[name][str(rows)] = row
            print(f"  N={rows} {name}: {row}", flush=True)
        del logits
        torch.cuda.empty_cache()
    return result


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not hasattr(_C, "fused_logp_backward"):
        raise SystemExit("needs a CUDA device and a _C build with fused_logp_backward")
    report = {
        "kind": "fused_logp_backward_evidence",
        "issue": "RL-Align/RL-Kernel#174",
        "git_commit": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "vocab": VOCAB,
        "dtype": "bfloat16",
        "accuracy": accuracy(),
        "row_invariance": row_invariance(),
        "performance": performance(),
    }
    for name, entry in report["row_invariance"].items():
        print(f"BI {name}: {entry}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out} (commit {report['git_commit'][:7]}, dirty={report['git_dirty']})")


if __name__ == "__main__":
    main()
