#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
# ruff: noqa: E402
"""Train DSv4 C0/C4/C128 attention blocks with synthetic candidate rows on CUDA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_engine.kernels.dsv4.attention.mqa_joint_attention_sink import MqaJointAttentionSinkOp
from rl_engine.kernels.dsv4.attention.candidate_plan import CandidatePlan
from rl_engine.kernels.dsv4.attention.contract import (
    ATTENTION_SCALE,
    GROUP_FLAT_DIM,
    HEAD_DIM,
    HIDDEN_SIZE,
    N_O_PROJ_GROUPS,
    N_Q_HEADS,
    O_LORA_RANK,
    RECENT_WINDOW,
)
from rl_engine.kernels.dsv4.attention.cuda_runtime import ensure_t06_cuda_kernel
from rl_engine.kernels.dsv4.attention.four_mode import verify_recorded_four_modes
from rl_engine.kernels.dsv4.attention.o_proj.o_proj_grouped import OProjGroupedOp, _det_gemm_linear
from rl_engine.kernels.dsv4.attention.o_proj.rope_consumer import (
    apply_gptj_interleaved_partial,
    fixture_cos_sin,
)
from rl_engine.kernels.dsv4.attention.state_gate import synthetic_pass_verdict


LAYER_TABLE = ("C0", "C4", "C128", "C4")


def _pool_rows(x: torch.Tensor, cr: int) -> torch.Tensor:
    tokens = x.shape[0]
    n = tokens // cr
    if n == 0:
        return x.new_zeros((0, x.shape[-1]))
    return x[: n * cr].reshape(n, cr, -1).mean(dim=1)


def build_candidates(layer_type: str, k: torch.Tensor, v: torch.Tensor):
    tokens = k.shape[0]
    recent_n = min(RECENT_WINDOW, tokens)
    recent_k = k[-recent_n:]
    recent_v = v[-recent_n:]
    if layer_type == "C0":
        n_c = 0
        ck = k.new_zeros((0, HEAD_DIM))
        cv = v.new_zeros((0, HEAD_DIM))
    elif layer_type == "C4":
        ck = _pool_rows(k, 4)
        cv = _pool_rows(v, 4)
        n_c = ck.shape[0]
    elif layer_type == "C128":
        ck = _pool_rows(k, 128)
        cv = _pool_rows(v, 128)
        n_c = ck.shape[0]
    else:
        raise ValueError(layer_type)
    k_cat = torch.cat([ck, recent_k], dim=0)
    v_cat = torch.cat([cv, recent_v], dim=0)
    plan = CandidatePlan(
        layer_type=layer_type,
        n_compressed=n_c,
        n_recent=recent_n,
        valid=torch.ones(n_c + recent_n, dtype=torch.bool, device=k.device),
    )
    return k_cat, v_cat, plan


class DSv4AttentionLayer(nn.Module):
    def __init__(self, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.rms_w = nn.Parameter(torch.ones(HIDDEN_SIZE))
        self.w_q = nn.Linear(HIDDEN_SIZE, N_Q_HEADS * HEAD_DIM, bias=False)
        self.w_k = nn.Linear(HIDDEN_SIZE, HEAD_DIM, bias=False)
        self.w_v = nn.Linear(HIDDEN_SIZE, HEAD_DIM, bias=False)
        self.w_a = nn.Parameter(0.02 * torch.randn(
            N_O_PROJ_GROUPS, O_LORA_RANK, GROUP_FLAT_DIM, dtype=torch.bfloat16
        ))
        self.w_b = nn.Parameter(0.02 * torch.randn(
            HIDDEN_SIZE, N_O_PROJ_GROUPS * O_LORA_RANK, dtype=torch.bfloat16
        ))
        self.sink = nn.Parameter(torch.zeros(N_Q_HEADS))
        self.attn = MqaJointAttentionSinkOp(backend="cuda")
        self.o_proj = OProjGroupedOp(backend="det_gemm")
        self.state_gate = synthetic_pass_verdict(tag=f"min-model-{layer_type}")

    def recorded_inputs(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        normed = F.rms_norm(hidden, (HIDDEN_SIZE,), self.rms_w, 1e-6)
        tokens = hidden.shape[0]
        q = self.w_q(normed).view(tokens, N_Q_HEADS, HEAD_DIM)
        q = apply_gptj_interleaved_partial(q, cos, sin, inverse=False)
        k = self.w_k(normed)
        v = self.w_v(normed)
        k_cat, v_cat, plan = build_candidates(self.layer_type, k, v)
        return q, k_cat, v_cat, plan

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q, k_cat, v_cat, plan = self.recorded_inputs(hidden, cos, sin)
        o = self.attn.apply_autograd(
            q,
            k_cat,
            v_cat,
            self.sink,
            plan,
            output_fp32=True,
            compare=True,
            state_gate=self.state_gate,
        )
        y = self.o_proj.forward(o.to(torch.bfloat16), self.w_a, self.w_b, cos, sin).y.float()
        return hidden + y


class DSv4AttentionMiniModel(nn.Module):
    def __init__(self, layer_table=LAYER_TABLE):
        super().__init__()
        self.layers = nn.ModuleList(DSv4AttentionLayer(kind) for kind in layer_table)
        self.final_norm = nn.Parameter(torch.ones(HIDDEN_SIZE))

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = hidden
        for layer in self.layers:
            x = layer(x, cos, sin)
        return F.rms_norm(x, (HIDDEN_SIZE,), self.final_norm, 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json-out", type=Path, default=Path("/tmp/dsv4_attention_min_model.json"))
    args = parser.parse_args()
    if args.tokens < 2 or args.steps < 2:
        parser.error("verification requires --tokens >= 2 and --steps >= 2")
    if torch.device(args.device).type != "cuda":
        parser.error("the mini-model requires CUDA attention and DetGemm")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    pending = {
        "model": "DSv4",
        "task_id": "T06",
        "experiment": "dsv4_attention_min_model",
        "status": "RUNNING",
        "four_mode_equal": False,
        "cuda_graph_status": "NOT_RUN",
    }
    args.json_out.write_text(json.dumps(pending, indent=2) + "\n")
    try:
        return _run(args)
    except Exception as exc:
        pending.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        args.json_out.write_text(json.dumps(pending, indent=2) + "\n")
        raise


def _run(args) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    cuda_source = ensure_t06_cuda_kernel() if device.type == "cuda" else "cpu"
    if device.type == "cuda":
        _det_gemm_linear()

    model = DSv4AttentionMiniModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tokens = args.tokens
    positions = torch.arange(tokens, device=device)
    cos, sin = fixture_cos_sin(positions)
    hidden = 0.02 * torch.randn(tokens, HIDDEN_SIZE, device=device)
    target = torch.zeros_like(hidden)

    print(
        f"[dsv4-attention-model] device={device} cuda_source={cuda_source} "
        f"layers={LAYER_TABLE} T={tokens} scale={ATTENTION_SCALE:.6f}",
        flush=True,
    )

    losses = []
    t0 = time.perf_counter()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(hidden, cos, sin)
        loss = F.mse_loss(out, target)
        loss.backward()
        finite = all(
            p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()
        )
        if not finite or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite at step {step} loss={float(loss)}")
        opt.step()
        losses.append(float(loss.detach()))
        print(f"[dsv4-attention-model] step={step} loss={losses[-1]:.6f}", flush=True)

    del opt
    model.zero_grad(set_to_none=True)
    model.eval()
    mode_reports = []
    x = hidden
    for index, layer in enumerate(model.layers):
        with torch.no_grad():
            q, k, v, plan = layer.recorded_inputs(x, cos, sin)
        try:
            mode_report = verify_recorded_four_modes(
                q, k, v, layer.sink, plan, layer.w_a, layer.w_b,
                cos, sin, layer.state_gate,
            )
        except Exception as exc:
            mode_reports.append({
                "layer": index, "status": "FAIL", "four_mode_equal": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            # Capture failures can invalidate the CUDA context.
            break
        mode_report["layer"] = index
        mode_reports.append(mode_report)
        print(
            f"[dsv4-attention-model] four_mode layer={index} status={mode_report['status']}",
            flush=True,
        )
        if index + 1 < len(model.layers):
            with torch.no_grad():
                x = layer(x, cos, sin)
    four_mode_equal = len(mode_reports) == len(model.layers) and all(
        r["four_mode_equal"] for r in mode_reports
    )

    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "model": "DSv4",
        "task_id": "T06",
        "experiment": "dsv4_attention_min_model",
        "device": str(device),
        "cuda_source": cuda_source,
        "layer_table": list(LAYER_TABLE),
        "tokens": tokens,
        "steps": args.steps,
        "n_params": n_params,
        "losses": losses,
        "loss_dropped": bool(losses[-1] < losses[0]),
        "status": "PASS" if four_mode_equal and losses[-1] < losses[0] else "FAIL",
        "cuda_graph_status": "PASS" if four_mode_equal else "FAIL",
        "four_mode_equal": four_mode_equal,
        "four_mode_scope": (
            "per-layer recorded attention + inverse RoPE + o-proj, including graph backward"
        ),
        "four_mode_layers": mode_reports,
        "elapsed_sec": elapsed,
        "output_shape": [tokens, HIDDEN_SIZE],
        "note": (
            "synthetic C4/C128 rows; frozen candidate state; "
            "no live cache/compressor or whole-model graph"
        ),
    }
    args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not four_mode_equal or not report["loss_dropped"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
