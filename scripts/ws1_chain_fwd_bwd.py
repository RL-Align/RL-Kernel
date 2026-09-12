#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C9 (#275): one-command full Qwen3-8B Dense fwd+bwd.

Assembly runnable only. Model-level EXIT requires C10 + C11.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import subprocess
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.alignment.qwen3_dense import Qwen3DenseSpec  # noqa: E402
from rl_engine.kernels.gtest.accelerator import (  # noqa: E402
    compute_capability,
    disable_tf32,
    manual_seed_all,
    resolve_device,
)
from rl_engine.kernels.gtest.chain_gate import build_model  # noqa: E402
from rl_engine.testing.ws1_workload import (  # noqa: E402
    apply_padding,
    build_logical_batch,
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WS1 C9 full Qwen3-8B Dense fwd+bwd")
    parser.add_argument(
        "--backend-profile",
        choices=("cuda_bf16", "triton_cuda_bf16", "ascend_bf16"),
        required=True,
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Defaults to the backend profile's own accelerator (cuda or npu).",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16",))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--weights",
        choices=("hf", "synthetic"),
        default="hf",
        help="hf = pinned snapshot (EXIT path). synthetic = official-shape wiring only.",
    )
    parser.add_argument("--weights-path", default=None, help="Directory of Qwen3-8B safetensors")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device, profile=args.backend_profile)
    except RuntimeError as exc:
        print(f"ERROR: C9 fwd+bwd needs a real device: {exc}", file=sys.stderr)
        return 2
    disable_tf32(device.type)
    manifest = load_manifest()
    execution_seed = manifest.seed if args.seed is None else int(args.seed)
    manual_seed_all(device.type, execution_seed)
    spec = Qwen3DenseSpec.from_manifest(manifest)
    log_stream = sys.stderr if args.json else sys.stdout
    with contextlib.redirect_stdout(log_stream):
        model = build_model(
            backend_profile=args.backend_profile,
            weights_mode=args.weights,
            weights_path=args.weights_path,
            device=device,
            dtype=torch.bfloat16,
            manifest=manifest,
        )
    batch = build_logical_batch(manifest)
    padded = apply_padding(batch, pad_side="right", manifest=manifest)
    input_ids = torch.tensor(padded.physical_token_ids, device=device, dtype=torch.long)
    attn = torch.tensor(padded.physical_attention_mask, device=device, dtype=torch.bool)
    pos = torch.tensor(padded.physical_position_ids, device=device, dtype=torch.long)
    loss_mask = torch.tensor(padded.physical_loss_mask, device=device, dtype=torch.bool)
    for tensor in model.weights.tensors.values():
        if tensor.is_floating_point():
            tensor.requires_grad_(True)
    with contextlib.redirect_stdout(log_stream):
        out = model.forward(
            input_ids,
            attention_mask=attn,
            position_ids=pos,
            target_ids=input_ids,
            loss_mask=loss_mask,
            capture_nodes=True,
        )
        loss = out["loss"]
        loss.backward()
    payload = {
        "disclaimer": "C9 assembly runnable only; model-level EXIT requires C10 + C11",
        "backend_profile": args.backend_profile,
        "workload_id": manifest.workload_id,
        "config_fingerprint": spec.__dict__,
        "weight_source": model.weights.source,
        "weight_hash": model.weights.content_hash,
        "loss": float(loss.detach().float().cpu()),
        "logits_shape": list(out["logits"].shape),
        "n_nodes": len(spec.node_names()),
        "captured_nodes": sorted(model.captured_node_outputs()),
        "provenance": model.profile_ops.provenance,
        "runtime_backend_observations": (model.profile_ops.validated_runtime_observations()),
        "device": str(device),
        "cc": compute_capability(device),
        "seed": execution_seed,
        "workload_seed": manifest.seed,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"C9 fwd+bwd ok profile={args.backend_profile} loss={payload['loss']:.6f} "
            f"layers={spec.num_hidden_layers} nodes={payload['n_nodes']} "
            f"weights={model.weights.source}"
        )
        print("assembly runnable only; model-level EXIT requires C10 + C11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
