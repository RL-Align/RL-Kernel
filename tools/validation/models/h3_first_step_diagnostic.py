#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Replay the first denoising step of a generation under one execution configuration.

The first transformer call of a generation sees the saved packed inputs (``initial.pt``)
and nothing has fed back yet, so the difference to the provider's saved ``step-000.pt``
is the model's own error on a real generation input. A configuration is the provider
(optionally with one forced SDPA backend or cuBLASLt), the candidate, or the candidate
with chosen block nodes run as their provider replay (``--provider-nodes``). Replaying
the provider default must reproduce ``step-000.pt`` bitwise; ``baseline_bitwise`` records
whether it did, which validates the replay itself. Diagnostic only: nothing here accepts
or rejects a candidate.

    python -m tools.validation.models.h3_first_step_diagnostic --checkpoint <dir> \\
        --generation <generation-reference dir> --seed 43 --mode candidate \\
        --provider-nodes attention --output out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from rl_engine.validation.models.h3_block import CandidateOps
from rl_engine.validation.models.h3_full_model import FullModelTrace, full_model_execution
from tools.validation.models.h3_full_generation import QueryChunkAttention
from tools.validation.models.h3_full_model import PROVIDER_VARIANTS, provider_variant_context


def errors(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """max_abs and relative_l2 of ``value`` against ``reference`` (float64 norms)."""

    delta = value.double() - reference.double()
    return dict(
        max_abs=float(delta.abs().max()),
        relative_l2=float(delta.norm() / reference.double().norm().clamp_min(1e-12)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["provider", "candidate"], required=True)
    parser.add_argument("--provider-variant", choices=PROVIDER_VARIANTS, default="default")
    parser.add_argument("--provider-nodes", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "candidate" and args.provider_variant != "default":
        parser.error("--provider-variant applies to --mode provider only")
    if args.mode == "provider" and args.provider_nodes:
        parser.error("--provider-nodes applies to --mode candidate only")
    nodes = tuple(n for n in args.provider_nodes.split(",") if n)

    from diffusers import MiniMaxH3Transformer3DModel

    directory = args.generation / str(args.seed)
    inputs = torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)
    saved = torch.load(directory / "step-000.pt", map_location="cpu", weights_only=True)
    contract = json.loads(args.contract.read_text())

    started = time.perf_counter()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        str(args.checkpoint / "transformer"), torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = model.to("cuda").eval()
    loaded = time.perf_counter() - started
    kwargs = {k: v.to("cuda") for k, v in inputs.items()}
    ops = None
    if args.mode == "candidate":
        ops = CandidateOps()
        ops.attention = QueryChunkAttention(ops.attention, contract["attention_query_chunk"])

    trace = FullModelTrace()
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        full_model_execution(
            model,
            candidate=args.mode == "candidate",
            trace=trace,
            ops=ops,
            provider_nodes=nodes,
        ),
        provider_variant_context(args.provider_variant),
    ):
        video, audio = model(**kwargs, attention_kwargs=None, return_dict=False)
        torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    video, audio = video.float().cpu(), audio.float().cpu()
    result = dict(
        seed=args.seed,
        mode=args.mode,
        provider_variant=args.provider_variant,
        provider_nodes=list(nodes),
        video=errors(video, saved[0]),
        audio=errors(audio, saved[1]),
        baseline_bitwise=bool(torch.equal(video, saved[0]) and torch.equal(audio, saved[1])),
        load_seconds=loaded,
        forward_seconds=forward_seconds,
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
